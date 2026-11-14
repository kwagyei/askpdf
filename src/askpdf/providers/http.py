"""The only network code in askpdf: JSON over HTTP with urllib, nothing else.

Two calls cover every provider: ``post_json`` for one request and one JSON reply, and
``stream_lines`` for replies that arrive a line at a time (Ollama's NDJSON, SSE).
"""

import contextlib
import http.client
import ipaddress
import json
import socket
import threading
import urllib.error
import urllib.request
from collections.abc import Iterator, Mapping
from typing import Any
from urllib.parse import urlsplit

TIMEOUT = 120.0  # seconds; embedding a batch on a CPU can take a while


class ProviderError(Exception):
    """The provider could not be reached, or answered with an error."""


class RemoteNotAllowedError(ProviderError):
    """The endpoint is not on this machine and remote use was not allowed."""


class UnreachableError(ProviderError):
    """Nothing answered: the provider is not running, or the network is down."""


class _NoRedirects(urllib.request.HTTPRedirectHandler):
    """Refuse redirects: a redirect could take document text, or the API key in the
    Authorization header, to a host the remote guard never approved."""

    def redirect_request(self, *args: object, **kwargs: object) -> None:
        return None  # urllib then raises HTTPError for the 3xx answer


# --- cutting a stream from another thread -----------------------------------------------
# A browser can leave while a worker thread is blocked reading the model's reply, before
# its first byte even (Ollama sends nothing until the prompt is read). Only shutting the
# socket from outside wakes that read, so streams record their socket per thread.

_LOCK = threading.Lock()
_SOCKETS: dict[int, socket.socket] = {}  # thread id -> socket of the stream it reads
_CUT: set[int] = set()  # threads whose stream was cut, connected yet or not
_streaming = threading.local()


def _register(sock: socket.socket) -> None:
    if not getattr(_streaming, "on", False):
        return
    me = threading.get_ident()
    with _LOCK:
        _SOCKETS[me] = sock
        cut = me in _CUT
    if cut:  # cut while it was still connecting
        _shut(sock)


def _shut(sock: socket.socket) -> None:
    with contextlib.suppress(OSError):
        sock.shutdown(socket.SHUT_RDWR)


def cut_stream(thread_id: int) -> None:
    """Break off the model stream that ``thread_id`` reads or is about to open: a read
    blocked in that thread returns, and a stream it opens later fails at once."""
    with _LOCK:
        _CUT.add(thread_id)
        sock = _SOCKETS.get(thread_id)
    if sock is not None:
        _shut(sock)


def forget_cut(thread_id: int) -> None:
    """Let ``thread_id`` stream again, once whatever it was doing has been closed."""
    with _LOCK:
        _CUT.discard(thread_id)
        _SOCKETS.pop(thread_id, None)


class _TrackedHTTPConnection(http.client.HTTPConnection):
    def connect(self) -> None:
        super().connect()
        _register(self.sock)


class _TrackedHTTPSConnection(http.client.HTTPSConnection):
    def connect(self) -> None:
        super().connect()
        _register(self.sock)


class _TrackedHTTP(urllib.request.HTTPHandler):
    def http_open(self, req: urllib.request.Request) -> http.client.HTTPResponse:
        return self.do_open(_TrackedHTTPConnection, req)


class _TrackedHTTPS(urllib.request.HTTPSHandler):
    def https_open(self, req: urllib.request.Request) -> http.client.HTTPResponse:
        return self.do_open(_TrackedHTTPSConnection, req, context=self._context)  # type: ignore[attr-defined]


# No proxies either: an http_proxy variable would send "local" traffic, excerpts and
# all, through whatever host it names.
_OPENER = urllib.request.build_opener(
    urllib.request.ProxyHandler({}), _NoRedirects(), _TrackedHTTP(), _TrackedHTTPS()
)
_FAILURES = (urllib.error.URLError, OSError, http.client.HTTPException, UnicodeDecodeError)


def is_local(url: str) -> bool:
    """True if ``url`` points at this machine (by name or loopback address)."""
    host = (urlsplit(url).hostname or "").rstrip(".").lower()
    if host == "localhost":
        return True
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return False
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        return ip.ipv4_mapped.is_loopback
    return ip.is_loopback


def require_local_or_allowed(url: str, allow_remote: bool) -> None:
    """Refuse to send document text off this machine unless the user allowed it."""
    if urlsplit(url).scheme not in {"http", "https"}:
        raise ProviderError(f"unsupported URL scheme {urlsplit(url).scheme!r}")
    if not is_local(url) and not allow_remote:
        host = urlsplit(url).hostname
        raise RemoteNotAllowedError(
            f"{host} is not this machine; pass --allow-remote or set ASKPDF_ALLOW_REMOTE=1"
            " to send document excerpts there"
        )


def _request(url: str, payload: Mapping[str, Any], headers: Mapping[str, str]) -> Any:
    if urlsplit(url).scheme not in {"http", "https"}:
        raise ProviderError(f"unsupported URL scheme {urlsplit(url).scheme!r}")
    data = json.dumps(payload).encode("utf-8")
    all_headers = {"Content-Type": "application/json", **headers}
    return urllib.request.Request(url, data=data, headers=all_headers, method="POST")  # noqa: S310 - scheme checked above


def where(url: str) -> str:
    """Host and port only: never userinfo, which may hold a password."""
    parts = urlsplit(url)
    host = f"[{parts.hostname}]" if ":" in (parts.hostname or "") else str(parts.hostname)
    return f"{host}:{parts.port}" if parts.port else host


def _redact(text: str, headers: Mapping[str, str]) -> str:
    for value in headers.values():
        for secret in {value, value.rsplit(" ", 1)[-1]}:
            if len(secret) >= 8:
                text = text.replace(secret, "[redacted]")
    return text


def _describe(url: str, exc: Exception, headers: Mapping[str, str]) -> ProviderError:
    # Request headers may carry an API key: they are never shown, and an error body that
    # echoes them is redacted.
    if isinstance(exc, urllib.error.HTTPError):
        detail = exc.read(500).decode("utf-8", "replace").strip()
        if 300 <= exc.code < 400:
            detail = "redirects are not followed"
        return ProviderError(f"{where(url)} answered HTTP {exc.code}: {_redact(detail, headers)}")
    if isinstance(exc, urllib.error.URLError | ConnectionError | TimeoutError):
        reason = getattr(exc, "reason", exc)
        return UnreachableError(f"cannot reach {where(url)}: {reason}")
    return ProviderError(f"bad answer from {where(url)}: {type(exc).__name__}")


def post_json(
    url: str,
    payload: Mapping[str, Any],
    headers: Mapping[str, str] | None = None,
    timeout: float = TIMEOUT,
) -> Any:
    headers = headers or {}
    try:
        with _OPENER.open(_request(url, payload, headers), timeout=timeout) as r:
            return json.load(r)
    except _FAILURES as exc:
        raise _describe(url, exc, headers) from exc
    except ValueError as exc:  # not JSON
        raise ProviderError(f"{where(url)} did not answer with JSON") from exc


def get_json(url: str, timeout: float = 10.0) -> Any:
    if urlsplit(url).scheme not in {"http", "https"}:
        raise ProviderError(f"unsupported URL scheme {urlsplit(url).scheme!r}")
    try:
        with _OPENER.open(url, timeout=timeout) as r:
            return json.load(r)
    except _FAILURES as exc:
        raise _describe(url, exc, {}) from exc
    except ValueError as exc:
        raise ProviderError(f"{where(url)} did not answer with JSON") from exc


def stream_lines(
    url: str,
    payload: Mapping[str, Any],
    headers: Mapping[str, str] | None = None,
    timeout: float = TIMEOUT,
) -> Iterator[str]:
    """Yield the reply's non-empty lines as they arrive; closing the iterator closes the
    connection. ``timeout`` bounds each wait for data, not the whole answer."""
    headers = headers or {}
    me = threading.get_ident()
    with _LOCK:
        if me in _CUT:
            raise ProviderError(f"the answer from {where(url)} was cancelled")
    _streaming.on = True
    try:
        response = _OPENER.open(_request(url, payload, headers), timeout=timeout)
    except _FAILURES as exc:
        raise _describe(url, exc, headers) from exc
    finally:
        _streaming.on = False
    with response:
        try:
            for raw in response:
                line = raw.decode("utf-8").strip()
                if line:
                    yield line
            # Reading line by line, http.client ends quietly when the connection closes
            # before the promised Content-Length; a body cut short is an error.
            if getattr(response, "length", None):
                raise http.client.IncompleteRead(b"", response.length)
        except _FAILURES as exc:
            raise ProviderError(f"the answer from {where(url)} broke off") from exc
