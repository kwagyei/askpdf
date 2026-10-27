"""The only network code in askpdf: JSON over HTTP with urllib, nothing else.

Two calls cover every provider: ``post_json`` for one request and one JSON reply, and
``stream_lines`` for replies that arrive a line at a time (Ollama's NDJSON, SSE).
"""

import http.client
import ipaddress
import json
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


# No proxies either: an http_proxy variable would send "local" traffic, excerpts and
# all, through whatever host it names.
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirects())
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
        raise ProviderError(f"unsupported URL scheme in {url!r}")
    if not is_local(url) and not allow_remote:
        host = urlsplit(url).hostname
        raise RemoteNotAllowedError(
            f"{host} is not this machine; pass --allow-remote or set ASKPDF_ALLOW_REMOTE=1"
            " to send document excerpts there"
        )


def _request(url: str, payload: Mapping[str, Any], headers: Mapping[str, str]) -> Any:
    if urlsplit(url).scheme not in {"http", "https"}:
        raise ProviderError(f"unsupported URL scheme in {url!r}")
    data = json.dumps(payload).encode("utf-8")
    all_headers = {"Content-Type": "application/json", **headers}
    return urllib.request.Request(url, data=data, headers=all_headers, method="POST")  # noqa: S310 - scheme checked above


def _where(url: str) -> str:
    """Host and port only: never userinfo, which may hold a password."""
    parts = urlsplit(url)
    return f"{parts.hostname}:{parts.port}" if parts.port else str(parts.hostname)


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
        return ProviderError(f"{_where(url)} answered HTTP {exc.code}: {_redact(detail, headers)}")
    if isinstance(exc, urllib.error.URLError | ConnectionError | TimeoutError):
        reason = getattr(exc, "reason", exc)
        return UnreachableError(f"cannot reach {_where(url)}: {reason}")
    return ProviderError(f"bad answer from {_where(url)}: {type(exc).__name__}")


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
        raise ProviderError(f"{_where(url)} did not answer with JSON") from exc


def get_json(url: str, timeout: float = 10.0) -> Any:
    if urlsplit(url).scheme not in {"http", "https"}:
        raise ProviderError(f"unsupported URL scheme in {url!r}")
    try:
        with _OPENER.open(url, timeout=timeout) as r:
            return json.load(r)
    except _FAILURES as exc:
        raise _describe(url, exc, {}) from exc
    except ValueError as exc:
        raise ProviderError(f"{_where(url)} did not answer with JSON") from exc


def stream_lines(
    url: str,
    payload: Mapping[str, Any],
    headers: Mapping[str, str] | None = None,
    timeout: float = TIMEOUT,
) -> Iterator[str]:
    """Yield the reply's non-empty lines as they arrive; closing the iterator closes the
    connection. ``timeout`` bounds each wait for data, not the whole answer."""
    headers = headers or {}
    try:
        response = _OPENER.open(_request(url, payload, headers), timeout=timeout)
    except _FAILURES as exc:
        raise _describe(url, exc, headers) from exc
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
            raise ProviderError(f"the answer from {_where(url)} broke off") from exc
