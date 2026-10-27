"""The only network code in askpdf: JSON over HTTP with urllib, nothing else.

Two calls cover every provider: ``post_json`` for one request and one JSON reply, and
``stream_lines`` for replies that arrive a line at a time (Ollama's NDJSON, SSE).
"""

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
    data = json.dumps(payload).encode("utf-8")
    all_headers = {"Content-Type": "application/json", **headers}
    return urllib.request.Request(url, data=data, headers=all_headers, method="POST")  # noqa: S310 - scheme checked by callers


def _describe(url: str, exc: Exception) -> str:
    # Never include request headers: they may carry an API key.
    if isinstance(exc, urllib.error.HTTPError):
        detail = exc.read(500).decode("utf-8", "replace").strip()
        return f"{urlsplit(url).netloc} answered HTTP {exc.code}: {detail}"
    reason = getattr(exc, "reason", exc)
    return f"cannot reach {urlsplit(url).netloc}: {reason}"


def post_json(
    url: str,
    payload: Mapping[str, Any],
    headers: Mapping[str, str] | None = None,
    timeout: float = TIMEOUT,
) -> Any:
    try:
        with urllib.request.urlopen(_request(url, payload, headers or {}), timeout=timeout) as r:  # noqa: S310
            return json.load(r)
    except (TimeoutError, urllib.error.URLError, OSError) as exc:
        raise ProviderError(_describe(url, exc)) from exc
    except ValueError as exc:  # not JSON
        raise ProviderError(f"{urlsplit(url).netloc} did not answer with JSON") from exc


def get_json(url: str, timeout: float = 10.0) -> Any:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:  # noqa: S310
            return json.load(r)
    except (urllib.error.URLError, OSError) as exc:
        raise ProviderError(_describe(url, exc)) from exc
    except ValueError as exc:
        raise ProviderError(f"{urlsplit(url).netloc} did not answer with JSON") from exc


def stream_lines(
    url: str,
    payload: Mapping[str, Any],
    headers: Mapping[str, str] | None = None,
    timeout: float = TIMEOUT,
) -> Iterator[str]:
    """Yield the reply's non-empty lines as they arrive; closing the iterator closes the
    connection."""
    try:
        response = urllib.request.urlopen(_request(url, payload, headers or {}), timeout=timeout)  # noqa: S310
    except (urllib.error.URLError, OSError) as exc:
        raise ProviderError(_describe(url, exc)) from exc
    with response:
        try:
            for raw in response:
                line = raw.decode("utf-8").strip()
                if line:
                    yield line
        except (TimeoutError, OSError) as exc:
            raise ProviderError(f"connection to {urlsplit(url).netloc} broke: {exc}") from exc
