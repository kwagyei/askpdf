"""Test-wide guards and settings."""

import hashlib
import io
import ipaddress
import os
import socket
from collections.abc import Iterator
from pathlib import Path
from typing import Any, BinaryIO

import pytest
from hypothesis import settings

from askpdf.extract.base import RawPage
from askpdf.extract.pypdf_ext import PypdfExtractor

# CI must be deterministic: same examples on every run, no example database, and no
# per-example deadline (shared runners are too noisy for timing to mean anything).
settings.register_profile("ci", derandomize=True, database=None, deadline=None, print_blob=True)
settings.register_profile("dev", max_examples=200)
_in_ci = os.environ.get("CI", "").lower() in {"1", "true"}
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "ci" if _in_ci else "dev"))


class NetworkBlockedError(RuntimeError):
    """Raised when a test tries to reach a non-loopback address."""


def is_loopback_host(host: object) -> bool:
    """True for names and literals that can only mean this machine."""
    if isinstance(host, bytes):
        host = host.decode("ascii", "replace")
    if not isinstance(host, str):
        return False
    if host.lower().rstrip(".") == "localhost":
        return True
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return False
    # Decided here rather than left to the stdlib, whose answer for IPv4-mapped
    # addresses has changed between Python versions.
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        return ip.ipv4_mapped.is_loopback
    return ip.is_loopback


def is_loopback_address(address: Any) -> bool:
    """True for socket addresses that never leave the machine."""
    if isinstance(address, tuple) and address:
        return is_loopback_host(address[0])
    # AF_UNIX paths never leave the machine.
    return isinstance(address, str | bytes)


def _check(address: Any) -> None:
    if not is_loopback_address(address):
        raise NetworkBlockedError(f"test tried to reach {address!r}")


_real_connect = socket.socket.connect
_real_connect_ex = socket.socket.connect_ex
_real_sendto = socket.socket.sendto
_real_getaddrinfo = socket.getaddrinfo


def _guarded_connect(self: socket.socket, address: Any) -> None:
    _check(address)
    _real_connect(self, address)


def _guarded_connect_ex(self: socket.socket, address: Any) -> int:
    _check(address)
    return _real_connect_ex(self, address)


def _guarded_sendto(self: socket.socket, data: Any, *args: Any) -> int:
    # sendto(data, address) or sendto(data, flags, address): the address is always last.
    _check(args[-1])
    return _real_sendto(self, data, *args)


def _guarded_getaddrinfo(host: Any, *args: Any, **kwargs: Any) -> Any:
    if host is not None and not is_loopback_host(host):
        raise NetworkBlockedError(f"test tried to resolve {host!r}")
    return _real_getaddrinfo(host, *args, **kwargs)


_GUARDS: list[tuple[Any, str, Any]] = [
    (socket.socket, "connect", _guarded_connect),
    (socket.socket, "connect_ex", _guarded_connect_ex),
    (socket.socket, "sendto", _guarded_sendto),
    (socket, "getaddrinfo", _guarded_getaddrinfo),
]
if hasattr(socket.socket, "sendmsg"):  # not on Windows

    def _guarded_sendmsg(self: socket.socket, buffers: Any, *args: Any) -> int:
        if len(args) >= 3:
            _check(args[2])
        return _real_sendmsg(self, buffers, *args)

    _real_sendmsg = socket.socket.sendmsg
    _GUARDS.append((socket.socket, "sendmsg", _guarded_sendmsg))

_session_patch = pytest.MonkeyPatch()


def pytest_configure(config: pytest.Config) -> None:
    """Block the network for the whole run: imports, collection and every fixture scope.

    No test may reach a model, arXiv or a remote API; they are faked. Loopback stays
    open for the local HTTP servers that provider tests start. One known gap: on
    Windows, asyncio's proactor loop connects through ConnectEx and bypasses
    socket.connect. askpdf's own HTTP client is synchronous (urllib), so this does
    not affect it.
    """
    for target, name, guard in _GUARDS:
        _session_patch.setattr(target, name, guard)


def pytest_unconfigure(config: pytest.Config) -> None:
    _session_patch.undo()


@pytest.fixture(autouse=True)
def _no_local_ollama(monkeypatch: pytest.MonkeyPatch) -> None:
    """Point askpdf at a closed port, so no test depends on an Ollama running here.

    Port 9 (discard) refuses at once. Provider tests start their own local server and
    set ASKPDF_OLLAMA_URL to it.
    """
    monkeypatch.setenv("ASKPDF_OLLAMA_URL", "http://127.0.0.1:9")
    monkeypatch.delenv("ASKPDF_ALLOW_REMOTE", raising=False)
    monkeypatch.delenv("ASKPDF_API_KEY", raising=False)


@pytest.fixture(autouse=True)
def _allow_network_if_marked(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> Iterator[None]:
    """Lift the guard for a single test marked ``@pytest.mark.network``."""
    if request.node.get_closest_marker("network"):
        monkeypatch.setattr(socket.socket, "connect", _real_connect)
        monkeypatch.setattr(socket.socket, "connect_ex", _real_connect_ex)
        monkeypatch.setattr(socket.socket, "sendto", _real_sendto)
        monkeypatch.setattr(socket, "getaddrinfo", _real_getaddrinfo)
        if hasattr(socket.socket, "sendmsg"):
            monkeypatch.setattr(socket.socket, "sendmsg", _real_sendmsg)
    yield


# --- shared extraction of the evaluation corpus ---------------------------------------------

_EVAL_CORPUS = Path(__file__).resolve().parent.parent / "eval" / "corpus"
_corpus_digests = (
    {hashlib.sha256(p.read_bytes()).digest() for p in _EVAL_CORPUS.glob("*.pdf")}
    if _EVAL_CORPUS.is_dir()
    else set()
)
_extracted: dict[bytes, list[RawPage]] = {}


@pytest.fixture(autouse=True)
def _reuse_corpus_extraction(monkeypatch: pytest.MonkeyPatch) -> None:
    """Extract each evaluation-corpus PDF once per test run, not once per test.

    Many evaluation tests index the whole corpus (129 NIST pages take seconds). Only files
    whose bytes match a committed corpus PDF are cached, so a test that fakes pypdf's
    behaviour on its own fixtures still sees the real extractor. The pages are frozen
    dataclasses, so sharing them between tests is safe.
    """
    real = PypdfExtractor.extract

    def extract(self: PypdfExtractor, source: Path | BinaryIO) -> list[RawPage]:
        data: bytes | None = None
        if isinstance(source, io.BytesIO):
            data = source.getvalue()
        elif isinstance(source, Path) and source.is_file():
            data = source.read_bytes()
        if data is None:  # missing, unreadable or streamed: the real extractor decides
            return real(self, source)
        digest = hashlib.sha256(data).digest()
        if digest not in _corpus_digests:
            return real(self, source)
        if digest not in _extracted:
            _extracted[digest] = real(self, io.BytesIO(data))
        return list(_extracted[digest])

    monkeypatch.setattr(PypdfExtractor, "extract", extract)
