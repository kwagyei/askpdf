import socket
import threading

import pytest
from hypothesis import given
from hypothesis import strategies as st

from tests.conftest import NetworkBlockedError, _is_loopback


def test_outbound_connection_is_blocked() -> None:
    with socket.socket() as s, pytest.raises(NetworkBlockedError):
        s.connect(("192.0.2.1", 80))  # TEST-NET-1, never routable


def test_loopback_connection_is_allowed() -> None:
    with socket.socket() as server:
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        port = server.getsockname()[1]
        accepted = threading.Thread(target=lambda: server.accept()[0].close())
        accepted.start()
        with socket.socket() as client:
            client.connect(("127.0.0.1", port))
        accepted.join(timeout=5)


@pytest.mark.parametrize("host", ["127.0.0.1", "127.8.9.10", "::1", "localhost"])
def test_loopback_hosts(host: str) -> None:
    assert _is_loopback((host, 80))


@pytest.mark.parametrize("host", ["192.0.2.1", "10.0.0.1", "example.com", "0.0.0.0"])  # noqa: S104
def test_non_loopback_hosts(host: str) -> None:
    assert not _is_loopback((host, 80))


def test_unix_socket_paths_count_as_local() -> None:
    assert _is_loopback("/tmp/ollama.sock")  # noqa: S108
    assert _is_loopback(b"/tmp/ollama.sock")


def test_empty_or_unknown_addresses_are_not_local() -> None:
    assert not _is_loopback(())
    assert not _is_loopback(None)


@given(st.ip_addresses())
def test_ip_classification_matches_stdlib(address: object) -> None:
    import ipaddress

    ip = ipaddress.ip_address(str(address))
    assert _is_loopback((str(ip), 0)) == ip.is_loopback
