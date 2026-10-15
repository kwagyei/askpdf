import socket
import sys
import threading

import pytest

from tests.conftest import NetworkBlockedError, is_loopback_address, is_loopback_host

BLACKHOLE = ("192.0.2.1", 9)  # TEST-NET-1: documentation range, never routable


def test_connect_is_blocked() -> None:
    with socket.socket() as s, pytest.raises(NetworkBlockedError):
        s.settimeout(1)
        s.connect(BLACKHOLE)


def test_connect_ex_is_blocked() -> None:
    with socket.socket() as s, pytest.raises(NetworkBlockedError):
        s.setblocking(False)
        s.connect_ex(BLACKHOLE)


def test_udp_sendto_is_blocked() -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s, pytest.raises(NetworkBlockedError):
        s.sendto(b"x", BLACKHOLE)


def test_udp_sendto_with_flags_is_blocked() -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s, pytest.raises(NetworkBlockedError):
        s.sendto(b"x", 0, BLACKHOLE)


@pytest.mark.skipif(not hasattr(socket.socket, "sendmsg"), reason="no sendmsg on Windows")
def test_udp_sendmsg_is_blocked() -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s, pytest.raises(NetworkBlockedError):
        s.sendmsg([b"x"], [], 0, BLACKHOLE)


def test_dns_lookup_is_blocked() -> None:
    with pytest.raises(NetworkBlockedError):
        socket.getaddrinfo("example.com", 80)


def test_create_connection_is_blocked() -> None:
    with pytest.raises(NetworkBlockedError):
        socket.create_connection(("example.com", 80), timeout=1)


def test_loopback_resolution_is_allowed() -> None:
    assert socket.getaddrinfo("127.0.0.1", 80)


def test_loopback_round_trip_is_allowed() -> None:
    with socket.socket() as server:
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        server.settimeout(5)
        port = server.getsockname()[1]
        accepted = threading.Thread(target=lambda: server.accept()[0].close(), daemon=True)
        accepted.start()
        with socket.create_connection(("127.0.0.1", port), timeout=5):
            pass
        accepted.join(timeout=5)
        assert not accepted.is_alive()


@pytest.mark.network
def test_network_marker_lifts_the_guard() -> None:
    # Nothing is sent: we only check that the real functions are back in place.
    assert socket.getaddrinfo.__name__ == "getaddrinfo"
    assert socket.socket.connect.__name__ == "connect"


def test_guard_is_back_after_a_marked_test() -> None:
    assert socket.getaddrinfo.__name__ == "_guarded_getaddrinfo"


@pytest.mark.parametrize(
    "host",
    [
        "127.0.0.1",
        "127.8.9.10",
        "::1",
        "::1%lo0",  # scoped
        "::ffff:127.0.0.1",  # IPv4-mapped loopback
        "localhost",
        "LOCALHOST",
        "localhost.",  # fully qualified
        b"localhost",
    ],
)
def test_loopback_hosts(host: str | bytes) -> None:
    assert is_loopback_host(host)


@pytest.mark.parametrize(
    "host",
    [
        "192.0.2.1",
        "10.0.0.1",
        "::ffff:10.0.0.1",  # IPv4-mapped, not loopback
        "0.0.0.0",  # noqa: S104 - "any address", not loopback
        "::",
        "example.com",
        "localhost.example.com",
        "",
        None,
        80,
    ],
)
def test_non_loopback_hosts(host: object) -> None:
    assert not is_loopback_host(host)


def test_unix_socket_paths_count_as_local() -> None:
    assert is_loopback_address("/tmp/ollama.sock")  # noqa: S108
    assert is_loopback_address(b"/tmp/ollama.sock")


def test_empty_or_unknown_addresses_are_not_local() -> None:
    assert not is_loopback_address(())
    assert not is_loopback_address(None)


def test_guard_is_active_outside_test_functions() -> None:
    # Installed in pytest_configure, so it already covered imports and collection.
    assert "tests.conftest" in sys.modules
    assert socket.socket.connect.__name__ == "_guarded_connect"
