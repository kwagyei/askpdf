"""Test-wide guards and settings."""

import ipaddress
import os
import socket
from collections.abc import Iterator
from typing import Any

import pytest
from hypothesis import settings

# CI must be deterministic: same examples on every run, no example database.
settings.register_profile("ci", derandomize=True, database=None, print_blob=True)
settings.register_profile("dev", max_examples=200)
settings.load_profile("ci" if os.environ.get("CI") else "dev")


def _is_loopback(address: Any) -> bool:
    if isinstance(address, tuple) and address:
        host = address[0]
        if host == "localhost":
            return True
        try:
            return ipaddress.ip_address(host).is_loopback
        except ValueError:
            return False
    # AF_UNIX paths and other families never leave the machine.
    return isinstance(address, str | bytes)


class NetworkBlockedError(RuntimeError):
    """Raised when a test tries to reach a non-loopback address."""


@pytest.fixture(autouse=True)
def _block_network(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> Iterator[None]:
    """No test may reach the network: models, arXiv and remote APIs are faked.

    Loopback stays open for the local HTTP servers that provider tests start.
    """
    if request.node.get_closest_marker("network"):
        yield
        return
    real_connect = socket.socket.connect

    def guarded_connect(self: socket.socket, address: Any) -> None:
        if not _is_loopback(address):
            raise NetworkBlockedError(f"test tried to connect to {address!r}")
        real_connect(self, address)

    monkeypatch.setattr(socket.socket, "connect", guarded_connect)
    yield
