"""Bring-your-own-key chat over an OpenAI-compatible API (ADR-0005): chat only.

The key is read from ``ASKPDF_API_KEY`` and nowhere else. It is sent only in the
Authorization header and never appears in ``repr``, logs or error messages. Excerpts of
the user's documents go to the endpoint, so a non-local one needs ``--allow-remote`` or
``ASKPDF_ALLOW_REMOTE=1``.
"""

import json
import os
from collections.abc import Iterator
from dataclasses import dataclass, field

from askpdf.providers import http


@dataclass(frozen=True)
class OpenAICompatChat:
    url: str  # e.g. https://api.example.com/v1
    model: str
    api_key: str = field(repr=False)

    def stream(self, system: str, user: str) -> Iterator[str]:
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "stream": True,
            "temperature": 0,
        }
        headers = {"Authorization": f"Bearer {self.api_key}"}
        for line in http.stream_lines(f"{self.url}/chat/completions", payload, headers):
            if not line.startswith("data:"):
                continue  # SSE comments and other fields
            data = line[len("data:") :].strip()
            if data == "[DONE]":
                return
            try:
                event = json.loads(data)
                piece = (event["choices"][0].get("delta") or {}).get("content") or ""
                if not isinstance(piece, str):
                    raise TypeError(piece)
            except (ValueError, KeyError, IndexError, TypeError, AttributeError) as exc:
                message = f"unexpected event from {self.url}: {data[:200]!r}"
                raise http.ProviderError(message) from exc
            if piece:
                yield piece
        # The stream closed without [DONE]: the answer may be cut short.
        raise http.ProviderError(f"the answer from {self.url} ended before it was complete")


def allow_remote(flag: bool = False) -> bool:
    return flag or os.environ.get("ASKPDF_ALLOW_REMOTE", "") in {"1", "true", "yes"}


def from_env(url: str, model: str, allow_remote_flag: bool = False) -> OpenAICompatChat:
    """The BYOK chat model, after the remote guard and with the key from the environment."""
    http.require_local_or_allowed(url, allow_remote(allow_remote_flag))
    key = os.environ.get("ASKPDF_API_KEY", "")
    if not key:
        raise http.ProviderError("set ASKPDF_API_KEY to use a remote chat model")
    return OpenAICompatChat(url.rstrip("/"), model, key)
