"""Ollama's native API, on this machine by default (ADR-0005)."""

import json
import os
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from askpdf.providers import http
from askpdf.providers.base import Matrix, normalise

DEFAULT_URL = "http://127.0.0.1:11434"
EMBED_MODEL = "nomic-embed-text"
CHAT_MODEL = "llama3.2:3b"
# nomic-embed-text v1.5 is trained with these task prefixes (its model card on Hugging
# Face); leaving them out lowers retrieval quality.
DOCUMENT_PREFIX = "search_document: "
QUERY_PREFIX = "search_query: "
BATCH = 16  # chunks per request: small enough to pause between batches while answering


def base_url() -> str:
    return os.environ.get("ASKPDF_OLLAMA_URL", DEFAULT_URL).rstrip("/")


@dataclass(frozen=True)
class OllamaEmbedder:
    url: str = DEFAULT_URL
    model: str = EMBED_MODEL

    @property
    def name(self) -> str:
        return f"ollama/{self.model}"

    def _embed(self, texts: Sequence[str]) -> Matrix:
        rows: list[list[float]] = []
        for start in range(0, len(texts), BATCH):
            batch = list(texts[start : start + BATCH])
            reply = http.post_json(
                f"{self.url}/api/embed",
                # truncate false: a chunk longer than the model's window is an error,
                # not a silently shortened vector.
                {"model": self.model, "input": batch, "truncate": False},
            )
            vectors = reply.get("embeddings") if isinstance(reply, dict) else None
            if not isinstance(vectors, list) or len(vectors) != len(batch):
                raise http.ProviderError(f"unexpected reply from Ollama /api/embed: {reply!r:.200}")
            rows.extend(vectors)
        if not rows:
            return np.zeros((0, 0), dtype=np.float32)
        return normalise(rows)

    def embed_documents(self, texts: Sequence[str]) -> Matrix:
        return self._embed([DOCUMENT_PREFIX + t for t in texts])

    def embed_query(self, text: str) -> Matrix:
        return self._embed([QUERY_PREFIX + text]).reshape(-1)


@dataclass(frozen=True)
class OllamaChat:
    """Streaming chat with ``temperature 0`` and ``seed 0``, for answers that repeat."""

    url: str = DEFAULT_URL
    model: str = CHAT_MODEL
    num_ctx: int = 8192

    def stream(self, system: str, user: str) -> Iterator[str]:
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "stream": True,
            "options": {"temperature": 0, "seed": 0, "num_ctx": self.num_ctx},
        }
        for line in http.stream_lines(f"{self.url}/api/chat", payload):
            event = _json(line)
            if "error" in event:
                raise http.ProviderError(f"Ollama: {event['error']}")
            piece = event.get("message", {}).get("content", "")
            if piece:
                yield piece
            if event.get("done"):
                return


def _json(line: str) -> dict[str, Any]:
    try:
        value = json.loads(line)
    except ValueError as exc:
        raise http.ProviderError(f"Ollama sent a line that is not JSON: {line[:200]!r}") from exc
    if not isinstance(value, dict):
        raise http.ProviderError(f"Ollama sent unexpected JSON: {line[:200]!r}")
    return value


def server_info(url: str = DEFAULT_URL, model: str = EMBED_MODEL) -> dict[str, str]:
    """Ollama's version and the model's digest, recorded next to every model-based result."""
    version = http.get_json(f"{url}/api/version")
    tags = http.get_json(f"{url}/api/tags")
    found = version.get("version", "?") if isinstance(version, dict) else "?"
    info = {"ollama_version": str(found)}
    for entry in tags.get("models", []) if isinstance(tags, dict) else []:
        if entry.get("name") in {model, f"{model}:latest"}:
            info["model_digest"] = str(entry.get("digest", "?"))
    return info
