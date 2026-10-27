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
CONTEXT = 8192  # nomic-embed-text's maximum input, in tokens


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
                {
                    "model": self.model,
                    "input": batch,
                    # A chunk longer than the window is an error, not a silently
                    # shortened vector; the window is the model's full 8192 tokens.
                    "truncate": False,
                    "options": {"num_ctx": CONTEXT},
                },
            )
            rows.extend(_vectors(reply, len(batch)))
        if not rows:
            return np.zeros((0, 0), dtype=np.float32)
        return normalise(rows)

    def embed_documents(self, texts: Sequence[str]) -> Matrix:
        return self._embed([DOCUMENT_PREFIX + t for t in texts])

    def embed_query(self, text: str) -> Matrix:
        return self._embed([QUERY_PREFIX + text]).reshape(-1)


def _vectors(reply: object, expected: int) -> list[list[float]]:
    """The reply's vectors, checked: one per input, all the same length, all finite."""
    vectors = reply.get("embeddings") if isinstance(reply, dict) else None
    try:
        matrix = np.asarray(vectors, dtype=np.float64)
    except (TypeError, ValueError):
        matrix = np.zeros((0,))
    if matrix.ndim != 2 or matrix.shape[0] != expected or matrix.shape[1] == 0:
        raise http.ProviderError(f"unexpected reply from Ollama /api/embed: {reply!r:.200}")
    if not np.isfinite(matrix).all():
        raise http.ProviderError("Ollama returned a vector with NaN or infinite values")
    rows: list[list[float]] = matrix.tolist()
    return rows


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
            message = event.get("message") or {}
            piece = message.get("content", "") if isinstance(message, dict) else None
            if not isinstance(piece, str):
                raise http.ProviderError(f"Ollama sent an unexpected message: {line[:200]!r}")
            if piece:
                yield piece
            if event.get("done"):
                return
        # The stream closed without "done": the answer may be cut short.
        raise http.ProviderError("Ollama's answer ended before it was complete")


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
