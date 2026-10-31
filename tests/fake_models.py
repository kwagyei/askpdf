"""A loopback HTTP server that stands in for Ollama and an OpenAI-compatible API."""

import hashlib
import json
import re
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import numpy as np

DIM = 32


def fake_vector(text: str) -> list[float]:
    """A deterministic bag-of-words vector: texts sharing words point the same way."""
    v = np.zeros(DIM)
    for word in re.findall(r"[a-z]+", text.lower()):
        word = word.rstrip("s")  # "cookies" and "cookie" alike
        v[int(hashlib.sha256(word.encode()).hexdigest(), 16) % DIM] += 1
    return v.tolist()


class FakeModels:
    def __init__(self) -> None:
        self.requests: list[tuple[str, dict[str, Any], dict[str, str]]] = []
        self.fail_with: int | None = None
        self.chat_reply = ["The ", "sky ", "is blue."]

    def handle(self, path: str, body: dict[str, Any]) -> tuple[int, str, list[str] | Any]:
        if self.fail_with:
            return self.fail_with, "application/json", {"error": "model not found"}
        if path == "/api/embed":
            return 200, "application/json", {"embeddings": [fake_vector(t) for t in body["input"]]}
        if path == "/api/chat":
            lines = [
                json.dumps({"message": {"content": p}, "done": False}) for p in self.chat_reply
            ]
            return 200, "application/x-ndjson", [*lines, json.dumps({"done": True})]
        if path == "/v1/chat/completions":
            events = [
                f"data: {json.dumps({'choices': [{'delta': {'content': p}}]})}"
                for p in self.chat_reply
            ]
            return 200, "text/event-stream", [": keep-alive", *events, "data: [DONE]"]
        if path == "/api/version":
            return 200, "application/json", {"version": "0.0-fake"}
        if path == "/api/tags":
            return (
                200,
                "application/json",
                {"models": [{"name": "nomic-embed-text:latest", "digest": "abc123"}]},
            )
        return 404, "application/json", {"error": "no such path"}


@contextmanager
def fake_model_server() -> Iterator[tuple[str, FakeModels]]:
    models = FakeModels()

    class Handler(BaseHTTPRequestHandler):
        def _answer(self, body: dict[str, Any]) -> None:
            models.requests.append((self.path, body, dict(self.headers)))
            status, kind, payload = models.handle(self.path, body)
            self.send_response(status)
            self.send_header("Content-Type", kind)
            self.end_headers()
            if isinstance(payload, list):
                for line in payload:
                    self.wfile.write(line.encode() + b"\n")
                    self.wfile.flush()
            else:
                self.wfile.write(json.dumps(payload).encode())

        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length", 0))
            self._answer(json.loads(self.rfile.read(length) or b"{}"))

        def do_GET(self) -> None:
            self._answer({})

        def log_message(self, *args: object) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}", models
    finally:
        server.shutdown()
        server.server_close()
