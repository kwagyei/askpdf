"""Model providers against a local fake server (ADR-0005)."""

import numpy as np
import pytest

from askpdf.providers import http
from askpdf.providers.base import normalise
from askpdf.providers.ollama import (
    DOCUMENT_PREFIX,
    QUERY_PREFIX,
    OllamaChat,
    OllamaEmbedder,
    server_info,
)
from askpdf.providers.openai_compat import OpenAICompatChat, from_env
from tests.fake_models import fake_model_server


def test_embeddings_carry_the_task_prefixes_and_are_normalised() -> None:
    with fake_model_server() as (url, models):
        embedder = OllamaEmbedder(url)
        docs = embedder.embed_documents(["alpha beta", "gamma"])
        query = embedder.embed_query("alpha")
    inputs = [body["input"] for path, body, _ in models.requests if path == "/api/embed"]
    assert inputs == [
        [DOCUMENT_PREFIX + "alpha beta", DOCUMENT_PREFIX + "gamma"],
        [QUERY_PREFIX + "alpha"],
    ]
    assert all(
        body["truncate"] is False for path, body, _ in models.requests if path == "/api/embed"
    )
    assert docs.dtype == np.float32 and docs.shape == (2, 32) and query.shape == (32,)
    assert np.allclose(np.linalg.norm(docs, axis=1), 1)
    assert embedder.name == "ollama/nomic-embed-text"


def test_embeddings_are_sent_in_batches(monkeypatch: pytest.MonkeyPatch) -> None:
    from askpdf.providers import ollama

    monkeypatch.setattr(ollama, "BATCH", 2)
    with fake_model_server() as (url, models):
        assert OllamaEmbedder(url).embed_documents([f"t{i}" for i in range(5)]).shape[0] == 5
    assert [len(body["input"]) for _, body, _ in models.requests] == [2, 2, 1]


def test_nothing_to_embed_makes_no_request() -> None:
    with fake_model_server() as (url, models):
        assert OllamaEmbedder(url).embed_documents([]).shape[0] == 0
    assert models.requests == []


def test_chat_streams_pieces_with_fixed_sampling() -> None:
    with fake_model_server() as (url, models):
        assert "".join(OllamaChat(url).stream("system", "question")) == "The sky is blue."
    body = models.requests[0][1]
    assert body["stream"] is True
    assert body["options"]["temperature"] == 0 and body["options"]["seed"] == 0
    assert [m["role"] for m in body["messages"]] == ["system", "user"]


def test_an_error_from_ollama_is_a_provider_error() -> None:
    with fake_model_server() as (url, models):
        models.fail_with = 404
        with pytest.raises(http.ProviderError, match=r"HTTP 404.*model not found"):
            OllamaEmbedder(url).embed_query("x")


def test_an_unreachable_server_is_a_provider_error() -> None:
    with pytest.raises(http.ProviderError, match=r"cannot reach 127\.0\.0\.1:9"):
        OllamaEmbedder("http://127.0.0.1:9").embed_query("x")


def test_server_info_names_version_and_model_digest() -> None:
    with fake_model_server() as (url, _):
        assert server_info(url) == {"ollama_version": "0.0-fake", "model_digest": "abc123"}


def test_byok_chat_streams_sse_and_sends_the_key_only_in_the_header() -> None:
    with fake_model_server() as (url, models):
        chat = OpenAICompatChat(f"{url}/v1", "some-model", "sk-secret")
        assert "".join(chat.stream("system", "question")) == "The sky is blue."
    _, body, headers = models.requests[0]
    assert headers["Authorization"] == "Bearer sk-secret"
    assert "sk-secret" not in str(body)
    assert "sk-secret" not in repr(chat)


def test_byok_reads_the_key_from_the_environment_only(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(http.ProviderError, match="ASKPDF_API_KEY"):
        from_env("http://127.0.0.1:8000/v1", "m")
    monkeypatch.setenv("ASKPDF_API_KEY", "sk-env")
    assert from_env("http://127.0.0.1:8000/v1/", "m").api_key == "sk-env"


@pytest.mark.parametrize(
    ("url", "allowed"),
    [
        ("http://127.0.0.1:11434", True),
        ("http://localhost:8000/v1", True),
        ("http://[::1]:8000", True),
        ("http://[::ffff:127.0.0.1]:8000", True),
        ("https://api.example.com/v1", False),
        ("http://10.0.0.5:11434", False),
        ("http://localhost.example.com", False),
    ],
)
def test_the_remote_guard(url: str, allowed: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ASKPDF_API_KEY", "k")
    if allowed:
        from_env(url, "m")
    else:
        with pytest.raises(http.RemoteNotAllowedError, match="--allow-remote"):
            from_env(url, "m")
        from_env(url, "m", allow_remote_flag=True)
        monkeypatch.setenv("ASKPDF_ALLOW_REMOTE", "1")
        from_env(url, "m")


def test_only_http_urls_are_accepted() -> None:
    with pytest.raises(http.ProviderError, match="scheme"):
        http.require_local_or_allowed("file:///etc/passwd", allow_remote=True)


def test_normalise_leaves_zero_vectors_alone() -> None:
    m = normalise([[3.0, 4.0], [0.0, 0.0]])
    assert np.allclose(m, [[0.6, 0.8], [0.0, 0.0]])
