"""Model providers (ADR-0005): local Ollama for embeddings and chat, optional BYOK chat.

`http.py` is the only module that opens sockets. Tests fake it or point it at a local
HTTP server; the test-wide network guard blocks anything else.
"""
