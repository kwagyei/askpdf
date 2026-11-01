"""The local web app: one page, an SSE answer stream, and the PDFs (design §12, ADR-0008).

The server binds to 127.0.0.1 only, but other websites open in the same browser can still
send it requests, so every request is checked before it reaches a route:

- the Host header must name this server (against DNS rebinding);
- ``/api`` and ``/pdf`` need the session cookie, which only someone holding the launch
  token from the terminal can get;
- a POST must come from this server's own origin (against cross-site requests).

PDFs are served by the id of their row in the index, never by a path from the request,
and only if the file still resolves to a regular PDF inside the folder.
"""

import asyncio
import hmac
import json
import logging
import os
import re
import secrets
import stat
import threading
from collections import OrderedDict
from collections.abc import AsyncIterator, Callable, Iterator, Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import (
    HTMLResponse,
    JSONResponse,
    RedirectResponse,
    Response,
    StreamingResponse,
)
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from askpdf.answer import ANSWER_K, Answer, Final, Pages, Token, answer, source_name
from askpdf.providers import http
from askpdf.providers.base import ChatModel
from askpdf.retrieve import PageHit, Search, search
from askpdf.snapshot import Snapshot
from askpdf.store import Store

log = logging.getLogger(__name__)
STATIC = Path(__file__).with_name("static")
MAX_QUESTION = 2000
HISTORY_TURNS = 3  # a follow-up is searched with up to this many questions joined
MAX_CONVERSATIONS = 100  # kept in memory only; the oldest is forgotten first
SNIPPET_CHARS = 240
CHUNK_BYTES = 64 * 1024
_FILE_ID = re.compile(r"[1-9][0-9]{0,17}")  # one spelling per id; fits a 64-bit rowid

CSP = "default-src 'self'; frame-ancestors 'none'; object-src 'none'; base-uri 'none'"
SECURITY_HEADERS = [
    (b"content-security-policy", CSP.encode()),
    (b"x-content-type-options", b"nosniff"),
    (b"referrer-policy", b"no-referrer"),
]

type ChatFor = Callable[[str, Sequence[PageHit], Mapping[tuple[str, int], str]], ChatModel]

START_FROM_TERMINAL = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>askpdf</title></head>
<body><h1>askpdf</h1>
<p>Open askpdf with the link printed in the terminal where you started it.
This page needs the one-time key in that link.</p></body></html>
"""


def cookie_name(port: int) -> str:
    # Cookies are not separated by port, so two askpdf servers need different names.
    return f"askpdf-{port}"


# --- request checks -----------------------------------------------------------------------


class Guard:
    """ASGI middleware: Host allow-list, session cookie, same-origin POSTs, and the
    security headers on every response."""

    def __init__(self, app: ASGIApp, port: int, token: str) -> None:
        self.app = app
        self.hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}
        self.cookie = cookie_name(port)
        self.token = token

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in scope["headers"]}
        host = headers.get("host", "")
        path: str = scope["path"]
        if host not in self.hosts:
            await _plain(send, 400, "unknown Host")
            return
        unsafe = scope["method"] not in ("GET", "HEAD", "OPTIONS")
        if unsafe and headers.get("origin") != f"http://{host}":
            await _plain(send, 403, "cross-origin request refused")
            return
        if path.startswith(("/api/", "/pdf/")) and not self.has_session(headers.get("cookie")):
            await _plain(send, 403, "no askpdf session; open the link from the terminal")
            return
        no_store = not path.startswith("/static/")

        async def send_with_headers(message: Message) -> None:
            if message["type"] == "http.response.start":
                extra = list(SECURITY_HEADERS)
                if no_store:
                    extra.append((b"cache-control", b"no-store"))
                message = {**message, "headers": [*message.get("headers", []), *extra]}
            await send(message)

        await self.app(scope, receive, send_with_headers)

    def has_session(self, header: str | None) -> bool:
        return has_session(header, self.cookie, self.token)


def has_session(header: str | None, name: str, token: str) -> bool:
    """True if the Cookie header holds ``name=token``. Parsed by hand and leniently:
    cookies ignore ports, so another app on localhost may have set cookies that a strict
    parser gives up on, which would hide ours."""
    for part in (header or "").split(";"):
        key, _, value = part.strip().partition("=")
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] == '"':
            value = value[1:-1]  # a quoted value, as RFC 6265 allows
        if key == name and same_secret(value, token):
            return True
    return False


def same_secret(given: str, token: str) -> bool:
    # As bytes: compare_digest refuses str holding non-ASCII, which anyone can send.
    return hmac.compare_digest(given.encode("utf-8", "surrogateescape"), token.encode())


async def _plain(send: Send, status: int, text: str) -> None:
    body = text.encode()
    await send(
        {
            "type": "http.response.start",
            "status": status,
            "headers": [
                (b"content-type", b"text/plain; charset=utf-8"),
                (b"content-length", str(len(body)).encode()),
                *SECURITY_HEADERS,
            ],
        }
    )
    await send({"type": "http.response.body", "body": body})


# --- state ---------------------------------------------------------------------------------


@dataclass
class Progress:
    """Where indexing stands, shared between the indexing thread and the event stream."""

    state: str = "indexing"  # indexing | ready | failed
    current: str = ""
    done: int = 0
    message: str = ""
    version: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def update(self, **changes: Any) -> None:
        with self._lock:
            for key, value in changes.items():
                setattr(self, key, value)
            self.version += 1

    def view(self) -> dict[str, Any]:
        with self._lock:
            return {
                "state": self.state,
                "current": self.current,
                "done": self.done,
                "message": self.message,
                "version": self.version,
            }


class Library:
    """The index as the web app sees it: a snapshot that is reloaded when the indexer
    commits, and the files behind its links."""

    def __init__(self, root: Path, index: Path) -> None:
        self.root = Path(os.path.realpath(root))
        self.index = index
        self._lock = threading.Lock()
        self._snapshot: Snapshot | None = None
        self._ids: dict[str, int] = {}

    def _store(self) -> Store:
        return Store(self.index, rebuild=False)  # readers never wipe the index

    def snapshot(self) -> tuple[Snapshot, dict[str, int]]:
        with self._lock, self._store() as store:
            if self._snapshot is None or not self._snapshot.is_current(store):
                self._snapshot = Snapshot.load(store)
                # /pdf checks every id again when it is opened.
                self._ids = store.file_ids()
            return self._snapshot, self._ids

    def documents(self) -> dict[str, Any]:
        with self._store() as store, store.read_transaction():
            ids = store.file_ids()
            documents = store.documents()
            failed = [f for f in store.files().values() if f.status == "failed"]
        return {
            "documents": [
                {
                    "id": ids[path],
                    "path": path,
                    "pages": doc.n_pages,
                    "scanned": doc.looks_scanned,
                }
                for doc in documents
                for path in doc.paths
                if path in ids
            ],
            "failed": [{"path": f.path, "reason": f.error or ""} for f in failed],
        }

    def pdf(self, file_id: int) -> tuple[BinaryIO, int] | None:
        """An indexed file, opened, and its size; None unless it is still a regular PDF
        inside the folder. The path is resolved again now: the file may have been replaced
        by a symlink since it was indexed."""
        with self._store() as store:
            rel = store.file_path(file_id)
        if rel is None:
            return None
        real = os.path.realpath(self.root / rel)
        try:
            if os.path.commonpath([self.root, real]) != str(self.root):
                return None
        except ValueError:  # Windows: on another drive
            return None
        if not real.casefold().endswith(".pdf"):
            return None
        # O_NOFOLLOW: if the last component became a symlink after the check, open fails.
        # O_NONBLOCK: a FIFO swapped in would otherwise block open() until a writer came.
        flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
        try:
            fd = os.open(real, flags)
        except OSError:
            return None
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):  # a folder or a FIFO opens too
            os.close(fd)
            return None
        return os.fdopen(fd, "rb"), info.st_size


class Conversations:
    """The questions of each conversation, in memory only (design §10)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._turns: OrderedDict[str, list[str]] = OrderedDict()

    def start(self) -> str:
        cid = secrets.token_urlsafe(16)
        with self._lock:
            self._turns[cid] = []
            while len(self._turns) > MAX_CONVERSATIONS:
                self._turns.popitem(last=False)
        return cid

    def earlier(self, cid: str) -> list[str] | None:
        with self._lock:
            turns = self._turns.get(cid)
            return None if turns is None else list(turns)

    def add(self, cid: str, question: str) -> None:
        with self._lock:
            if cid in self._turns:
                self._turns[cid] = [*self._turns[cid], question][-(HISTORY_TURNS - 1) :]
                self._turns.move_to_end(cid)


# --- serialising events --------------------------------------------------------------------


def sse(event: str, data: object) -> str:
    # ASCII only: a lone surrogate (from a question, the model or a PDF) is escaped
    # instead of failing to encode as UTF-8 and killing the stream.
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def _snippet(text: str) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= SNIPPET_CHARS else flat[: SNIPPET_CHARS - 1].rstrip() + "…"


def _page(hit: PageHit, ids: Mapping[str, int]) -> dict[str, Any]:
    file_id = ids.get(hit.paths[0]) if hit.paths else None
    return {
        "file": source_name(hit),
        "path": hit.paths[0] if hit.paths else "",
        "page": hit.page,
        "label": hit.label,
        "href": None if file_id is None else f"/pdf/{file_id}#page={hit.page}",
    }


def final_view(result: Answer, ids: Mapping[str, int]) -> dict[str, Any]:
    return {
        "not_found": result.not_found,
        "text": result.text,
        "grounded": result.grounded,
        "dropped": list(result.dropped),
        "citations": [
            {
                "number": c.number,
                **_page(c.hit, ids),
                "verified": c.verified,
                "quotes": [
                    {"quote": q.quote, "verified": q.verified, "reason": q.reason} for q in c.quotes
                ],
                "quote_on": None if c.quote_on is None else _page(c.quote_on, ids),
            }
            for c in result.citations
        ],
    }


# --- the app -------------------------------------------------------------------------------


class AskRequest(BaseModel):
    conversation_id: str = Field(min_length=1, max_length=64)
    question: str = Field(min_length=1, max_length=MAX_QUESTION)


def create_app(
    *,
    root: Path,
    index: Path,
    port: int,
    token: str,
    chat_for: ChatFor,
    retrieve: Callable[[Snapshot, str, int], Search] = search,
    index_folder: Callable[[Progress], None] | None = None,
    on_start: Callable[[], None] | None = None,
    k: int = ANSWER_K,
) -> ASGIApp:
    """The app, wrapped in its request checks. ``index_folder`` runs in a thread at
    start-up and reports through ``Progress``; without it the index is taken as ready."""
    library = Library(root, index)
    progress = Progress() if index_folder else Progress(state="ready")
    conversations = Conversations()
    answering = threading.Lock()  # one answer at a time: the model would share the CPU

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if index_folder is not None:
            threading.Thread(
                target=index_folder, args=(progress,), name="askpdf-index", daemon=True
            ).start()
        if on_start is not None:
            on_start()
        yield

    app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    @app.exception_handler(RequestValidationError)
    async def invalid(request: Request, exc: RequestValidationError) -> JSONResponse:
        # FastAPI's own handler echoes the input back, which fails to encode (a 500)
        # when the input holds a lone surrogate. Where and why is enough.
        errors = [{"loc": list(e["loc"]), "msg": e["msg"]} for e in exc.errors()]
        return JSONResponse({"detail": errors}, status_code=422)

    @app.get("/", response_model=None)
    def home(request: Request) -> Response:
        given = request.query_params.get("token")
        if given is not None:
            if not same_secret(given, token):
                return HTMLResponse(START_FROM_TERMINAL, status_code=403)
            # Swap the token in the URL for a cookie, and drop it from the address bar.
            response = RedirectResponse("/", status_code=303)
            response.set_cookie(
                cookie_name(port), token, httponly=True, samesite="strict", path="/"
            )
            return response
        if not has_session(request.headers.get("cookie"), cookie_name(port), token):
            return HTMLResponse(START_FROM_TERMINAL, status_code=403)
        return HTMLResponse((STATIC / "index.html").read_text(encoding="utf-8"))

    @app.get("/api/docs")
    def docs() -> dict[str, Any]:
        return {"folder": library.root.name, **library.documents()}

    @app.post("/api/conversations")
    def new_conversation() -> dict[str, str]:
        return {"id": conversations.start()}

    @app.get("/api/events")
    async def events() -> StreamingResponse:
        async def stream() -> AsyncIterator[str]:
            seen = -1
            while True:
                view = progress.view()
                if view["version"] != seen:
                    seen = view["version"]
                    yield sse("progress", view)
                if view["state"] != "indexing":
                    return  # the page closes its EventSource; PR 10 keeps it open
                await asyncio.sleep(0.25)

        return StreamingResponse(stream(), media_type="text/event-stream")

    @app.post("/api/ask")
    def ask(body: AskRequest) -> StreamingResponse:
        earlier = conversations.earlier(body.conversation_id)
        if earlier is None:
            raise HTTPException(404, "unknown conversation; start a new one")
        question = body.question.strip()
        if not question:
            raise HTTPException(422, "the question is empty")
        return StreamingResponse(
            _answer_stream(question, earlier, body.conversation_id),
            media_type="text/event-stream",
        )

    def _answer_stream(question: str, earlier: list[str], cid: str) -> Iterator[str]:
        if not answering.acquire(blocking=False):
            yield sse("error", {"message": "another question is being answered; try again"})
            return
        try:
            snapshot, ids = library.snapshot()
            # Follow-ups are searched with the conversation's questions joined: the
            # "concat" condition of the conversation eval (design revision 3).
            query = " ".join([*earlier, question])
            found = retrieve(snapshot, query, k)
            if earlier or found.notes:
                # The query is shown only for follow-ups, where it differs from the question.
                shown = query if earlier else None
                yield sse("search", {"query": shown, "used": found.used, "notes": [*found.notes]})
            hits = found.hits
            chat = chat_for(question, hits, snapshot.texts)
            for event in answer(question, hits, snapshot.texts, chat):
                if isinstance(event, Pages):
                    pages = [
                        {
                            "number": n,
                            **_page(hit, ids),
                            "snippet": _snippet(hit.best_chunk.text),
                        }
                        for n, hit in enumerate(event.hits, start=1)
                    ]
                    yield sse("pages", {"pages": pages})
                elif isinstance(event, Token):
                    yield sse("token", {"text": event.text})
                elif isinstance(event, Final):
                    conversations.add(cid, question)
                    yield sse("final", final_view(event.answer, ids))
        except http.UnreachableError as exc:
            yield sse("error", {"message": f"{exc}; start Ollama (ollama serve) and try again"})
        except http.ProviderError as exc:
            yield sse("error", {"message": str(exc)})
        except Exception:
            log.exception("answering failed")
            yield sse("error", {"message": "askpdf failed while answering; see the terminal"})
        finally:
            answering.release()

    @app.get("/pdf/{file_id}")
    def pdf(file_id: str) -> Response:
        # Only plain digits name a file; anything else is "not found", not a 422 that
        # describes the parameter.
        opened = library.pdf(int(file_id)) if _FILE_ID.fullmatch(file_id) else None
        if opened is None:
            raise HTTPException(404, "no such document")
        f, size = opened

        def chunks() -> Iterator[bytes]:
            with f:  # streamed, so a large PDF is never held in memory whole
                while chunk := f.read(CHUNK_BYTES):
                    yield chunk

        return StreamingResponse(
            chunks(),
            media_type="application/pdf",
            headers={"Content-Disposition": "inline", "Content-Length": str(size)},
        )

    return Guard(app, port, token)
