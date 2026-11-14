"""A browser that disconnects mid-answer must not leave the answer running (ADR-0005).

Starlette cancels the response's task when the client goes away. The answer stream holds
the answering lock until its ``finally`` runs, and only closing it closes the model's
HTTP stream, which is what tells Ollama to stop.
"""

import asyncio
import contextlib
import time
from collections.abc import AsyncIterator, Generator

from starlette.concurrency import iterate_in_threadpool
from starlette.requests import ClientDisconnect

from askpdf.web.app import closing


def _answer(state: dict[str, object]) -> Generator[str, None, None]:
    """Stands in for _answer_stream: holds a lock while it runs, frees it in finally."""
    state["held"] = True
    try:
        for n in range(100):
            time.sleep(0.02)  # a model token arriving
            state["sent"] = n + 1
            yield f"token {n}"
    finally:
        state["held"] = False


async def _browser_leaves(stream: AsyncIterator[str]) -> None:
    async def read() -> None:
        async for _ in stream:
            pass

    task = asyncio.ensure_future(read())
    await asyncio.sleep(0.1)  # a few tokens in, and inside a worker thread's next()
    task.cancel()  # what Starlette does on http.disconnect
    with contextlib.suppress(asyncio.CancelledError):
        await task
    await asyncio.sleep(0.2)  # time for a stream that kept going to show it


def test_a_disconnect_closes_the_answer_and_frees_its_lock() -> None:
    state: dict[str, object] = {}
    asyncio.run(_browser_leaves(closing(_answer(state))))
    assert state["held"] is False
    sent = state["sent"]
    assert isinstance(sent, int) and 1 <= sent < 100  # 100: it ran to the end


def test_starlettes_own_wrapper_leaves_the_lock_held() -> None:
    # Why closing() exists: the stream Starlette makes from a plain generator is never
    # closed after a disconnect, so the lock stayed held until askpdf was restarted.
    state: dict[str, object] = {}
    answer = _answer(state)

    async def run() -> None:
        await _browser_leaves(iterate_in_threadpool(answer))
        assert state["held"] is True

    asyncio.run(run())
    answer.close()


def test_a_finished_answer_is_served_whole() -> None:
    state: dict[str, object] = {}

    async def all_of_it() -> list[str]:
        return [piece async for piece in closing(_answer(state))]

    assert len(asyncio.run(all_of_it())) == 100
    assert state["held"] is False


def test_a_response_that_ends_mid_send_still_closes_its_stream() -> None:
    # A disconnect can land while a piece is being sent, with the stream paused at its
    # yield rather than inside a worker; the response itself closes it then.
    from askpdf.web.app import ClosingStream

    state: dict[str, object] = {}

    async def run() -> None:
        response = ClosingStream(closing(_answer(state)), media_type="text/event-stream")
        sent: list[bytes] = []

        async def send(message: dict[str, object]) -> None:
            if message.get("type") == "http.response.body":
                sent.append(message["body"])  # type: ignore[arg-type]
                if len(sent) == 2:
                    raise OSError("the browser has gone")  # the socket is closed

        async def receive() -> dict[str, object]:
            await asyncio.sleep(10)  # no disconnect message before the failed send
            return {"type": "http.disconnect"}

        scope = {"type": "http", "asgi": {"spec_version": "2.4"}}
        with contextlib.suppress(ClientDisconnect):  # how Starlette reports it
            await response(scope, receive, send)  # type: ignore[arg-type]
        # Checked while the loop runs: asyncio.run closes leftover generators on exit,
        # which a server that keeps running never does.
        assert state["held"] is False

    asyncio.run(run())
