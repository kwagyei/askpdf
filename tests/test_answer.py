"""Answering: the prompt, the event order, and validation of what the model writes."""

from collections.abc import Iterator
from pathlib import Path

import pytest

from askpdf.answer import (
    MAX_PAGE_CHARS,
    NOT_FOUND,
    SYSTEM,
    Final,
    Pages,
    Token,
    answer,
    build_prompt,
    context_window,
    parse,
)
from askpdf.cli import main
from askpdf.retrieve import PageHit
from askpdf.snapshot import Chunk
from tests.fake_models import fake_model_server
from tests.pdfs import make_pdf, text_page

TEXTS = {
    ("a" * 64, 3): "Passwords SHALL be at least fifteen characters long for single-factor use.",
    ("a" * 64, 7): "Accounts are locked after one hundred failed attempts in a row.",
    ("b" * 64, 1): "The wing was tested in the Langley transonic blowdown tunnel in 1956.",
}


def _hit(sha: str, page: int, label: str | None = None, path: str = "docs/a.pdf") -> PageHit:
    text = TEXTS[(sha, page)]
    return PageHit(sha, page, label or str(page), (path,), 1.0, Chunk(1, sha, page, 0, text))


HITS = (
    _hit("a" * 64, 3, label="iii"),
    _hit("a" * 64, 7),
    _hit("b" * 64, 1, path="reports/wing.pdf"),
)


class ScriptedChat:
    def __init__(self, *pieces: str) -> None:
        self.pieces = pieces
        self.calls: list[tuple[str, str]] = []

    def stream(self, system: str, user: str) -> Iterator[str]:
        self.calls.append((system, user))
        yield from self.pieces


# --- the prompt ---------------------------------------------------------------------------


def test_the_prompt_numbers_one_block_per_page_with_index_names() -> None:
    prompt = build_prompt("How long?", HITS, TEXTS)
    assert "[1] a.pdf, page 3 (printed iii)\nPasswords SHALL" in prompt
    assert "[2] a.pdf, page 7\nAccounts" in prompt  # no "printed" when it is the same
    assert "[3] wing.pdf, page 1\n" in prompt  # the file name, not its folders
    assert prompt.endswith("Question: How long?")


def test_a_long_page_is_shortened_in_the_prompt_only() -> None:
    sha = "c" * 64
    texts = {(sha, 1): "word " * 2000}
    hit = PageHit(sha, 1, "1", ("c.pdf",), 1.0, Chunk(9, sha, 1, 0, "word"))
    prompt = build_prompt("q", [hit], texts)
    assert len(prompt) < MAX_PAGE_CHARS + 200 and " …" in prompt


def test_the_context_window_fits_the_prompt_and_stays_bounded() -> None:
    assert context_window("", 768) == 2048
    assert context_window("x" * 30_000, 768) >= (len(SYSTEM) + 30_000) // 3 + 768
    assert context_window("x" * 30_000, 768) % 1024 == 0
    assert context_window("x" * 10**6, 768) == 32768


# --- events -------------------------------------------------------------------------------


def test_events_come_in_order_pages_tokens_final() -> None:
    chat = ScriptedChat(
        "Fifteen ", "characters [1].\nQuotes:\n", '[1] "be at least fifteen characters long"'
    )
    events = list(answer("How long?", HITS, TEXTS, chat))
    assert isinstance(events[0], Pages) and events[0].hits == HITS
    assert [e.text for e in events[1:-1] if isinstance(e, Token)] == list(chat.pieces)
    final = events[-1]
    assert isinstance(final, Final) and final.answer.grounded
    assert chat.calls[0][0] == SYSTEM


def test_with_no_pages_the_model_is_not_asked() -> None:
    chat = ScriptedChat("should not be used")
    events = list(answer("anything", [], TEXTS, chat))
    assert chat.calls == []
    assert isinstance(events[-1], Final) and events[-1].answer.not_found


# --- validation ---------------------------------------------------------------------------


def test_numbers_are_renumbered_by_first_use() -> None:
    raw = (
        "Locked after 100 attempts [2]. Passwords are 15 characters [1].\n"
        "Quotes:\n"
        '[1] "Passwords SHALL be at least fifteen characters"\n'
        '[2] "locked after one hundred failed attempts"\n'
    )
    result = parse(raw, HITS, TEXTS)
    assert result.text == "Locked after 100 attempts [1]. Passwords are 15 characters [2]."
    assert [(c.number, c.hit.page) for c in result.citations] == [(1, 7), (2, 3)]
    assert all(c.verified for c in result.citations)
    assert (result.sentences, result.supported) == (2, 2)


def test_numbers_outside_the_context_are_dropped() -> None:
    raw = 'It is 15 [1][4]. Also [0] and [12].\nQuotes:\n[4] "made up words from nowhere at all"'
    result = parse(raw, HITS, TEXTS)
    assert result.text == "It is 15 [1]. Also  and ."
    assert result.dropped == (4, 0, 12)
    assert [c.hit.page for c in result.citations] == [3]


def test_lists_of_numbers_in_one_bracket() -> None:
    result = parse("Both [1, 3]. Again [3,1].", HITS, TEXTS)
    assert result.text == "Both [1][2]. Again [2][1]."
    assert [c.hit.page for c in result.citations] == [3, 1]


def test_a_citation_without_a_quote_is_unverified() -> None:
    result = parse("Fifteen characters [1].", HITS, TEXTS)
    assert not result.grounded and result.citations[0].quotes == ()
    assert (result.sentences, result.supported) == (1, 0)


def test_one_verified_quote_grounds_the_answer() -> None:
    raw = (
        "Fifteen characters [1]. Tested in 1956 [3].\nQuotes:\n"
        '[1] "be at least fifteen characters long"\n'
        '[3] "tested in the Langley wind tunnel"\n'
    )
    result = parse(raw, HITS, TEXTS)
    assert result.grounded
    assert [c.verified for c in result.citations] == [True, False]
    assert result.citations[1].quotes[0].reason == "not found on the page"
    assert (result.sentences, result.supported) == (2, 1)


def test_a_quote_is_checked_against_its_own_page_only() -> None:
    raw = 'Locked [2].\nQuotes:\n[2] "be at least fifteen characters long"'
    assert not parse(raw, HITS, TEXTS).grounded  # true words, but from page [1]


@pytest.mark.parametrize(
    "section",
    [
        'Quotes:\n[1] "be at least fifteen characters long"',
        '**Quotes:**\n- [1] "be at least fifteen characters long"',
        "## Quote\n[1]: \u201cbe at least fifteen characters long\u201d.",
        'quotes: [1] "be at least fifteen characters long"',
        "Quotes:\n\n  [1] \u2013 be at least fifteen characters long.",
    ],
)
def test_quote_sections_in_the_forms_small_models_write(section: str) -> None:
    result = parse(f"Fifteen [1].\n\n{section}", HITS, TEXTS)
    assert result.text == "Fifteen [1]."
    assert result.grounded


def test_a_number_only_quoted_still_becomes_a_source() -> None:
    raw = 'Fifteen characters.\nQuotes:\n[1] "be at least fifteen characters long"'
    result = parse(raw, HITS, TEXTS)
    assert result.grounded and result.citations[0].number == 1
    assert (result.sentences, result.supported) == (1, 0)  # the sentence carries no marker


@pytest.mark.parametrize(
    "raw", [NOT_FOUND, " NOT_FOUND.\n", "**NOT_FOUND**", "`NOT_FOUND`\n\nSorry."]
)
def test_not_found_in_its_forms(raw: str) -> None:
    result = parse(raw, HITS, TEXTS)
    assert result.not_found and result.text == NOT_FOUND and not result.citations


def test_not_found_with_a_valid_citation_is_an_answer() -> None:
    result = parse("NOT_FOUND\nBut it is 15 characters [1].", HITS, TEXTS)
    assert not result.not_found and len(result.citations) == 1


def test_not_found_inside_a_sentence_is_an_answer() -> None:
    assert not parse("The value was NOT_FOUND in the table.", HITS, TEXTS).not_found


def test_file_names_in_model_text_do_not_become_sources() -> None:
    raw = 'See secrets.pdf page 99 [1].\nQuotes:\n[1] "be at least fifteen characters long"'
    result = parse(raw, HITS, TEXTS)
    assert [(c.hit.paths, c.hit.page) for c in result.citations] == [(("docs/a.pdf",), 3)]


def test_an_empty_reply_is_an_ungrounded_answer() -> None:
    result = parse("", HITS, TEXTS)
    assert not result.not_found and not result.grounded and result.sentences == 0


# --- the ask command ----------------------------------------------------------------------


@pytest.fixture
def folder(tmp_path: Path) -> Path:
    make_pdf(
        tmp_path / "f" / "policy.pdf",
        [
            text_page("Passwords need fifteen characters or more in every case."),
            text_page("Cookies need the Secure flag on every response."),
        ],
    )
    return tmp_path / "f"


def _ask(folder: Path, question: str = "how many characters do passwords need") -> list[str]:
    return ["ask", str(folder), question, "--index", str(folder.parent / "i.db")]


def _indexed(folder: Path) -> None:
    assert main(["index", str(folder), "--index", str(folder.parent / "i.db")]) == 0


def test_ask_prints_the_answer_and_checked_sources(
    folder: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    _indexed(folder)
    capsys.readouterr()
    with fake_model_server() as (url, models):
        monkeypatch.setenv("ASKPDF_OLLAMA_URL", url)
        models.chat_reply = [
            "Fifteen characters [1].\n",
            'Quotes:\n[1] "Passwords need fifteen characters or more"',
        ]
        assert main(_ask(folder)) == 0
        body = models.requests[-1][1]
    out = capsys.readouterr().out
    assert "Fifteen characters [1]." in out
    assert "[1] policy.pdf  page 1" in out and "✓" in out
    assert body["options"]["num_ctx"] >= 2048 and body["options"]["num_predict"] == 768


def test_an_ungrounded_answer_exits_3_with_a_warning(
    folder: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    _indexed(folder)
    capsys.readouterr()
    with fake_model_server() as (url, models):
        monkeypatch.setenv("ASKPDF_OLLAMA_URL", url)
        models.chat_reply = ['Twenty [1].\nQuotes:\n[1] "Passwords need twenty characters or more"']
        assert main(_ask(folder)) == 3
    captured = capsys.readouterr()
    assert "✗ not found on the page" in captured.out
    assert "ungrounded" in captured.err


def test_not_found_is_said_plainly(
    folder: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    _indexed(folder)
    capsys.readouterr()
    with fake_model_server() as (url, models):
        monkeypatch.setenv("ASKPDF_OLLAMA_URL", url)
        models.chat_reply = [NOT_FOUND]
        assert main(_ask(folder)) == 0
    assert "not in these documents" in capsys.readouterr().out


def test_no_matching_page_answers_without_the_model(
    folder: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _indexed(folder)
    capsys.readouterr()
    # ASKPDF_OLLAMA_URL points at a closed port, so asking the model would fail.
    assert main(_ask(folder, "zebra giraffe")) == 0
    assert "not in these documents" in capsys.readouterr().out


def test_ask_without_ollama_says_how_to_start_it(
    folder: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _indexed(folder)
    capsys.readouterr()
    assert main(_ask(folder)) == 1
    assert "ollama serve" in capsys.readouterr().err


def test_ask_refuses_a_remote_ollama_without_permission(
    folder: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    _indexed(folder)
    capsys.readouterr()
    monkeypatch.setenv("ASKPDF_OLLAMA_URL", "https://models.example.com")
    assert main(_ask(folder)) == 1
    assert "--allow-remote" in capsys.readouterr().err


def test_ask_with_an_openai_compatible_endpoint(
    folder: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    _indexed(folder)
    capsys.readouterr()
    with fake_model_server() as (url, models):
        monkeypatch.setenv("ASKPDF_API_KEY", "sk-test-not-real")
        models.chat_reply = [
            'Fifteen [1].\nQuotes:\n[1] "Passwords need fifteen characters or more"'
        ]
        args = [*_ask(folder), "--api-url", f"{url}/v1", "--model", "m"]
        assert main(args) == 0
        assert models.requests[-1][2]["Authorization"] == "Bearer sk-test-not-real"
    assert "✓" in capsys.readouterr().out
    assert main([*_ask(folder), "--api-url", "http://127.0.0.1:9/v1"]) == 1
    assert "--model" in capsys.readouterr().err


def test_show_context_prints_the_pages_given_to_the_model(
    folder: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    _indexed(folder)
    capsys.readouterr()
    with fake_model_server() as (url, models):
        monkeypatch.setenv("ASKPDF_OLLAMA_URL", url)
        models.chat_reply = [NOT_FOUND]
        assert main([*_ask(folder), "--show-context"]) == 0
    assert "[1] policy.pdf, page 1\nPasswords need" in capsys.readouterr().err
