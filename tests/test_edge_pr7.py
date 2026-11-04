"""Edge cases for PR 7: the quote check, reply validation, terminal output, remote guard.

Tests marked xfail(strict=True) expose real defects found by the tester; each names the
finding in its reason. When a defect is fixed its test starts to XPASS and fails the run,
which is the cue to drop the marker.
"""

import time
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from askpdf.answer import parse
from askpdf.cli import main
from askpdf.providers import http
from askpdf.quotes import check, match_key
from tests.fake_models import fake_model_server
from tests.pdfs import make_pdf, text_page
from tests.test_answer import HITS, TEXTS

PAGE = "Passwords SHALL be at least fifteen characters long. Multi-factor tokens are optional."


def _verified(quote: str, page: str = PAGE) -> bool:
    return check(quote, page).verified


# --- the quote check: near misses and Unicode tricks ----------------------------------------


@pytest.mark.parametrize(
    "quote",
    [
        "Passwords SHALL NOT be at least fifteen characters long",  # inserted "not"
        "Passwords be at least fifteen characters long",  # dropped word
        "Passwords SHALL be at least fifteen charactors long",  # one letter
        "Passwords SHALL be at least fifteen characters long Multi-factor tokens",  # skips a word
    ],
)
def test_near_misses_fail(quote: str) -> None:
    assert not _verified(quote)


@pytest.mark.parametrize("quote", ["", " ", "\n\t ", "\u200b", "\u200b \u200b", "\u00ad"])
def test_empty_and_invisible_quotes_never_verify(quote: str) -> None:
    assert not _verified(quote)
    assert not _verified(quote, " ")
    assert not _verified(quote, "")


def test_zero_width_characters_in_a_quote_do_not_hide_a_changed_word() -> None:
    assert not _verified("Passwords SHALL be at least fif\u200bteen charactors long")
    assert not _verified("Passwords SHALL be at lea\u200bst sixteen characters long")


def test_zero_width_characters_are_only_removed_not_matched_against() -> None:
    # Legitimate: the invisible character is not part of the words.
    assert _verified("Passwords SHALL be at lea\u200bst fifteen characters")


def test_turkish_i_does_not_collapse_into_plain_i() -> None:
    tail = "stanbul is a very large city today"
    for page, quote in [
        ("I" + tail, "\u0130" + tail),
        ("I" + tail, "\u0131" + tail),
        ("\u0130" + tail, "i" + tail),
        ("\u0130" + tail, "I" + tail),
        ("\u0131" + tail, "i" + tail),
    ]:
        assert not _verified(quote, page), (page, quote)
    assert _verified("\u0130" + tail, "\u0130" + tail)


def test_combining_marks_are_not_ignored() -> None:
    page = "the r\u00e9sum\u00e9 is on the desk today"
    assert not _verified("the resume is on the desk today", page)
    assert _verified("the re\u0301sume\u0301 is on the desk today", page)  # NFC equal
    assert not _verified("the r\u00e9sum\u00e9\u0323 is on the desk today", page)


def test_nfkc_compatibility_characters_match_the_nfkc_page_only() -> None:
    # The stored page is NFKC already; a quote written with a ligature or fullwidth
    # letters is brought to the same form, so this is a style difference, not a loophole.
    assert _verified("the o\ufb03ce is open on every day", "the office is open on every day")
    assert not _verified("the o\ufb03ce is open on every day", "the orfice is open on every day")
    assert _verified(
        "\uff54\uff48\uff45 office is open on every day", "the office is open on every day"
    )


def test_a_quote_joining_the_end_of_one_page_and_the_start_of_the_next_fails_everywhere() -> None:
    a, b = TEXTS[("a" * 64, 3)], TEXTS[("a" * 64, 7)]
    joined = " ".join(a.split()[-3:] + b.split()[:3])
    assert len(joined.split()) >= 5
    assert not check(joined, a).verified and not check(joined, b).verified
    raw = f'Both [1][2].\nQuotes:\n[1] "{joined}"\n[2] "{joined}"'
    result = parse(raw, HITS, TEXTS)
    assert not result.grounded and result.supported == 0


def test_a_quote_is_not_checked_against_the_prompt_text_of_neighbouring_blocks() -> None:
    # The wing page is [3]; quoting it under [1] must fail even though it was in the prompt.
    raw = 'Ok [1].\nQuotes:\n[1] "tested in the Langley transonic blowdown tunnel"'
    assert not parse(raw, HITS, TEXTS).grounded


@pytest.mark.parametrize(
    "quote",
    ["SP 80063B rev three is the guide", "SP 800 63B rev three is the guide"],
)
def test_digit_hyphens_are_kept(quote: str) -> None:
    assert not _verified(quote, "see SP 800-63B rev three is the guide")


def test_the_four_hyphen_spellings_still_agree_with_odd_neighbours() -> None:
    page = "the multi\u2010factor token is the one we use"
    for spelling in ("multi-factor", "multi\u2010factor", "multi- factor", "multifactor"):
        assert _verified(f"the {spelling} token is the one we use", page)
    assert not _verified(
        "the multi factor token is the one we use", page
    )  # a space is a word break


@pytest.mark.parametrize(
    "text",
    [
        "a- -b",
        "a-- b",
        "\u0130-a",
        "\u00df-a-\u00df",
        "x\u2010\u2010y",
        "a- - b",
        "\u0307-a",
        "A\u2212B",
    ],
)
def test_the_key_is_idempotent_on_hard_cases(text: str) -> None:
    assert match_key(match_key(text)) == match_key(text)
    assert match_key(match_key(text, fold_hyphens=False), fold_hyphens=False) == match_key(
        text, fold_hyphens=False
    )


_ODD = st.text(
    alphabet=st.sampled_from(
        list("aAbB1 \n\t-_'\"")
        + list("\u0130\u00df\u0307\u2010\u2212\u201c\u2019\u00a0\u3000\u200b")
    ),
    max_size=40,
)


@given(_ODD)
def test_the_key_is_idempotent_over_odd_alphabets(text: str) -> None:
    assert match_key(match_key(text)) == match_key(text)


_WORDS = st.lists(
    st.sampled_from(["alpha", "beta", "gamma", "delta", "eps", "Zeta", "eta"]),
    min_size=6,
    max_size=20,
)


@given(_WORDS, st.data())
def test_any_run_of_words_verifies_and_one_foreign_word_never_does(
    words: list[str], data: st.DataObject
) -> None:
    page = " ".join(words)
    start = data.draw(st.integers(0, len(words) - 5))
    end = data.draw(st.integers(start + 5, len(words)))
    run = words[start:end]
    assert check(" ".join(run), page).verified
    run[data.draw(st.integers(0, len(run) - 1))] = "omega"
    assert not check(" ".join(run), page).verified


# --- the quote check: defects ---------------------------------------------------------------


def test_a_quote_may_not_start_inside_a_word() -> None:
    page = "This system is unsafe to use on all machines."
    assert not _verified("safe to use on all machines", page)


def test_a_quote_may_not_end_inside_a_word() -> None:
    page = "Passwords SHALL be at least fifteen characters long."
    assert not _verified("Passwords SHALL be at least fifteen charact", page)


@pytest.mark.parametrize("quote", ["- - - - -", "* * * * *", "1 2 3 4 5", "| | | | |"])
def test_a_quote_without_words_is_not_evidence(quote: str) -> None:
    assert not _verified(quote, f"table {quote} end of table")


def test_eszett_matches_ss_as_casefold_defines() -> None:
    # F6, accepted: ADR-0006 Amendment 1 specifies str.casefold, which folds "\u00df" to
    # "ss". Both are standard spellings of the same word, so the quote's words are on the page.
    assert _verified("the strasse is long and wide", "The Stra\u00dfe is long and wide")


# --- parse: robustness ----------------------------------------------------------------------

_TOKENS = [
    *"[]0123456789, \n\t.:*#\"'_-QuotesNFD",
    "NOT_FOUND",
    "Quotes:",
    "[1]",
    "[2,3]",
    "\u0663",
]
_REPLY = st.lists(st.sampled_from(_TOKENS), max_size=40).map("".join)


@settings(max_examples=300)
@given(_REPLY)
def test_parse_never_crashes_and_stays_consistent(raw: str) -> None:
    result = parse(raw, HITS, TEXTS)
    assert result.raw == raw
    if result.not_found:
        assert result.citations == () and result.supported == 0
    numbers = [c.number for c in result.citations]
    assert numbers == list(range(1, len(numbers) + 1))
    assert 0 <= result.supported <= result.sentences
    assert all(
        q.verified is False for c in result.citations for q in c.quotes if not q.quote.strip()
    )
    import re

    for marker in re.findall(r"\[(\d+)\]", result.text):
        assert 1 <= int(marker) <= max(len(numbers), 0) or result.not_found


@given(st.text(max_size=200))
def test_parse_never_crashes_on_arbitrary_text(raw: str) -> None:
    parse(raw, HITS, TEXTS)
    parse(raw, (), {})


def test_hostile_replies_parse_in_linear_time() -> None:
    n = 50_000
    replies = [
        "[" * n,
        "[1" + ",1" * n,
        " " * n + "x",
        "*" * n + "\n" + "[1]" * 5000,
        "\n".join(["Quote"] * 5000),
        "Quotes:\n" + "[1] " * 5000,
        "a [1]\nQuotes:\n[1] " + "a " * n + "b",
        "[1] " + "\u00a0" * 1000,
        "Quotes:\n" + "-\n" * n,  # punctuation lines: a prefix crossing lines is quadratic
        "Quotes:\n" + "_\n" * n,
    ]
    start = time.perf_counter()
    for raw in replies:
        parse(raw, HITS, TEXTS)
    # About 1 s when linear; a quadratic pass at n = 50,000 takes minutes. The margin is
    # wide because a 2 s limit failed under parallel load with Ollama busy.
    assert time.perf_counter() - start < 10.0


def test_a_whitespace_run_inside_a_quote_line_is_not_quadratic() -> None:
    raw = 'a [1]\nQuotes:\n[1] "a' + " " * 20_000 + 'b"'
    start = time.perf_counter()
    parse(raw, HITS, TEXTS)
    assert time.perf_counter() - start < 1.0


def test_a_huge_marker_number_in_the_answer_is_dropped() -> None:
    result = parse("It is so [" + "9" * 5000 + "].", HITS, TEXTS)
    assert result.text == "It is so ."


def test_a_huge_marker_number_in_a_quote_line_is_ignored() -> None:
    parse("x [1]\nQuotes:\n[" + "9" * 5000 + '] "some quoted words go right here"', HITS, TEXTS)


def test_a_stray_quote_heading_does_not_swallow_the_answer() -> None:
    result = parse("Quote\nFifteen characters [1].", HITS, TEXTS)
    assert "Fifteen" in result.text


# --- parse: markers and NOT_FOUND -----------------------------------------------------------


def test_renumbering_follows_first_use_and_quotes_follow_the_page() -> None:
    raw = (
        "Wing [3][3]. Lock [2, 3]. Pass [1].\n"
        "Quotes:\n"
        '[3] "tested in the Langley transonic blowdown tunnel"\n'
        '[2] "Accounts are locked after one hundred failed"\n'
    )
    result = parse(raw, HITS, TEXTS)
    assert result.text == "Wing [1][1]. Lock [2][1]. Pass [3]."
    assert [(c.number, c.hit.page) for c in result.citations] == [(1, 1), (2, 7), (3, 3)]
    assert [c.verified for c in result.citations] == [True, True, False]
    assert (result.sentences, result.supported) == (3, 2)


def test_an_invalid_number_never_lends_its_quote_to_another_page() -> None:
    raw = 'Ok [1].\nQuotes:\n[9] "Passwords SHALL be at least fifteen characters"'
    result = parse(raw, HITS, TEXTS)
    assert not result.grounded and result.citations[0].quotes == ()


@pytest.mark.parametrize("raw", ["not_found", "Not found", "NOT FOUND", "NOT_FOUNDS", "NOTFOUND"])
def test_other_spellings_of_not_found_are_answers_not_abstentions(raw: str) -> None:
    result = parse(raw, HITS, TEXTS)
    assert not result.not_found and not result.grounded


@pytest.mark.parametrize("raw", ["\n\n  **NOT_FOUND**  \n", "`NOT_FOUND`", "NOT_FOUND."])
def test_not_found_with_only_decoration_or_invalid_numbers(raw: str) -> None:
    assert parse(raw, HITS, TEXTS).not_found


def test_not_found_with_a_bogus_number_is_an_ungrounded_answer_not_an_abstention() -> None:
    result = parse("NOT_FOUND [9]", HITS, TEXTS)
    assert not result.grounded and result.dropped == (9,)


def test_not_found_followed_by_a_cited_claim_is_not_an_abstention() -> None:
    result = parse("NOT_FOUND\nBut passwords need fifteen characters [1].", HITS, TEXTS)
    assert not result.not_found and result.citations


# --- terminal output ------------------------------------------------------------------------

BAD = "\x1b[2J\x1b]0;pwned\x07\x9b31m\u202e\r\x08\u2028\u200b"
BAD_CHARS = "\x1b\x07\x9b\u202e\r\x08\u2028\u200b\x00"


@pytest.fixture
def folder(tmp_path: Path) -> Path:
    make_pdf(
        tmp_path / "f" / "policy.pdf",
        [text_page("Passwords need fifteen characters or more in every case.")],
    )
    return tmp_path / "f"


def _ask(folder: Path, *extra: str) -> list[str]:
    return [
        "ask",
        str(folder),
        "how many characters do passwords need",
        "--index",
        str(folder.parent / "i.db"),
        *extra,
    ]


def _index(folder: Path) -> None:
    assert main(["index", str(folder), "--index", str(folder.parent / "i.db")]) == 0


def test_control_characters_in_answer_text_and_quotes_are_escaped(
    folder: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    _index(folder)
    capsys.readouterr()
    with fake_model_server() as (url, models):
        monkeypatch.setenv("ASKPDF_OLLAMA_URL", url)
        models.chat_reply = [
            f"Fifteen {BAD} [1].\nQuotes:\n",
            '[1] "Passwords need fifteen characters or more"\n',
            f'[1] "{BAD} five words that are not on the page"',
        ]
        assert main(_ask(folder, "--show-context")) == 0
    captured = capsys.readouterr()
    for stream in (captured.out, captured.err):
        assert not any(ch in stream for ch in BAD_CHARS)
    assert "\\x1b" in captured.out


def test_a_hostile_file_name_is_escaped_everywhere_it_is_printed(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    name = "po\x1b[31mli\x07cy\u202e.pdf"
    try:
        make_pdf(tmp_path / "f" / name, [text_page("Passwords need fifteen characters or more.")])
    except OSError:
        pytest.skip("file system refuses control characters in names")
    folder = tmp_path / "f"
    _index(folder)
    capsys.readouterr()
    with fake_model_server() as (url, models):
        monkeypatch.setenv("ASKPDF_OLLAMA_URL", url)
        models.chat_reply = ['Ok [1].\nQuotes:\n[1] "Passwords need fifteen characters or more"']
        assert main(_ask(folder, "--show-context")) == 0
    captured = capsys.readouterr()
    for stream in (captured.out, captured.err):
        assert not any(ch in stream for ch in BAD_CHARS)
    assert "\\x1b" in captured.out


def test_sources_come_from_the_index_not_from_the_reply(
    folder: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    _index(folder)
    capsys.readouterr()
    with fake_model_server() as (url, models):
        monkeypatch.setenv("ASKPDF_OLLAMA_URL", url)
        models.chat_reply = [
            "Fifteen [1] (evil.pdf, page 99).\nQuotes:\n",
            '[1] "Passwords need fifteen characters or more" (evil.pdf page 99)\n',
            '[evil.pdf p. 99] "Passwords need fifteen characters or more"',
        ]
        main(_ask(folder))
    out = capsys.readouterr().out
    sources = out.split("\nSources:\n", 1)[1]
    headings = [line for line in sources.splitlines() if line.startswith("[")]
    assert headings == ["[1] policy.pdf  page 1"]
    assert "[1] policy.pdf  page 1" in sources


# --- the key and the remote guard -----------------------------------------------------------

KEY = "sk-live-SECRET-0123456789"


@pytest.mark.parametrize(
    "url",
    [
        "https://api.example.com/v1",
        "http://127.0.0.1.evil.com/v1",
        "http://localhost.evil.com/v1",
        "http://127.0.0.1@evil.com/v1",
        "http://evil.com/v1#@127.0.0.1",
        "http://evil.com/v1?@localhost",
        "http://0.0.0.0/v1",
        "http://[::ffff:8.8.8.8]/v1",
        "http://localtest.me/v1",
        "http://127.0.0.1:80@evil.com/v1",
    ],
)
def test_the_remote_guard_refuses_lookalike_hosts_and_never_prints_the_key(
    url: str, folder: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    _index(folder)
    capsys.readouterr()
    monkeypatch.setenv("ASKPDF_API_KEY", KEY)
    monkeypatch.delenv("ASKPDF_ALLOW_REMOTE", raising=False)
    assert main(_ask(folder, "--api-url", url, "--model", "m")) == 1
    captured = capsys.readouterr()
    assert "--allow-remote" in captured.err
    assert KEY not in captured.out + captured.err


@pytest.mark.parametrize("value", ["", "0", "no", "false", "TRUE", "Yes", " 1"])
def test_only_the_documented_environment_values_allow_remote(
    value: str, folder: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    _index(folder)
    capsys.readouterr()
    monkeypatch.setenv("ASKPDF_API_KEY", KEY)
    monkeypatch.setenv("ASKPDF_ALLOW_REMOTE", value)
    assert main(_ask(folder, "--api-url", "https://api.example.com/v1", "--model", "m")) == 1
    assert "--allow-remote" in capsys.readouterr().err


@pytest.mark.parametrize(
    "url", ["http://127.0.0.1/v1", "http://localhost:8080/v1", "http://[::1]/v1"]
)
def test_loopback_hosts_are_local(url: str) -> None:
    assert http.is_local(url)


def test_the_key_is_never_printed_on_success_failure_or_show_context(
    folder: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    _index(folder)
    capsys.readouterr()
    monkeypatch.setenv("ASKPDF_API_KEY", KEY)
    with fake_model_server() as (url, models):
        args = _ask(folder, "--api-url", f"{url}/v1", "--model", "m", "--show-context")
        models.chat_reply = ['Ok [1].\nQuotes:\n[1] "Passwords need fifteen characters or more"']
        assert main(args) == 0
        models.chat_reply = [f"echo {KEY[:6]} [1]"]
        main(args)
        models.fail_with = 500
        assert main(args) == 1
        models.fail_with = 401
        assert main(args) == 1
    captured = capsys.readouterr()
    assert KEY not in captured.out + captured.err


def test_the_ollama_path_sends_no_key_even_when_one_is_set(
    folder: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _index(folder)
    monkeypatch.setenv("ASKPDF_API_KEY", KEY)
    with fake_model_server() as (url, models):
        monkeypatch.setenv("ASKPDF_OLLAMA_URL", url)
        models.chat_reply = ["NOT_FOUND"]
        assert main(_ask(folder)) == 0
        assert all(KEY not in str(headers) for _, _, headers in models.requests)
