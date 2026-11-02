# ruff: noqa: RUF001, RUF005
"""Adversarial edge cases for the v0.1.0 release (PR 11): the linear hyphen resolver, the
bench command, ``eval --check``, the release workflow and the README's claims.

Tests marked ``xfail(strict=True)`` are genuine findings (F<n> in the report,
~/Vault/Projects/code-projects/tester/2026-11-03-askpdf-pr11.md); they flip to failures
when the problem is fixed, so the marker gets removed then.

Nothing here runs Ollama or scores the real test split: guard tests use a synthetic eval
directory, and the committed results are only read.
"""

import itertools
import json
import re
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
from hypothesis import example, given, settings
from hypothesis import strategies as st

from askpdf.bench import percentile, spread
from askpdf.cli import main
from askpdf.extract import clean
from askpdf.extract.clean import HYPHENS, Lexicon, resolve_hyphens
from askpdf.providers import http, ollama
from tests.eval_pins import EVAL, synthetic_eval
from tests.fake_models import fake_model_server

ROOT = EVAL.parent
in_repo = pytest.mark.skipif(not (EVAL / "questions").is_dir(), reason="eval/ is not in the sdist")

# --- the old resolver, as it was on main, for comparison -------------------------------------

_OLD_TAIL = re.compile(r"[\w\-‐]*\Z")
_OLD_HEAD = re.compile(r"\w+")


def _old_join(line: str, following: str, lexicon: Lexicon) -> str:
    left_word = _OLD_TAIL.findall(line[:-1])[0].lstrip(HYPHENS)
    right_word = _OLD_HEAD.findall(following)[0]
    drop = (
        line[-2].isalpha()
        and following[0].islower()
        and not any(h in left_word for h in HYPHENS)
        and lexicon.prefers_solid(left_word, right_word)
    )
    return (line[:-1] if drop else line) + following


def old_resolve_hyphens(
    pages: Sequence[str],
    lexicon: Lexicon | None = None,
    protected: Sequence[set[int]] | None = None,
) -> list[str]:
    """Verbatim from ``git show main:src/askpdf/extract/clean.py`` (quadratic)."""
    import unicodedata

    if lexicon is None:
        lexicon = Lexicon.build(pages)
    resolved = []
    for n, page in enumerate(pages):
        keep_apart = protected[n] if protected is not None else set()
        out: list[str] = []
        last_protected = False
        for i, line in enumerate(page.split("\n")):
            joinable = out and not last_protected and i not in keep_apart
            if joinable and clean._ends_with_break(out[-1], line):
                out[-1] = _old_join(out[-1], line, lexicon)
            else:
                out.append(line)
                last_protected = i in keep_apart
        resolved.append(unicodedata.normalize("NFC", "\n".join(out)))
    return resolved


def lexicon_preferring_solid(*words: str) -> Lexicon:
    """A lexicon in which each word is spelled solid, so a break inside it is rejoined."""
    lexicon = Lexicon()
    for word in words:
        lexicon.add(word)
    return lexicon


TOKENS = [
    "ab",
    "cd",
    "abcd",
    "a",
    "x",
    "é",
    "é",  # decomposed: a base letter and a combining mark
    "ß",
    "Straße",
    "x1",
    "1",
    "12",
    "_",
    "²",
    "Ⅳ",
    "ǅ",
    "٣",  # Arabic-indic digit: Nd and \w
    "AB",
    "multi",
    "factor",
    "multifactor",
]
SEPARATORS = ["-\n", "‐\n", "\n", " ", "-", "‐", "--\n", "-‐\n", "\n\n", " -\n", "- \n"]


@st.composite
def pages_of_pieces(draw: st.DrawFn) -> list[str]:
    """Pages made of word pieces and the separators that make hyphen events."""
    pages = []
    for _ in range(draw(st.integers(1, 3))):
        pieces = draw(st.lists(st.tuples(st.sampled_from(TOKENS), st.sampled_from(SEPARATORS))))
        pages.append("".join(a + b for a, b in pieces))
    return pages


EXTRA_SOLID = ("abcd", "multifactor", "exé", "Straße", "ab1", "x1", "ß", "1212")


@given(pages=pages_of_pieces(), data=st.data())
@settings(max_examples=600)
@example(pages=["ab-\ncd"], data=None)
def test_new_resolver_equals_the_old_one_on_piece_text(
    pages: list[str], data: st.DataObject | None
) -> None:
    protected = None
    if data is not None and data.draw(st.booleans()):
        protected = [
            set(
                data.draw(
                    st.lists(st.integers(0, max(0, p.count("\n"))), max_size=3),
                )
            )
            for p in pages
        ]
    for lexicon in (
        Lexicon.build(pages),
        lexicon_preferring_solid(*EXTRA_SOLID),
        lexicon_preferring_solid("a", "ab", "abcd", "xb", "12"),
    ):
        assert resolve_hyphens(pages, lexicon, protected) == old_resolve_hyphens(
            pages, lexicon, protected
        )


CHARS = "ab c\n-‐­_1é́²Ⅳ\t　xB"


@given(
    pages=st.lists(st.text(alphabet=CHARS, max_size=40), min_size=1, max_size=3),
    solid=st.lists(st.text(alphabet="abcxé1B", min_size=2, max_size=4), max_size=4),
)
@settings(max_examples=800)
def test_new_resolver_equals_the_old_one_on_arbitrary_characters(
    pages: list[str], solid: list[str]
) -> None:
    for lexicon in (Lexicon.build(pages), lexicon_preferring_solid(*solid)):
        assert resolve_hyphens(pages, lexicon) == old_resolve_hyphens(pages, lexicon)


@given(text=st.text(alphabet="ab-‐\n", max_size=14))
def test_every_short_string_over_the_hyphen_alphabet(text: str) -> None:
    """Dense coverage of the hyphen/newline/letter structure: lines of one character,
    hyphen-only lines, runs of hyphens."""
    lexicon = lexicon_preferring_solid("aa", "ab", "ba", "bb", "aaa", "abab")
    assert resolve_hyphens([text], lexicon) == old_resolve_hyphens([text], lexicon)


def test_exhaustive_short_strings() -> None:
    lexicon = lexicon_preferring_solid("aa", "ab", "ba", "bb", "ab1")
    for n in range(1, 8):
        for combo in itertools.product("ab-‐\n1", repeat=n):
            text = "".join(combo)
            assert resolve_hyphens([text], lexicon) == old_resolve_hyphens([text], lexicon), text


EDGE_CASES = {
    "one-character lines": "a-\nb\na-\nb\nc",
    "single hyphen then letter": "-\nb",
    "hyphen-only lines": "a\n-\n-\nb",
    "hyphen-only line then break": "-\n-\nab-\ncd",
    "digit before break": "x1-\n2",
    "digit after break is not lower": "ab-\n1d",
    "chain dropped then kept": "ab-\ncd-\nef-\ngh",
    "chain kept then dropped": "multi-\nfactor-\nab-\ncd",
    "u2010 break": "ab‐\ncd",
    "u2010 inside the word": "out‐of-\nband",
    "hyphen run before the break": "ab--\ncd",
    "mixed hyphen run": "ab-‐\ncd",
    "leading hyphens": "--ab-\ncd",
    "word after a space": "see ab-\ncd",
    "word after punctuation": "(ab-\ncd",
    "underscore word": "ab_cd-\nef",
    "unicode word": "Straße-\nxx\nstra-\nße",
    "combining mark in the word": "é-\nx",
    "superscript digit": "ab²-\ncd",
    "uppercase right half": "ab-\nCd",
    "blank lines": "ab-\n\ncd",
    "trailing hyphen at the end": "ab-\ncd-",
}


@pytest.mark.parametrize("text", EDGE_CASES.values(), ids=EDGE_CASES.keys())
def test_edge_case_gives_the_same_text_as_the_old_resolver(text: str) -> None:
    for lexicon in (
        Lexicon.build([text]),
        lexicon_preferring_solid("abcd", "multifactor", "ab", "efgh", "x12", "ab_cd", "Straße"),
    ):
        assert resolve_hyphens([text], lexicon) == old_resolve_hyphens([text], lexicon)


def test_a_chain_mixing_dropped_and_kept_hyphens() -> None:
    lexicon = lexicon_preferring_solid("abcd", "efgh")
    assert resolve_hyphens(["ab-\ncd-\nef-\ngh"], lexicon) == [
        "abcd-ef-gh"
    ]  # a word already holding a hyphen is never dropped


def test_protected_lines_are_never_joined() -> None:
    lexicon = lexicon_preferring_solid("abcd")
    assert resolve_hyphens(["ab-\ncd"], lexicon, [{1}]) == ["ab-\ncd"]
    assert resolve_hyphens(["ab-\ncd"], lexicon, [{0}]) == ["ab-\ncd"]
    assert resolve_hyphens(["ab-\ncd"], lexicon, [set()]) == ["abcd"]
    # a protected line that is followed by a plain line stays a barrier for the line after it
    assert resolve_hyphens(["x\nab-\ncd-\nef"], lexicon, [{2}]) == ["x\nab-\ncd-\nef"]


def test_hyphen_only_and_one_character_lines_are_left_alone() -> None:
    lexicon = lexicon_preferring_solid("ab")
    assert resolve_hyphens(["-\nb"], lexicon) == ["-\nb"]
    assert resolve_hyphens(["a\n-\n-\nb"], lexicon) == ["a\n-\n-\nb"]


def test_u2010_breaks_like_an_ascii_hyphen() -> None:
    lexicon = lexicon_preferring_solid("abcd")
    assert resolve_hyphens(["ab‐\ncd"], lexicon) == ["abcd"]
    assert resolve_hyphens(["out‐of-\nband"], lexicon) == ["out‐of-band"]


def test_a_unicode_word_before_the_hyphen_is_found_whole() -> None:
    lexicon = lexicon_preferring_solid("Straße")
    assert resolve_hyphens(["Stra-\nße"], lexicon) == ["Straße"]
    assert resolve_hyphens(["Strañ-\nße"], lexicon) == ["Strañ-ße"]


def test_a_very_long_chain_is_resolved_without_recursion_or_copying() -> None:
    lexicon = lexicon_preferring_solid("ab")
    text = "\n".join(["a-"] * 30_000 + ["b"])
    assert resolve_hyphens([text], lexicon) == ["a-" * 30_000 + "b"]


# --- bench: percentile and spread -------------------------------------------------------------

finite = st.floats(min_value=0, max_value=1e6, allow_nan=False)


@given(values=st.lists(finite, min_size=1, max_size=60), share=st.floats(0.0, 1.0))
def test_percentile_is_always_a_measured_value_within_range(
    values: list[float], share: float
) -> None:
    result = percentile(values, share)
    assert result in values and min(values) <= result <= max(values)


@given(values=st.lists(finite, min_size=1, max_size=60))
def test_p50_never_exceeds_p95_and_p95_of_few_values_is_the_maximum(values: list[float]) -> None:
    s = spread(values)
    assert s["p50"] <= s["p95"] and s["n"] == len(values)
    if len(values) < 20:
        assert s["p95"] == round(max(values), 3)  # nearest rank: below 20 runs p95 is the max


# --- bench: the command ------------------------------------------------------------------------


@pytest.fixture
def eval_dir(tmp_path: Path) -> Path:
    synthetic_eval(tmp_path / "eval")
    return tmp_path / "eval"


@in_repo
@pytest.mark.parametrize("args", [["--runs", "0"], ["--answers", "-1"], ["--answers", "x"]])
def test_bench_rejects_bad_counts(args: list[str], eval_dir: Path) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["bench", "--eval-dir", str(eval_dir), *args])
    assert exc.value.code == 2


@in_repo
def test_bench_accepts_zero_answers_and_records_the_version(
    eval_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(["bench", "--eval-dir", str(eval_dir), "--answers", "0", "--runs", "1"])
    assert code == 0
    printed = json.loads(capsys.readouterr().out)
    from askpdf import __version__

    assert printed["askpdf"] == __version__
    assert printed["indexing"]["documents"] == len(list((eval_dir / "corpus").glob("*.pdf")))
    assert set(printed) == {"askpdf", "date", "machine", "python", "indexing", "search_ms"}


@in_repo
def test_bench_with_no_dev_questions_fails_cleanly(
    eval_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (eval_dir / "questions" / "dev.jsonl").write_text("", encoding="utf-8")
    assert main(["bench", "--eval-dir", str(eval_dir), "--answers", "0", "--runs", "1"]) == 1
    assert "no entries" in capsys.readouterr().err


@in_repo
def test_bench_unwritable_json_path_is_an_error_not_a_traceback(
    eval_dir: Path, tmp_path: Path
) -> None:
    target = tmp_path / "is-a-directory"
    target.mkdir()
    code = main(
        ["bench", "--eval-dir", str(eval_dir), "--answers", "0", "--runs", "1"]
        + ["--json", str(target)]
    )
    assert code == 1


def test_unload_asks_ollama_to_drop_the_chat_model() -> None:
    with fake_model_server() as (url, fake), pytest.raises(http.ProviderError):
        ollama.unload(url)  # the fake has no /api/generate
    assert fake.requests[0][0] == "/api/generate"
    assert fake.requests[0][1] == {"model": ollama.CHAT_MODEL, "keep_alive": 0}


# --- eval --check ------------------------------------------------------------------------------


def _release_file(eval_dir: Path, tag: str, stages: dict[str, Any]) -> Path:
    path = eval_dir / "results" / f"test-{tag}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"release": tag, "stages": stages}), encoding="utf-8")
    return path


@in_repo
def test_check_needs_a_release_tag(eval_dir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert (
        main(["eval", "retrieval", "--split", "test", "--check", "--eval-dir", str(eval_dir)]) == 1
    )
    assert "--release" in capsys.readouterr().err


@in_repo
@pytest.mark.parametrize("tag", ["v1", "v1.2", "1.2.3", "v1.2.3-rc1", "v01.2.3x", "v١.٢.٣"])
def test_check_rejects_a_malformed_release_tag(
    tag: str, eval_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(
        ["eval", "retrieval", "--split", "test", "--release", tag, "--check"]
        + ["--eval-dir", str(eval_dir)]
    )
    assert code == 1 and "--release vX.Y.Z" in capsys.readouterr().err


@in_repo
def test_check_a_release_path_traversal_tag_is_rejected(
    eval_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(
        ["eval", "retrieval", "--split", "test", "--release", "v1.2.3/../../x", "--check"]
        + ["--eval-dir", str(eval_dir)]
    )
    assert code == 1


@in_repo
def test_check_with_a_corrupt_release_file_is_an_error(
    eval_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (eval_dir / "results").mkdir(parents=True)
    (eval_dir / "results" / "test-v9.9.9.json").write_text("{not json", encoding="utf-8")
    code = main(
        ["eval", "retrieval", "--split", "test", "--release", "v9.9.9", "--check"]
        + ["--eval-dir", str(eval_dir)]
    )
    assert code == 1 and "cannot read" in capsys.readouterr().err


@in_repo
@pytest.mark.parametrize("content", ["[]", '{"stages": []}', '{"stages": null}', "null"])
def test_check_with_a_release_file_of_the_wrong_shape(
    content: str, eval_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (eval_dir / "results").mkdir(parents=True)
    (eval_dir / "results" / "test-v9.9.9.json").write_text(content, encoding="utf-8")
    code = main(
        ["eval", "retrieval", "--split", "test", "--release", "v9.9.9", "--check"]
        + ["--eval-dir", str(eval_dir)]
    )
    assert code == 1 and "cannot read" in capsys.readouterr().err


@in_repo
def test_check_does_not_write_a_json_file_even_when_asked(
    eval_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    base = ["eval", "conversation", "--split", "test", "--release", "v9.9.9"]
    base += ["--eval-dir", str(eval_dir)]
    assert main(base) == 0
    target = tmp_path / "out.json"
    assert main([*base, "--check", "--json", str(target)]) == 0
    assert not target.exists()


@in_repo
def test_check_fails_when_a_stage_gains_or_loses_a_key(
    eval_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    base = ["eval", "conversation", "--split", "test", "--release", "v9.9.9"]
    base += ["--eval-dir", str(eval_dir)]
    assert main(base) == 0
    release = eval_dir / "results" / "test-v9.9.9.json"
    data = json.loads(release.read_text(encoding="utf-8"))
    data["stages"]["conversation"]["surplus"] = 1
    release.write_text(json.dumps(data), encoding="utf-8")
    assert main([*base, "--check"]) == 1
    assert "surplus" in capsys.readouterr().err
    del data["stages"]["conversation"]["surplus"]
    del data["stages"]["conversation"]["n_follow_ups"]
    release.write_text(json.dumps(data), encoding="utf-8")
    assert main([*base, "--check"]) == 1
    assert "n_follow_ups" in capsys.readouterr().err


@in_repo
def test_check_is_exact_on_floats(eval_dir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    base = ["eval", "conversation", "--split", "test", "--release", "v9.9.9"]
    base += ["--eval-dir", str(eval_dir)]
    assert main(base) == 0
    release = eval_dir / "results" / "test-v9.9.9.json"
    data = json.loads(release.read_text(encoding="utf-8"))
    overall = data["stages"]["conversation"]["overall"]
    condition = next(iter(overall))
    overall[condition]["hit@1"] += 1e-9
    release.write_text(json.dumps(data), encoding="utf-8")
    assert main([*base, "--check"]) == 1
    assert "overall" in capsys.readouterr().err


@in_repo
def test_check_ignores_stages_other_than_the_one_asked_for(
    eval_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    base = ["eval", "conversation", "--split", "test", "--release", "v9.9.9"]
    base += ["--eval-dir", str(eval_dir)]
    assert main(base) == 0
    release = eval_dir / "results" / "test-v9.9.9.json"
    data = json.loads(release.read_text(encoding="utf-8"))
    data["stages"]["retrieval"] = {"anything": 1}
    release.write_text(json.dumps(data), encoding="utf-8")
    assert main([*base, "--check"]) == 0


@in_repo
def test_check_on_the_model_stage_is_refused_without_calling_the_model(
    eval_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _release_file(eval_dir, "v9.9.9", {"answers": {"stage": "answers"}})
    with fake_model_server() as (url, fake):
        monkeypatch.setenv("ASKPDF_OLLAMA_URL", url)
        code = main(
            ["eval", "answers", "--split", "test", "--release", "v9.9.9", "--check"]
            + ["--eval-dir", str(eval_dir)]
        )
        assert code == 1
        assert [p for p, _, _ in fake.requests if p in ("/api/chat", "/api/generate")] == []


@in_repo
def test_check_on_the_committed_release_passes_for_the_two_model_free_stages(
    capsys: pytest.CaptureFixture[str],
) -> None:
    import pypdf

    pin = re.search(r"^pypdf==(\S+)", (EVAL / "constraints.txt").read_text(encoding="utf-8"), re.M)
    assert pin is not None
    if pypdf.__version__ != pin[1]:
        pytest.skip(f"results are pinned to pypdf {pin[1]}")
    for stage in ("retrieval", "conversation"):
        assert main(["eval", stage, "--split", "test", "--release", "v0.1.0", "--check"]) == 0
        assert "matches" in capsys.readouterr().err


# --- the release workflow ------------------------------------------------------------------------

RELEASE_YML = ROOT / ".github" / "workflows" / "release.yml"
CI_YML = ROOT / ".github" / "workflows" / "ci.yml"
workflow = pytest.mark.skipif(not RELEASE_YML.is_file(), reason=".github is not in the sdist")


@workflow
def test_release_workflow_actions_are_pinned_by_sha_like_ci() -> None:
    uses = re.findall(r"uses:\s*(\S+)", RELEASE_YML.read_text(encoding="utf-8"))
    assert uses and all(re.fullmatch(r"[\w./-]+@[0-9a-f]{40}", u) for u in uses)
    ci = set(re.findall(r"uses:\s*(\S+)", CI_YML.read_text(encoding="utf-8")))
    assert set(uses) <= ci  # the same pins CI already runs with


@workflow
def test_release_workflow_has_least_privilege_and_orders_its_jobs() -> None:
    text = RELEASE_YML.read_text(encoding="utf-8")
    assert re.search(r"^permissions:\n  contents: read\n", text, re.M)
    assert text.count("contents: write") == 1
    reproduce, release = text.split("  release:\n", 1)
    assert "contents: write" in release and "contents: write" not in reproduce
    assert "needs: reproduce" in release
    assert "--verify-tag" in release
    assert "tags:" in text and "branches:" not in text


@workflow
def test_release_workflow_never_puts_the_tag_in_a_shell_line() -> None:
    """A tag name is attacker-chosen text; it must reach the shell as an env variable."""
    for run in re.findall(
        r"run:\s*(?:\|\n((?:[ ]{10,}.*\n)+)|(.*)\n)", RELEASE_YML.read_text(encoding="utf-8")
    ):
        assert "${{" not in "".join(run)


@workflow
def test_release_workflow_runs_both_checks_on_the_test_split() -> None:
    text = RELEASE_YML.read_text(encoding="utf-8")
    for stage in ("retrieval", "conversation"):
        assert re.search(rf'askpdf eval {stage} --split test --release "\$TAG" --check', text)
    assert 'test "$(askpdf --version)" = "askpdf ${TAG#v}"' in text
    assert "eval answers" not in text  # the model stage cannot be reproduced on a runner


@workflow
def test_pull_requests_exercise_the_release_checks_before_a_tag_exists() -> None:
    ci = CI_YML.read_text(encoding="utf-8")
    assert re.search(r"askpdf eval (retrieval|conversation) --split test .*--check", ci)


# --- README claims against the results files -----------------------------------------------------

README = ROOT / "README.md"
results = pytest.mark.skipif(not (EVAL / "results").is_dir(), reason="eval/ is not in the sdist")


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


@results
def test_label_counts_in_the_readme_follow_from_the_raw_labels() -> None:
    labels = json.loads((EVAL / "results" / "test-v0.1.0-labels.json").read_text(encoding="utf-8"))[
        "labels"
    ]
    questions = {q["id"]: q for q in _jsonl(EVAL / "questions" / "test.jsonl")}
    assert set(labels) == set(questions)  # every test question labelled, none invented
    answerable = {k: v for k, v in labels.items() if questions[k]["answerable"]}
    count = Counter(v["label"] for v in answerable.values())
    assert (len(answerable), len(labels) - len(answerable)) == (46, 7)
    assert (count["correct"], count["partial"], count["wrong"], count["abstained"]) == (38, 1, 4, 3)
    unanswerable = [v for k, v in labels.items() if not questions[k]["answerable"]]
    assert sum(v["label"] == "correct" for v in unanswerable) == 5
    causes = Counter(v["cause"] for v in answerable.values() if v["label"] != "correct")
    assert causes == {"retrieval": 5, "reading": 3}
    assert all(v["cause"] is None for v in answerable.values() if v["label"] == "correct")
    assert all(v["reason"].strip() for v in labels.values())  # each label has its reason
    text = README.read_text(encoding="utf-8")
    for fragment in ("38 of 46", "5 of 7", "5 never had the page", "3 had it and misread it"):
        assert fragment in text


@results
def test_causes_agree_with_what_the_retriever_returned() -> None:
    """`retrieval` iff no gold page was in the 3 pages the model read."""
    release = json.loads((EVAL / "results" / "test-v0.1.0.json").read_text(encoding="utf-8"))[
        "stages"
    ]
    labels = json.loads((EVAL / "results" / "test-v0.1.0-labels.json").read_text(encoding="utf-8"))[
        "labels"
    ]
    questions = {q["id"]: q for q in _jsonl(EVAL / "questions" / "test.jsonl")}
    for qid, label in labels.items():
        q = questions[qid]
        if not q["answerable"] or label["label"] == "correct":
            continue
        top3 = {tuple(x) for x in release["retrieval"]["ranked"][qid][:3]}
        has_gold = any((q["doc"], p) in top3 for p in q["pages"])
        assert label["cause"] == ("reading" if has_gold else "retrieval"), qid


@results
def test_abstained_label_matches_a_not_found_reply() -> None:
    release = json.loads((EVAL / "results" / "test-v0.1.0.json").read_text(encoding="utf-8"))[
        "stages"
    ]
    labels = json.loads((EVAL / "results" / "test-v0.1.0-labels.json").read_text(encoding="utf-8"))[
        "labels"
    ]
    replies = release["answers"]["per_question"]
    for qid, label in labels.items():
        if label["label"] == "abstained":
            assert replies[qid]["not_found"], qid
        if replies[qid]["not_found"]:
            assert label["label"] in ("abstained", "correct"), qid


@results
def test_the_split_is_hash_pinned_and_the_results_cover_it() -> None:
    import hashlib

    for folder, name in (("questions", "test.jsonl"), ("conversations", "test.jsonl")):
        digest = hashlib.sha256((EVAL / folder / name).read_bytes()).hexdigest()
        assert digest == (EVAL / folder / "test.sha256").read_text(encoding="utf-8").strip()
    release = json.loads((EVAL / "results" / "test-v0.1.0.json").read_text(encoding="utf-8"))
    assert release["release"] == "v0.1.0"
    questions = _jsonl(EVAL / "questions" / "test.jsonl")
    assert release["stages"]["answers"]["n_questions"] == len(questions) == 53
    assert release["stages"]["retrieval"]["n_questions"] == sum(q["answerable"] for q in questions)


@results
def test_readme_latency_rows_equal_the_bench_file() -> None:
    bench = json.loads((EVAL / "results" / "bench-v0.1.0.json").read_text(encoding="utf-8"))
    block = re.search(
        r"<!-- results:latency -->\n(.*?)<!-- /results -->",
        README.read_text(encoding="utf-8"),
        re.S,
    )
    assert block is not None
    text = block[1]
    for needle in (bench["machine"], bench["model"], bench["ollama_version"], bench["date"]):
        assert needle in text
    warm = bench["answers_seconds"]["warm"]
    assert f"| Answer, warm: first token | {warm['first_token']['n']} |" in text
    assert bench["askpdf"] == "0.1.0"
    assert bench["answers_seconds"]["cold"]["first_token"] > warm["first_token"]["p50"]
    # nearest-rank p95 of nine runs is the slowest one; the table says n so a reader can tell
    assert warm["first_token"]["n"] < 20 and warm["total"]["n"] < 20


@results
def test_the_bench_file_is_internally_consistent() -> None:
    bench = json.loads((EVAL / "results" / "bench-v0.1.0.json").read_text(encoding="utf-8"))
    assert bench["indexing"]["seconds_per_document"]["n"] == (
        bench["indexing"]["documents"] * bench["indexing"]["runs"]
    )
    questions = _jsonl(EVAL / "questions" / "dev.jsonl")
    assert bench["search_ms"]["n"] == len(questions) * bench["indexing"]["runs"]
    for spread_ in (bench["search_ms"], bench["indexing"]["seconds_per_document"]):
        assert spread_["p50"] <= spread_["p95"]
    assert bench["model"] == ollama.CHAT_MODEL


def _block(name: str) -> str:
    match = re.search(
        rf"<!-- results:{name} -->\n(.*?)<!-- /results -->",
        README.read_text(encoding="utf-8"),
        re.S,
    )
    assert match is not None
    return match[1]


@results
@pytest.mark.parametrize("name", ["answers", "test-split"])
def test_every_model_based_readme_block_names_model_version_machine_and_date(name: str) -> None:
    text = _block(name)
    assert "llama3.2:3b" in text and re.search(r"Ollama \d+\.\d+", text)
    assert re.search(r"Intel|MacBook", text) and re.search(r"20\d\d-\d\d-\d\d", text)
