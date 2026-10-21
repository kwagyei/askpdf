"""BM25, tokenizer, snapshot and `search` edge cases (tester, PR 3)."""

import math
import random
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

from askpdf.bm25 import BM25, tokenize
from askpdf.chunking import ChunkParams
from askpdf.cli import main
from askpdf.extract.pypdf_ext import PypdfExtractor
from askpdf.indexer import pins
from askpdf.retrieve import search_bm25
from askpdf.snapshot import Snapshot
from askpdf.store import ChunkRecord, FileRow, PageRecord, Store, folder_key


def _put(store: Store, path: str, sha: str, pages: dict[int, list[str]]) -> None:
    store.put_document(
        FileRow(path, sha, 1, 1, "indexed", None),
        [PageRecord(n, str(n), " ".join(t), (), False) for n, t in pages.items()],
        [
            ChunkRecord(n, i, 0, len(x.split()), x)
            for n, t in pages.items()
            for i, x in enumerate(t)
        ],
    )


# --- BM25 against an independent brute-force implementation ------------------------------


def _brute(docs: list[list[str]], query: list[str]) -> list[float]:
    n_docs = len(docs)
    avg = sum(map(len, docs)) / n_docs
    out = []
    for doc in docs:
        score = 0.0
        for term in set(query):
            df = sum(1 for d in docs if term in d)
            if df == 0:
                continue
            tf = doc.count(term)
            idf = math.log(1 + (n_docs - df + 0.5) / (df + 0.5))
            score += idf * tf * 2.2 / (tf + 1.2 * (0.25 + 0.75 * len(doc) / avg))
        out.append(score)
    return out


def test_scores_match_a_brute_force_reference_on_random_corpora() -> None:
    rng = random.Random(7)  # noqa: S311
    vocab = [f"w{i}" for i in range(40)]
    for _ in range(200):
        docs = [
            [rng.choice(vocab) for _ in range(rng.randint(0, 25))]
            for _ in range(rng.randint(1, 20))
        ]
        query = [rng.choice(vocab) for _ in range(rng.randint(0, 6))]
        got = BM25.build(docs).scores(query)
        assert list(got) == pytest.approx(_brute(docs, query), abs=1e-9)


def test_empty_corpora() -> None:
    assert len(BM25.build([]).scores(["a"])) == 0
    assert list(BM25.build([[], []]).scores(["a"])) == [0.0, 0.0]


# --- tokenizer ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "\uff33\uff30\u3000\uff18\uff10\uff10\uff0d\uff16\uff13\uff22",
        "SP\u00a0800\u201363B",
        "SP 800\u201363B",
    ],
)
def test_width_and_dash_variants_of_a_document_number(text: str) -> None:
    assert tokenize(text) == ["sp", "800", "63b"]


def test_emoji_separate_tokens_and_long_tokens_survive() -> None:
    assert tokenize("pass😀word") == ["pass", "word"]
    assert tokenize("x" * 200_000) == ["x" * 200_000]


@pytest.mark.parametrize("word", ["हिंदी", "ก็", "कि", "e͡e"])
def test_words_with_combining_marks_stay_whole(word: str) -> None:
    assert tokenize(word) == [word]


def test_turkish_dotted_capital_i_is_one_token() -> None:
    assert tokenize("İstanbul") == ["i\u0307stanbul"]


# --- search -------------------------------------------------------------------------------


def test_ties_across_documents_follow_chunk_id_and_are_repeatable(tmp_path: Path) -> None:
    with Store(tmp_path / "i.db") as store:
        for n in range(6):
            _put(store, f"{n}.pdf", f"{n:064d}", {1: ["alpha beta"]})
        snap = Snapshot.load(store)
        first = [h.sha256 for h in search_bm25(snap, "alpha", k=6)]
        assert first == [f"{n:064d}" for n in range(6)]
        assert first == [h.sha256 for h in search_bm25(Snapshot.load(store), "alpha", k=6)]


def test_empty_index_gives_no_hits(tmp_path: Path) -> None:
    with Store(tmp_path / "i.db") as store:
        snap = Snapshot.load(store)
        assert search_bm25(snap, "anything") == []
        assert search_bm25(snap, "") == []


def test_snapshot_never_mixes_generations_with_a_writer_in_another_process(
    tmp_path: Path,
) -> None:
    db = tmp_path / "i.db"
    with Store(db) as store:
        _put(store, "seed.pdf", "f" * 64, {1: ["seed"]})
    writer = textwrap.dedent(
        """
        import sys
        from pathlib import Path
        from askpdf.store import ChunkRecord, FileRow, PageRecord, Store
        with Store(Path(sys.argv[1])) as s:
            for n in range(60):
                pages = {p: [f"doc{n} p{p}"] for p in (1, 2, 3)}
                s.put_document(
                    FileRow(f"{n}.pdf", f"{n:064d}", 1, 1, "indexed", None),
                    [PageRecord(p, str(p), "x", (), False) for p in pages],
                    [ChunkRecord(p, 0, 0, 2, t[0]) for p, t in pages.items()],
                )
        """
    )
    proc = subprocess.Popen([sys.executable, "-I", "-c", writer, str(db)])  # noqa: S603
    try:
        with Store(db) as reader:
            seen: set[int] = set()
            while proc.poll() is None or not seen:
                snap = Snapshot.load(reader)
                per_doc: dict[str, int] = {}
                for c in snap.chunks:
                    per_doc[c.sha256] = per_doc.get(c.sha256, 0) + 1
                for sha, count in per_doc.items():
                    assert count == (1 if sha == "f" * 64 else 3)
                assert set(per_doc) == set(snap.paths)
                assert len(per_doc) == snap.generation
                seen.add(snap.generation)
    finally:
        proc.wait()
    assert proc.returncode == 0
    assert len(seen) > 1


def test_ten_thousand_chunks_load_and_search_quickly(tmp_path: Path) -> None:
    rng = random.Random(3)  # noqa: S311
    words = [f"term{i}" for i in range(5000)]
    with Store(tmp_path / "i.db") as store:
        for d in range(100):
            pages = {
                p: [" ".join(rng.choices(words, k=120)) for _ in range(2)] for p in range(1, 51)
            }
            _put(store, f"{d}.pdf", f"{d:064d}", pages)
        start = time.perf_counter()
        snap = Snapshot.load(store)
        load = time.perf_counter() - start
        assert len(snap.chunks) == 10_000
        start = time.perf_counter()
        hits = search_bm25(snap, "term1 term20 term300", k=10)
        query = time.perf_counter() - start
    assert len(hits) == 10
    assert load < 10 and query < 0.5


# --- CLI ----------------------------------------------------------------------------------


@pytest.fixture
def folder(tmp_path: Path) -> Path:
    root = tmp_path / "f"
    root.mkdir()
    with Store(tmp_path / "i.db") as store:
        store.claim_folder(folder_key(root))
        # An index records the settings it was built with; `search` refuses one without.
        store.ensure_pins(pins(PypdfExtractor(), ChunkParams()))
        _put(store, "bad\x1b[31mname\n9. forged.pdf", "a" * 64, {1: ["alpha\x1b[2Jbeta‮gamma"]})
    return root


def test_control_characters_in_names_and_text_are_escaped(
    folder: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["search", str(folder), "alpha", "--index", str(tmp_path / "i.db")]) == 0
    out = capsys.readouterr().out
    assert "\x1b" not in out and "‮" not in out
    assert out.count("\n") == 2


def test_stopword_only_and_blank_questions_exit_zero_with_a_message(
    folder: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    for q in ["the and of", "", "   ", "!!!"]:
        assert main(["search", str(folder), q, "--index", str(tmp_path / "i.db")]) == 0
    assert capsys.readouterr().out == ""


def test_a_missing_or_foreign_index_exits_one(
    folder: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["search", str(folder), "alpha", "--index", str(tmp_path / "none.db")]) == 1
    assert main(["search", str(tmp_path), "alpha", "--index", str(tmp_path / "i.db")]) == 1
    (tmp_path / "junk.db").write_text("junk")
    assert main(["search", str(folder), "alpha", "--index", str(tmp_path / "junk.db")]) == 1
    assert capsys.readouterr().out == ""
    with pytest.raises(SystemExit) as exc:
        main(["search", str(folder), "alpha", "-k", "0"])
    assert exc.value.code == 2
