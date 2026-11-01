"""Questions that name a page or a line: read the page instead of searching for it."""

from collections.abc import Iterator
from pathlib import Path

import pytest

from askpdf.cli import main
from askpdf.pagelookup import SHOWN_NAMES, find_page
from askpdf.snapshot import Snapshot
from askpdf.store import ChunkRecord, FileRow, PageRecord, Store
from tests.pdfs import make_pdf, text_page

# report.pdf: 4 pages, printed i, ii, 1, 2 (so page 2 of the file is printed "ii").
REPORT = {
    1: ("i", "Title page\nAnnual report"),
    2: ("ii", "Contents\n\nIntroduction\nMethods"),
    3: ("1", "Introduction\nThe first line after the heading.\nA third line here."),
    4: ("2", "Methods\nWe measured everything twice."),
}


def _put(store: Store, path: str, sha: str, pages: dict[int, tuple[str, str]]) -> None:
    store.put_document(
        FileRow(path, sha, 1, 1, "indexed", None),
        [PageRecord(n, label, text, (), False) for n, (label, text) in pages.items()],
        [
            ChunkRecord(n, 0, 0, len(text.split()), text)
            for n, (_, text) in pages.items()
            if text.strip()
        ],
    )


@pytest.fixture
def store(tmp_path: Path) -> Iterator[Store]:
    with Store(tmp_path / "i.db") as s:
        _put(s, "annual-report-2024.pdf", "a" * 64, REPORT)
        _put(s, "nist-sp-800-63b-4.pdf", "b" * 64, {1: ("1", "Passwords\nat least 15")})
        _put(s, "nist-sp-800-53.pdf", "c" * 64, {1: ("1", "Controls")})
        yield s


@pytest.fixture
def snapshot(store: Store) -> Snapshot:
    return Snapshot.load(store)


def _pages(question: str, snapshot: Snapshot, earlier: tuple[str, ...] = ()) -> list[int]:
    lookup = find_page(question, snapshot, earlier)
    assert lookup is not None
    return [h.page for h in lookup.hits]


# --- recognising the question ---------------------------------------------------------------


@pytest.mark.parametrize(
    "question",
    [
        "how long must passwords be?",
        "which page talks about methods?",
        "summarise the annual report",
        "it has 5 pages",
        "what do pages 3 to 4 say about annual-report-2024",
    ],
)
def test_questions_without_a_page_number_are_searched(question: str, snapshot: Snapshot) -> None:
    assert find_page(question, snapshot) is None


@pytest.mark.parametrize(
    "question",
    [
        "what is on page 3 of annual-report-2024.pdf",
        "What is on PAGE 3 of Annual-Report-2024.PDF?",
        "page 3 in annual report 2024",
        "annual_report_2024 p. 3",
        "annual report pg 3",  # most of the name, in a row
        "summarise p.3 of the annual report",
    ],
)
def test_the_document_is_found_however_it_is_written(question: str, snapshot: Snapshot) -> None:
    # Page 3 of the file is printed "1"; printed "3" does not exist, so only the 3rd page.
    assert _pages(question, snapshot) == [3]


def test_one_distinctive_word_of_a_name_is_enough_when_only_one_name_has_it(
    snapshot: Snapshot,
) -> None:
    assert _pages("what is on page 2 of the annual one?", snapshot) == [4, 2]


def test_a_word_shared_by_two_names_asks_which_of_them(snapshot: Snapshot) -> None:
    lookup = find_page("page 1 of the nist document", snapshot)
    assert lookup is not None and lookup.hits == []
    assert lookup.notes == (
        "which document do you mean? Name it, e.g. nist-sp-800-53.pdf, nist-sp-800-63b-4.pdf",
    )


def test_the_longer_match_wins_over_a_shared_prefix(snapshot: Snapshot) -> None:
    lookup = find_page("page 1 of NIST SP 800-63B", snapshot)
    assert lookup is not None and lookup.hits[0].paths == ("nist-sp-800-63b-4.pdf",)


def test_without_a_document_name_it_asks_which_one(snapshot: Snapshot) -> None:
    lookup = find_page("what does page 3 say?", snapshot)
    assert lookup is not None and lookup.hits == []
    assert lookup.notes[0].startswith("which document do you mean?")


def test_the_list_of_documents_is_kept_short(tmp_path: Path) -> None:
    with Store(tmp_path / "many.db") as store:
        for i in range(SHOWN_NAMES + 2):
            _put(store, f"doc{i}.pdf", str(i) * 64, {1: ("1", "text")})
        snapshot = Snapshot.load(store)
    lookup = find_page("what does page 1 say?", snapshot)
    assert lookup is not None and lookup.notes[0].endswith(", …")
    assert lookup.notes[0].count(".pdf") == SHOWN_NAMES


def test_in_a_folder_of_one_document_the_name_is_not_needed(tmp_path: Path) -> None:
    with Store(tmp_path / "one.db") as store:
        _put(store, "only.pdf", "d" * 64, REPORT)
        snapshot = Snapshot.load(store)
    assert _pages("what is on page 4?", snapshot) == [4]


def test_nothing_indexed_is_left_to_the_search(tmp_path: Path) -> None:
    with Store(tmp_path / "empty.db") as store:
        assert find_page("page 2 of x.pdf", Snapshot.load(store)) is None


def test_a_follow_up_takes_the_document_from_earlier_questions(snapshot: Snapshot) -> None:
    assert _pages("and page 4?", snapshot, ("what is on page 3 of the annual report?",)) == [4]
    # The newest earlier question wins.
    earlier = ("page 1 of the annual report", "page 1 of nist-sp-800-53")
    lookup = find_page("and page 1?", snapshot, earlier)
    assert lookup is not None and lookup.hits[0].paths == ("nist-sp-800-53.pdf",)


# --- printed labels and physical pages ------------------------------------------------------


def test_a_printed_label_and_a_file_page_that_differ_are_both_read(snapshot: Snapshot) -> None:
    lookup = find_page("page 2 of the annual report", snapshot)
    assert lookup is not None
    assert [(h.page, h.label) for h in lookup.hits] == [(4, "2"), (2, "ii")]
    assert lookup.notes == (
        "page 2 of annual-report-2024.pdf could mean page 4 and page 2; reading both",
    )


def test_pdf_page_means_the_page_of_the_file(snapshot: Snapshot) -> None:
    assert _pages("PDF page 2 of the annual report", snapshot) == [2]


def test_roman_page_labels_are_found(snapshot: Snapshot) -> None:
    assert _pages("page ii of the annual report", snapshot) == [2]
    assert _pages("page II of the annual report", snapshot) == [2]


def test_a_page_that_does_not_exist_is_said_plainly(snapshot: Snapshot) -> None:
    lookup = find_page("page 99 of the annual report", snapshot)
    assert lookup is not None and lookup.hits == [] and lookup.answer is None
    assert lookup.notes == ("annual-report-2024.pdf has no page 99 (4 pages)",)


def test_the_hit_carries_the_index_label_and_paths(snapshot: Snapshot) -> None:
    lookup = find_page("page 1 of the annual report", snapshot)
    assert lookup is not None
    hit = lookup.hits[0]
    assert hit.paths == ("annual-report-2024.pdf",) and hit.label == "1" and hit.page == 3
    assert hit.best_chunk.text.startswith("Introduction")


# --- lines ----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("question", "line"),
    [
        ("what is the first line of PDF page 3 of the annual report", "Introduction"),
        ("what is the second line in pdf page 3 of the annual report?", "The first line after"),
        ("the 3rd line of PDF page 3, annual report", "A third line here."),
        ("line 3 of PDF page 3 of the annual report", "A third line here."),
        ("the last line on pdf page 3 of the annual report", "A third line here."),
    ],
)
def test_a_line_is_read_from_the_page_without_the_model(
    question: str, line: str, snapshot: Snapshot
) -> None:
    lookup = find_page(question, snapshot)
    assert lookup is not None and lookup.answer is not None
    answer = lookup.answer
    assert line in answer.text and answer.text.endswith("[1].")
    (citation,) = answer.citations
    assert (
        citation.verified
        and citation.hit.page == 3
        and line.startswith(citation.quotes[0].quote[:10])
    )
    assert answer.grounded and answer.supported == 1 and answer.raw == ""
    assert lookup.notes == ("read from annual-report-2024.pdf without the model",)


def test_blank_lines_are_not_counted(snapshot: Snapshot) -> None:
    lookup = find_page("the second line of pdf page 2 of the annual report", snapshot)
    assert lookup is not None and lookup.answer is not None
    assert "“Introduction”" in lookup.answer.text


def test_a_line_past_the_end_says_how_many_there_are(snapshot: Snapshot) -> None:
    lookup = find_page("line 9 of pdf page 4 of the annual report", snapshot)
    assert lookup is not None and lookup.answer is not None
    assert lookup.answer.text == "The page printed 2 (PDF page 4) has 2 lines of text."
    assert lookup.answer.citations == () and not lookup.answer.grounded


def test_a_line_of_an_ambiguous_page_is_read_from_both(snapshot: Snapshot) -> None:
    lookup = find_page("the first line of page 2 of the annual report", snapshot)
    assert lookup is not None and lookup.answer is not None
    text = lookup.answer.text
    assert "Line 1 of the page printed 2 (PDF page 4) reads: “Methods” [1]." in text
    assert "Line 1 of the page printed ii (PDF page 2) reads: “Contents” [2]." in text
    assert [c.number for c in lookup.answer.citations] == [1, 2]


def test_line_wording_without_a_page_is_searched(snapshot: Snapshot) -> None:
    assert find_page("what is the second line of the annual report", snapshot) is None


# --- the ask command ------------------------------------------------------------------------


@pytest.fixture
def folder(tmp_path: Path) -> Path:
    make_pdf(
        tmp_path / "f" / "policy.pdf",
        [
            text_page("Passwords need fifteen characters or more in every case."),
            text_page("Cookies need the Secure flag.", "Sessions end after one hour."),
        ],
    )
    make_pdf(tmp_path / "f" / "other.pdf", [text_page("Unrelated words.")])
    index = str(tmp_path / "i.db")
    assert main(["index", str(tmp_path / "f"), "--index", index]) == 0
    return tmp_path / "f"


def _ask(folder: Path, question: str) -> list[str]:
    return ["ask", str(folder), question, "--index", str(folder.parent / "i.db")]


def test_ask_reads_a_line_without_reaching_the_model(
    folder: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    capsys.readouterr()
    # Ollama is unreachable in tests (conftest), so any model call would fail.
    assert main(_ask(folder, "what is the second line of page 2 of policy.pdf")) == 0
    captured = capsys.readouterr()
    assert "Line 2 of page 2 reads: “Sessions end after one hour.” [1]." in captured.out
    assert "✓ “Sessions end after one hour.”" in captured.out
    assert "read from policy.pdf without the model" in captured.err


def test_ask_says_when_the_page_does_not_exist(
    folder: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    capsys.readouterr()
    assert main(_ask(folder, "what is on page 7 of policy.pdf")) == 0
    assert capsys.readouterr().out.strip() == "policy.pdf has no page 7 (2 pages)"


def test_ask_asks_which_document(folder: Path, capsys: pytest.CaptureFixture[str]) -> None:
    capsys.readouterr()
    assert main(_ask(folder, "what is on page 1?")) == 0
    assert capsys.readouterr().out.startswith("which document do you mean?")
