"""Command-line entry point.

Results go to stdout and progress to stderr, so output can be piped. Exit codes:
0 success, 1 a run-time problem (missing folder, unusable or busy index), 2 bad
options, 130 interrupted.
The remaining subcommands (serve, ask, bench) and the other eval stages arrive in later
pull requests.
"""

import argparse
import errno
import io
import json
import os
import re
import sqlite3
import sys
import unicodedata
from collections.abc import Sequence
from pathlib import Path
from typing import Any, TextIO, cast

from askpdf import __version__
from askpdf.chunking import ChunkParams
from askpdf.extract.pypdf_ext import PypdfExtractor
from askpdf.indexer import build_records, index_folder, pins
from askpdf.retrieve import DEFAULT_K, PageHit, search_bm25
from askpdf.snapshot import Snapshot
from askpdf.store import (
    NotAnIndexError,
    OutdatedIndexError,
    Store,
    WrongFolderError,
    default_index_path,
    folder_key,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="askpdf",
        description="Ask questions about a folder of PDFs; answers cite the page.",
    )
    parser.add_argument("--version", action="version", version=f"askpdf {__version__}")
    commands = parser.add_subparsers(dest="command", metavar="COMMAND")

    def with_target(sub: argparse.ArgumentParser) -> None:
        sub.add_argument("path", type=Path, metavar="FOLDER_OR_PDF")
        sub.add_argument(
            "--index",
            type=Path,
            metavar="PATH",
            help="index file to use (default: one per folder in your cache directory)",
        )

    with_target(commands.add_parser("index", help="read new or changed PDFs into the index"))
    with_target(commands.add_parser("docs", help="list indexed documents and failed files"))
    search = commands.add_parser("search", help="list the pages that best match a question")
    with_target(search)
    search.add_argument("question", metavar="QUESTION")
    search.add_argument(
        "-k", type=_positive, default=DEFAULT_K, metavar="N", help="pages to show (default: 5)"
    )
    search.add_argument(
        "--retriever",
        choices=["bm25"],
        default="bm25",
        help="ranking method (dense and hybrid come with the model providers)",
    )
    evaluate = commands.add_parser("eval", help="score one stage on the evaluation data")
    evaluate.add_argument("stage", choices=["extraction", "retrieval", "conversation"])
    evaluate.add_argument(
        "--split",
        choices=["dev", "test"],
        default="dev",
        help="question set for retrieval (test is scored once per release)",
    )
    evaluate.add_argument(
        "--release", metavar="vX.Y.Z", help="required with --split test (ADR-0007)"
    )
    evaluate.add_argument(
        "--eval-dir",
        type=Path,
        default=Path("eval"),
        metavar="DIR",
        help="the evaluation data (default: ./eval, as in the repository)",
    )
    evaluate.add_argument("--json", type=Path, metavar="PATH", help="also write the results here")
    return parser


def _positive(text: str) -> int:
    try:
        value = int(text)
    except ValueError:
        value = 0
    if value < 1:
        raise argparse.ArgumentTypeError(f"expected a whole number of at least 1, got {text!r}")
    return value


def shown(text: str) -> str:
    """Escape control and invisible characters, so a file name cannot forge output lines
    or send escape sequences to the terminal."""
    return "".join(
        ch.encode("unicode_escape").decode("ascii")
        if unicodedata.category(ch) in {"Cc", "Cf", "Cs", "Zl", "Zp"}
        else ch
        for ch in text
    )


def _plural(n: int, word: str) -> str:
    return f"{n} {word}{'' if n == 1 else 's'}"


def _error(err: TextIO, message: object) -> int:
    print(f"askpdf: error: {shown(str(message))}", file=err)
    return 1


def _index_path(args: argparse.Namespace) -> Path:
    return Path(args.index) if args.index else default_index_path(args.path)


def _open_store(args: argparse.Namespace, err: TextIO, *, reading: bool = False) -> Store | None:
    """Open the index; ``reading`` commands never rebuild or wipe it."""
    if not args.path.exists():
        _error(err, f"{args.path} does not exist")
        return None
    try:
        return Store(_index_path(args), rebuild=not reading)
    except OutdatedIndexError as exc:
        _error(err, f"{exc}; run: askpdf index {args.path}")
    except NotAnIndexError as exc:
        _error(err, exc)
    except (OSError, sqlite3.Error) as exc:
        _error(err, f"cannot open the index at {_index_path(args)}: {exc}")
    return None


def cmd_index(args: argparse.Namespace, out: TextIO, err: TextIO) -> int:
    store = _open_store(args, err)
    if store is None:
        return 1
    with store:
        try:
            report = index_folder(
                args.path,
                store,
                PypdfExtractor(),
                ChunkParams(),
                on_file=lambda rel: print(f"reading {shown(rel)}", file=err),
            )
        except WrongFolderError as exc:
            return _error(err, f"{exc}; pass a different --index")
        except sqlite3.OperationalError as exc:
            if "locked" in str(exc) or "busy" in str(exc):
                return _error(err, "the index is busy (another askpdf is indexing)")
            return _error(err, f"index error: {exc}")
    if report.rebuilt:
        print("settings changed since the last run: rebuilt the index", file=err)
    for rel, reason in sorted(report.failed.items()):
        print(f"failed: {shown(rel)}: {shown(reason)}", file=err)
    if "." in report.unreadable:
        return _error(err, f"cannot list {args.path}; the index was left as it was")
    for folder in report.unreadable:
        print(f"skipped unreadable folder {shown(folder)} (its files stay indexed)", file=err)
    for rel, reason in sorted(report.out_of_reach.items()):
        print(f"skipped {shown(rel)}: {shown(reason)} (kept from the last run)", file=err)
    parts = [
        f"{len(report.added)} added",
        f"{len(report.reused)} already known (copies or renames)",
        f"{len(report.removed)} removed",
        f"{report.unchanged} unchanged",
        f"{len(report.failed)} failed",
    ]
    print(", ".join(parts), file=out)
    print(f"index: {store.path}", file=err)
    return 0


def cmd_docs(args: argparse.Namespace, out: TextIO, err: TextIO) -> int:
    if args.path.exists() and not _index_path(args).exists():
        print(f"nothing indexed yet; run: askpdf index {args.path}", file=err)
        return 0
    store = _open_store(args, err, reading=True)
    if store is None:
        return 1
    with store:
        try:
            store.check_folder(folder_key(args.path))
        except WrongFolderError as exc:
            return _error(err, exc)
        if not store.pins_match(pins(PypdfExtractor(), ChunkParams())):
            print(
                f"note: askpdf's settings changed since this index was built;"
                f" run: askpdf index {args.path}",
                file=err,
            )
        documents = store.documents()
        failed = [f for f in store.files().values() if f.status == "failed"]
    if not documents and not failed:
        print(f"nothing indexed yet; run: askpdf index {args.path}", file=err)
        return 0
    for doc in documents:
        name = shown(doc.paths[0]) if doc.paths else doc.sha256[:12]
        line = (
            f"{name}  {_plural(doc.n_pages, 'page')}, {_plural(doc.n_chunks, 'chunk')}"
            f", {doc.n_image_only} image-only"
        )
        if len(doc.paths) > 1:
            line += f"  (also: {', '.join(shown(p) for p in doc.paths[1:])})"
        if doc.looks_scanned:
            line += "  [looks scanned; no OCR, so most pages have no searchable text]"
        print(line, file=out)
    for f in sorted(failed, key=lambda f: f.path):
        print(f"{shown(f.path)}  FAILED: {shown(f.error or '')}", file=out)
    return 0


RELEASE = re.compile(r"v[0-9]+\.[0-9]+\.[0-9]+", re.ASCII)


def cmd_eval(args: argparse.Namespace, out: TextIO, err: TextIO) -> int:
    corpus = args.eval_dir / "corpus"
    if not corpus.is_dir():
        return _error(err, f"no evaluation data in {args.eval_dir} (run from the repository)")
    if args.stage == "extraction" and (args.split != "dev" or args.release):
        return _error(err, "extraction has no question split; drop --split and --release")
    if args.split == "dev" and args.release:
        return _error(err, "--release goes with --split test only")
    release_file, release = None, None
    if args.split == "test":
        if not RELEASE.fullmatch(args.release or ""):
            return _error(err, "the test split is scored once per release: pass --release vX.Y.Z")
        # The frozen split is scored once per release, into its own file (ADR-0007).
        release_file = args.eval_dir / "results" / f"test-{args.release}.json"
        release = {"release": args.release, "stages": {}}
        if release_file.exists():
            try:
                release = json.loads(release_file.read_text(encoding="utf-8"))
                if not isinstance(release, dict) or not isinstance(release.get("stages"), dict):
                    raise ValueError("not an askpdf release results file")
            except (OSError, ValueError) as exc:
                return _error(err, f"cannot read {release_file}: {exc}")
        if args.stage in release["stages"]:
            return _error(
                err,
                f"{args.stage} on the test split was already scored for {args.release}"
                f" ({release_file}); the test split is scored once per release",
            )
    run = {"retrieval": _eval_retrieval, "conversation": _eval_conversation}
    try:
        summary = run.get(args.stage, _eval_extraction)(args, out)
    except ValueError as exc:
        return _error(err, exc)
    try:
        if args.json:
            _write_json(args.json, summary)
        if release_file is not None and release is not None:
            release["stages"][args.stage] = summary
            release_file.parent.mkdir(parents=True, exist_ok=True)
            _write_json(release_file, release)
            print(f"wrote {release_file}", file=err)
    except OSError as exc:
        return _error(err, f"cannot write the results: {exc.strerror or exc}")
    return 0


def _write_json(path: Path, data: object) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _eval_retrieval(args: argparse.Namespace, out: TextIO) -> dict[str, Any]:
    from askpdf.evaluate import questions, retrieval  # only `eval` needs them

    path = args.eval_dir / "questions" / f"{args.split}.jsonl"
    if not path.is_file():
        raise ValueError(f"no question set at {path}")
    snapshot = retrieval.load_corpus_snapshot(args.eval_dir / "corpus")
    result = retrieval.evaluate(questions.load(path), retrieval.bm25_retriever(snapshot))
    summary = {"stage": "retrieval", "split": args.split, "retriever": "bm25", **result}
    if args.release:
        summary["release"] = args.release
    print(f"BM25 on {args.split}, {result['n_questions']} answerable questions", file=out)
    print("  " + "  ".join(f"{m} {v:.3f}" for m, v in result["overall"].items()), file=out)
    for field in ("by_kind", "by_doc"):
        for name, scores in result[field].items():
            print(
                f"  {name}: hit@1 {scores['hit@1']:.3f}  ndcg@10 {scores['ndcg@10']:.3f}", file=out
            )
    return summary


def _eval_conversation(args: argparse.Namespace, out: TextIO) -> dict[str, Any]:
    from askpdf.evaluate import conversation, retrieval  # only `eval` needs them

    path = args.eval_dir / "conversations" / f"{args.split}.jsonl"
    if not path.is_file():
        raise ValueError(f"no conversations at {path}")
    snapshot = retrieval.load_corpus_snapshot(args.eval_dir / "corpus")
    result = conversation.evaluate(conversation.load(path), retrieval.bm25_retriever(snapshot))
    summary = {"stage": "conversation", "split": args.split, "retriever": "bm25", **result}
    if args.release:
        summary["release"] = args.release
    n, switches = result["n_follow_ups"], result["n_topic_switches"]
    print(f"BM25 on {args.split}, {n} follow-ups ({switches} topic switches)", file=out)
    for condition, scores in result["overall"].items():
        hit, ndcg = scores["hit@1"], scores["ndcg@10"]
        print(f"  {condition}: hit@1 {hit:.3f}  ndcg@10 {ndcg:.3f}", file=out)
    return summary


def _eval_extraction(args: argparse.Namespace, out: TextIO) -> dict[str, Any]:
    from askpdf.evaluate import extraction  # only `eval` needs it

    corpus, transcripts = args.eval_dir / "corpus", args.eval_dir / "transcripts"
    if not transcripts.is_dir():
        raise ValueError(f"no transcriptions in {args.eval_dir}")
    try:
        import jiwer  # noqa: F401 - only checking it is installed
    except ImportError:
        raise ValueError("the eval stages need the eval group: pip install --group eval") from None
    pages = {}
    for pdf in sorted(corpus.glob("*.pdf")):
        records, _ = build_records(PypdfExtractor().extract(pdf), ChunkParams())
        pages[pdf.stem] = records
    scanned_gold = json.loads((corpus / "scanned-pages.json").read_text(encoding="utf-8"))
    gold = {(doc, n) for doc, numbers in scanned_gold.items() if doc in pages for n in numbers}
    results = extraction.evaluate_pages(extraction.load_transcripts(transcripts), pages)
    summary = extraction.summarise(results, extraction.scanned_detection(pages, gold))
    summary = {"stage": "extraction", "extractor": f"pypdf {PypdfExtractor.version}", **summary}
    for row in summary["pages"]:
        print(f"{row['doc']}  page {row['page']}  CER {row['cer']:.3f}", file=out)
    print(f"mean CER {summary['mean_cer']:.3f} over {len(results)} pages", file=out)
    for name in ("boilerplate", "scanned"):
        pr = summary[name]
        print(f"{name}: precision {_fmt(pr['precision'])}, recall {_fmt(pr['recall'])}", file=out)
    return summary


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


SNIPPET_CHARS = 160


def page_ref(page: int, label: str) -> str:
    """ "p. 9 (PDF page 20)" when the printed label differs from the physical page."""
    return f"page {page}" if label == str(page) else f"p. {label} (PDF page {page})"


def _snippet(text: str) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= SNIPPET_CHARS else flat[: SNIPPET_CHARS - 1].rstrip() + "…"


def _print_hit(rank: int, hit: PageHit, out: TextIO) -> None:
    name = shown(hit.paths[0]) if hit.paths else hit.sha256[:12]
    where = page_ref(hit.page, shown(hit.label))
    print(f"{rank}. {name}  {where}  score {hit.score:.2f}", file=out)
    print(f"   {shown(_snippet(hit.best_chunk.text))}", file=out)


def cmd_search(args: argparse.Namespace, out: TextIO, err: TextIO) -> int:
    if args.path.exists() and not _index_path(args).exists():
        return _error(err, f"nothing indexed yet; run: askpdf index {args.path}")
    store = _open_store(args, err, reading=True)
    if store is None:
        return 1
    with store:
        try:
            store.check_folder(folder_key(args.path))
        except WrongFolderError as exc:
            return _error(err, exc)
        if not store.pins_match(pins(PypdfExtractor(), ChunkParams())):
            # Stale text (e.g. an older cleaning) would give wrong results and quotes.
            return _error(
                err,
                f"the index was built with other settings or an older askpdf;"
                f" run: askpdf index {args.path}",
            )
        snapshot = Snapshot.load(store)
    hits = search_bm25(snapshot, args.question, args.k)
    if not hits:
        print("no page matches any word of the question", file=err)
        return 0
    for rank, hit in enumerate(hits, start=1):
        _print_hit(rank, hit, out)
    return 0


def _as_pipe_error(exc: OSError) -> OSError:
    """On Windows a write to a pipe whose reader has gone fails with EINVAL, not EPIPE."""
    if sys.platform == "win32" and exc.errno == errno.EINVAL:
        return BrokenPipeError(errno.EPIPE, "the reader closed the pipe")
    return exc


class _Stdout(io.TextIOBase):
    """stdout, with Windows' EINVAL on a write reported as the broken pipe it is.

    Only writes to stdout are translated, so an EINVAL from anywhere else (a bad path,
    say) still surfaces as an error instead of a silent exit.
    """

    def __init__(self, stream: TextIO) -> None:
        self._stream = stream

    def write(self, text: str) -> int:
        try:
            return self._stream.write(text)
        except OSError as exc:
            raise _as_pipe_error(exc) from exc

    def flush(self) -> None:
        try:
            self._stream.flush()
        except OSError as exc:
            raise _as_pipe_error(exc) from exc


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    commands = {"index": cmd_index, "docs": cmd_docs, "search": cmd_search, "eval": cmd_eval}
    if args.command not in commands:
        build_parser().print_help(sys.stderr)
        return 2
    out = cast(TextIO, _Stdout(sys.stdout))
    try:
        code = commands[args.command](args, out, sys.stderr)
        out.flush()  # a closed pipe shows up here, not in Python's exit flush
        return code
    except BrokenPipeError:
        # The reader went away (`askpdf search … | head`). Point stdout at devnull so
        # the interpreter's final flush does not print a second error.
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, sys.stdout.fileno())
        os.close(devnull)
        return 1
    except KeyboardInterrupt:
        # Every document is committed in its own transaction, so the index is consistent.
        print("\ninterrupted; run the same command again to continue", file=sys.stderr)
        return 130
