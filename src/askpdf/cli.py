"""Command-line entry point.

Results go to stdout and progress to stderr, so output can be piped. Exit codes:
0 success, 1 a run-time problem (missing folder, unusable or busy index), 2 bad
options, 130 interrupted.
The remaining subcommands (serve, ask, eval, bench) arrive in later pull requests.
"""

import argparse
import errno
import os
import sqlite3
import sys
import unicodedata
from collections.abc import Sequence
from pathlib import Path
from typing import TextIO

from askpdf import __version__
from askpdf.chunking import ChunkParams
from askpdf.extract.pypdf_ext import PypdfExtractor
from askpdf.indexer import index_folder, pins
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


def _closed_pipe(exc: OSError) -> bool:
    """Writing to a pipe whose reader has gone: EPIPE, or EINVAL on Windows."""
    return isinstance(exc, BrokenPipeError) or (
        sys.platform == "win32" and exc.errno == errno.EINVAL
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    commands = {"index": cmd_index, "docs": cmd_docs, "search": cmd_search}
    if args.command not in commands:
        build_parser().print_help(sys.stderr)
        return 2
    try:
        code = commands[args.command](args, sys.stdout, sys.stderr)
        sys.stdout.flush()  # a closed pipe shows up here, not in Python's exit flush
        return code
    except OSError as exc:
        if not _closed_pipe(exc):
            raise
        # The reader went away (`askpdf search … | head`). Point stdout at devnull so
        # the interpreter's final flush does not print a second error.
        os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
        return 1
    except KeyboardInterrupt:
        # Every document is committed in its own transaction, so the index is consistent.
        print("\ninterrupted; run the same command again to continue", file=sys.stderr)
        return 130
