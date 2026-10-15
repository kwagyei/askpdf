"""Command-line entry point.

Results go to stdout and progress to stderr, so output can be piped. Exit codes:
0 success, 1 a run-time problem (missing folder, unusable or busy index), 2 bad
options, 130 interrupted.
The remaining subcommands (serve, search, ask, eval, bench) arrive in later pull
requests.
"""

import argparse
import sqlite3
import sys
import unicodedata
from collections.abc import Sequence
from pathlib import Path
from typing import TextIO

from askpdf import __version__
from askpdf.chunking import ChunkParams
from askpdf.extract.pypdf_ext import PypdfExtractor
from askpdf.indexer import index_folder
from askpdf.store import NotAnIndexError, Store, WrongFolderError, default_index_path


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
    return parser


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


def _open_store(args: argparse.Namespace, err: TextIO) -> Store | None:
    if not args.path.exists():
        _error(err, f"{args.path} does not exist")
        return None
    try:
        return Store(_index_path(args))
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
    store = _open_store(args, err)
    if store is None:
        return 1
    with store:
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


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    commands = {"index": cmd_index, "docs": cmd_docs}
    if args.command not in commands:
        build_parser().print_help(sys.stderr)
        return 2
    try:
        return commands[args.command](args, sys.stdout, sys.stderr)
    except KeyboardInterrupt:
        # Every document is committed in its own transaction, so the index is consistent.
        print("\ninterrupted; run the same command again to continue", file=sys.stderr)
        return 130
