"""Command-line entry point.

Results go to stdout and progress to stderr, so output can be piped. Exit codes:
0 success, 1 a run-time problem (missing folder, unusable index), 2 bad options.
The remaining subcommands (serve, search, ask, eval, bench) arrive in later pull
requests.
"""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import TextIO

from askpdf import __version__
from askpdf.chunking import ChunkParams
from askpdf.extract.pypdf_ext import PypdfExtractor
from askpdf.indexer import index_folder
from askpdf.store import NotAnIndexError, Store, default_index_path


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


def _plural(n: int, word: str) -> str:
    return f"{n} {word}{'' if n == 1 else 's'}"


def _open_store(args: argparse.Namespace, err: TextIO) -> Store | None:
    if not args.path.exists():
        print(f"askpdf: error: {args.path} does not exist", file=err)
        return None
    try:
        return Store(args.index or default_index_path(args.path))
    except NotAnIndexError as exc:
        print(f"askpdf: error: {exc}", file=err)
        return None


def cmd_index(args: argparse.Namespace, out: TextIO, err: TextIO) -> int:
    store = _open_store(args, err)
    if store is None:
        return 1
    with store:
        report = index_folder(
            args.path,
            store,
            PypdfExtractor(),
            ChunkParams(),
            on_file=lambda rel: print(f"reading {rel}", file=err),
        )
    if report.rebuilt:
        print("settings changed since the last run: rebuilt the index", file=err)
    for rel, reason in sorted(report.failed.items()):
        print(f"failed: {rel}: {reason}", file=err)
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
        name = doc.paths[0] if doc.paths else doc.sha256[:12]
        line = (
            f"{name}  {_plural(doc.n_pages, 'page')}, {_plural(doc.n_chunks, 'chunk')}"
            f", {doc.n_image_only} image-only"
        )
        if len(doc.paths) > 1:
            line += f"  (also: {', '.join(doc.paths[1:])})"
        if doc.looks_scanned:
            line += "  [looks scanned; no OCR, so most pages have no searchable text]"
        print(line, file=out)
    for f in sorted(failed, key=lambda f: f.path):
        print(f"{f.path}  FAILED: {f.error}", file=out)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "index":
        return cmd_index(args, sys.stdout, sys.stderr)
    if args.command == "docs":
        return cmd_docs(args, sys.stdout, sys.stderr)
    build_parser().print_help(sys.stderr)
    return 2
