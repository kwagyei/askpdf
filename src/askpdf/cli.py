"""Command-line entry point.

The subcommands (serve, index, docs, search, ask, eval, bench) arrive in later pull
requests; until then the CLI only reports its version.
"""

import argparse
from collections.abc import Sequence

from askpdf import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="askpdf",
        description="Ask questions about a folder of PDFs; answers cite the page.",
    )
    parser.add_argument("--version", action="version", version=f"askpdf {__version__}")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    build_parser().parse_args(argv)
    return 0
