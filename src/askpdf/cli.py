"""Command-line entry point.

Results go to stdout and progress to stderr, so output can be piped. Exit codes:
0 success, 1 a run-time problem (missing folder, unusable or busy index, no model), 2 bad
options, 3 an answer none of whose quotes was found on its cited page, 130 interrupted.
`askpdf FOLDER` is short for `askpdf serve FOLDER`. `bench` arrives in a later pull request.
"""

import argparse
import errno
import io
import json
import os
import re
import secrets
import socket
import sqlite3
import sys
import unicodedata
import webbrowser
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, TextIO, cast

from askpdf import __version__
from askpdf.answer import (
    ANSWER_K,
    Answer,
    Final,
    Pages,
    answer,
    build_prompt,
    context_window,
    source_name,
)
from askpdf.chunking import ChunkParams
from askpdf.extract.pypdf_ext import PypdfExtractor
from askpdf.indexer import build_records, embed_missing, index_folder, pins
from askpdf.providers import http, ollama, openai_compat
from askpdf.providers.base import ChatModel
from askpdf.providers.ollama import OllamaEmbedder
from askpdf.providers.openai_compat import allow_remote
from askpdf.retrieve import (
    DEFAULT_K,
    DEFAULT_RETRIEVER,
    RETRIEVERS,
    RRF_K,
    PageHit,
    search_bm25,
    search_dense,
    search_hybrid,
)
from askpdf.snapshot import Snapshot
from askpdf.store import (
    NotAnIndexError,
    OutdatedIndexError,
    Store,
    WrongFolderError,
    default_index_path,
    folder_key,
)

if TYPE_CHECKING:
    from askpdf.web.app import Progress  # imported only by serve, at run time


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
            "--allow-remote",
            action="store_true",
            help="allow a model endpoint that is not this machine (it receives excerpts)",
        )
        sub.add_argument(
            "--index",
            type=Path,
            metavar="PATH",
            help="index file to use (default: one per folder in your cache directory)",
        )

    def with_model(sub: argparse.ArgumentParser) -> None:
        sub.add_argument(
            "--model", metavar="NAME", help=f"chat model (default: Ollama's {ollama.CHAT_MODEL})"
        )
        sub.add_argument(
            "--api-url",
            metavar="URL",
            help="an OpenAI-compatible endpoint (e.g. https://host/v1) instead of Ollama;"
            " the key is read from ASKPDF_API_KEY",
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
        choices=list(RETRIEVERS),
        default=DEFAULT_RETRIEVER,
        help=f"ranking method (default: {DEFAULT_RETRIEVER}; dense and hybrid need Ollama)",
    )
    ask = commands.add_parser("ask", help="answer a question, citing pages with checked quotes")
    with_target(ask)
    ask.add_argument("question", metavar="QUESTION")
    ask.add_argument(
        "-k",
        type=_positive,
        default=ANSWER_K,
        metavar="N",
        help=f"pages to answer from (default: {ANSWER_K})",
    )
    ask.add_argument(
        "--retriever",
        choices=list(RETRIEVERS),
        default=DEFAULT_RETRIEVER,
        help=f"ranking method (default: {DEFAULT_RETRIEVER})",
    )
    ask.add_argument(
        "--show-context", action="store_true", help="print the pages given to the model"
    )
    with_model(ask)
    serve = commands.add_parser(
        "serve", help="open the web app for a folder (the default: askpdf FOLDER)"
    )
    with_target(serve)
    with_model(serve)
    serve.add_argument(
        "--port", type=_port, metavar="N", help="port on 127.0.0.1 (default: first free from 8765)"
    )
    serve.add_argument(
        "--no-browser", action="store_true", help="print the link instead of opening it"
    )
    evaluate = commands.add_parser("eval", help="score one stage on the evaluation data")
    evaluate.add_argument("stage", choices=["extraction", "retrieval", "conversation", "answers"])
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
    evaluate.add_argument(
        "--retriever",
        choices=list(RETRIEVERS),
        default="bm25",
        help="for retrieval (default: bm25)",
    )
    evaluate.add_argument(
        "--rrf-k", type=_positive, default=RRF_K, metavar="N", help="RRF constant for hybrid"
    )
    evaluate.add_argument("--allow-remote", action="store_true", help=argparse.SUPPRESS)
    return parser


def _port(text: str) -> int:
    value = _positive(text)
    if value > 65535:
        raise argparse.ArgumentTypeError(f"{text} is not a port number")
    return value


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
        embedded = _embed(args, store, err)
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
    if embedded is not None:
        parts.append(f"{embedded} chunks embedded")
    print(", ".join(parts), file=out)
    print(f"index: {store.path}", file=err)
    return 0


def _embedder(args: argparse.Namespace) -> OllamaEmbedder:
    url = ollama.base_url()
    http.require_local_or_allowed(url, allow_remote(args.allow_remote))
    return OllamaEmbedder(url)


def _embed(args: argparse.Namespace, store: Store, err: TextIO) -> int | None:
    """Embed the chunks that have no vector yet; None if that was not possible."""
    try:
        report = embed_missing(
            store,
            _embedder(args),
            on_batch=lambda done, total: print(f"embedding {done}/{total} chunks", file=err),
        )
    except http.UnreachableError as exc:
        print(
            f"semantic search unavailable ({shown(str(exc))}); indexed for keyword search."
            " Start Ollama (ollama serve) and run index again to add it.",
            file=err,
        )
        return None
    except http.ProviderError as exc:
        print(f"semantic search unavailable: {shown(str(exc))}", file=err)
        return None
    if report.refused:
        print(
            f"{_plural(len(report.refused), 'chunk')} could not be embedded (the model refused"
            " them, e.g. too long); they are found by keyword search only",
            file=err,
        )
    return report.embedded


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
    run = {
        "retrieval": _eval_retrieval,
        "conversation": _eval_conversation,
        "answers": _eval_answers,
    }
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
    corpus = args.eval_dir / "corpus"
    provenance: dict[str, Any] = {}
    if args.retriever == "bm25":
        retriever = retrieval.bm25_retriever(retrieval.load_corpus_snapshot(corpus))
    else:
        # Needs Ollama, so it runs locally, never in the CI gate (design §16).
        try:
            embedder = _embedder(args)
            provenance = {"embedder": embedder.name, **ollama.server_info(embedder.url)}
            snapshot = retrieval.load_embedded_snapshot(
                corpus, embedder, default_index_path(corpus)
            )
        except http.ProviderError as exc:
            raise ValueError(f"{exc}; dense and hybrid evaluation need Ollama running") from exc
        queries = retrieval.QueryCache(embedder)
        retriever = retrieval.model_retriever(snapshot, queries, args.retriever, args.rrf_k)
        if args.retriever == "hybrid":
            provenance["rrf_k"] = args.rrf_k
    result = retrieval.evaluate(questions.load(path), retriever)
    summary = {
        "stage": "retrieval",
        "split": args.split,
        "retriever": args.retriever,
        **provenance,
        **result,
    }
    if args.release:
        summary["release"] = args.release
    label = {"bm25": "BM25", "dense": "Dense", "hybrid": f"Hybrid (RRF {args.rrf_k})"}
    n = result["n_questions"]
    print(f"{label[args.retriever]} on {args.split}, {n} answerable questions", file=out)
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


def _eval_answers(args: argparse.Namespace, out: TextIO) -> dict[str, Any]:
    from askpdf.evaluate import answers, questions, retrieval  # only `eval` needs them

    path = args.eval_dir / "questions" / f"{args.split}.jsonl"
    if not path.is_file():
        raise ValueError(f"no question set at {path}")
    # Needs Ollama, so it runs locally, never in the CI gate (design §16).
    url = ollama.base_url()
    try:
        http.require_local_or_allowed(url, allow_remote(args.allow_remote))
        provenance = {"model": ollama.CHAT_MODEL, **ollama.server_info(url, ollama.CHAT_MODEL)}
    except http.ProviderError as exc:
        raise ValueError(f"{exc}; the answers stage needs Ollama running") from exc
    snapshot = retrieval.load_corpus_snapshot(args.eval_dir / "corpus")

    def chat(question: str, hits: Sequence[PageHit]) -> ollama.OllamaChat:
        prompt = build_prompt(question, hits, snapshot.texts)
        return ollama.OllamaChat(url, num_ctx=context_window(prompt, 768))

    def progress(done: int, total: int, qid: str) -> None:
        print(f"answering {done}/{total} ({qid})", file=sys.stderr, flush=True)

    try:
        records = answers.run(
            questions.load(path),
            snapshot,
            lambda q, k: search_bm25(snapshot, q, k),
            chat,
            progress,
        )
    except http.ProviderError as exc:
        raise ValueError(f"{exc}; the answers stage needs Ollama running") from exc
    result = answers.summarise(records)
    summary = {
        "stage": "answers",
        "split": args.split,
        "retriever": "bm25",
        "k": ANSWER_K,
        **provenance,
        **result,
        "per_question": dict(sorted(records.items())),
    }
    if args.release:
        summary["release"] = args.release
    g, a = result["grounding"], result["abstention"]
    print(
        f"answers on {args.split}, {result['n_questions']} questions, {ollama.CHAT_MODEL}", file=out
    )
    print(
        f"  grounding: quote-verified {_fmt(g['quote_verified_rate'])}"
        f"  (folding only {_fmt(g['folding_only_share'])})"
        f"  supported sentences {_fmt(g['supported_sentence_rate'])}"
        f"  grounded answers {_fmt(g['grounded_rate'])}",
        file=out,
    )
    print(
        f"  abstention: false answers {_fmt(a['false_answer_rate'])}"
        f"  NOT_FOUND precision {_fmt(a['not_found_precision'])}"
        f" recall {_fmt(a['not_found_recall'])}"
        f"  over-abstention {_fmt(a['over_abstention_rate'])}",
        file=out,
    )
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


# The three retrievers score on different scales, so the number says which one it is.
SCORE_NAMES = {"bm25": "score", "dense": "cosine", "hybrid": "rrf"}


def _print_hit(rank: int, hit: PageHit, out: TextIO, retriever: str = "bm25") -> None:
    name = shown(hit.paths[0]) if hit.paths else hit.sha256[:12]
    where = page_ref(hit.page, shown(hit.label))
    digits = 4 if retriever == "hybrid" else 2  # RRF scores are around 1/60
    print(f"{rank}. {name}  {where}  {SCORE_NAMES[retriever]} {hit.score:.{digits}f}", file=out)
    print(f"   {shown(_snippet(hit.best_chunk.text))}", file=out)


def _load_snapshot(args: argparse.Namespace, err: TextIO) -> Snapshot | None:
    if args.path.exists() and not _index_path(args).exists():
        _error(err, f"nothing indexed yet; run: askpdf index {args.path}")
        return None
    store = _open_store(args, err, reading=True)
    if store is None:
        return None
    with store:
        try:
            store.check_folder(folder_key(args.path))
        except WrongFolderError as exc:
            _error(err, exc)
            return None
        if not store.pins_match(pins(PypdfExtractor(), ChunkParams())):
            # Stale text (e.g. an older cleaning) would give wrong results and quotes.
            _error(
                err,
                f"the index was built with other settings or an older askpdf;"
                f" run: askpdf index {args.path}",
            )
            return None
        return Snapshot.load(store)


def cmd_search(args: argparse.Namespace, out: TextIO, err: TextIO) -> int:
    snapshot = _load_snapshot(args, err)
    if snapshot is None:
        return 1
    hits, used = _search(args, snapshot, err)
    if not hits:
        print("no page matches any word of the question", file=err)
        return 0
    for rank, hit in enumerate(hits, start=1):
        _print_hit(rank, hit, out, used)
    return 0


def _search(args: argparse.Namespace, snapshot: Snapshot, err: TextIO) -> tuple[list[PageHit], str]:
    """Search with the chosen retriever, falling back to keywords when vectors are
    missing; return the hits and the retriever actually used."""
    if args.retriever != "bm25":
        if not snapshot.embedded.any():
            print(
                "no usable semantic index (Ollama was not running when the folder was indexed,"
                " or the embedding model changed); using keyword search",
                file=err,
            )
        else:
            try:
                query = _embedder(args).embed_query(args.question)
                if not snapshot.fully_embedded:
                    print("some chunks are not embedded yet; run index to finish", file=err)
                if args.retriever == "dense":
                    return search_dense(snapshot, query, args.k), "dense"
                return search_hybrid(snapshot, args.question, query, args.k), "hybrid"
            except http.ProviderError as exc:
                print(f"{shown(str(exc))}; using keyword search", file=err)
            except ValueError:  # the query vector's size does not match the index
                print(
                    "the embedding model changed since this folder was indexed; using keyword"
                    f" search. Delete {_index_path(args)} and index again to fix it.",
                    file=err,
                )
    return search_bm25(snapshot, args.question, args.k), "bm25"


def _chat(args: argparse.Namespace) -> ChatModel:
    """Ollama on this machine by default; an OpenAI-compatible endpoint with --api-url."""
    if args.api_url:
        if not args.model:
            raise http.ProviderError("--api-url needs --model")
        return openai_compat.from_env(args.api_url, args.model, args.allow_remote)
    url = ollama.base_url()
    http.require_local_or_allowed(url, allow_remote(args.allow_remote))
    return ollama.OllamaChat(url, args.model or ollama.CHAT_MODEL)


def sized(
    chat: ChatModel,
    question: str,
    hits: Sequence[PageHit],
    texts: Mapping[tuple[str, int], str],
) -> ChatModel:
    """Ollama's context window sized to this prompt; other endpoints manage their own."""
    if isinstance(chat, ollama.OllamaChat):
        prompt = build_prompt(question, hits, texts)
        return replace(chat, num_ctx=context_window(prompt, chat.max_tokens))
    return chat


def cmd_ask(args: argparse.Namespace, out: TextIO, err: TextIO) -> int:
    snapshot = _load_snapshot(args, err)
    if snapshot is None:
        return 1
    try:
        chat = _chat(args)
    except http.ProviderError as exc:
        return _error(err, exc)
    hits, _ = _search(args, snapshot, err)
    chat = sized(chat, args.question, hits, snapshot.texts)
    final: Answer | None = None
    try:
        for event in answer(args.question, hits, snapshot.texts, chat):
            if isinstance(event, Pages):
                print(f"answering from {_plural(len(event.hits), 'page')}…", file=err)
                if args.show_context and event.hits:
                    print(
                        _shown_lines(build_prompt(args.question, event.hits, snapshot.texts)),
                        file=err,
                    )
            elif isinstance(event, Final):
                final = event.answer
    except http.UnreachableError as exc:
        return _error(err, f"{exc}; start Ollama (ollama serve) and try again")
    except http.ProviderError as exc:
        return _error(err, exc)
    if final is None:  # answer() always ends with Final; this is for the type checker
        return _error(err, "the answer did not finish")
    return _print_answer(final, out, err)


FIRST_PORT = 8765


def free_port(first: int = FIRST_PORT, tries: int = 50) -> int | None:
    for port in range(first, first + tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            try:
                probe.bind(("127.0.0.1", port))
            except OSError:
                continue
            return port
    return None


def _index_in_background(args: argparse.Namespace, err: TextIO) -> Callable[["Progress"], None]:
    def run(progress: "Progress") -> None:
        done = 0

        def on_file(rel: str) -> None:
            nonlocal done
            progress.update(current=rel, done=done)
            done += 1

        try:
            # Its own connection: SQLite connections stay in the thread that made them.
            with Store(_index_path(args)) as store:
                report = index_folder(
                    args.path, store, PypdfExtractor(), ChunkParams(), on_file=on_file
                )
        except (WrongFolderError, OSError, sqlite3.Error) as exc:
            print(f"askpdf: indexing failed: {shown(str(exc))}", file=err)
            progress.update(state="failed", current="", message=str(exc))
            return
        for rel, reason in sorted(report.failed.items()):
            print(f"failed: {shown(rel)}: {shown(reason)}", file=err)
        progress.update(state="ready", current="", done=len(report.added))

    return run


def cmd_serve(args: argparse.Namespace, out: TextIO, err: TextIO) -> int:
    import uvicorn  # only the web app needs the server

    from askpdf.web.app import create_app

    if not args.path.is_dir():
        return _error(err, f"{args.path} is not a folder")
    try:
        chat = _chat(args)
    except http.ProviderError as exc:
        return _error(err, exc)
    store = _open_store(args, err)  # creates the index, so readers find one
    if store is None:
        return 1
    with store:
        try:
            store.check_folder(folder_key(args.path))
        except WrongFolderError as exc:
            return _error(err, f"{exc}; pass a different --index")
    port = args.port or free_port()
    if port is None:
        return _error(err, f"no free port from {FIRST_PORT}; pass --port")
    token = secrets.token_urlsafe(32)
    url = f"http://127.0.0.1:{port}/?token={token}"

    def started() -> None:
        print(f"askpdf is running at {url}", file=out, flush=True)
        print("press Ctrl-C to stop", file=err, flush=True)
        if not args.no_browser:
            webbrowser.open(url)

    app = create_app(
        root=args.path,
        index=_index_path(args),
        port=port,
        token=token,
        chat_for=lambda question, hits, texts: sized(chat, question, hits, texts),
        index_folder=_index_in_background(args, err),
        on_start=started,
    )
    # 127.0.0.1 only, with no option to change it (ADR-0008).
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)
    server.run()
    return 0 if server.started else _error(err, f"could not listen on port {port}")


def _shown_lines(text: str) -> str:
    """``shown``, keeping the line breaks."""
    return "\n".join(shown(line) for line in text.split("\n"))


def _print_answer(result: Answer, out: TextIO, err: TextIO) -> int:
    if result.not_found:
        print("The answer is not in these documents.", file=out)
        return 0
    print(_shown_lines(result.text), file=out)
    if result.citations:
        print("\nSources:", file=out)
    for c in result.citations:
        where = page_ref(c.hit.page, shown(c.hit.label))
        print(f"[{c.number}] {shown(source_name(c.hit))}  {where}", file=out)
        if not c.quotes:
            print("    no quote given", file=out)
        if c.quote_on is not None:
            elsewhere = page_ref(c.quote_on.page, shown(c.quote_on.label))
            print(
                f"    the quoted words are on {shown(source_name(c.quote_on))} {elsewhere};"
                " the answer numbered its source wrongly",
                file=out,
            )
        for q in c.quotes:
            mark = "\u2713" if q.verified else f"\u2717 {q.reason}:"
            print(f"    {mark} \u201c{shown(q.quote)}\u201d", file=out)
    if result.dropped:
        numbers = ", ".join(str(n) for n in result.dropped)
        print(f"removed citations of pages the model was not given: {numbers}", file=err)
    if not result.grounded:
        print(
            "ungrounded: no quote in this answer was found on its cited page; check it"
            " against the documents",
            file=err,
        )
        return 3
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
    argv = list(sys.argv[1:] if argv is None else argv)
    commands = {
        "index": cmd_index,
        "docs": cmd_docs,
        "search": cmd_search,
        "ask": cmd_ask,
        "eval": cmd_eval,
        "serve": cmd_serve,
    }
    if argv and argv[0] not in commands and not argv[0].startswith("-"):
        argv.insert(0, "serve")  # `askpdf FOLDER` opens the web app
    args = build_parser().parse_args(argv)
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
