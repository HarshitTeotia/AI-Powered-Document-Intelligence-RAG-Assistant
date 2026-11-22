"""Benchmark the local RAG pipeline in isolated processes without logging content."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import platform
import statistics
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from contextlib import ExitStack
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CSV = ROOT / "eval/questions.csv"
UPLOADS = ROOT / "data/uploads"
STORE = ROOT / "data/vector_store"
APP = ROOT / "distribuai/ui/app.py"
RUNS = 3
IMPORTS = ("torch", "sentence_transformers", "chromadb", "distribuai.ui.app")


def _ms(start):
    return (perf_counter() - start) * 1000


def _hash_file(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _hash_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _page_hashes(pages):
    return [(p.page_number, _hash_text(p.text), p.needs_ocr) for p in pages]


def _cpu():
    cpu = platform.processor() or "unknown"
    cores = None
    if sys.platform == "win32":
        import ctypes
        import winreg

        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
            cpu = winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
        query = ctypes.windll.kernel32.GetLogicalProcessorInformationEx
        length = ctypes.c_ulong()
        query(0, None, ctypes.byref(length))  # RelationProcessorCore
        if length.value:
            buffer = ctypes.create_string_buffer(length.value)
            if query(0, buffer, ctypes.byref(length)):
                offset = cores = 0
                while offset < length.value:
                    size = int.from_bytes(buffer[offset + 4:offset + 8], "little")
                    if size < 8:
                        raise RuntimeError("Windows returned invalid processor topology.")
                    offset += size
                    cores += 1
    return cpu, cores, os.cpu_count()


class _Timers:
    def __init__(self):
        self.times = defaultdict(float)
        self.calls = Counter()
        self.native = []
        self.native_hashes = []

    def timed(self, label, func, *, native=False):
        def run(*args, **kwargs):
            table = self.times.get("table detection", 0)
            started = perf_counter()
            try:
                return_value = func(*args, **kwargs)
            finally:
                elapsed = _ms(started)
                self.times[label] += elapsed
                self.calls[label] += 1
                if native:
                    self.native.append((elapsed, self.times.get("table detection", 0) - table))
            if native:
                self.native_hashes.append(_page_hashes(return_value))
            return return_value
        return run

    def instrument(self, stack, obj, attr, label, *, native=False):
        original = getattr(obj, attr)
        stack.enter_context(patch.object(obj, attr, self.timed(label, original, native=native)))


def _imports():
    times = {}
    for name in IMPORTS:
        for state in ("cold", "warm"):
            start = perf_counter()
            importlib.import_module(name)
            times[f"Import / {name} / {state}"] = _ms(start)
    return {"times": times}


def _model():
    importlib.import_module("sentence_transformers")  # Load imports before timing.
    from distribuai.rag.embeddings import load_embedding_model

    times = {}
    for state in ("cold", "warm"):
        start = perf_counter()
        try:
            load_embedding_model()
        except Exception as exc:
            return {"times": {}, "error": type(exc.__cause__ or exc).__name__}
        times[state] = _ms(start)
    return {"times": times}


def _ingestion(collection, pdfs, phase, out, guard):
    from pdfplumber.page import Page
    from pdfplumber.table import Table

    from distribuai.ingestion import ocr_fallback as ocr, pdf_extractor as pdf, pipeline
    from distribuai.rag import embeddings

    for index, path in enumerate(pdfs, start=1):
        timer = _Timers()
        final_pages = []
        extract = pipeline._extract_pages

        def save_pages(*args, **kwargs):
            pages, count = extract(*args, **kwargs)
            final_pages.extend(pages)
            return pages, count

        with ExitStack() as stack:
            for obj, attr, label in (
                (pdf, "_tagged_table_blocks", "table detection"),
                (Page, "find_tables", "table detection"),
                (Table, "extract", "table detection"),
                (ocr, "_ocr_page", "OCR"),
                (pipeline, "chunk_pdf_pages", "chunking"),
                (embeddings, "_count_tokens", "token counting"),
                (pipeline, "embed_chunks", "embedding"),
                (type(collection), "upsert", "ChromaDB upsert"),
                (pipeline, "_source_chunk_ids", "stale-chunk cleanup"),
                (pipeline, "_delete_source_chunks", "stale-chunk cleanup"),
            ):
                timer.instrument(stack, obj, attr, label)
            for obj in (pipeline, ocr):
                timer.instrument(stack, obj, "extract_pdf_pages", "native pass", native=True)
            stack.enter_context(patch.object(pipeline, "_extract_pages", save_pages))
            start = perf_counter()
            result = pipeline.ingest_pdf(path, collection)
            total = _ms(start)
        primary, tables = timer.native[-1]
        times = {
            "total": total,
            "native extraction": max(0, primary - tables),
            "table detection and rendering": tables,
            "extra native pass": sum(elapsed for elapsed, _ in timer.native[:-1]),
            "OCR": timer.times["OCR"],
            "chunking": timer.times["chunking"],
            "token counting": timer.times["token counting"],
            "embedding": max(0, timer.times["embedding"] - timer.times["token counting"]),
            "ChromaDB upsert": timer.times["ChromaDB upsert"],
            "stale-chunk cleanup": timer.times["stale-chunk cleanup"],
        }
        for name, ms in times.items():
            out[f"PDF {index} / {name} / {phase}"] = ms
        ids = collection.get(where={"source_file": path.name}, include=[])["ids"]
        guard[path.name] = {
            "ingestion": asdict(result), "native": timer.native_hashes[-1],
            "extracted": _page_hashes(final_pages), "chunk_ids": sorted(ids),
        }


def _mock_request(_client, _model, content, _settings):
    source = json.loads(content)["untrusted_document_excerpts"][0]
    return SimpleNamespace(text=json.dumps({
        "status": "answered", "answer": "Synthetic benchmark response.",
        "evidence": [{"source_id": source["source_id"], "quote": source["text"]}],
    }))


def _questions(collection, questions, phase, out, *, live=False):
    from distribuai.rag import generation, retrieval
    from distribuai.ui import service

    per_question = defaultdict(list)
    status = {}
    guard = {}
    for question in questions:
        timer = _Timers()
        request_hash = []
        original_build = generation._build_request

        def build(*args, **kwargs):
            content, settings = original_build(*args, **kwargs)
            request_hash.append(_hash_text(content + settings.model_dump_json()))
            return content, settings

        with ExitStack() as stack:
            stack.enter_context(patch.object(generation, "_build_request", build))
            for obj, attr, label in (
                (retrieval, "embed_query", "query embedding"),
                (type(collection), "count", "ChromaDB count"),
                (type(collection), "query", "ChromaDB query"),
                (generation, "_build_request", "prompt build"),
                (generation, "_validate_response", "post-processing"),
                (generation, "_create_client", "Gemini client creation"),
                (generation, "_close_client", "Gemini client close"),
            ):
                timer.instrument(stack, obj, attr, label)
            request = generation._request if live else _mock_request
            stack.enter_context(patch.object(generation, "_request", timer.timed("Gemini call", request)))
            start = perf_counter()
            try:
                answer = service.ask_documents(question.text, collection)
            except (generation.GeminiGenerationError, generation.GeminiResponseError,
                    generation.GeminiConfigurationError) as exc:
                if not live:
                    raise
                status[question.id] = type(exc).__name__
            else:
                status[question.id] = answer.status
                if not live:
                    guard[question.id] = [request_hash, _hash_text(json.dumps(asdict(answer), sort_keys=True))]
            timer.times["total"] = _ms(start)
        assert timer.calls["Gemini call"] <= 1
        per_question["Gemini calls per question"].append(timer.calls["Gemini call"])
        for name, value in timer.times.items():
            if live or name != "Gemini call":
                per_question[name].append(value)
    for name, values in per_question.items():
        if name != "Gemini calls per question":
            out[f"Question / {name} / {phase}"] = statistics.median(values)
    return {"calls": per_question["Gemini calls per question"], "statuses": status, "guard": guard}


def _retrieval_guard(collection, questions):
    from distribuai.evaluation.scoring import aggregate_retrieval_scores, score_retrieval
    from distribuai.rag.retrieval import retrieve

    scores = []
    hits = {}
    for question in questions:
        results = retrieve(collection, question.text)
        hits[question.id] = [(hit.chunk_id, hit.similarity) for hit in results]
        if question.category == "answerable":
            scores.append(score_retrieval(results, question.source, question.pages, question.terms))
    return {"metrics": asdict(aggregate_retrieval_scores(scores)), "hits": hits}


def _streamlit(pdfs, out):
    import streamlit as st
    from streamlit.testing.v1 import AppTest

    from distribuai.rag import embeddings, generation, vector_store
    from distribuai.ui import service

    before = embeddings._cached_model.cache_info()
    counts = Counter()
    snapshots = {}
    tokenizer_ids = set()

    def track(label, original):
        def call(*args, **kwargs):
            counts[label] += 1
            result = original(*args, **kwargs)
            if label == "tokenizer uses":
                tokenizer_ids.add(id(result[0]))
            return result
        return call

    def run(label, action):
        previous = counts.copy()
        misses = embeddings._cached_model.cache_info().misses
        time = perf_counter()
        result = action()
        out[f"Streamlit / {label}"] = _ms(time)
        assert not result.exception
        snapshots[label] = {
            **{key: counts[key] - previous[key] for key in (
                "reruns", "collection opens", "list_documents", "Gemini clients", "uploads", "tokenizer uses",
            )},
            "model loads": embeddings._cached_model.cache_info().misses - misses,
        }
        return result

    key = "benchmark-placeholder"
    upload_data = pdfs[0].read_bytes()
    with ExitStack() as stack:
        for obj, attr, label in (
            (st, "set_page_config", "reruns"),
            (vector_store, "open_vector_collection", "collection opens"),
            (vector_store, "list_documents", "list_documents"),
            (generation, "_create_client", "Gemini clients"),
            (service, "ingest_uploaded_pdf", "uploads"),
            (embeddings, "_model_tokenizer", "tokenizer uses"),
        ):
            stack.enter_context(patch.object(obj, attr, track(label, getattr(obj, attr))))
        stack.enter_context(patch.object(generation, "_request", _mock_request))
        # AppTest has no uploader setter. Keep the file selected through reruns
        # and click the real form submit button once to verify the event guard.
        stack.enter_context(patch.object(st, "file_uploader", return_value=SimpleNamespace(
            name="Benchmark upload.pdf", size=len(upload_data), getvalue=lambda: upload_data,
        )))
        stack.enter_context(patch.dict(os.environ, {"GEMINI_API_KEY": key, "GEMINI_CHAT_ENABLED": "true"}))
        app = AppTest.from_file(str(APP), default_timeout=60)
        run("initial", app.run)
        run("explicit rerun", app.run)
        run("submit question", app.chat_input[0].set_value("What is the return policy?").run)
        run("upload", next(button for button in app.button if button.label == "Add to knowledge base").click().run)
        run("rerun after upload", app.run)
    assert before.misses == embeddings._cached_model.cache_info().misses
    return {"interactions": snapshots, "distinct_tokenizers": len(tokenizer_ids)}


def _pipeline(directory, *, live=False):
    times = {}
    if not live:
        for phase in ("cold", "warm"):
            start = perf_counter()
            importlib.import_module("distribuai.ui.app")
            times[f"Import / app standalone / {phase}"] = _ms(start)
    import torch
    from eval.evaluate_retrieval import _discover_pdfs, _load_questions
    from distribuai import config
    from distribuai.rag import embeddings, vector_store

    config.CHROMA_STORAGE_DIR = directory
    pdfs = _discover_pdfs(UPLOADS)
    assert len(pdfs) == 4, "Expected the four benchmark PDFs."
    questions = _load_questions([CSV])
    first_five = [q for q in questions if q.category == "answerable"][:5]
    assert len(first_five) == 5
    start = perf_counter()
    collection = vector_store.open_vector_collection(directory)
    times["ChromaDB / open new store"] = _ms(start)
    embeddings.load_embedding_model()
    guards = {}
    for phase in ("cold", "warm"):
        state = {}
        _ingestion(collection, pdfs, phase, times, state)
        guard = _questions(collection, first_five, phase, times, live=live)
        guards[phase] = {"pdfs": state, "questions": guard}
    if live:
        return {
            "times": times,
            "calls": {phase: item["questions"]["calls"] for phase, item in guards.items()},
            "statuses": {phase: item["questions"]["statuses"] for phase, item in guards.items()},
        }
    guards["retrieval"] = _retrieval_guard(collection, questions)
    hardware = {"CPU": _cpu(), "torch threads": torch.get_num_threads(), "CUDA": torch.cuda.is_available()}
    return {"times": times, "guard": guards, "ui": _streamlit(pdfs, times), "hardware": hardware}


def _worker(mode, directory):
    if mode == "imports":
        return _imports()
    if mode == "model":
        return _model()
    directory = directory.resolve(strict=True)
    temp = Path(tempfile.gettempdir()).resolve()
    if (directory.is_relative_to(ROOT) or not directory.is_relative_to(temp) or any(directory.iterdir())
            or not directory.parent.name.startswith("rag-pipeline-benchmark-")):
        raise ValueError("Worker requires an empty temporary store from the benchmark parent.")
    if mode == "pipeline":
        os.environ["GEMINI_API_KEY"] = "benchmark-placeholder"
    os.environ["GEMINI_CHAT_ENABLED"] = "true"
    return _pipeline(directory, live=mode == "live")


def _child(mode, directory, *, offline=True, inherit_key=False):
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONDONTWRITEBYTECODE": "1",
           "PYTHON_DOTENV_DISABLED": "1"}
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "GOOGLE_API_KEY"):
        env.pop(name, None)
    if offline:
        env["HF_HUB_OFFLINE"] = "1"
    if not inherit_key:
        env.pop("GEMINI_API_KEY", None)
    cmd = [sys.executable, "-B", str(Path(__file__).resolve()), "--_worker", mode]
    if directory:
        cmd.extend(["--_directory", str(directory)])
    timeout = 45 if mode == "model" and not offline else 240
    try:
        completed = subprocess.run(cmd, env=env, cwd=ROOT, capture_output=True, encoding="utf-8", timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        if mode == "model" and not offline:
            stderr = exc.stderr or ""
            if isinstance(stderr, bytes):
                stderr = stderr.decode("utf-8", errors="replace")
            return {"times": {}, "error": f"exceeded {timeout} s including imports",
                    "network_retry_notices": stderr.count("Retrying in")}
        raise RuntimeError(f"{mode} child exceeded {timeout} s.") from None
    lines = completed.stdout.splitlines()
    try:
        result = json.loads(lines[-1])
    except (IndexError, json.JSONDecodeError):
        raise RuntimeError(f"{mode} child exited without a benchmark result.") from None
    if completed.returncode:
        error = result.get("error", "") if isinstance(result, dict) else ""
        error = error if isinstance(error, str) and error.isidentifier() else "details withheld"
        raise RuntimeError(f"{mode} child failed ({error}).")
    if mode == "model":
        result["network_retry_notices"] = completed.stderr.count("Retrying in")
    return result


def _protected_files():
    paths = [CSV, *UPLOADS.glob("*.pdf")]
    paths.extend(p for p in STORE.rglob("*") if p.is_file())
    return {path: _hash_file(path) for path in paths}


def _report(out, report):
    if report is not None:
        target = report.resolve()
        if target.is_relative_to(ROOT):
            raise ValueError("Report output must be outside the repository.")
        with target.open("x", encoding="utf-8") as stream:
            json.dump(out, stream, indent=2, ensure_ascii=True, sort_keys=True)
            stream.write("\n")
    print(f"Median of {RUNS} runs for each local measurement, sorted by milliseconds.")
    for label, duration in sorted(out["median_ms"].items(), key=lambda item: (-item[1], item[0])):
        print(f"{duration:10.2f}  {label}")
    for field, value in out["hardware"].items():
        print(f"{field}: {value}")
    for index, name in enumerate(out["pdf_order"], start=1):
        print(f"PDF {index}: {name}")
    for name, counts in out["ui"][0]["interactions"].items():
        print(f"Streamlit {name}: {counts}")
    print("Distinct tokenizer objects used per Streamlit session:", out["ui"][0]["distinct_tokenizers"])
    print("Gemini calls per question in offline simulation:", out["guard"][0]["cold"]["questions"]["calls"])
    print("Router: N/A (no separate local route). Order lookup: N/A (no order system).")
    print("Gemini network:", "unverified (offline run)" if out["network"] is None else out["network"])
    print("Hugging Face retry notices per model load:", out["model_retry_notices"])
    if out["model_errors"]:
        print("Model load conditions unavailable:", out["model_errors"])
    if out["network_check_skipped"]:
        print("Network-enabled Hugging Face model-load check was skipped.")
    print("Read-only input and persistent storage hashes verified.")


def _parent(args):
    if args.live_gemini and not os.environ.get("GEMINI_API_KEY", "").strip():
        raise ValueError("Set GEMINI_API_KEY in the process environment before a live run.")
    temp = Path(tempfile.gettempdir()).resolve()
    if temp.is_relative_to(ROOT):
        raise ValueError("The system temporary directory must be outside the repository.")
    if len([path for path in UPLOADS.glob("*.pdf") if path.is_file() and not path.is_symlink()]) != 4:
        raise ValueError("Benchmark requires the four sample PDFs from docs/DATA_SOURCE_MANIFEST.md in data/uploads.")
    protected = _protected_files()
    outcomes = defaultdict(list)
    guard, ui = [], []
    hardware = network = None
    model_errors = Counter()
    model_retries = defaultdict(list)
    try:
        with tempfile.TemporaryDirectory(prefix="rag-pipeline-benchmark-", dir=temp) as tmp:
            folder = Path(tmp)
            for index in range(RUNS):
                print(f"Local benchmark run {index + 1}/{RUNS}", flush=True)
                for label, ms in _child("imports", None)["times"].items():
                    outcomes[label].append(ms)
                model = _child("model", None)
                model_retries["set"].append(model["network_retry_notices"])
                for phase, ms in model["times"].items():
                    outcomes[f"Model load / HF_HUB_OFFLINE set / {phase}"].append(ms)
                if "error" in model:
                    model_errors[f"set: {model['error']}"] += 1
                directory = folder / str(index)
                directory.mkdir()
                result = _child("pipeline", directory)
                for label, ms in result["times"].items():
                    outcomes[label].append(ms)
                guard.append(result["guard"])
                ui.append(result["ui"])
                hardware = result["hardware"]
            if not args.skip_network_check:
                for index in range(RUNS):
                    print(f"Network-enabled Hugging Face check {index + 1}/{RUNS}", flush=True)
                    model = _child("model", None, offline=False)
                    model_retries["unset"].append(model["network_retry_notices"])
                    for phase, ms in model["times"].items():
                        outcomes[f"Model load / HF_HUB_OFFLINE unset / {phase}"].append(ms)
                    if "error" in model:
                        model_errors[f"unset: {model['error']}"] += 1
            if args.live_gemini:
                directory = folder / "live"
                directory.mkdir()
                result = _child("live", directory, inherit_key=True)
                network = {key: value for key, value in result["times"].items() if "Gemini call" in key}
                network["attempts by pass"] = result["calls"]
                network["statuses by pass"] = result["statuses"]
    finally:
        if _protected_files() != protected:
            raise RuntimeError("A protected input or persistent store file changed.")
    _report({
        "median_ms": {key: statistics.median(samples) for key, samples in outcomes.items() if len(samples) == RUNS},
        "samples_ms": outcomes, "guard": guard, "ui": ui, "hardware": hardware,
        "network": network, "model_errors": model_errors, "model_retry_notices": model_retries,
        "network_check_skipped": args.skip_network_check,
        "pdf_order": [path.name for path in sorted(UPLOADS.glob("*.pdf"), key=lambda path: path.name.casefold())],
    }, args.report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, help="Write timing and regression JSON to a new file outside the repo.")
    parser.add_argument("--live-gemini", action="store_true", help="Opt into at most 10 real Gemini requests.")
    parser.add_argument("--skip-network-check", action="store_true", help="Skip the network-enabled Hugging Face check.")
    parser.add_argument("--_worker", choices=("imports", "model", "pipeline", "live"), help=argparse.SUPPRESS)
    parser.add_argument("--_directory", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    try:
        if args._worker:
            print(json.dumps(_worker(args._worker, args._directory), ensure_ascii=True))
        else:
            _parent(args)
    except Exception as exc:
        if args._worker:
            print(json.dumps({"error": type(exc).__name__}))
        else:
            print(f"Benchmark failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
