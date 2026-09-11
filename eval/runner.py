"""Evaluation runs over the golden dataset: retrieval-only and end-to-end, with reports."""

import json
import re
import statistics
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

from app.core.config import Settings
from app.core.container import (
    build_chat_model,
    build_container,
    build_embedder,
    build_rag_service,
)
from app.domain.models import Chunk, ChunkType
from app.generation.guard import strip_disclaimers
from app.ingestion.fixed_chunker import FixedSizeChunker
from app.ingestion.pipeline import load_document
from app.retrieval.chunk_store import ChunkStore
from app.retrieval.lexical import BM25Index
from app.retrieval.protocols import Retriever
from app.retrieval.references import ReferenceRouter
from app.retrieval.retriever import HybridRetriever
from app.retrieval.vector_store import InMemoryVectorStore
from app.text.normalize import lexical_form
from eval.dataset import EvalItem, load_dataset, validate_dataset
from eval.metrics import (
    TARGETS,
    E2EOutcome,
    contains_fact,
    e2e_metrics,
    hit_at_k,
    meets_target,
    percentile,
    reciprocal_rank,
    suggest_threshold,
)

REPORTS_DIR = Path(__file__).resolve().parent / "reports"
_LEADING_NUMBER = re.compile(r"^\d+(?:-\d+)?\.\s*")
_SPACES = re.compile(r"\s+")


@dataclass
class Report:
    mode: str
    created_at: str
    git_revision: str
    config: dict[str, Any]
    metrics: dict[str, Any]
    details: list[dict[str, Any]] = field(default_factory=list)
    tables: dict[str, list[dict[str, Any]]] = field(default_factory=dict)


# --- retrieval ---------------------------------------------------------------------------------


async def run_retrieval(settings: Settings, *, compare_chunking: bool = False) -> Report:
    container = await build_container(settings)
    try:
        retriever = container.require_retriever()
        store = container.require_store()
        items = load_dataset()
        validate_dataset(items, {c.id for c in store.all()})
        structural = await _retrieval_rows(retriever, items)
        metrics = _retrieval_metrics(structural, settings.refusal_threshold)
        tables: dict[str, list[dict[str, Any]]] = {"threshold_sweep": _sweep(structural)}
        if compare_chunking:
            fixed_retriever = await _fixed_retriever(settings)
            fixed = await _retrieval_rows(fixed_retriever, items)
            tables["chunking"] = [
                {"strategy": "structural", **_snippet_hits(structural, store)},
                {"strategy": "fixed", **_snippet_hits(fixed, store)},
            ]
        return Report(
            mode="retrieval",
            created_at=_now(),
            git_revision=_git_revision(),
            config=_config(settings),
            metrics=metrics,
            details=structural,
            tables=tables,
        )
    finally:
        container.close()


async def _retrieval_rows(retriever: Retriever, items: list[EvalItem]) -> list[dict[str, Any]]:
    rows = []
    for item in items:
        result = await retriever.retrieve(item.question, top_k=5)
        ranking = [s.chunk.id for s in result.chunks if not s.expansion]
        rows.append(
            {
                "id": item.id,
                "kind": item.kind,
                "question": item.question,
                "expected": list(item.expected_chunks),
                "ranking": ranking,
                "texts": [s.chunk.text for s in result.chunks if not s.expansion],
                "top_similarity": round(result.top_similarity, 4),
                "reference_match": result.reference_match,
                "lexical_anchor": result.lexical_anchor,
            }
        )
    return rows


def _retrieval_metrics(rows: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    scored = [r for r in rows if r["expected"]]
    inside = [r for r in rows if r["kind"] in ("in_doc", "partial")]
    outside = [r for r in rows if r["kind"] in ("out_of_doc", "trap")]

    def mean(values: list[float]) -> float:
        return round(sum(values) / len(values), 3)

    def blocked(group: list[dict[str, Any]]) -> str:
        count = sum(
            r["top_similarity"] < threshold and not (r["reference_match"] or r["lexical_anchor"])
            for r in group
        )
        return f"{count}/{len(group)}"

    in_doc_similarity = [r["top_similarity"] for r in inside]
    gate_dependent = [  # questions that pass only if the similarity clears the threshold
        r["top_similarity"] for r in inside if not (r["reference_match"] or r["lexical_anchor"])
    ]
    return {
        "questions": len(scored),
        "hit_at_1": mean([hit_at_k(r["ranking"], set(r["expected"]), 1) for r in scored]),
        "hit_at_5": mean([hit_at_k(r["ranking"], set(r["expected"]), 5) for r in scored]),
        "mrr": mean([reciprocal_rank(r["ranking"], set(r["expected"])) for r in scored]),
        "similarity_in_doc": _distribution(in_doc_similarity),
        "similarity_out_of_doc": _distribution([r["top_similarity"] for r in outside]),
        "suggested_threshold": suggest_threshold(gate_dependent),
        "threshold": threshold,
        "gate_blocks_out_of_doc": blocked(outside),
        "gate_blocks_in_doc": blocked(inside),
    }


def _sweep(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """How many questions the gate alone would stop at several thresholds."""
    table = []
    for threshold in (0.40, 0.45, 0.50, 0.55, 0.60):
        metrics = _retrieval_metrics(rows, threshold)
        table.append(
            {
                "threshold": threshold,
                "stopped_out_of_doc": metrics["gate_blocks_out_of_doc"],
                "stopped_in_doc": metrics["gate_blocks_in_doc"],
            }
        )
    return table


def _distribution(values: list[float]) -> dict[str, float]:
    return {
        "min": round(min(values), 3),
        "p10": round(percentile(values, 10), 3),
        "median": round(statistics.median(values), 3),
        "max": round(max(values), 3),
    }


async def _fixed_retriever(settings: Settings) -> HybridRetriever:
    """A throwaway in-memory index of fixed-size windows, for the chunking comparison."""
    fixed = settings.model_copy(update={"chunk_strategy": "fixed"})
    chunks = FixedSizeChunker(fixed.fixed_chunk_size, fixed.fixed_chunk_overlap).chunk(
        load_document(fixed)
    )
    embedder = build_embedder(fixed)
    vectors = InMemoryVectorStore()
    await vectors.add(chunks, await embedder.embed([c.embedding_text for c in chunks]))
    store = ChunkStore(chunks)
    return HybridRetriever(
        store=store, embedder=embedder, vector_store=vectors, lexical=BM25Index(chunks),
        router=ReferenceRouter.from_store(store), top_k=5, candidates=fixed.retrieval_candidates,
        expansion_max=0, token_budget=100_000,
    )  # fmt: skip


def _snippet(chunk: Chunk) -> str | None:
    """Text that identifies a structural chunk inside any plain-text rendering of the page."""
    if chunk.type is ChunkType.SCHEME_STAGE or chunk.type is ChunkType.OVERVIEW:
        return None
    text = chunk.text.split(" Toifasi:")[0] if chunk.type is ChunkType.TABLE_ROW else chunk.text
    text = _LEADING_NUMBER.sub("", text.split("\n")[0])
    return _SPACES.sub(" ", lexical_form(text))[:60] or None


def _snippet_hits(rows: list[dict[str, Any]], store: ChunkStore) -> dict[str, Any]:
    """hit@1 / hit@5 judged by content, so structural chunks and windows are comparable."""
    judged = hit1 = hit5 = 0
    for row in rows:
        snippets = [s for c in row["expected"] if (s := _snippet(store.get(c)))]
        if not snippets:
            continue
        judged += 1
        texts = [_SPACES.sub(" ", lexical_form(t)) for t in row["texts"]]
        found = [any(s in t for s in snippets) for t in texts]
        hit1 += bool(found[:1] and found[0])
        hit5 += any(found[:5])
    return {
        "questions": judged,
        "hit_at_1": round(hit1 / judged, 3),
        "hit_at_5": round(hit5 / judged, 3),
    }


# --- end to end --------------------------------------------------------------------------------


async def run_e2e(settings: Settings, *, models: list[str] | None = None) -> Report:
    container = await build_container(settings)
    try:
        retriever = container.require_retriever()
        items = load_dataset()
        validate_dataset(items, {c.id for c in container.require_store().all()})
        comparison, details = [], []
        for model in models or [settings.llm_model]:
            rag = build_rag_service(settings, retriever, build_chat_model(settings, model))
            await rag.ask("Qaror qachon kuchga kiradi?")  # warm-up: loads the model into memory
            outcomes = []
            for item in items:
                started = perf_counter()
                answer = await rag.ask(item.question)
                latency = int((perf_counter() - started) * 1000)
                outcome, detail = judge(item, answer.status.value, answer.text, answer.citations)
                outcomes.append(
                    E2EOutcome(
                        item.id,
                        item.kind,
                        answer.status.value,
                        outcome,
                        detail["cited_expected"],
                        latency,
                    )
                )
                details.append({"model": model, **detail, "latency_ms": latency})  # fmt: skip
            comparison.append({"model": model, **e2e_metrics(outcomes)})
        best = comparison[0]
        return Report(
            mode="e2e",
            created_at=_now(),
            git_revision=_git_revision(),
            config=_config(settings) | {"models": models or [settings.llm_model]},
            metrics={k: v for k, v in best.items() if k != "model"},
            details=details,
            tables={"models": comparison},
        )
    finally:
        container.close()


def judge(
    item: EvalItem, status: str, text: str, citations: list[str]
) -> tuple[bool, dict[str, Any]]:
    facts = all(contains_fact(text, fact) for fact in item.must_include)
    claims = strip_disclaimers(text)  # "X haqida ma'lumot yo'q" does not assert X
    forbidden = [f for f in item.forbidden if contains_fact(claims, f)]
    if item.kind == "in_doc":
        correct = status != "not_found" and facts
    elif item.kind == "partial":
        correct = status == "partial" and facts
    elif item.kind == "trap":
        correct = status in ("not_found", "partial") and not forbidden
    else:
        correct = status == "not_found"
    return correct, {
        "id": item.id,
        "kind": item.kind,
        "question": item.question,
        "status": status,
        "answer": text,
        "citations": citations,
        "expected": list(item.expected_chunks),
        "cited_expected": any(c in item.expected_chunks for c in citations),
        "correct": correct,
    }


# --- reports -----------------------------------------------------------------------------------


def write_report(report: Report, directory: Path = REPORTS_DIR) -> tuple[Path, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    stamp = report.created_at.replace(":", "").replace("-", "")[:15]
    base = directory / f"{stamp}-{report.mode}"
    json_path = base.with_suffix(".json")
    json_path.write_text(json.dumps(asdict(report), ensure_ascii=False, indent=2), encoding="utf-8")
    md_path = base.with_suffix(".md")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    return md_path, json_path


def render_markdown(report: Report) -> str:
    lines = [
        f"# Baholash hisoboti: {report.mode}",
        "",
        f"- Sana: {report.created_at}",
        f"- Git: `{report.git_revision}`",
        f"- Sozlamalar: `{json.dumps(report.config, ensure_ascii=False)}`",
        "",
        "## Metrikalar",
        "",
        "| Metrika | Qiymat | Maqsad | Holat |",
        "|---|---|---|---|",
    ]
    for name, value in report.metrics.items():
        target = TARGETS.get(name)
        verdict = meets_target(name, value if isinstance(value, float | int) else None)
        mark = "" if verdict is None else ("✅" if verdict else "❌")
        shown = json.dumps(value, ensure_ascii=False) if isinstance(value, dict) else value
        lines.append(
            f"| {name} | {shown} | {' '.join(map(str, target)) if target else ''} | {mark} |"
        )
    for title, rows in report.tables.items():
        if not rows:
            continue
        lines += ["", f"## Taqqoslash: {title}", "", "| " + " | ".join(rows[0]) + " |"]
        lines.append("|" + "---|" * len(rows[0]))
        lines += ["| " + " | ".join(str(v) for v in row.values()) + " |" for row in rows]
    failures = [d for d in report.details if d.get("correct") is False]
    if report.mode == "retrieval":
        failures = [
            d
            for d in report.details
            if d["expected"] and not hit_at_k(d["ranking"], set(d["expected"]), 5)
        ]
    if failures:
        lines += ["", "## Muvaffaqiyatsiz savollar", ""]
        for d in failures:
            got = d.get("answer", d.get("ranking"))
            who = d.get("model", d["kind"])
            lines.append(
                f"- **{d['id']}** ({who}) {d['question']} → kutilgan {d['expected']}, "
                f"olingan: {got} {d.get('citations', '')}"
            )
    return "\n".join(lines) + "\n"


def _config(settings: Settings) -> dict[str, Any]:
    keys = (
        "llm_model", "embed_model", "chunk_strategy", "chunk_max_chars", "retrieval_top_k",
        "retrieval_candidates", "refusal_threshold", "llm_num_ctx", "llm_temperature",
    )  # fmt: skip
    return {k: getattr(settings, k) for k in keys}


def _git_revision() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")
