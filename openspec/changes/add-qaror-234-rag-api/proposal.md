## Why

Cabinet of Ministers resolution No. 234 (11.05.2026) replaces the 2020 environmental impact assessment regime (resolution No. 541) with a 9-appendix legal text — eight regulations, a 221-row activity/category/fee table, workflow schemes and form templates. Businesses, project developers and eco-experts need fast, exact answers from it ("which category is a 100 MW solar plant, how many working days and how many BXM does the review take?"), and a wrong answer to a legal question is worse than no answer. General-purpose chatbots hallucinate on this material and cannot run inside a closed environment, so we need a local, verifiable, document-bound question-answering API.

## What Changes

- New Python/FastAPI service that answers questions about resolution No. 234 using a Retrieval-Augmented Generation pipeline running fully locally on Ollama.
- Ingestion of the official lex.uz page (committed HTML snapshot for offline, reproducible builds; optional refresh from the live URL) into a structured document tree: resolution → appendix → regulation → chapter → numbered item → sub-items, plus tables, schemes and footnotes, each keeping its lex.uz element id for deep links.
- Structure-aware ("legal semantic") chunking: one chunk per numbered item with its sub-items, one chunk per defined term, one chunk per row of the Appendix 1 activity table, one chunk per scheme stage; every chunk carries a breadcrumb path and a deep link.
- Hybrid retrieval: multilingual dense embeddings (bge-m3 via Ollama) + BM25 over normalized, lightly stemmed Uzbek tokens, fused with Reciprocal Rank Fusion; explicit references in the question ("2-ilova 15-band") are resolved directly; cross-referenced items are pulled into the context.
- Layered anti-hallucination: retrieval confidence gate, strict grounding prompt at temperature 0, JSON-schema-constrained output with mandatory citations, citation and number verification against the cited text, and an exact refusal string — `Hujjatda bu haqida ma'lumot yo'q` — whenever the document does not cover the question.
- HTTP API: `POST /api/v1/ask`, `POST /api/v1/search`, `GET /api/v1/chunks/{chunk_id}`, `GET /health`, OpenAPI docs, and a minimal built-in web page for live demos.
- Evaluation harness with a golden question set (in-document, out-of-document and trap questions) that measures retrieval hit@k/MRR, refusal accuracy, fact accuracy and latency, and compares chunking strategies and models.
- One-command local runtime via Docker Compose (Ollama + API, optional GPU override) and a native Python path; a step-by-step Uzbek README and a presentation document.

## Capabilities

### New Capabilities
- `document-ingestion`: loading the resolution (snapshot or live), parsing lex.uz HTML into a typed document tree, and text normalization (apostrophe variants, Cyrillic → Latin, whitespace).
- `structural-chunking`: turning the document tree into retrievable chunks with breadcrumbs, deep links and metadata; rules for items, definitions, table rows, schemes, oversize and undersize pieces.
- `hybrid-retrieval`: dense + lexical search with RRF fusion, Uzbek-aware tokenization, explicit-reference routing, cross-reference context expansion and a calibrated relevance signal for the refusal gate.
- `grounded-answering`: prompt construction, constrained generation through Ollama, citation and numeric verification, answer statuses (answered / partial / not_found) and the exact refusal contract.
- `qa-api`: the async HTTP interface, request/response schemas, error handling, health reporting and the demo page.
- `answer-evaluation`: golden dataset format, retrieval-only and end-to-end evaluation runs, metrics and report output.
- `local-runtime`: configuration through environment variables, model provisioning, index lifecycle, Docker Compose (CPU default, GPU override) and native execution.

### Modified Capabilities
<!-- None: this is a greenfield project with no existing specs. -->

## Impact

- **New code**: `app/` package (core, api, domain, text, ingestion, retrieval, generation, services), `eval/`, `tests/`, `data/raw/` snapshot.
- **New dependencies**: fastapi, uvicorn, pydantic-settings, ollama (Python client), chromadb, rank-bm25, beautifulsoup4 + lxml for HTML parsing; dev: pytest, pytest-asyncio, httpx, ruff.
- **External systems**: local Ollama server only (models `bge-m3` plus one chat model, default chosen by evaluation among `qwen2.5:7b`, `qwen3.5:4b`, `qwen3.5:9b`); no network access needed at query time. Live lex.uz access is optional and used only for snapshot refresh.
- **Resources**: ~5–8 GB disk for models; runs on an 8 GB VRAM GPU, degrades to CPU with higher latency.
- **Docs**: `README.md` (setup and usage, Uzbek) and `docs/PRESENTATION.md` (detailed plan, architecture and results, Uzbek).
