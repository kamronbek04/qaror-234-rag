## Context

Motivation and scope: see `proposal.md`. Requirements: see `specs/*/spec.md`. This document records how we build it and why.

**What the source looks like** (measured on the lex.uz snapshot, 1.49 MB HTML):

| Property | Value |
|---|---|
| Text volume | ~190 000 characters (~60–90 K tokens) — does not fit a local 4–9 B model context, so retrieval is required |
| Structure | main resolution (8 items) + 9 appendices; appendices 2–8 are regulations (`N-bob` → `N.` item → unnumbered sub-items) |
| Numbered items | 277 — median 264 chars, p90 ≈ 1 070, max 5 837 (appendix 9, item 1, 24 sub-items); 62 items under 150 chars |
| Definitions | 68 `term — definition` lines in "asosiy tushunchalar" items |
| Appendix 1 table | 221 rows; 3 categories (I high, II medium, III low hazard) × 13–16 sectors; columns: activity, deadline (working days), fee (BXM) |
| Other | 7 workflow schemes (SXEMA), footnotes, application/certificate forms, 31 item cross-references, 23 appendix references |
| Markup | server-rendered; every element has a semantic class (`ACT_TEXT`, `TEXT_HEADER_DEFAULT`, `APPL_BANNER_LANDSCAPE_TITLE`, `ACT_TITLE_APPL`, `ACT_FORM`, `TABLE_STD2`, `FOOTNOTE`, `TEXT_CENTER`) and sits in a `<div id="-82054xx">` usable as a deep-link anchor; interface text ("Hujjatga taklif yuborish…") is embedded inside elements |

**Constraints:** fully local (Ollama), baseline hardware is an 8 GB VRAM laptop GPU (RTX 4060) with 32 GB RAM, CPU-only must still work; the reviewer must be able to start everything with one command; tests must run without Ollama.

**Model landscape on Ollama (September 2026):** `qwen2.5` (7b = 4.7 GB; officially 29+ languages, Uzbek not among the listed ones), `qwen3.5` (4b = 3.4 GB, 9b = 6.6 GB; 201 languages), `llama3.1:8b`, `gemma4` (12b and up — too large for 8 GB next to the embedder); embeddings `bge-m3` (567 M, 1.2 GB, 100+ languages, 8 K context), `qwen3-embedding` (0.6b+), `embeddinggemma` (300 M).

## Goals / Non-Goals

**Goals:**
- Exactness over fluency: when in doubt, refuse.
- Every answer verifiable in one click (quote + deep link to the exact item).
- Guarantees enforced by code, not by prompt wording: the refusal string, citation validity and number fidelity are checked deterministically.
- Swappable infrastructure: vector store, embedder, chat model and retriever sit behind protocols; the pipeline does not know which implementation it uses.
- Every major choice backed by an evaluation number.

**Non-Goals:**
- Multiple documents, document upload or generic legal search.
- Conversation memory and follow-up questions.
- Authentication, rate limiting, multi-tenancy.
- Token streaming (see D11).
- Fine-tuning or training models.
- Answers in Russian or English (questions in Cyrillic are accepted, answers are Uzbek Latin).

## Decisions

### D1. Structure-aware chunking, not fixed-size or similarity-based "semantic chunking"
Chunk boundaries follow the legal structure (spec `structural-chunking`): one item with its sub-items, one definition, one table row, one scheme stage. In legal text the meaning unit *is* the item: it is what users cite, it keeps enumerations ("quyidagilar:" + list) intact, and it maps 1:1 to a lex.uz anchor.
- Each chunk text is prefixed with its breadcrumb (contextual chunk header), which gives short items ("8. … 1-ilovadagi sxemaga muvofiq oʻtkaziladi.") enough context to embed well.
- Table rows are rendered as full sentences ("1-ilova … I toifa (yuqori darajada xavfli). Soha: … 2-qator: Aeroportlar. Ekspertiza muddati: 25 ish kuni. Toʻlov miqdori: 25 BXM.") so that the numbers are never separated from their meaning.
- Max chunk size 1 500 characters (configurable); longer items split at sub-item boundaries, each part repeating breadcrumb and lead sentence.
- Short items are **not** merged with neighbours.
- Identifier grammar: `q-b{n}` (main resolution item), `a{N}-b{n}` (appendix item), `-p{k}` (part), `-d{k}` (definition), `a1-r{n}` (table row), `a{N}-fn{n}` (footnote), `a{N}-x{m}-s{k}` (stage k of the scheme in attachment m), `a{N}-x{m}-form` (form template).

*Rejected:* fixed-size windows — cut enumerations and numbers mid-sentence; kept only as the evaluation baseline. Embedding-similarity splitters — non-deterministic across embedding models, split lists at arbitrary points, and lose the item identity needed for citations. Merging short items to a minimum size — would make one chunk cite two items and break the 1:1 deep link.

### D2. HTML parsing with BeautifulSoup (lxml backend) driven by lex.uz CSS classes
A single-pass walker over `div#divCont` classifies elements by class and builds the tree with a small state machine (current appendix → attachment → chapter → item). Integrity checks (spec `document-ingestion`) run before indexing.
*Rejected:* regex over HTML — fragile for nested tables. selectolax — faster, but speed is irrelevant for one 1.5 MB page and BeautifulSoup is more readable to reviewers.

### D3. One normalization function for index and query; light Uzbek stemmer for BM25 only
- `normalize()` maps all apostrophe variants to ASCII `'` (both oʻ/gʻ and tutuq belgisi — users type `'`), transliterates Uzbek Cyrillic (ў→o', ғ→g', қ→q, ҳ→h, ш→sh, ч→ch, ё→yo, ю→yu, я→ya, е→ye word-initially/after vowels else e, ц→ts, ъ→', ь→∅), NFC-normalizes and collapses whitespace.
- Dense text keeps casing; lexical text is lowercased, tokenized on non-letters (keeping in-word `'` and `-`), and stemmed.
- Stemmer: ordered longest-first inflectional suffixes (plural `-lar`; possessive `-imiz -ingiz -lari -si -im -ing -i`; case `-ning -ni -ga -ka -qa -da -ta -dan -tan -dagi -gacha`), up to 3 stripping passes, minimum stem length 3.

*Rejected:* no stemming — Uzbek is agglutinative ("ekspertiza / ekspertizasi / ekspertizasidan / ekspertizaning"), so BM25 misses matches. Character n-grams — larger index, noisier scores, harder to explain. A full morphological analyzer — no maintained lightweight Python package for Uzbek.

### D4. Embeddings: `bge-m3` through Ollama `/api/embed`
The model is multilingual (XLM-R base covers Uzbek), has an 8 K context, needs no query/passage instruction prefixes, and is small enough (1.2 GB) to share the GPU with the chat model. Chunks are embedded in batches of 32 at ingestion; queries are embedded one at a time. Vectors are L2-normalized and compared by cosine similarity.
*Rejected:* `nomic-embed-text` — English-centric. `multilingual-e5` — not in the Ollama library, so it would break "fully Ollama". `qwen3-embedding:0.6b` stays a configured alternative that evaluation can promote (config change + reindex).

### D5. Vector store: ChromaDB embedded `PersistentClient`, behind a `VectorStore` protocol
Chroma is listed in the task, needs no extra container, persists to disk, and stores metadata next to vectors. At ~1 300 vectors, its HNSW index is effectively exact and deterministic for a fixed index. Its synchronous calls run in `asyncio.to_thread`.
*Rejected:* FAISS — vectors only, so metadata and persistence would have to be built by hand. Qdrant / Milvus — an extra service to run for about a thousand vectors. pgvector — needs PostgreSQL, and Postgres full-text search has no Uzbek configuration.

### D6. Lexical index: in-memory BM25 (`rank-bm25`) rebuilt from the chunk store at startup
`chunks.jsonl` is the single source of truth for chunk lookup, the BM25 corpus and the Chroma contents. Building BM25 over ~1 300 short documents takes milliseconds, so it is not persisted.
*Rejected:* Chroma `$contains` filters — not ranked. `bm25s` — faster, but unnecessary at this scale.

### D7. Fusion: Reciprocal Rank Fusion + explicit-reference routing + cross-reference expansion
- The top 30 dense and top 30 BM25 candidates are fused with RRF (k = 60); the final `top_k` defaults to 6.
- A reference router recognizes `N-ilova`, `N-bob`, `N-band`, `qarorning N-bandi` and `N-qator`/`N-bandida` on Appendix 1, and pins the matching chunks first.
- Up to 3 cross-referenced chunks are appended as expansions.
- The context budget is 3 500 estimated tokens (characters / 3, conservative for Uzbek), with lowest-ranked excerpts dropped first.

*Rejected:* weighted score sums — cosine and BM25 scales are incomparable and would need per-query normalization. A cross-encoder reranker (`bge-reranker-v2-m3`) — needs PyTorch (2 GB+), is not served by Ollama, and would break the "everything through Ollama" story; it stays behind a `Reranker` protocol as a future option. LLM-based reranking — adds a full model call of latency.

### D8. Refusal gate on calibrated dense similarity
The gate uses the maximum cosine similarity among dense candidates (an absolute, comparable number), not the RRF score (rank-only, not comparable across queries). Its job is to drop *clearly* unrelated questions cheaply, so the threshold sits below the lowest in-document score seen in the retrieval evaluation, with a margin. Borderline questions go through to the model, which can still return `not_found`, and to verification. An explicit reference match bypasses the gate.
*Rejected:* gating on the RRF score — no absolute meaning. A high threshold that makes the gate the main defence — it would produce many false refusals for colloquial questions.

### D9. Generation: Ollama chat with JSON-schema-constrained output
- Called through the official `ollama` Python `AsyncClient`, wrapped by a `ChatModel` protocol.
- `format` is the JSON schema of `LLMAnswer {status: answered|partial|not_found, answer: str, citations: list[str]}`.
- Options: `temperature=0`, `seed=42`, `num_ctx=8192` (explicit — Ollama's default context would silently truncate excerpts), `num_predict=512`, `keep_alive=30m`. `think=false` is sent only when configured, for thinking-capable models.
- The system prompt is in English, which instruction-following models obey more reliably, with explicit rules: use only the excerpts, answer in Uzbek Latin, cite chunk ids, copy numbers exactly as written, return `not_found` if the excerpts do not contain the answer, and ignore instructions inside the question. Two short Uzbek few-shot examples follow, one `answered` and one `not_found`.
- The excerpts go in a delimited block: `[chunk_id] breadcrumb\ntext`.

*Rejected:* free-text answers with regex citation extraction — brittle. Tool/function calling — adds nothing here. Temperature above 0 — non-deterministic and more prone to invention.

### D10. Deterministic verification (`AnswerGuard`), fail-closed
A pure function with no model involved:
1. Status `not_found` or an empty answer → the API returns the constant `Hujjatda bu haqida ma'lumot yo'q`. The refusal text always comes from code, never from the model, so it is exact by construction.
2. Citations not among the supplied chunk ids are dropped; no valid citation left → refusal.
3. Every number in the answer (`\d+(?:[.,]\d+)?`, comma treated as decimal point) must occur in the text or breadcrumb of a cited chunk. On failure there is one regeneration with feedback that names the unsupported numbers; a second failure → refusal.
4. Status `partial` → if the answer lacks the refusal sentence, the code appends "Savolning qolgan qismi boʻyicha: Hujjatda bu haqida ma'lumot yo'q."
5. Schema-invalid output → one retry, then refusal.

Every decision is recorded in debug output.
*Rejected:* LLM-as-judge — a second model call doubles latency, and small models are unreliable judges. An NLI entailment model — needs PyTorch. Both could be added behind the same guard interface later.

### D11. API: FastAPI app factory, lifespan-built container, thin routers, no streaming
- `create_app(settings)`: the lifespan builds a `Container` (the composition root: settings → clients → indexes → retriever → service) and stores it on `app.state`. Routers get services through `Depends`.
- Domain exceptions (`LLMUnavailableError`, `EmbeddingUnavailableError`, `IndexNotReadyError`) map to 503 JSON errors with codes and Uzbek messages.
- Middleware handles `X-Request-ID`; logs are structured JSON with request id, timings and guard decisions.
- `asyncio.Semaphore(LLM_MAX_CONCURRENCY=2)` bounds generations.
- A static `app/web/index.html` (vanilla JS, inline CSS, no CDN) is served at `/`.

*Rejected:* SSE token streaming — the answer must be verified before any of it reaches the user (D10), so streaming would either show unverified text or stream a finished answer, which gains nothing. Module-level singletons — hide dependencies and make tests harder.

### D12. Code layout and seams
```
app/
  main.py            # create_app(), lifespan, middleware, error handlers
  cli.py             # ingest | ask | eval  (argparse, no extra dependency)
  core/              # config.py (pydantic-settings), logging.py, container.py, errors.py
  domain/            # models.py: DocumentNode, Chunk, ScoredChunk, RetrievalResult, Answer
  text/              # normalize.py, stemmer.py, numbers.py
  ingestion/         # source.py (snapshot/refresh), lex_parser.py, chunker.py, fixed_chunker.py, pipeline.py
  retrieval/         # protocols.py, embedder.py, vector_store.py, lexical.py, fusion.py, references.py, retriever.py
  generation/        # protocols.py, ollama_chat.py, prompts.py, guard.py
  services/          # rag_service.py (retrieve → gate → generate → verify → respond)
  api/               # schemas.py, routes_ask.py, routes_search.py, routes_system.py, deps.py
  web/index.html
eval/                # dataset.jsonl, metrics.py, runner.py, reports/
tests/               # unit/, api/, integration/ (marked, needs Ollama)
data/raw/lex_8193120.html   # committed snapshot
```
Protocols: `Embedder`, `VectorStore`, `LexicalIndex`, `ChatModel`, `Retriever`, `Reranker` (unused). Tests use `FakeEmbedder` (deterministic hashed bag-of-stems vectors) and `FakeChatModel` (scripted replies).

### D13. Index lifecycle: versioned directories + `CURRENT` pointer
- Ingestion writes `data/index/<fingerprint>/`, which contains `chunks.jsonl`, `chroma/` and `manifest.json`. The fingerprint is sha256 of: snapshot bytes, chunking config, embedding model name and schema version.
- The directory is built in full, then `data/index/CURRENT` is atomically replaced (write temp file + `os.replace`). Older versions beyond the last 2 are pruned.
- The API opens the version named in `CURRENT`. It auto-ingests on startup when `AUTO_INGEST=true` and the index is missing or stale; otherwise `/health` reports `stale`.

*Rejected:* renaming the live Chroma directory — Windows keeps its files locked while a client has them open.

### D14. Default models
- Embedding model: `bge-m3`.
- Chat model: chosen by the end-to-end evaluation among `qwen2.5:7b` (the task's suggestion and a strong JSON follower), `qwen3.5:4b` (201 languages, 3.4 GB) and `qwen3.5:9b` (best quality, but 6.6 GB plus the 1.2 GB embedder approaches the 8 GB VRAM limit). Until evaluated, the default is `qwen2.5:7b`.
- Settings: `OLLAMA_MAX_LOADED_MODELS=2`, `keep_alive=30m`. The documented CPU-only fallback is `qwen3.5:4b`.

### D15. Containers
- `Dockerfile`: `python:3.12-slim`, non-root user, dependencies installed before the source is copied (for layer caching), `uvicorn app.main:app`.
- `docker-compose.yml`:
  - `ollama` (pinned `ollama/ollama` tag, named volume, healthcheck).
  - `models`, a one-shot container from the same image that runs `ollama pull` for both configured models against `OLLAMA_HOST=http://ollama:11434`.
  - `api`, which depends on `models` having completed successfully, with `AUTO_INGEST=true`, the index on a named volume and port 8000.
- `docker-compose.gpu.yml` adds NVIDIA device reservations to `ollama`.

*Rejected:* GPU in the base file — would break `docker compose up` on machines without an NVIDIA GPU (macOS, most CI).

### D16. Testing
- Unit tests, all without network:
  - normalizer (apostrophes, Cyrillic);
  - stemmer;
  - parser against the committed snapshot (9 appendices, 221 rows, no UI noise);
  - chunker (breadcrumbs, ids, table sentences, splitting, cross-references);
  - RRF;
  - reference router;
  - guard (citations, numbers, partial, refusal constant);
  - metrics.
- Service and API tests: the real retriever over a small fixture index built with `FakeEmbedder`, `FakeChatModel` scripts, and `httpx.AsyncClient` with `ASGITransport`. Tests cover 422, 404 and 503 mapping, request-id round trip and the concurrency bound.
- Integration tests are marked `ollama` and skipped when the server is unreachable.

### D17. No RAG orchestration framework
The pipeline is about 1 500 lines of plain Python behind explicit protocols: parse → chunk → embed/index → retrieve → gate → generate → verify.
*Rejected:* LangChain / LlamaIndex. Their generic splitters, retrievers and output parsers hide exactly the parts this task is judged on (structural chunking, fusion, verification). They add large, fast-moving dependency trees, and they make the fail-closed control flow harder to read and test. The protocols keep the door open to wrap any of their components later.

### D18. Evaluation
- `eval/dataset.jsonl` has about 50 items: `{id, question, kind, expected_chunks[], must_include[]}`.
- `python -m app.cli eval retrieval` computes hit@k, MRR and similarity distributions with a threshold suggestion; `--compare-chunking` rebuilds a temporary fixed-size index.
- `python -m app.cli eval e2e [--models a,b,c]` runs `RagService` directly (same code path as the API) and computes refusal precision/recall, fact accuracy, citation accuracy and latency p50/p95.
- Reports are written to `eval/reports/<utc-timestamp>-<mode>.{md,json}` with the git revision and configuration. The headline numbers are copied into `README.md` and `docs/PRESENTATION.md`.

## Risks / Trade-offs

- [Small models write imperfect Uzbek and sometimes slip into Russian or Turkish words] → temperature 0, short extractive-leaning answers, source quotes shown next to every answer, model chosen by evaluation with the 201-language `qwen3.5` as a candidate.
- [`bge-m3` retrieval quality on Uzbek is only moderate] → BM25 with Uzbek stemming covers exact terms and numbers, breadcrumbs add context, reference routing handles explicit citations, and hit@5 is tracked against a target.
- [A miscalibrated threshold causes false refusals or lets junk through] → the gate is only a coarse first line, biased to let borderline questions pass; the model's `not_found` and deterministic verification are the second and third lines; the threshold is derived from measured distributions.
- [The number guard rejects correct answers that restate numbers differently, e.g. "uch oy" → "3 oy"] → the prompt says to copy numbers exactly; one feedback retry; the residual false refusals are an accepted, fail-closed bias.
- [Numbers written as words ("uch oy", "ikki nafar") escape the number guard] → accepted; the citation check still applies and trap questions in evaluation watch for invented dates and amounts.
- [lex.uz layout changes] → the committed snapshot keeps builds working; integrity checks turn silent breakage into a clear failure.
- [Ollama's default context window truncates prompts] → explicit `num_ctx` on every request plus a prompt budget below it.
- [8 GB VRAM shared by two models] → the default pair needs ≤ 6 GB; `qwen3.5:9b` is documented as partially offloading to CPU; a CPU profile is documented.
- [No streaming, so 3–8 s perceived latency on GPU and more on CPU] → accepted: verified-before-shown is the core promise; the UI shows progress.
- [Chroma brings a heavy dependency tree (onnxruntime, etc.)] → accepted in exchange for zero extra services and alignment with the task; the protocol allows swapping it.
- [Prompt injection via the question] → the question sits in a delimited user message and the rules in the system prompt; injected claims cannot be cited, so verification rejects them.
- [The resolution may be amended after the snapshot date] → the snapshot date is shown in `/health` and the README; `ingest --refresh` updates it.

## Migration Plan

Greenfield service — nothing to migrate. Deploy with `docker compose up -d` (optionally with the GPU override). Rollback of a bad index: point `data/index/CURRENT` back to the previous version directory, which is kept, and restart the API.

## Open Questions

- Which chat model becomes the default (`qwen2.5:7b` vs `qwen3.5:4b` vs `qwen3.5:9b`) — decided by the end-to-end evaluation; configuration-only change.
- The exact refusal-threshold value — set from the retrieval evaluation; configuration-only change.
- Whether `qwen3-embedding:0.6b` beats `bge-m3` on this corpus — decided by the retrieval evaluation; configuration change plus reindex.
