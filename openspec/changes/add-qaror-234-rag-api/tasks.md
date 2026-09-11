## 1. Project foundation

- [x] 1.1 Create `pyproject.toml` (Python ≥ 3.11; runtime deps: fastapi, uvicorn[standard], pydantic-settings, ollama, chromadb, rank-bm25, beautifulsoup4, lxml, httpx; dev deps: pytest, pytest-asyncio, ruff) with ruff and pytest configuration; verify `pip install -e ".[dev]"` succeeds in a fresh Python 3.12 virtualenv
- [x] 1.2 Create the package skeleton from design D12 (`app/core`, `app/domain`, `app/text`, `app/ingestion`, `app/retrieval`, `app/generation`, `app/services`, `app/api`, `app/web`, `eval/`, `tests/unit`, `tests/api`, `tests/integration`); verify `python -c "import app"` and `pytest` (empty suite) both succeed
- [x] 1.3 Implement `app/core/config.py` (pydantic-settings, every tunable from spec `local-runtime`, validation) and `.env.example` documenting defaults; verify a unit test that an invalid `REFUSAL_THRESHOLD` fails with an error naming the variable
- [x] 1.4 Implement `app/core/errors.py` (domain exceptions with error codes) and `app/core/logging.py` (JSON logs with request id); verify unit tests for error codes and a log record containing `request_id`
- [x] 1.5 Add `.gitignore`, commit the lex.uz snapshot to `data/raw/lex_8193120.html` with `data/raw/SOURCE.md` (URL, download date, sha256); verify the recorded sha256 matches the file

## 2. Text utilities

- [x] 2.1 Implement `app/text/normalize.py` (apostrophe unification, Uzbek Cyrillic → Latin, NFC, whitespace, case folding for lexical form); verify unit tests for the spec scenarios (four apostrophe variants equal; "Экологик экспертиза" → "Ekologik ekspertiza")
- [x] 2.2 Implement `app/text/stemmer.py` (tokenizer + ordered inflectional suffix stripping); verify unit tests that "ekspertiza", "ekspertizasi", "ekspertizasidan", "ekspertizaning" share one stem and short words stay intact
- [x] 2.3 Implement `app/text/numbers.py` (number extraction with decimal-comma normalization); verify unit tests ("7,5" == "7.5"; "2026-yil 11-may" → {2026, 11}; "110 kV" → {110})

## 3. Parsing the resolution

- [ ] 3.1 Implement `app/domain/models.py` (`DocumentNode`, `TableRow`, `Chunk`, `ScoredChunk`, `RetrievalResult`, `Answer`, enums for node/chunk types and answer status); verify models construct and serialize in a unit test
- [ ] 3.2 Implement `app/ingestion/source.py` (load snapshot; optional refresh that validates before replacing and keeps the old snapshot on failure); verify unit tests with a mocked HTTP transport for success, HTTP error and invalid page
- [ ] 3.3 Implement `app/ingestion/lex_parser.py` (class-driven walker: main resolution, appendices, titles, document types, chapters, items with sub-items, attachments, element ids, UI-noise stripping); verify tests on the snapshot: 9 appendices, appendix 2 has 8 chapters, item 4 of appendix 2 has 6 sub-items, no UI noise strings anywhere
- [ ] 3.4 Parse the Appendix 1 table into 221 typed rows (category, hazard level, sector, number, activity, deadline days, fee BXM or fee text); verify tests for rows 2 (I, 25, 25), 69 (II, 25, 15) and 134 (III, 15, 7,5)
- [ ] 3.5 Implement parse integrity checks; verify a test where appendix headers are removed from the HTML raises an error naming the missing appendices

## 4. Structural chunking

- [ ] 4.1 Implement item chunks with breadcrumbs, stable ids (design D1 grammar), element ids and deep links; verify tests for `a2-b6` breadcrumb text, `q-b6` content and deep-link format
- [ ] 4.2 Implement definition chunks; verify the `ekolog-ekspert` definition chunk exists, contains "kamida uzluksiz uch yil ish stajiga ega" and references `a2-b2`
- [ ] 4.3 Implement Appendix 1 row chunks as standalone sentences plus metadata; verify tests for `a1-r2` and `a1-r69` wording and metadata
- [ ] 4.4 Implement scheme-stage, footnote and form chunks; verify the footnote chunk about unlisted activity types and at least one scheme stage chunk per scheme
- [ ] 4.5 Implement oversize splitting at sub-item boundaries with repeated breadcrumb and lead sentence; verify `a9-b1` is split into parts and no part exceeds the max size unless a single sub-item does
- [ ] 4.6 Implement cross-reference extraction; verify `a2-b5` lists `a2-b4` and appendix references resolve to existing chunks
- [ ] 4.7 Implement the fixed-size baseline chunker; verify window size and overlap in a unit test
- [ ] 4.8 Verify determinism: chunking the snapshot twice yields identical ids and texts, and all ids are unique

## 5. Indexing and hybrid retrieval

- [ ] 5.1 Define retrieval protocols and implement `OllamaEmbedder` (batched `/api/embed`, normalization, timeout, errors → `EmbeddingUnavailableError`); verify unit tests with a fake Ollama client
- [ ] 5.2 Implement `ChromaVectorStore` (persistent, cosine, metadata, calls via `asyncio.to_thread`); verify an add/query round trip in a temp directory with fake vectors
- [ ] 5.3 Implement `BM25Index` over stemmed lexical text; verify that an inflected query form scores the chunk containing another form
- [ ] 5.4 Implement RRF fusion with document-order tie-break; verify unit tests for ranking and ties
- [ ] 5.5 Implement the explicit-reference router; verify "2-ilovaning 6-bandida nima deyilgan?" → `a2-b6`, "Qarorning 7-bandi" → `q-b7`, "1-ilova 2-qator" → `a1-r2`
- [ ] 5.6 Implement `HybridRetriever` (query normalization, dense + BM25 candidates, fusion, pinned references, cross-reference expansion, context budget, relevance signal); verify tests over a fixture index built with `FakeEmbedder`
- [ ] 5.7 Implement the ingestion pipeline with fingerprinted versioned index directories, atomic `CURRENT` pointer and pruning, plus `python -m app.cli ingest [--refresh] [--strategy]`; verify a FakeEmbedder test of the atomic swap and a real run against Ollama that writes a manifest with the expected chunk count

## 6. Grounded answering

- [ ] 6.1 Implement `app/generation/prompts.py` (system rules, Uzbek few-shot examples, excerpt block, token budget); verify a unit test on the rendered prompt structure and budget trimming
- [ ] 6.2 Implement `OllamaChatModel` (JSON-schema `format`, temperature 0, seed, explicit `num_ctx`, optional `think`, timeouts, errors → `LLMUnavailableError`); verify unit tests with a fake client that the request carries these options
- [ ] 6.3 Implement `AnswerGuard` (refusal constant, citation filtering, number verification, partial suffix); verify unit tests for every `grounded-answering` scenario that does not need a real model
- [ ] 6.4 Implement `RagService` (retrieve → gate → generate under semaphore → verify → one retry → respond, with debug trace); verify tests with `FakeChatModel`: the gate skips the model, a wrong number triggers retry then refusal, a fabricated citation yields refusal, malformed output twice yields refusal, a correct answer passes

## 7. HTTP API and demo page

- [ ] 7.1 Implement API schemas, `create_app()`, lifespan container, request-id middleware and error handlers; verify the app starts with a fake container in tests
- [ ] 7.2 Implement `/api/v1/ask`, `/api/v1/search`, `/api/v1/chunks/{chunk_id}` and `/health`; verify API tests for 200, 422 (empty, too long, bad `top_k`), 404 and 503 (backend down, missing model, stale index) and the request-id round trip
- [ ] 7.3 Verify bounded concurrency: 5 simultaneous asks with a bound of 2 all succeed while `/health` answers in under one second (API test with a slow fake model)
- [ ] 7.4 Implement the self-contained demo page at `/` (question box, status, answer, clickable sources, no external assets); verify an API test that `/` returns the page and a manual browser check against the running stack
- [ ] 7.5 Add OpenAPI examples for every endpoint; verify `/docs` shows them

## 8. Runtime and containers

- [ ] 8.1 Write the `Dockerfile` (python:3.12-slim, non-root, cached dependency layer); verify `docker build .` succeeds
- [ ] 8.2 Write `docker-compose.yml` (pinned `ollama`, one-shot `models` puller, `api` with auto-ingest and index volume) and `docker-compose.gpu.yml`; verify `docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d` reaches `/health` 200 on this machine and Ollama reports GPU use
- [ ] 8.3 Verify the native path: run the API from a virtualenv against an Ollama endpoint on `localhost:11434` and get `/health` 200
- [ ] 8.4 Add a `Makefile` for Unix shells (install, ingest, run, test, eval, up, down) mirroring the raw commands documented in the README; verify `make test` passes

## 9. Evaluation

- [ ] 9.1 Write `eval/dataset.jsonl` (~50 items: in_doc with expected chunks and required facts, out_of_doc, trap, partial; ≥ 25% out_of_doc/trap; ≥ 5 Cyrillic or non-canonical apostrophes) and the dataset validator; verify validation passes against the built index
- [ ] 9.2 Implement `eval/metrics.py` (hit@k, MRR, refusal precision/recall, fact accuracy, citation accuracy, latency percentiles); verify unit tests with hand-computed examples
- [ ] 9.3 Implement the retrieval runner with threshold suggestion and chunking comparison, writing Markdown and JSON reports; verify a report is produced for structural and fixed-size strategies
- [ ] 9.4 Implement the end-to-end runner with model comparison and target marking; verify a report is produced listing failed questions
- [ ] 9.5 Run both evaluations on this machine for `qwen2.5:7b`, `qwen3.5:4b` and `qwen3.5:9b`; set the default chat model and refusal threshold from the results in `.env.example` and config defaults; verify the chosen configuration meets the targets or the gaps are documented

## 10. Documentation and delivery

- [ ] 10.1 Finalize `README.md` in Uzbek (overview, mermaid architecture diagram, Docker quick start, native setup for Windows/Linux/macOS, configuration table, curl and PowerShell examples, evaluation results, troubleshooting); verify every command by following it on a fresh clone
- [ ] 10.2 Finalize `docs/PRESENTATION.md` in Uzbek with the measured evaluation numbers and demo script; verify every number matches the latest report in `eval/reports/`
- [ ] 10.3 Run final quality gates: `ruff check`, `ruff format --check`, full `pytest`, `openspec validate --strict`, and review `git ls-files` so that only project files are tracked; verify all pass
- [ ] 10.4 Push to `github.com/kamronbek04/qaror-234-rag` and verify the repository page renders the README and the latest commit
