## Purpose

Exposes question answering, retrieval inspection and service health over an asynchronous HTTP API (plus a tiny demo page) with stable, documented request and response contracts.

## ADDED Requirements

### Requirement: Ask endpoint
The system SHALL expose `POST /api/v1/ask` accepting JSON `{"question": string, "top_k"?: integer, "debug"?: boolean}` and returning `{"answer", "status", "found", "sources", "meta"}`, where `status` is one of `answered`, `partial`, `not_found`; each source contains `chunk_id`, `breadcrumb`, `quote` (original text), `url` (lex.uz deep link) and `score`; and `meta` contains the model name, a request id and timings in milliseconds for retrieval, generation and total. When `debug` is true the response MUST additionally include the retrieval candidates with all scores and the verification decisions.

#### Scenario: Covered question
- **WHEN** a client posts `{"question": "Ekolog-ekspert kim?"}`
- **THEN** the response is HTTP 200 with status `answered`, `found` true, and at least one source whose `url` starts with `https://lex.uz/uz/docs/-8193120#`

#### Scenario: Uncovered question
- **WHEN** a client posts a question the resolution does not cover
- **THEN** the response is HTTP 200 with answer "Hujjatda bu haqida ma'lumot yo'q", status `not_found`, `found` false and an empty `sources` list

### Requirement: Input validation
The ask and search endpoints SHALL reject a missing, empty or whitespace-only question and a question longer than 1000 characters with HTTP 422, and SHALL reject `top_k` outside 1–20 with HTTP 422.

#### Scenario: Empty question
- **WHEN** a client posts `{"question": "   "}`
- **THEN** the response is HTTP 422 and no retrieval or generation is performed

### Requirement: Search endpoint
The system SHALL expose `POST /api/v1/search` accepting `{"query": string, "top_k"?: integer}` and returning the ranked chunks with chunk id, type, breadcrumb, text, deep link and their semantic, lexical and fused scores, without invoking the language model.

#### Scenario: Retrieval inspection
- **WHEN** a client posts `{"query": "jamoatchilik eshituvi", "top_k": 5}`
- **THEN** the response lists 5 chunks with their scores and at least one comes from appendix 4

### Requirement: Chunk lookup
The system SHALL expose `GET /api/v1/chunks/{chunk_id}` returning the full chunk with metadata, and HTTP 404 for an unknown identifier.

#### Scenario: Unknown chunk
- **WHEN** a client requests `/api/v1/chunks/does-not-exist`
- **THEN** the response is HTTP 404

### Requirement: Health endpoint
The system SHALL expose `GET /health` reporting whether the Ollama server is reachable, whether each required model is available, and whether the index is loaded (with its chunk count and build fingerprint). It MUST return HTTP 200 when everything is ready and HTTP 503 otherwise, naming the failing component.

#### Scenario: Missing model
- **WHEN** the configured chat model has not been pulled into Ollama
- **THEN** `/health` returns HTTP 503 and names the missing model

### Requirement: Backend failure handling
If the language model or embedding backend is unreachable, times out or errors, the ask and search endpoints SHALL respond with HTTP 503 and a JSON error containing a machine-readable code and an Uzbek message. The API MUST NOT return a generated or guessed answer in that case.

#### Scenario: Ollama is down
- **WHEN** Ollama is stopped and a client calls `/api/v1/ask`
- **THEN** the response is HTTP 503 with error code `llm_unavailable` or `embedding_unavailable`

### Requirement: Asynchronous, bounded concurrency
Request handling SHALL be asynchronous end to end: calls to Ollama are non-blocking and CPU-bound or synchronous index work runs off the event loop. The number of simultaneous generations MUST be limited by a configurable bound; excess requests wait instead of failing, and lightweight endpoints stay responsive meanwhile.

#### Scenario: Concurrent load
- **WHEN** 5 ask requests are sent at the same time with a generation bound of 2
- **THEN** all 5 complete successfully and `/health` answers within one second while they are in progress

### Requirement: Traceability and documentation
Every response SHALL carry an `X-Request-ID` header (echoing the client's value when provided) that also appears in structured logs and in `meta.request_id`. Interactive OpenAPI documentation with example requests MUST be available at `/docs`.

#### Scenario: Request id round trip
- **WHEN** a client sends `X-Request-ID: demo-1` to `/api/v1/ask`
- **THEN** the response header `X-Request-ID` and `meta.request_id` both equal `demo-1`

### Requirement: Demo page
The system SHALL serve at `/` a single self-contained page where a user can type a question, see the answer and its status, and open each cited source via its lex.uz deep link. The page MUST work without internet access.

#### Scenario: Asking from the browser
- **WHEN** a user opens `http://localhost:8000/`, types "Aeroport ekspertizasi qancha turadi?" and submits
- **THEN** the page shows the answer with clickable sources
