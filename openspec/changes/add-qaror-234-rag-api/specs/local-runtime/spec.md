## Purpose

Guarantees that the whole system runs locally and offline on ordinary developer hardware, with explicit configuration, reproducible model provisioning, a persistent index, and a one-command start.

## ADDED Requirements

### Requirement: Fully local inference
All language-model and embedding calls SHALL go to a local Ollama server at a configurable URL. After models are pulled and the index is built, answering questions MUST NOT require internet access, and no question, answer or document text may be sent to any third-party service.

#### Scenario: Offline operation
- **WHEN** the machine is disconnected from the internet after setup
- **THEN** `/api/v1/ask` still answers covered questions and refuses uncovered ones

### Requirement: Environment-based configuration
Every tunable (Ollama URL, chat and embedding model names, context window size, temperature, retrieval sizes, refusal threshold, generation concurrency, timeouts, index location, chunking strategy, automatic ingestion) SHALL be set through environment variables or a `.env` file, with defaults documented in `.env.example`. Invalid values MUST stop startup with a message naming the variable.

#### Scenario: Invalid value
- **WHEN** the refusal threshold is set to "abc"
- **THEN** the service fails to start and the error names the threshold variable

### Requirement: Explicit context window
Every generation request SHALL set the model context window explicitly from configuration, and the assembled prompt MUST fit within it, so that retrieved excerpts are never silently truncated by the model server's default context size.

#### Scenario: Prompt budget
- **WHEN** the selected excerpts would exceed the configured prompt budget
- **THEN** the lowest-ranked excerpts are dropped before sending, and the request declares the configured context size

### Requirement: Model provisioning
The required models (one chat model, one embedding model) SHALL be declared in configuration. The Docker Compose setup MUST pull them automatically before the API starts; the native setup MUST document the exact `ollama pull` commands; `/health` MUST report any model that is missing.

#### Scenario: First start with Compose
- **WHEN** `docker compose up` runs on a machine that has never pulled the models
- **THEN** the models are pulled before the API reports healthy

### Requirement: Persistent, fingerprinted index
The vector and lexical indexes SHALL persist on disk together with a fingerprint of the source snapshot, chunking configuration and embedding model. On startup the API MUST reuse a matching index without re-embedding; if the index is missing or its fingerprint does not match, the API MUST build it when automatic ingestion is enabled, or report the index as stale in `/health` otherwise. An explicit ingestion command MUST also be available.

#### Scenario: Warm restart
- **WHEN** the API is restarted without configuration changes
- **THEN** it becomes ready without calling the embedding model for any chunk

#### Scenario: Embedding model changed
- **WHEN** the embedding model name is changed and automatic ingestion is disabled
- **THEN** `/health` reports the index as stale and returns HTTP 503

### Requirement: One-command container start
`docker compose up` SHALL start Ollama and the API with CPU inference by default, and an additional Compose override file MUST enable NVIDIA GPU acceleration. The API MUST be reachable at `http://localhost:8000` once healthy.

#### Scenario: GPU machine
- **WHEN** the stack is started with the GPU override on a machine with an NVIDIA GPU and container toolkit
- **THEN** Ollama runs the models on the GPU

### Requirement: Native execution
The system SHALL run without containers using Python 3.11 or newer in a virtual environment plus a natively installed Ollama, with step-by-step instructions for Windows, Linux and macOS in the README.

#### Scenario: Native start on Windows
- **WHEN** a user follows the README native path on Windows with Ollama installed
- **THEN** the API starts on `http://localhost:8000` and `/health` returns HTTP 200

### Requirement: Hardware profiles
The default model set SHALL fit in 8 GB of GPU memory together with the embedding model, and the README MUST name a lighter chat model for CPU-only machines and a larger one for machines with more memory.

#### Scenario: Default profile on an 8 GB GPU
- **WHEN** the default models are loaded on an 8 GB GPU
- **THEN** both the chat and the embedding model are served without running out of GPU memory
