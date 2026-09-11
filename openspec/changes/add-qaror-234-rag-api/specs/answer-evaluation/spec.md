## Purpose

Measures, with a versioned golden question set, how well the system retrieves the right passages, answers correctly and refuses when it should, so that design choices (chunking, models, thresholds) are justified by numbers.

## ADDED Requirements

### Requirement: Golden dataset
The repository SHALL contain a versioned golden dataset of at least 40 questions in Uzbek, each labelled with a kind — `in_doc` (with the expected chunk identifiers and the facts that must appear in the answer), `out_of_doc` (the answer must be the refusal), `trap` (plausible questions whose answer is absent or must not be computed, for example the content of repealed resolution No. 541 or a calendar date of entry into force) or `partial`. At least 25% of the questions MUST be `out_of_doc` or `trap`, and at least 5 MUST be written in Cyrillic or with non-canonical apostrophes.

#### Scenario: Dataset validation
- **WHEN** the evaluation tool loads the dataset
- **THEN** it verifies that every expected chunk identifier exists in the current index and fails with a list of unknown identifiers otherwise

### Requirement: Retrieval evaluation
The system SHALL provide a retrieval-only evaluation run that, without invoking the language model, reports hit@1, hit@5 and MRR over `in_doc` questions, and the relevance-signal distribution of in-document versus out-of-document questions for threshold calibration.

#### Scenario: Retrieval report
- **WHEN** the retrieval evaluation runs
- **THEN** it prints hit@1, hit@5 and MRR and a suggested refusal threshold derived from the two score distributions

### Requirement: End-to-end evaluation
The system SHALL provide an end-to-end evaluation run through the same pipeline the API uses, reporting: refusal precision and recall (refusals on `out_of_doc`/`trap` versus false refusals on `in_doc`), fact accuracy (share of required facts present in answers), citation accuracy (share of answers citing at least one expected chunk) and latency percentiles (p50, p95).

#### Scenario: End-to-end report
- **WHEN** the end-to-end evaluation runs against a running Ollama
- **THEN** it reports all four metric groups and lists every failed question with the expected and actual output

### Requirement: Configuration comparison
The evaluation SHALL run the same dataset against alternative configurations — at least chunking strategy (structural versus fixed-size) and chat model — and produce a side-by-side comparison table.

#### Scenario: Chunking comparison
- **WHEN** the retrieval evaluation runs for both chunking strategies
- **THEN** the report shows hit@5 and MRR for each strategy in one table

### Requirement: Reproducible reports with targets
Each run SHALL write a Markdown and a JSON report containing the timestamp, git revision, full configuration and metrics, and SHALL mark each metric as meeting or missing its target: out-of-document refusal recall ≥ 0.95, in-document false-refusal rate ≤ 0.10, retrieval hit@5 ≥ 0.90, fact accuracy ≥ 0.85.

#### Scenario: Target check
- **WHEN** a run finishes with out-of-document refusal recall of 0.93
- **THEN** the report marks that metric as missing its 0.95 target
