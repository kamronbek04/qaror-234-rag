## Purpose

Turns the official lex.uz publication of resolution No. 234 into a faithful, typed, normalized document tree that every later stage (chunking, retrieval, citation) can rely on, reproducibly and without network access.

## ADDED Requirements

### Requirement: Offline-first source loading
The system SHALL build its knowledge base from the official lex.uz HTML of resolution No. 234 (`https://lex.uz/uz/docs/-8193120`). A snapshot of that page MUST be stored in the repository and used by default, so that ingestion works without network access and produces the same result on every machine. Refreshing the snapshot from the live URL MUST happen only on explicit request.

#### Scenario: Default ingestion uses the committed snapshot
- **WHEN** ingestion runs without the refresh option on a machine with no internet access
- **THEN** it completes successfully using the committed snapshot

#### Scenario: Explicit refresh replaces the snapshot
- **WHEN** ingestion runs with the refresh option and lex.uz responds with HTTP 200 and a page whose title contains "234-сон"
- **THEN** the snapshot file is replaced with the downloaded page and ingestion continues from it

#### Scenario: Failed refresh keeps the previous snapshot
- **WHEN** ingestion runs with the refresh option and the download fails or returns a page that does not pass the integrity checks
- **THEN** the existing snapshot is left untouched, the error is reported, and the process exits with a non-zero status without modifying the index

### Requirement: Structural parsing of the resolution
The system SHALL parse the page into an ordered hierarchy that preserves the legal structure: the resolution preamble and numbered items; appendices 1–9 with their titles and document type (regulation, list, scheme, form); chapters (`N-bob`); numbered items (`N.` → band) with their unnumbered sub-items; tables with header and rows; footnotes; and attachments of a regulation (its own schemes and forms). Every node MUST keep its lex.uz element id and its position in document order. Interface text of the lex.uz page (for example "Hujjatga taklif yuborish", "Audioni tinglash", "Hujjat elementidan havola olish") and lex.uz classifier annotations (for example `[OKOZ: …]`) MUST NOT appear in parsed text.

#### Scenario: All appendices are recovered
- **WHEN** the committed snapshot is parsed
- **THEN** the tree contains the main resolution and exactly 9 appendices numbered 1 to 9, and appendix 2 is the regulation "Davlat ekologik ekspertizasini oʻtkazish tartibi toʻgʻrisida" with 8 chapters

#### Scenario: Numbered items keep their sub-items
- **WHEN** item 4 of appendix 2 ("Davlat ekologik ekspertizasi obyektlari quyidagilar hisoblanadi:") is parsed
- **THEN** its node contains the lead sentence and its 6 unnumbered sub-items in original order

#### Scenario: Activity table is parsed row by row
- **WHEN** the Appendix 1 table is parsed
- **THEN** it yields 221 activity rows, each with its category (I, II or III), sector heading, row number, activity description, review deadline in working days and fee in BXM (or the textual fee when the cell is not numeric)

#### Scenario: Interface noise is removed
- **WHEN** any parsed text is inspected
- **THEN** it contains none of the lex.uz interface strings

### Requirement: Parse integrity checks
Ingestion MUST validate structural invariants of the parse result before anything is indexed: the expected appendix count, a non-zero number of numbered items per regulation, the Appendix 1 row count, and uniqueness of node identifiers. A violation MUST abort ingestion with a message naming the failed invariant, and the previously built index MUST remain in service.

#### Scenario: Layout change is detected
- **WHEN** the source HTML no longer contains recognizable appendix headers
- **THEN** ingestion aborts with an error that names the missing appendices, and the existing index is still used by the API

### Requirement: Text normalization
The system SHALL normalize text for search in exactly the same way at ingestion time and at query time: all apostrophe variants used for oʻ/gʻ and the tutuq belgisi (`ʻ ʼ ' ` ‘ ’ ´`) MUST map to one canonical form; Uzbek Cyrillic MUST be transliterated to Uzbek Latin; case MUST be folded; runs of whitespace MUST collapse to a single space. The original, unnormalized text MUST be preserved for display and quotations.

#### Scenario: Apostrophe variants are equivalent
- **WHEN** the strings "oʻtkazish", "o'tkazish", "o`tkazish" and "o‘tkazish" are normalized
- **THEN** all four produce the same normalized string

#### Scenario: Cyrillic input is transliterated
- **WHEN** the string "Экологик экспертиза" is normalized
- **THEN** the result equals the normalized form of "Ekologik ekspertiza"

#### Scenario: Quotes keep the original spelling
- **WHEN** a source passage is shown to a user
- **THEN** it uses the original characters of the resolution text, not the normalized form

### Requirement: Deterministic, atomic re-ingestion
Ingesting the same snapshot with the same configuration MUST produce identical chunk identifiers and contents. A new index MUST replace the old one only after it has been fully built, so that queries never observe a half-built index.

#### Scenario: Re-running ingestion is idempotent
- **WHEN** ingestion runs twice on the same snapshot and configuration
- **THEN** the set of chunk identifiers and their texts is identical after both runs
