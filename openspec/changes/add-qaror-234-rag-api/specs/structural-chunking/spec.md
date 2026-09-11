## Purpose

Splits the parsed resolution into self-contained, citable retrieval units whose boundaries follow the legal structure (items, definitions, table rows, scheme stages), so that retrieval returns exactly the passage a lawyer would cite.

## ADDED Requirements

### Requirement: One chunk per numbered item
The system SHALL create one chunk per numbered item (band) of the main resolution and of every regulation, containing the item's lead sentence together with all of its unnumbered sub-items. Every chunk text MUST start with a breadcrumb that names its position, in the form `<appendix> › <document title> › <chapter> › <item>` (levels that do not exist are omitted).

#### Scenario: Regulation item carries its breadcrumb
- **WHEN** item 6 of appendix 2 is chunked
- **THEN** the chunk text starts with "2-ilova › Davlat ekologik ekspertizasini oʻtkazish tartibi toʻgʻrisida nizom › 1-bob. Umumiy qoidalar › 6-band" and contains "Davlat ekologik ekspertizasi buyurtmachining (tashabbuskorning) mablagʻlari hisobidan oʻtkaziladi"

#### Scenario: Main resolution item
- **WHEN** item 6 of the main resolution is chunked
- **THEN** a chunk exists whose breadcrumb names the main resolution and item 6, and whose text states that resolution No. 541 of 7 September 2020 is declared void

### Requirement: Definition chunks
Each defined term in a "asosiy tushunchalar" item (lines of the form `<term> — <definition>`) SHALL additionally be emitted as its own chunk labelled with the term, so that "what is X" questions retrieve the definition directly. The definition chunk MUST reference the item it came from.

#### Scenario: A definition is retrievable on its own
- **WHEN** appendix 2 is chunked
- **THEN** a definition chunk for "ekolog-ekspert" exists whose text includes "kamida uzluksiz uch yil ish stajiga ega" and which references item 2 of appendix 2

### Requirement: Self-contained table-row chunks
Each of the 221 rows of the Appendix 1 activity table SHALL become one chunk written as a complete, standalone statement that includes the category number with its hazard level, the sector, the row number, the activity, the review deadline in working days and the fee in BXM. The same values MUST also be stored as structured metadata.

#### Scenario: Airport row
- **WHEN** row 2 ("Aeroportlar.") is chunked
- **THEN** the chunk states category I (yuqori darajada xavfli), sector "Transport, elektrotexnika va yoʻl xoʻjaligi", deadline 25 working days and fee 25 BXM

#### Scenario: Medium-hazard row
- **WHEN** row 69 (10–100 MW solar and wind power plants) is chunked
- **THEN** the chunk states category II (oʻrtacha darajada xavfli), deadline 25 working days and fee 15 BXM

### Requirement: Scheme, footnote and form chunks
Each stage of a workflow scheme (SXEMA) SHALL become one chunk naming the scheme, stage number, responsible party, action and deadline. Each footnote SHALL become one chunk attached to the table or appendix it annotates. Each application or certificate form template SHALL become one chunk.

#### Scenario: Footnote about unlisted activities
- **WHEN** the footnotes of Appendix 1 are chunked
- **THEN** a chunk exists stating that activity types not listed in the table also undergo state ecological expertise and that their category is determined by the expert council

### Requirement: Size bounds without losing context
A chunk whose text exceeds the configured maximum size SHALL be split only at sub-item boundaries, and every part MUST repeat the breadcrumb and the item's lead sentence and be marked with its part number. Items are never merged with neighbouring items, so that every chunk maps to exactly one citable item.

#### Scenario: Long amendment item is split
- **WHEN** item 1 of appendix 9 (about 5,800 characters with 24 sub-items) is chunked with the default maximum size
- **THEN** it is emitted as several parts, each starting with the same breadcrumb and lead sentence, and no sub-item is cut in the middle

### Requirement: Stable identifiers, metadata and deep links
Every chunk SHALL have a stable, human-readable identifier that encodes its structural position (for example `q-b6` for main resolution item 6, `a2-b6` for appendix 2 item 6, `a1-r2` for Appendix 1 row 2, with suffixes for parts and definitions). Every chunk MUST carry: its type (item, definition, table_row, scheme_stage, footnote, form), appendix number, chapter, item or row number, breadcrumb, original text, normalized search text, the lex.uz element id and a deep link of the form `https://lex.uz/uz/docs/-8193120#<element-id>`.

#### Scenario: Deep link points at the exact item
- **WHEN** the chunk for appendix 2 item 6 is inspected
- **THEN** its deep link is the resolution URL followed by `#` and the lex.uz element id of that item

### Requirement: Cross-reference extraction
The chunker SHALL detect references to other items of the same regulation (for example "ushbu Nizomning 4-bandida") and references to appendices (for example "1-ilovaga muvofiq") and record the referenced chunk identifiers in the chunk metadata.

#### Scenario: Reference to another item
- **WHEN** item 5 of appendix 2 ("Ushbu Nizomning 4-bandida nazarda tutilgan ekologik ekspertiza obyektlari…") is chunked
- **THEN** its metadata lists `a2-b4` as a referenced chunk

### Requirement: Baseline strategy for comparison
The system SHALL offer a fixed-size chunking strategy (character windows with overlap, no structural awareness), selectable by configuration, used only to measure the benefit of structural chunking in evaluation.

#### Scenario: Switching strategies
- **WHEN** the chunking strategy is configured as fixed-size and ingestion runs
- **THEN** the index is built from fixed-size windows and the evaluation report records the strategy name
