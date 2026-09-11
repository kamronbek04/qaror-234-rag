## Purpose

Finds the few chunks of resolution No. 234 that answer a question, robustly for Uzbek morphology, spelling variants, exact numbers and explicit legal references, and produces a relevance signal the answering stage can trust.

## ADDED Requirements

### Requirement: Hybrid semantic and lexical search
The system SHALL rank chunks by combining a semantic similarity signal (multilingual dense embeddings) and a lexical signal (BM25 over normalized tokens) using rank fusion. Each returned chunk MUST expose its semantic score, lexical score, fused score and final rank. Results for the same index and query MUST be deterministic, with ties broken by document order.

#### Scenario: Paraphrased question finds the table row
- **WHEN** the query is "Aeroport qurish uchun ekologik ekspertiza necha kun davom etadi?"
- **THEN** chunk `a1-r2` (Aeroportlar) is among the top 3 results

#### Scenario: Exact term is found lexically
- **WHEN** the query is "541-son qaror"
- **THEN** chunk `q-b6`, which declares resolution No. 541 void, is among the top 3 results

### Requirement: Uzbek-aware lexical matching
Lexical matching SHALL treat common inflected forms of the same Uzbek word as matches by reducing tokens to a stem (plural, possessive and case suffixes), after normalization.

#### Scenario: Inflected forms match
- **WHEN** the query contains "ekspertizasidan" and a chunk contains only "ekspertizasi" and "ekspertizaning"
- **THEN** the chunk receives a non-zero lexical score for that term

### Requirement: Script- and spelling-independent queries
Queries SHALL be normalized exactly like indexed text, so that a query written in Uzbek Cyrillic or with any apostrophe variant retrieves the same chunks as its canonical Latin spelling.

#### Scenario: Cyrillic query
- **WHEN** the query "Аэропорт учун экспертиза муддати" is searched
- **THEN** the top 5 results equal the top 5 results of "Aeroport uchun ekspertiza muddati"

### Requirement: Explicit reference routing
When a query explicitly names a location in the document — an appendix (`N-ilova`), a chapter (`N-bob`), an item (`N-band`) or a table row, alone or combined — the referenced chunks SHALL be placed at the top of the results ahead of fused results.

#### Scenario: Appendix and item named
- **WHEN** the query is "2-ilovaning 6-bandida nima deyilgan?"
- **THEN** the first result is chunk `a2-b6`

#### Scenario: Item of the main resolution
- **WHEN** the query is "Qarorning 7-bandi"
- **THEN** the first result is chunk `q-b7` (entry into force)

### Requirement: Cross-reference context expansion
When a selected chunk references other chunks, the referenced chunks SHALL be added to the answering context after the selected chunks, within a configurable context budget, and MUST be marked as expansions so that they can be cited but are distinguishable from direct hits.

#### Scenario: Referenced item is pulled in
- **WHEN** chunk `a2-b5` is selected for a question about mandatory expertise objects
- **THEN** chunk `a2-b4`, which lists those objects, is included in the context as an expansion

### Requirement: Relevance signal for the refusal gate
Retrieval SHALL report, for every query, the highest semantic similarity among the candidates, whether the query produced any explicit reference match, and whether a multi-digit number from the query (other than the resolution's own number 234) occurs in a selected chunk (lexical anchor). These values MUST be computed deterministically and exposed to the answering stage and in debug output.

#### Scenario: Exact number is reported as an anchor
- **WHEN** the query is "541-son qaror nima boʻldi?"
- **THEN** the lexical anchor is reported because chunk `q-b6` contains 541, while "234-son qaror" and "2-ilovaning 6-bandi" report no anchor

#### Scenario: Unrelated question has low relevance
- **WHEN** the query is "Toshkentda ertaga ob-havo qanday boʻladi?"
- **THEN** the reported top semantic similarity is below the configured refusal threshold

### Requirement: Result diversity
The final selection SHALL contain at most two parts of the same split item, so that one long item cannot crowd other relevant chunks out of the context.

#### Scenario: Long amendment item does not flood the context
- **WHEN** five parts of item 1 of appendix 9 rank above chunk `q-b6` for a query and `top_k` is 4
- **THEN** exactly two of those parts are selected and `q-b6` is also selected
