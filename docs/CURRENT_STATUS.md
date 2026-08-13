# Current Status — video-transcription

_Last updated 2026-08-12 19:41 MST by an AI session · transcript: `ce166dfb-eca1-4c5a-b935-71755aed3e98` — term corrector built and tested (config/terms.yml, scripts/terms.py, preview tool, 56-assertion suite); added the 83-transcript corpus measurement._

**Last Updated:** 2026-08-12 19:15 MST

## What's Done

- Pipeline is in production use: audio extraction → AssemblyAI transcription → Claude analysis → Notion page → S3 upload → local cleanup
- Standard doc set scaffolded (this file, `NEXT_STEPS.md`, `DECISIONS.md`, `history/NEXT_STEPS-archive.md`, root `README.md`)
- **Term normalization, step 1 of the build order — built and tested (2026-08-12).** The pipeline had been writing wrong domain terms into Notion pages that feed meeting reconciliations and, from there, decision documents. Step 1 is the deterministic corrector:
  - `config/terms.yml` — the term list, seeded from a frequency analysis of 83 cached transcripts. Six terms; every entry carries its measured occurrence count, and terms deliberately **excluded** are documented in-file with the reason.
  - `scripts/terms.py` — `load_terms()` and `apply_corrections(text) -> (text, changes)`. Classifies each variant at load time so the author never has to.
  - `scripts/preview_corrections.py` — read-only before/after preview against cached transcripts. Costs nothing to run.
  - `tests/test_terms.py` — 56 assertions, 0 failures, including a corpus sweep asserting no ordinary-English word is altered anywhere.
  - `docs/new-term-testing.md` — the add-a-term workflow.
  - Eight decisions recorded: [DEC-002] through [DEC-008].

### What the corpus measurement showed

Mining the 83 cached transcripts corrected a finding the plan had carried since 2026-07-31, which had generalised from a single meeting:

| form | count | |
|---|---:|---|
| `bookio` | 361 | the dominant wrong token — **not** the "mush" the plan predicted |
| `booking` | 259 | ordinary English, almost always legitimate |
| `bookeo` | 177 | already correct |
| `karam` / `haram` | 69 / 37 | both are "Khurram" |
| `senay` | 13 | "Cenay" |

⚠️ **`Khurram` and `Cenay` are transcribed correctly ZERO times across 83 meetings.** The engine is reliable on English morphology and fails completely outside it, so every meeting note naming either of them has been wrong.

## In Progress

- **Step 2 of the build order: wiring `apply_corrections()` into `scripts/pipeline.py:180`.** Not started — this is the first change that affects live runs.

## Blockers

- None. One small ruling is wanted (`Nik` vs `nick`), but it does not block wiring.

<!-- link-doc-refs:start (auto-generated — edit the IDs in prose, not this block) -->
[DEC-002]: DECISIONS.md#dec-002-term-corrections-are-a-substitution-pass-over-pipeline-output-from-a-hand-authored-list
[DEC-008]: DECISIONS.md#dec-008-add-term-will-auto-commit-its-one-file--an-exception-to-the-no-auto-commit-rule
<!-- link-doc-refs:end -->
