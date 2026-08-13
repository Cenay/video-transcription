# Current Status — video-transcription

_Last updated 2026-08-13 12:22 MST by an AI session · transcript: `2fa5b28a-7c93-4f78-8239-fc20e8d6cc8f` — wired the corrector into pipeline.py and protected the analysis stage two ways; recorded the verification measurements_

<details>
<summary>📜 <strong>Stamp history</strong> — the 1 previous update (older ones: <code>history/CURRENT_STATUS-stamp-history.md</code>)</summary>

- _Prior: 2026-08-12 19:41 MST by an AI session · transcript: `ce166dfb-eca1-4c5a-b935-71755aed3e98` — term corrector built and tested (config/terms.yml, scripts/terms.py, preview tool, 56-assertion suite); added the 83-transcript corpus measurement._

</details>

**Last Updated:** 2026-08-13 12:22 MST

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
- **Steps 2 and 3 — wired into the pipeline and the analysis stage (2026-08-13).** The corrector now affects live runs.
  - `scripts/pipeline.py` — `apply_corrections()` at the `--from-cache`/fresh convergence point ([DEC-004]); prints the report and appends to `logs/term-corrections.log` ([DEC-006]).
  - `scripts/analyzer.py` — `spelling_constraint()` renders `config/terms.yml` into `ANALYSIS_PROMPT`, so the prompt can never drift from the term list.
  - `scripts/pipeline.py` — `correct_structure()` walks the analysis JSON afterwards. **It should normally find nothing**; anything it reports is a term the model *invented* ([DEC-009]).
  - `tests/test_terms.py` — 19 new assertions, **75 total, 0 failures**.

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

### What the verification showed

Measured on a real `--from-cache` pass over a cached 126-minute meeting (free — no re-transcription, and `--transcribe-only` stops before Claude and Notion):

| check | before | after |
|---|---:|---:|
| `bookio` in the published transcript | 28 | **0** |
| `karam` / `haram` / `senay` | 8 | **0** — `Khurram` ×7, `Cenay` ×1 |
| `booking` / `book it` (ordinary English) | 3 / 3 | **3 / 3 — untouched** |
| raw cache `temp/…-raw-transcript.json` | 56 `bookio` | **56 — unmodified, by design** |

The analysis seam was verified separately by stubbing the Claude call with a poisoned analysis and running the real `process_video()`: 0 `bookio` reached Notion, `bookio_product_groups` → `bookeo_product_groups`, `_usage` and schema keys intact.

## In Progress

- **Nothing in the working tree.** The one open thread is a **measurement, not a build**: the prompt constraint is proven to render and reach the API payload, but whether Claude obeys it is unverified. Cenay is testing against a live meeting on 2026-08-13, and the residual printed by the analysis post-pass is the number that answers it.

## Blockers

- None. Two small items remain unruled (`Nik` vs `nick`; whether corrections also appear on the Notion page), neither blocking.

<!-- link-doc-refs:start (auto-generated — edit the IDs in prose, not this block) -->
[DEC-002]: DECISIONS.md#dec-002-term-corrections-are-a-substitution-pass-over-pipeline-output-from-a-hand-authored-list
[DEC-004]: DECISIONS.md#dec-004-the-substitution-runs-at-the-pipelinepy180-convergence-point
[DEC-006]: DECISIONS.md#dec-006-every-correction-is-logged-to-logs
[DEC-008]: DECISIONS.md#dec-008-add-term-will-auto-commit-its-one-file--an-exception-to-the-no-auto-commit-rule
[DEC-009]: DECISIONS.md#dec-009-the-analysis-stage-is-protected-twice--a-prompt-constraint-and-a-post-pass-and-the-gap-between-them-is-the-measurement
<!-- link-doc-refs:end -->
