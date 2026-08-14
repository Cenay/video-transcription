# Current Status — video-transcription

_Last updated 2026-08-14 00:28 MST by an AI session · transcript: `4c61a822-47ec-4195-b344-607007d9c624` — recorded that /add-term shipped and is in production use; closed the live-model measurement (two real runs, zero ANALYSIS residual)_

<details>
<summary>📜 <strong>Stamp history</strong> — the 2 previous updates (older ones: <code>history/CURRENT_STATUS-stamp-history.md</code>)</summary>

- _Prior: 2026-08-13 12:22 MST by an AI session · transcript: `2fa5b28a-7c93-4f78-8239-fc20e8d6cc8f` — wired the corrector into pipeline.py and protected the analysis stage two ways; recorded the verification measurements_
- _Prior: 2026-08-12 19:41 MST by an AI session · transcript: `ce166dfb-eca1-4c5a-b935-71755aed3e98` — term corrector built and tested (config/terms.yml, scripts/terms.py, preview tool, 56-assertion suite); added the 83-transcript corpus measurement._

</details>

**Last Updated:** 2026-08-14 00:28 MST

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
- **`/add-term` built, tested and in production use (2026-08-13)** — tooling under [DEC-008], *not* a build-order step (steps 4 and 5 are `word_boost` and `custom_spelling`, both still outstanding). All three designed pieces plus an unplanned `test-add-term.py`, in `claude-personal-toolkit`. The script writes `config/terms.yml` by absolute path, rolls the file back from a backup if the edited list fails to reload, and commits with an explicit pathspec — never `-a`. Five `chore(terms):` commits landed the same evening, each pushing that one file. Details in [`TODOS.md`](TODOS.md) → Completed.
- **Test suite now 82 assertions, 0 failures** (`./venv/bin/python tests/test_terms.py --corpus`). The count of `force:` entries is no longer pinned — see [`LESSONS_LEARNED.md`](LESSONS_LEARNED.md).

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

### What the live runs showed — the last open measurement, now closed

Two real meetings ran the full pipeline on 2026-08-13 (`trfa-tamp-new-class-bookeo` 10:08 MST, `trfaapi-deletes-new-bookeo-classes` 23:00 MST):

| check | result |
|---|---|
| transcript-stage corrections applied | **24** across the two runs (`bookio`, `karam`/`haram`, `senay`/`cna`, `milosh`, `brandash`, plus forced `active campaign` / `fran dash`) |
| `[ANALYSIS — …]` entries in `logs/term-corrections.log` | **0** |

✅ **The prompt constraint holds against a live model.** That was the unverified half of [DEC-009]. Because the transcript pass runs first, the analysis input was already clean — so any residual could only have been a term the model *invented*, and it invented none.

## In Progress

- **Nothing in the working tree**, and both open measurements are closed.
- ⚠️ **The plan is not finished.** Build-order steps 1–3 are shipped; **steps 4 (Layer 2 `word_boost`) and 5 (Layer 4 `custom_spelling`) are still outstanding.** Step 4 is the recommended next build and is the first item that **cannot be verified from cache** — it changes what AssemblyAI returns, so it costs a real transcription to test. See [`NEXT_STEPS.md`](NEXT_STEPS.md).

## Blockers

- None. Two small items remain unruled (`Nik` vs `nick`; whether corrections also appear on the Notion page) plus dead code in `analyzer.py`, none blocking.

<!-- link-doc-refs:start (auto-generated — edit the IDs in prose, not this block) -->
[DEC-002]: DECISIONS.md#dec-002-term-corrections-are-a-substitution-pass-over-pipeline-output-from-a-hand-authored-list
[DEC-004]: DECISIONS.md#dec-004-the-substitution-runs-at-the-pipelinepy180-convergence-point
[DEC-006]: DECISIONS.md#dec-006-every-correction-is-logged-to-logs
[DEC-008]: DECISIONS.md#dec-008-add-term-will-auto-commit-its-one-file--an-exception-to-the-no-auto-commit-rule
[DEC-009]: DECISIONS.md#dec-009-the-analysis-stage-is-protected-twice--a-prompt-constraint-and-a-post-pass-and-the-gap-between-them-is-the-measurement
<!-- link-doc-refs:end -->
