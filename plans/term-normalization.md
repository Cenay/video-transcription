# Plan — domain term normalization ("when you hear XXX, write YYY")

_Last updated 2026-07-31 02:03 MST by an AI session · transcript: `d3cd0438-54be-4e0f-bbef-9e2cc664c92d`_

## Why this exists

The 2026-07-30 TRFA API meeting notes rendered the Bookeo table prefix as **`bookio_`** — ten times, in the Overview, a Notes heading, and a Key Decision. That is a table name in a production schema. It was caught only because a reconciliation session cross-checked it against the repo, where `bookeo_` appears 49 times and `bookio` appears zero. Adopting the notes verbatim would have written a wrong table name into `fran-dash/docs/DECISIONS.md`, corrupting [DEC-154], which had closed the day before.

The correction is now permanent on the reading side (a session-memory rule normalizes `bookio` → `bookeo` on sight). This plan fixes the **writing** side, so the bad token is never generated.

## What the evidence actually showed — read this before choosing an approach

The obvious fix is a "misheard → correct" spelling map. **On this data it would have fixed almost nothing.** Counts from the 2026-07-30 meeting page:

| Source | `bookio` | `bookeo` | What it really contained |
|---|---:|---:|---|
| Raw transcript (AssemblyAI, 32.7 KB) | 1 | 1 | "book you", "book your", "bookie", "book. You", "booking" |
| LLM summary / distillation (16.8 KB) | 10 | 12 | — |

Representative transcript lines:

> "Just get rid of your **bookie** underscore stuff."
> "Drop **book you** underscore prefix from your tables."
> "So now your table should be **book. You** underscore all stuff."
> "Categories, **booking**, underscore schedules."

**The speech engine does not produce a consistent wrong token — it produces mush.** The LLM summarizer then resolves that mush into a single confident, wrong identifier (`bookio_`) and repeats it. So:

- The error is **minted at the summarization step**, not the transcription step.
- A `custom_spelling` map keyed on `"bookio"` would have caught **1 occurrence in the transcript and 0 in the summary**, because `custom_spelling` is applied by AssemblyAI to *its own* output and never sees the LLM stage at all.
- Therefore the highest-value intervention is the **analyzer prompt**, not the transcription config.

This is the whole reason to build it in this order.

## The four layers, in value order

### Layer 1 — Glossary in the analyzer prompt ★ highest value

Pass the domain term list into the summarizing LLM's prompt as an explicit spelling constraint, e.g.:

> These domain terms appear in this meeting. Spell them exactly as written, including case and underscores, regardless of how they were transcribed: `Bookeo`, `bookeo_*` (table prefix), `site_*`, `TRFA`, `ActiveCampaign`, `fran-dash`, `Khurram`, `Eloquent`, `Blade`, `Laravel`, `Bookeo widget`. If the transcript contains a garbled form ("book you", "bookie", "book io"), it refers to Bookeo.

This attacks the step that actually produced the error, and it also fixes the class of error `custom_spelling` structurally cannot reach: identifiers the LLM *constructs* (`bookio_product_groups`) that were never spoken as a single token.

**Wire-in:** `scripts/analyzer.py` — the prompt that generates the summary. ⚠️ Not yet read; confirm the prompt's shape before editing.

### Layer 2 — `word_boost` at transcription

Bias recognition toward the domain vocabulary so "Bookeo" wins over "book you" in the first place. Verified present in the installed SDK:

- `venv/lib/python3.12/site-packages/assemblyai/types.py:856` — `word_boost: Optional[List[str]]`
- same file `:858` — `boost_param: Optional[WordBoost]`; `WordBoost` enum at `:237`

**Wire-in:** `scripts/transcriber.py`, the `aai.TranscriptionConfig(...)` call in `transcribe_audio()` (currently `speaker_labels` / `language_code` / `punctuate` / `format_text` only). Add `word_boost=TERMS.boost_list()` and `boost_param=aai.WordBoost.high`.

Cheap, native, no correctness risk — boosting only shifts probability, it never rewrites text.

### Layer 3 — Deterministic post-pass over transcript **and** summary

A term map applied after generation, to both artifacts. This is the guarantee layer: testable offline, reviewable in a diff, and it works on transcripts already cached from past meetings.

Design:

- **`config/terms.yml`** — the single source of truth, one entry per term:
  ```yaml
  - correct: Bookeo
    aliases: [bookio, "book io"]        # safe, unambiguous
    identifier_prefix: bookeo_           # rewrites bookio_foo -> bookeo_foo
  - correct: TRFA
    aliases: [tarifa, "t r f a"]
  - correct: ActiveCampaign
    aliases: ["active campaign", "AC"]
  ```
- **`scripts/terms.py`** — `load_terms()` and `apply_corrections(text) -> (text, [changes])`. Word-boundary regex, case-preserving where it matters. **Returns the list of substitutions made** so the pipeline can print them; a silent corrector is one nobody trusts.
- **Wire-in:** `scripts/pipeline.py`, after transcription and again after analysis.

**Keep the raw cache pristine.** `transcriber.py:59` writes the raw transcript to the cache *before* any of this runs. That ordering is already correct and must stay — corrections apply downstream, so the original is always recoverable when a correction turns out to be wrong.

### Layer 4 — `custom_spelling` ⚠️ use narrowly

AssemblyAI's literal "hear X, write Y". Verified: `custom_spelling` at `types.py:895` / `:1014`, setter `set_custom_spelling` at `types.py:1832`.

**The footgun:** it is safe for unambiguous tokens (`bookio` → `Bookeo`) and dangerous for the forms that actually dominate this transcript. Mapping `"book you"` → `Bookeo` would corrupt ordinary English — *"I'll book you for Monday"* becomes *"I'll Bookeo for Monday"*. Given Layer 1 and Layer 3 already cover the ground, this layer earns its place only for single-word confusions with no English meaning.

Populate it from the same `terms.yml` `aliases`, filtered to single tokens flagged `safe_for_asr: true`.

## Suggested build order

1. `config/terms.yml` + `scripts/terms.py` + unit tests (pure functions, no API calls — testable without spending a cent on transcription).
2. Layer 3 wiring in `pipeline.py`, run against the cached 2026-07-30 transcript as the fixture. **Success criterion: the summary comes out with `bookeo_` and zero `bookio`.**
3. Layer 1 glossary in `analyzer.py`, re-run the same fixture — the corrections list from Layer 3 should come back empty or near-empty, proving Layer 1 did the work upstream.
4. Layer 2 `word_boost` on the next real meeting; compare spoken-form noise against this baseline.
5. Layer 4 only if 1–4 leave residue.

## Open questions

- Does `analyzer.py` build one prompt or several (per-section)? The glossary must reach every call that emits prose.
- Should `terms.yml` live in this repo or be shared across TRFA projects? It is TRFA-domain vocabulary, not transcription-tool vocabulary — but this is the only consumer today. Start local; promote if a second consumer appears.
- Worth logging every correction to `logs/` so a wrong entry in `terms.yml` is discoverable after the fact rather than silently rewriting meetings.
