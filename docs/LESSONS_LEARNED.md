# Lessons Learned

_Last updated 2026-08-12 19:41 MST by an AI session · transcript: `ce166dfb-eca1-4c5a-b935-71755aed3e98` — added three lessons: four defects all found by running not reading, the engine's total failure on non-English names, and the one-meeting generalisation that produced a wrong plan finding._

Running log of non-obvious failures and how we diagnosed/fixed them. Append new entries at the top.

---

## 2026-08-12 — Four defects in one session, every one found by running, none by reading

### Workflow: a claim verified only against examples you invented proves nothing

**Date:** 2026-08-12 19:15 MST
**Context:** Building the term corrector — designing a rule to decide which transcription mishearings are safe to auto-replace.
**Problem:** Four separate defects shipped into design or code, and each survived review:

1. **The classifier rule was wrong as specified.** "Refuse a variant whose every token is ordinary English" refuses `book io` — because `io` really is in `/usr/share/dict/words`. The obvious repair (require tokens of 3+ characters) is *worse*: it then accepts `book it` and `book he`, trading one false refusal for two false acceptances in the direction that corrupts prose.
2. **The seeded term list swallowed possessives.** Listing `bookio's` as its own variant beats the bare `bookio` under longest-first matching, so *"bookio's widget"* became *"Bookeo widget"*. The corrections log would have printed `14x 'bookio's' -> Bookeo` and looked perfectly healthy.
3. **The automated miner silently missed a variant.** `haram` (37 occurrences) scored below the fuzzy-match threshold against "Khurram" and never appeared in the seed. It surfaced only when a preview printed a real line: *"And Haram Karam has had to do that before"* — the engine mangling one name two ways in a single sentence.
4. **A documented workflow the tool could not perform.** `docs/new-term-testing.md` step 3 said to vet a candidate word with `--grep`, but `--grep` filtered to lines the term list *already changed* — so vetting a word not yet in the list returned nothing. A second bug in the same code path reported 3 matching lines where there were 42.

**Solution:** Run every claim against data nobody authored for the purpose. Specifically: mine the 83 real cached transcripts rather than reasoning about them; execute the classifier against `/usr/share/dict/words` before describing its behaviour; run each command in a doc before shipping the doc; and mutation-test the test suite itself (disabling the guard must turn assertions red — it turns 13 red here).
**Why:** Every one of these defects was invisible to review and obvious on execution. The pattern is not carelessness — the rules *sounded* right, which is exactly why reading them again could not help. **A silent miss looks identical to a clean result**, whether it comes from a similarity threshold, a regex that matched nothing, or a checker that never ran.

### Tool: the speech engine fails completely on names outside English morphology

**Date:** 2026-08-12 19:15 MST
**Context:** Frequency analysis across 83 cached AssemblyAI transcripts (27,014 utterances).
**Problem:** `Khurram` appears **zero** times — it arrives as `karam` (69) or `haram` (37). `Cenay` appears **zero** times — it arrives as `senay` (13), `cinay`, `cnay`, `cna`. Meanwhile `Arthur` is correct 97 times, `Milos` 104, `Laravel` 198, `TRFA` 169. Every meeting note naming either person has been wrong, in a repo full of records attributing work to people.
**Solution:** Term corrections for the names (`config/terms.yml`), and names are the obvious payload for AssemblyAI's `word_boost` when Layer 2 is built — boosting only shifts probability and cannot corrupt text.
**Why:** The engine's language model is strong on English morphology and has nothing to anchor a name it has never seen. The failure is total rather than occasional, which is *good* news for a term map: a consistently wrong token is trivially correctable, whereas the "mush" the original plan predicted would not have been.

### Architecture: generalising from one meeting produced a wrong plan finding

**Date:** 2026-08-12 19:15 MST
**Context:** `plans/term-normalization.md` was written 2026-07-31 from the 2026-07-30 meeting.
**Problem:** It concluded *"the speech engine does not produce a consistent wrong token — it produces mush"*, and ranked the deterministic term map third on that basis. True of that one meeting (1 `bookio` in the transcript). **False across the archive:** `bookio` appears **361 times**.
**Solution:** The corpus measurement is now recorded *above* the section it revises, rather than the stale text being quietly edited. Layer 3's justification no longer rests on "testable for free" alone.
**Why:** Scope of test = scope of claim. A finding drawn from one sample described as a property of the engine will be wrong at exactly the moment someone relies on it. The counter-discipline is cheap here: 83 transcripts sat in `temp/transcribe-cache/` the whole time.

---

## 2026-06-09 — Notion page published with empty summary ("No summary available")

### Symptom
A `transcribe-this` run produced a Notion page with **only** a "Summary → No summary available." line. Overview, Notes, Keywords, Action Items, and Key Decisions were all missing. The page was still marked Complete and was uploaded to S3.

### Root cause
It was **not** a code regression (yesterday's collapsible-transcript commit was innocent) and **not** a `max_tokens` truncation. On that one run, Claude returned text that failed `json.loads()`. The old `analyzer.py` caught the error and returned `{"error", "raw_response"}`. Then `notion_output.py` silently dropped every section gated by `if items:` — only `summary` had a fallback default (`"No summary available."`), so that lone line survived. The pipeline never noticed and published the empty page anyway.

The trigger was a **non-deterministic bad JSON response** from the model. Replaying the identical cached transcript through the same code parsed perfectly the second time.

### Fixes applied
- `analyzer.py`: `analyze_transcript()` now **retries** (default 3 attempts) on `JSONDecodeError`, nudging the model to return JSON-only; added `_extract_json()` to strip fences/stray prose; bumped `max_tokens` to 8192 and warns loudly on `stop_reason == "max_tokens"`. Only returns the `error` marker after all attempts fail.
- `pipeline.py`: if analysis still contains `"error"`, **aborts before Notion/S3**, saves the raw response to `…-FAILED-ANALYSIS.txt`, and prints a loud message. Never publishes an empty page.
- `notion_output.py`: refactored body-building into `build_meeting_blocks()` + `_append_body()`, shared by `create_meeting_page()` and the new `repair_meeting_page()` (rebuilds an existing page in place, preserving its S3 Meeting Link).

### How to diagnose this again (cheap — no re-transcription)
The raw transcript is cached at `$TEMP_DIR/transcribe-cache/<stem>-raw-transcript.json`, so you can replay analysis for ~$0.09 without paying the AssemblyAI cost again.

```bash
source venv/bin/activate
# Replay analysis on a cached transcript; prints stop_reason, token counts,
# parse result, and saves the raw model reply to <stem>-RAW-ANALYSIS.txt.
python scripts/diagnose_analysis.py <stem>
# e.g. trfa-global-options-status-ninthroot-issue
```

Read `stop_reason` (truncation = `max_tokens`) and the `PARSE` line. The saved `-RAW-ANALYSIS.txt` shows exactly what the model returned.

### How to repair a published page from a recovered analysis (no new Claude call)
If `diagnose_analysis.py` produced a valid `-RAW-ANALYSIS.txt`, rebuild the page body in place:

```python
from notion_output import repair_meeting_page
# repair_meeting_page(page_id, date, duration_min, analysis, transcript, costs)
# Deletes the existing body, rebuilds all sections, re-applies the S3 link.
```

`scripts/repair_global_options.py` is a worked example (finds the page by title via `data_sources.query`, loads the recovered analysis, calls `repair_meeting_page`). Copy and adjust `STEM`/`TITLE`/`DATE`/`COSTS` for a different page.

### Gotcha: notion-client 3.0.0 query API
`notion.databases.query(...)` no longer exists in notion-client 3.0.0. Query a data source instead:
```python
ds_id = notion.databases.retrieve(db_id)["data_sources"][0]["id"]
notion.data_sources.query(data_source_id=ds_id, filter={...})
```
(`pages.create(parent={"database_id": ...})` still works.)
