# Lessons Learned

Running log of non-obvious failures and how we diagnosed/fixed them. Append new entries at the top.

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

`scripts/repair_global_options.py` is a worked example (finds the page by title via
`data_sources.query`, loads the recovered analysis, calls `repair_meeting_page`). Copy and adjust
`STEM`/`TITLE`/`DATE`/`COSTS` for a different page.

### Gotcha: notion-client 3.0.0 query API
`notion.databases.query(...)` no longer exists in notion-client 3.0.0. Query a data source instead:
```python
ds_id = notion.databases.retrieve(db_id)["data_sources"][0]["id"]
notion.data_sources.query(data_source_id=ds_id, filter={...})
```
(`pages.create(parent={"database_id": ...})` still works.)
