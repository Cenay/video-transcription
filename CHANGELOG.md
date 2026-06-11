# Changelog

| Date | Time | Change |
|------|------|--------|
| 2026-06-09 | | Fix silent analysis failure: a one-off malformed-JSON response from Claude was swallowed, publishing a Notion page with only "No summary available." `analyzer.py` now retries 3× with robust JSON extraction + truncation warning; `pipeline.py` aborts before Notion/S3 on failure; `notion_output.py` adds `repair_meeting_page()` (shared body builders). Added `scripts/diagnose_analysis.py`, `scripts/repair_global_options.py`, `docs/LESSONS_LEARNED.md`. |
