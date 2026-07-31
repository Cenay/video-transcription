# TODOs — video-transcription

**Last updated:** 2026-07-31 02:19 MST

Small quality-of-life polish items for the transcription pipeline. Add new ideas to
**Active** as they come up; move them to **Completed** with a date, time and timezone
when shipped.

> Larger workflow/architecture items live in [`planning.md`](planning.md).
> Migrated 2026-07-31 from `todos/qol-improvements.md` (folder retired — see
> [`DECISIONS.md`](DECISIONS.md)).

## Active

### Clean up dead code in `analyzer.py`
Pylance flags two unused items (harmless, pre-existing):
- Line 4: `import os` is unused — remove it.
- `estimate_analysis_cost()` takes a `model` parameter it never uses in the body —
  either use it (e.g. for per-model pricing) or drop the param.

## Backlog

<!-- Future tasks -->

## Completed

### Remove the "Quotes" section — _2026-06-08_
Dropped the Key Quotes section from the Notion output (`notion_output.py`) and the
`key_quotes` sample in the `__main__` test block. Also removed `key_quotes` from the
prompt/JSON schema and the "Pull quotes" guideline in `analyzer.py` so we no longer pay
tokens to generate quotes we throw away.

### Put the transcript inside a collapsible Heading 3 toggle — _2026-06-08_
Transcript now lives in a Heading 3 toggle (`is_toggleable = true`), hidden by default.
Implementation in `notion_output.py`: non-transcript blocks append to the page first,
then the toggle heading is created and its id captured, then the transcript paragraphs
are appended as `children` of that heading in 100-block batches (handles long
transcripts that exceed a single request).
