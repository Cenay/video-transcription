# QOL Improvements Backlog

Small quality-of-life polish items for the transcription pipeline. Add new ideas
to the **Open** list as they come up; move them to **Done** with a date when shipped.

> Larger workflow/architecture items live in [`docs/planning.md`](../docs/planning.md).

---

## Open

### 1. Remove the "Quotes" section
The Key Quotes section is unneeded — drop it from the Notion output.

- **Where:** `scripts/notion_output.py` ~lines 201–227 (the `key_quotes` / "Key Quotes"
  heading + `quote` blocks). Also remove the `key_quotes` sample in the `__main__` test
  block (~lines 319–321).
- **Also consider:** stop asking Claude for quotes at all — drop `key_quotes` from the
  prompt/JSON schema in `scripts/analyzer.py` so we don't pay tokens to generate data we
  throw away. (Safe to leave the key in the response and just ignore it if simpler.)
- **Effort:** small.

### 2. Put the transcript inside a collapsible Heading 3 toggle
The full transcript should live inside a **Heading 3 toggle** so it's hidden by default
and can be expanded only when needed.

- **Where:** `scripts/notion_output.py` ~lines 235–264 (currently a plain `heading_3`
  "Transcript" followed by paragraph blocks for each speaker turn).
- **How (Notion API):** make the heading a toggle by setting
  `heading_3.is_toggleable = true`, and nest the transcript blocks as `children` of that
  heading block instead of appending them as siblings.
- **Watch out for:**
  - The 100-blocks-per-request and 2000-char-per-block limits still apply. Long
    transcripts may exceed 100 children on a single toggle — may need to append children
    in batches via a follow-up `blocks.children.append` call targeting the heading block.
  - Verify nesting depth limits aren't hit (toggle → children paragraphs is fine).
- **Effort:** medium (the batching for long transcripts is the tricky part).

---

## Done

_(nothing yet)_
