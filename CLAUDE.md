# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Video transcription pipeline that processes MP4 files through:
1. Audio extraction (FFmpeg → 16kHz mono WAV)
2. Transcription (AssemblyAI with speaker diarization)
3. Analysis (Claude API extracts overview, notes, keywords, action items, decisions, quotes)
4. Output (Notion page in Fireflies-style format)
5. S3 upload with prefix-based routing + Notion link update
6. Local cleanup (safe delete via gio trash)

## Commands

```bash
# Full pipeline (transcribe + analyze + Notion + S3 upload + cleanup)
transcribe-this /path/to/video.mp4

# Estimate costs without processing
transcribe-this /path/to/video.mp4 --dry-run

# Re-run from cached transcript (no transcription cost)
transcribe-this /path/to/video.mp4 --from-cache

# Transcribe only (no Claude analysis, no Notion, no S3)
transcribe-this /path/to/video.mp4 --transcribe-only

# Keep the local folder (skip cleanup) — for multi-segment meetings (alias: --keep)
transcribe-this /path/to/video.mp4 --no-cleanup

# Right-click: Nautilus > Scripts > "Transcribe This" (opens terminal)
#   "Transcribe This --no-cleanup" — same flow but keeps the local folder

# Direct pipeline (no S3/cleanup)
source venv/bin/activate
python scripts/pipeline.py /path/to/video.mp4
python scripts/pipeline.py /path/to/video.mp4 --from-cache

# Test connections
python scripts/test_connections.py
python scripts/verify_notion_setup.py
```

## Architecture

```
scripts/
├── pipeline.py        # Main orchestrator - coordinates all steps
├── audio_extractor.py # FFmpeg wrapper, handles chunking at silence points
├── transcriber.py     # AssemblyAI integration with speaker identification
├── analyzer.py        # Claude API for transcript analysis
├── notion_output.py   # Creates structured Notion pages
└── test_connections.py
```

**Data flow:** `pipeline.py` calls modules sequentially: extract_audio → transcribe_audio → analyze_transcript → create_meeting_page

## Key Implementation Details

**Transcription (transcriber.py):**
- Uses AssemblyAI (not OpenAI Whisper) despite setup guide references
- Interactive speaker identification prompts user to identify themselves
- Cost: $0.0062/minute with speaker diarization
- `identify_user_speaker()` replaces speaker labels with user's name (default: "Cenay")

**Audio chunking (audio_extractor.py):**
- AssemblyAI handles large files natively, but chunking code exists for Whisper fallback
- Splits at silence points (700ms threshold, -40dB)
- 30-second overlap between chunks for context continuity

**Analysis (analyzer.py):**
- Prompt requests JSON with: overview (bullets), summary (narrative), notes (topical sections with emojis), keywords, action_items (grouped by owner), decisions, meeting_metadata
- Handles markdown code blocks in response
- Tracks token usage for cost reporting

**Notion output (notion_output.py):**
- Fireflies-style format: metadata header → overview → summary → notes → keywords → action items (by person) → decisions → costs → transcript
- `update_meeting_link()` updates the page with S3 URL after upload
- Splits transcript by speaker turns (blank lines) not arbitrary character counts
- Transcript lives inside a collapsible Heading 3 toggle (`is_toggleable`), hidden by default: the toggle heading is created first, then transcript paragraphs are appended as its `children` in 100-block batches
- 2000-char block limit, 100 blocks per API request

**Transcript caching (transcriber.py):**
- Raw transcript + utterances saved to `TEMP_DIR/transcribe-cache/` immediately after AssemblyAI returns
- Enables `--from-cache` re-runs without re-transcription cost
- Warns user if transcript is empty/very short

**S3 upload (transcribe-this.sh):**
- Prefix-based routing: `trfa-` → s3://cn-client-meetings/TRFA/, `trfaapi-` → s3://cn-team-videos/TRFA API/, else → s3://cn-client-meetings/ root
- Guard rail: script aborts if upload would create a new S3 prefix not in KNOWN_PREFIXES
- Verifies upload via `aws s3 ls` before cleanup
- Safe delete via `gio trash` (fallback: mv to .archived/)

## Environment Variables

Required in `.env`:
- `ASSEMBLYAI_API_KEY` - Transcription
- `ANTHROPIC_API_KEY` - Claude analysis
- `NOTION_API_KEY` - Output integration
- `NOTION_DATABASE_ID` - Target database (32-char with dashes)
- `TEMP_DIR` - Optional, defaults to system temp

## Dependencies

System: `ffmpeg`, `redis-server` (optional for queue)
Python: See `requirements.txt` (assemblyai, anthropic, pydub, notion-client, etc.)

## Notion Database Schema

Required properties: Name (title), Date (date), Duration (text), Status (select), Cost (number), Source File (text)

## Notion Setup

**Database:** Transcriptions (ID: `2e39a9bc-da8f-80d4-a29ec248064b1bad`)
- Original location: https://www.notion.so/cenay/Transcriptions-1-2e39a9bcda8f80d4a29ec248064b1bad
- Linked view on Client Meetings page (created manually, see NOTION_SETUP_GUIDE.md)

**Important:** The Notion API does not support creating linked database views programmatically. To display the Transcriptions database on the Client Meetings page:
1. See detailed instructions in `NOTION_SETUP_GUIDE.md`
2. Use `/linked` command in Notion UI to create the view
3. Position between "Random Links I Need To Save" and "Fireflies Summary" sections

**Verify setup:** Run `python scripts/verify_notion_setup.py` to confirm configuration

# context-mode — MANDATORY routing rules

You have context-mode MCP tools available. These rules are NOT optional — they protect your context window from flooding. A single unrouted command can dump 56 KB into context and waste the entire session.

## BLOCKED commands — do NOT attempt these

### curl / wget — BLOCKED
Any Bash command containing `curl` or `wget` is intercepted and replaced with an error message. Do NOT retry.
Instead use:
- `ctx_fetch_and_index(url, source)` to fetch and index web pages
- `ctx_execute(language: "javascript", code: "const r = await fetch(...)")` to run HTTP calls in sandbox

### Inline HTTP — BLOCKED
Any Bash command containing `fetch('http`, `requests.get(`, `requests.post(`, `http.get(`, or `http.request(` is intercepted and replaced with an error message. Do NOT retry with Bash.
Instead use:
- `ctx_execute(language, code)` to run HTTP calls in sandbox — only stdout enters context

### WebFetch — BLOCKED
WebFetch calls are denied entirely. The URL is extracted and you are told to use `ctx_fetch_and_index` instead.
Instead use:
- `ctx_fetch_and_index(url, source)` then `ctx_search(queries)` to query the indexed content

## REDIRECTED tools — use sandbox equivalents

### Bash (>20 lines output)
Bash is ONLY for: `git`, `mkdir`, `rm`, `mv`, `cd`, `ls`, `npm install`, `pip install`, and other short-output commands.
For everything else, use:
- `ctx_batch_execute(commands, queries)` — run multiple commands + search in ONE call
- `ctx_execute(language: "shell", code: "...")` — run in sandbox, only stdout enters context

### Read (for analysis)
If you are reading a file to **Edit** it → Read is correct (Edit needs content in context).
If you are reading to **analyze, explore, or summarize** → use `ctx_execute_file(path, language, code)` instead. Only your printed summary enters context. The raw file content stays in the sandbox.

### Grep (large results)
Grep results can flood context. Use `ctx_execute(language: "shell", code: "grep ...")` to run searches in sandbox. Only your printed summary enters context.

## Tool selection hierarchy

1. **GATHER**: `ctx_batch_execute(commands, queries)` — Primary tool. Runs all commands, auto-indexes output, returns search results. ONE call replaces 30+ individual calls.
2. **FOLLOW-UP**: `ctx_search(queries: ["q1", "q2", ...])` — Query indexed content. Pass ALL questions as array in ONE call.
3. **PROCESSING**: `ctx_execute(language, code)` | `ctx_execute_file(path, language, code)` — Sandbox execution. Only stdout enters context.
4. **WEB**: `ctx_fetch_and_index(url, source)` then `ctx_search(queries)` — Fetch, chunk, index, query. Raw HTML never enters context.
5. **INDEX**: `ctx_index(content, source)` — Store content in FTS5 knowledge base for later search.

## Subagent routing

When spawning subagents (Agent/Task tool), the routing block is automatically injected into their prompt. Bash-type subagents are upgraded to general-purpose so they have access to MCP tools. You do NOT need to manually instruct subagents about context-mode.

## Output constraints

- Keep responses under 500 words.
- Write artifacts (code, configs, PRDs) to FILES — never return them as inline text. Return only: file path + 1-line description.
- When indexing content, use descriptive source labels so others can `ctx_search(source: "label")` later.

## ctx commands

| Command | Action |
|---------|--------|
| `ctx stats` | Call the `ctx_stats` MCP tool and display the full output verbatim |
| `ctx doctor` | Call the `ctx_doctor` MCP tool, run the returned shell command, display as checklist |
| `ctx upgrade` | Call the `ctx_upgrade` MCP tool, run the returned shell command, display as checklist |
