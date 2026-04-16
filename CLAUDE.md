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

# Right-click: Nautilus > Scripts > "Transcribe This" (opens terminal)

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
- Prompt requests JSON with: overview (bullets), summary (narrative), notes (topical sections with emojis), keywords, action_items (grouped by owner), decisions, key_quotes, meeting_metadata
- Handles markdown code blocks in response
- Tracks token usage for cost reporting

**Notion output (notion_output.py):**
- Fireflies-style format: metadata header → overview → summary → notes → keywords → action items (by person) → decisions → quotes → costs → transcript
- `update_meeting_link()` updates the page with S3 URL after upload
- Splits transcript by speaker turns (blank lines) not arbitrary character counts
- 2000-char block limit, 100 blocks per API request

**Transcript caching (transcriber.py):**
- Raw transcript + utterances saved to `TEMP_DIR/transcribe-cache/` immediately after AssemblyAI returns
- Enables `--from-cache` re-runs without re-transcription cost
- Warns user if transcript is empty/very short

**S3 upload (transcribe-this.sh):**
- Prefix-based routing: `trfa-` → TRFA/, `trfaapi-` → cn-team-videos/TRFA API/, else → root
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