# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Video transcription pipeline that processes MP4 files through:
1. Audio extraction (FFmpeg → 16kHz mono MP3)
2. Transcription (AssemblyAI with speaker diarization)
3. Analysis (Claude API extracts summaries, action items, decisions, quotes)
4. Output (Notion page with structured meeting notes)

## Commands

```bash
# Activate environment
source venv/bin/activate

# Process a video (full pipeline)
python scripts/pipeline.py /path/to/video.mp4

# Estimate costs without processing
python scripts/pipeline.py /path/to/video.mp4 --dry-run

# Keep temp files for debugging
python scripts/pipeline.py /path/to/video.mp4 --keep-temp

# Save metadata to JSON
python scripts/pipeline.py /path/to/video.mp4 --output-json results.json

# Test all API connections
python scripts/test_connections.py

# Test individual modules
python scripts/audio_extractor.py /path/to/video.mp4
python scripts/transcriber.py /path/to/audio.mp3
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
- Prompt requests JSON with: summary, action_items, decisions, key_quotes, topics_discussed, follow_up_items, meeting_metadata
- Handles markdown code blocks in response
- Tracks token usage for cost reporting

**Notion output (notion_output.py):**
- Splits transcript by speaker turns (blank lines) not arbitrary character counts
- 2000-char block limit, 100 blocks per API request
- Full transcript in collapsible toggle

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
