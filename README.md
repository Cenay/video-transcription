# video-transcription

Video transcription pipeline that turns an MP4 into a structured Notion meeting page: FFmpeg audio extraction → AssemblyAI transcription with speaker diarization → Claude analysis (overview, notes, keywords, action items, decisions) → Notion page → S3 upload with prefix-based routing → safe local cleanup.

## Getting Started

Requires `ffmpeg`, Python 3, and a `.env` with `ASSEMBLYAI_API_KEY`, `ANTHROPIC_API_KEY`, `NOTION_API_KEY`, `NOTION_DATABASE_ID`.

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python scripts/test_connections.py
```

See [QUICK_START.md](QUICK_START.md) and [docs/video-transcription-setup-guide.md](docs/video-transcription-setup-guide.md).

## Development

```bash
transcribe-this /path/to/video.mp4            # full pipeline
transcribe-this /path/to/video.mp4 --dry-run  # cost estimate only
transcribe-this /path/to/video.mp4 --from-cache
transcribe-this /path/to/video.mp4 --no-cleanup
```

Architecture and implementation notes live in [CLAUDE.md](CLAUDE.md); gotchas in [docs/LESSONS_LEARNED.md](docs/LESSONS_LEARNED.md).

## License

Private — internal TRFA tooling.
