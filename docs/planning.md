# Workflow Automation Planning

## Automated Workflow (Completed 2026-04-15)

**Trigger:** Right-click MP4 in Nautilus > "Transcribe This"

A terminal opens and runs all steps sequentially:

1. **Transcribe** — Extract audio (FFmpeg), transcribe (AssemblyAI), identify speakers (interactive prompt)
2. **Analyze** — Claude analyzes transcript, generates structured output (overview, notes, keywords, action items, etc.)
3. **Notion** — Creates page in Fireflies-style format with meeting link placeholder
4. **S3 Upload** — Uploads MP4 based on filename prefix routing:
   - `trfa-*` → `s3://cn-client-meetings/TRFA/`
   - `trfaapi-*` → `s3://cn-team-videos/TRFA API/`
   - Everything else → `s3://cn-client-meetings/`
5. **Notion Update** — Replaces meeting link placeholder with S3 URL
6. **S3 Verify** — Confirms file exists in S3 via `aws s3 ls`
7. **Cleanup** — Moves Zoom folder to trash (`gio trash`, recoverable)
8. **Toast notification** on completion

### Safety Features
- Transcript cached immediately after AssemblyAI returns (before speaker ID or analysis)
- `--from-cache` flag to re-run without re-transcribing (no cost)
- Empty transcript warning with abort option
- S3 verification required before local cleanup
- Safe delete via `gio trash` (recoverable from trash can)
- Known acronyms (TRFA, API, FDD, etc.) stay uppercase in titles

### CLI Usage
```bash
transcribe-this /path/to/file.mp4                 # Full pipeline
transcribe-this /path/to/file.mp4 --dry-run       # Estimate costs only
transcribe-this /path/to/file.mp4 --from-cache    # Skip transcription, use cached
transcribe-this /path/to/file.mp4 --transcribe-only # Transcribe only, no analysis/Notion/S3
transcribe-this /path/to/file.mp4 --no-cleanup    # Keep local folder (alias: --keep)
```

**`--no-cleanup` (added 2026-06-04):** Skips Step 4 entirely so the local folder is
not trashed. Use for multi-segment meetings where several files share one folder —
processing segment 1 would otherwise trash segments 2 and 3. Run the last segment
*without* the flag to clean up when done. The flag is parsed out in the shell script
and stripped before args reach `pipeline.py` (argparse would reject it).

### Right-Click Setup
- File manager: Nautilus (GNOME Files)
- Scripts (both open `gnome-terminal` for interactive speaker ID):
  - `~/.local/share/nautilus/scripts/Transcribe This` — full pipeline incl. cleanup
  - `~/.local/share/nautilus/scripts/Transcribe This --no-cleanup` — keeps local folder
- Supports: mp4, m4a, webm, mkv, wav, mp3
- Note: new scripts only appear after the scripts folder is re-scanned (reopen the Files window)

---

## Future / Open Items

- [ ] TRFA dual-Notion routing — Art's Teamspace needs its own Notion integration (separate workspace, separate API key). Currently all pages go to Cenay's Notion.
- [ ] ExpanDrive doesn't refresh Nautilus view after `aws s3 cp` uploads — cosmetic issue, files are confirmed in S3

> **Quality-of-life backlog:** smaller polish items live in [`todos/qol-improvements.md`](../todos/qol-improvements.md).
