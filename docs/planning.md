# Workflow Automation Planning

## Automated Workflow (Completed 2026-04-15)

**Trigger:** Right-click MP4 in Nautilus > "Transcribe This"

A terminal opens and runs all steps sequentially:

1. **Transcribe** — Extract audio (FFmpeg), transcribe (AssemblyAI), identify speakers (interactive prompt)
2. **Analyze** — Claude analyzes transcript, generates structured output (overview, notes, keywords, action items, etc.)
3. **Notion** — Creates page in Fireflies-style format with meeting link placeholder
4. **S3 Upload** — Uploads MP4 based on filename prefix routing:
   - `trfa-*` → `s3://cn-client-meetings/TRFA/`
   - `trfaapi-*` → `s3://cn-client-meetings/cn-team-videos/TRFA API/`
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
transcribe-this /path/to/file.mp4              # Full pipeline
transcribe-this /path/to/file.mp4 --dry-run    # Estimate costs only
transcribe-this /path/to/file.mp4 --from-cache # Skip transcription, use cached
```

### Right-Click Setup
- File manager: Nautilus (GNOME Files)
- Script: `~/.local/share/nautilus/scripts/Transcribe This`
- Opens `gnome-terminal` for interactive speaker identification
- Supports: mp4, m4a, webm, mkv, wav, mp3

---

## Future / Open Items

- [ ] TRFA dual-Notion routing — Art's Teamspace needs its own Notion integration (separate workspace, separate API key). Currently all pages go to Cenay's Notion.
- [ ] ExpanDrive doesn't refresh Nautilus view after `aws s3 cp` uploads — cosmetic issue, files are confirmed in S3
