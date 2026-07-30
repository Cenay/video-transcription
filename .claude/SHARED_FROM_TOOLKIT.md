# Shared from the toolkit — do not edit here

These `.claude/skills/`, `.claude/commands/`, `.claude/guides/`, `.claude/scripts/`
and `.github/ISSUE_TEMPLATE/` entries are **managed copies** synced from Cenay's
`claude-personal-toolkit`, which owns the originals. They travel with this repo so every
developer gets the same `/wrap-up`, `/init-project`, `/resume`, `/next-step`, `/bug`,
`/file-bug` and `/build-timer-note` behaviour — including the doc timestamp standard
(`YYYY-MM-DD HH:MM TZ`), the shared doc-style / bug-report guides, and the bug-report
Issue Form.

`/checkpoint` and the `doc-reconcile` skill are deliberately NOT here (retired
2026-07-29). Under the single-writer doc model, `docs/DECISIONS.md`,
`CURRENT_STATUS.md`, `NEXT_STEPS.md`, `TODOS.md` and `LESSONS_LEARNED.md` have exactly
one writer; those two tools belong to that writer. Use `/ship` or `/wrap-up` to commit,
and record decisions in your PR body instead — they are harvested from there.

**Edit them in the toolkit, not here.** A local edit here is not merged back: the next sync
detects it, refuses to overwrite it, and asks for it to be back-ported. That protects the
work, but it also stalls the sync — so make the change in the toolkit in the first place.

Managed items: skills = wrap-up next-step; commands = init-project resume ship bug file-bug build-timer-note; guides = doc-style-reference.md bug-report-style.md; scripts = link-doc-refs.py reflow-md.py; github = bug_report.yml config.yml.
Last synced: 2026-07-30 00:37 MST
