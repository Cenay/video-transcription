---
name: checkpoint
description: End-of-session save — update project status, TODOs, next steps, and capture lessons learned. Use when wrapping up a session, hitting token limits, or context-switching between projects. Also captures gotchas and lessons discovered during work.
---

You are performing an end-of-session checkpoint for the current project. This preserves context so the next session (or a different agent) can resume without loss.

**Formatting:** the docs you write below follow the documentation style reference — `guides/doc-style-reference.md` (this toolkit) or `.claude/guides/doc-style-reference.md` (a shared repo). It's guidance to keep docs consistent, not a gate — never let a formatting nitpick block a checkpoint.

## Step 1: Assess Current State

Read the following files if they exist:
- `CLAUDE.md` (for project name and context)
- `docs/CURRENT_STATUS.md` (previous status)
- `docs/NEXT_STEPS.md` (previous next steps)
- `docs/TODO(S).md` (previous TODOs)
- `docs/LESSONS_LEARNED.md` (previous lessons)
- `docs/CHANGELOG.md` (previous changelog)
- `plans/*.md` (any active plans — check for `status: in-progress`)

Also run `git log --oneline -20` and `git diff --stat` to see what changed this session. Note the branch currently in use.

## Step 2: Update docs/CURRENT_STATUS.md

Rewrite this file based on the current conversation and git history. **Get the stamp from the machine's own clock — run `date "+%Y-%m-%d %H:%M %Z"` and paste the output verbatim; never guess the time or the zone.** Use this structure:

```markdown
# Current Status
**Last updated:** YYYY-MM-DD HH:MM TZ (session N)   <!-- exact format, e.g. 2026-07-17 09:47 MST — 24-hour time AND timezone, taken verbatim from `date "+%Y-%m-%d %H:%M %Z"`. /resume echoes this line back exactly, so the format must not drift. Two devs across zones (Arizona MST, Pakistan PKT) ship in one day — the date alone can't tell them apart, and the TZ makes two times comparable. -->
```

**One `**Last updated:**` line, in this exact shape, is what `/resume` reads and echoes back — keep it the first non-heading line so it's unambiguous to find.**

```markdown
## Session Summary
{1-3 sentences summarizing what was accomplished}

## Summary
{Brief project description — carry forward from previous version}

## What's Working
{Bullet list — carry forward + add new items}

## What's In Progress
{Anything started but not finished}

## What's Not Started
{Carry forward from previous version, remove anything now started/done}
```

Create `docs/` directory if it doesn't exist.

## Step 3: Update docs/TODO(S).md and docs/NEXT_STEPS.md

**TODOS.md** — review the conversation for any TODOs that were discovered, completed, or added:
- Mark completed items: strike through (`~~…~~`) + `✅ DONE {TODAY'S DATE + TIME}` (or move to a Completed section with today's date **and time**)
- Annotate abandoned items (don't silently delete) and add any new TODOs discovered
- Keep the Active/Backlog/Completed structure

**NEXT_STEPS.md** — reconcile it to reality (this drifts fastest); keep it to **live work only**:
- Refresh "Where We Left Off" with this session's shipped work
- For any "Pick Up Here" / open item now **done, abandoned, or superseded**, **move it out** of NEXT_STEPS.md into `docs/history/NEXT_STEPS-archive.md` (create the file and `history/` folder if missing) — don't leave it here and don't silently delete it. Append it under a `## {TODAY'S DATE + TIME}` block with its outcome (`✅ Done` / `🚫 Abandoned` / `↪️ Superseded by …`). The archive is append-only and grows freely; `doc-reconcile` skips it. **Never keep a "Recently Closed" pile inside NEXT_STEPS.md** — that overflow lives in the archive.
- Add follow-ups the session's work spawned
- Bump the "Last updated" date **and time** + session number

**Close the loop per-item:** every task that mapped to an item in these files must be marked done/abandoned here in the same session it finished. Don't leave a shipped feature invisible in the tracking docs.

## Step 4: Capture Lessons Learned

Review the conversation for notable lessons. Look for:
- Solutions that required multiple attempts
- Surprising platform/API behaviors
- Approaches that were abandoned and why
- Techniques that worked better than expected
- API endpoints that don't work as documented

If you find lessons worth capturing, append them to `docs/LESSONS_LEARNED.md` using this format:

```markdown
### {Category}: {Title}
**Date:** {TODAY'S DATE + TIME}
**Context:** What we were trying to do
**Problem:** What went wrong or was discovered
**Solution:** What works
**Why:** Root cause or reasoning
```

Categories: Workflow, Code, Tool, Architecture, Integration, Data, Deployment

If nothing notable happened, skip this step — don't invent lessons.

**Bugs found this session?** A lesson explains something you learned; a *bug* is a place the code does the wrong thing. If the session uncovered a real, verifiable defect, log it in the project's bug ledger (`docs/BUG-REPORT.md`) per the bug-report style guide — `guides/bug-report-style.md` (this toolkit) or `.claude/guides/bug-report-style.md` (a shared repo) — or just run `/bug`. If instead you have an *unconfirmed suspicion* that needs research, don't force it or drop it — file it as a `SUSP-` (suspected) entry via `/bug`, which labels it honestly and records how to confirm it. Either way, never dress a hunch as a confirmed bug. This is optional and never blocks the checkpoint.

## Step 5: Update docs/CHANGELOG.md

Add a new entry at the top of the changelog (below the `# Changelog` header) summarizing what was built/changed/fixed this session. Use Keep a Changelog format:

```markdown
## [{TODAY'S DATE + TIME}] - {Short Title}

### Added
- {New features, pages, API routes, migrations}

### Changed
- {Modified behavior, refactors, UI updates}

### Fixed
- {Bug fixes}
```

Only include sections (Added/Changed/Fixed) that have entries. Be specific — include file paths, component names, and migration numbers where relevant.

## Step 6: Update Plan Status

If any plans in `plans/` or `docs/plans/` were worked on during this session, update their status:
- `in-progress` — if work started but isn't finished
- `completed` — if the feature is fully built

## Step 7: DB Migrations (apply-state hygiene)

If the project applies migrations by hand (e.g. Supabase SQL editor) and uses a `migrations/` + `migrations/archive/` split:
- **Never move a migration into `archive/` yourself** — that move is the user's "I applied it" signal. Leave any migration you wrote this session at the top level and flag it as **pending** in the summary.
- Don't claim a migration is "live" without verifying the actual DB state (e.g. `pg_class.relrowsecurity`, a `GET /rest/v1/<table>?limit=1`). A file in the repo ≠ an applied migration.

## Step 8: Summary + Ship

Show a brief summary:
- Files updated (with what changed)
- Key status for next session
- Any TODO(s) that need attention

Then ask:

> "Ready to commit and push? I can run `/ship` to commit these doc updates."

If the user says yes, invoke `/ship`.
