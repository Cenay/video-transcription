---
name: checkpoint
description: "End-of-session save — update project status, TODOs, next steps, and capture lessons learned. Use when wrapping up a session, hitting token limits, or context-switching between projects. Also captures gotchas and lessons discovered during work."
---

# Checkpoint

Saves session state to project files so the next session (or agent) can resume without context loss. Also serves as the system for capturing lessons learned during work and writing clear next-steps handoff notes.

**Related guides (advisory, never a gate):** docs written here follow `doc-style-reference.md`; verifiable defects get logged per `bug-report-style.md`. Both live at `guides/` in the toolkit, or `.claude/guides/` in a shared repo. They keep output consistent — they do not block a checkpoint.

## When to Use

- Wrapping up a work session
- Before a DISTILL or compact
- Context-switching between projects
- When a lesson or gotcha is discovered mid-work

## Session Closeout

**Order of operations matters.** Always follow this sequence.

### Step 0 — Reconcile the docs (MANDATORY, before anything else)

**Invoke the `doc-reconcile` skill and let it finish before touching a single doc below.**

This is a step, not a suggestion. Checkpoint *appends* to docs it does not verify: if `NEXT_STEPS.md` still calls a decision open that the ledger closed last week, Step 1 writes this session's notes on top of that lie and the next reader inherits it with a fresh timestamp vouching for it. Reconcile first and the save lands on true docs.

- **Do not skip because the docs "look fine"** — staleness is a cross-reference problem, invisible from any single file.
- **Already ran it this session?** Say so in one line and continue to Step 1.
- **Skill unavailable** (a project outside the shared set): don't block the checkpoint. Do the equivalent by hand — grep the docs for every decision/task you closed this session and confirm none still lists it as open — then note in your report that reconciliation was manual.

### Step 1 — Update docs (BEFORE any commit/push)

**Timestamp source:** every stamp these docs get — the `CURRENT_STATUS.md` and `NEXT_STEPS.md` `**Last updated:** YYYY-MM-DD HH:MM TZ` headers, LESSONS `**Date:**` lines, Recently-Closed and `✅ DONE` marks — comes verbatim from `date "+%Y-%m-%d %H:%M %Z"`. Never guess the time or zone. The `CURRENT_STATUS.md` header must be a single `**Last updated:**` line in exactly this shape as the first non-heading line, because `/resume` reads and echoes it back verbatim.

Update these files in the project's `docs/` folder:

1. **`docs/CURRENT_STATUS.md`** — what's done, where we left off
2. **`docs/NEXT_STEPS.md`** — plain-English handoff note for resuming next session (see below)
3. **`docs/TODOS.md`** — any new TODOs discovered during the session. **Canonical name is `TODOS.md`; if the repo already has `TODO.md`, write to that. Never create a second one** — two backlog files both look authoritative and items get lost between them.
4. **`docs/LESSONS_LEARNED.md`** — if any gotchas/lessons were found (create if doesn't exist)

### Step 1b — Un-wrap prose, then deep-link decision references

**First, reflow.** Markdown prose and list items must each be ONE continuous
physical line (editors soft-wrap); never hand-wrap prose mid-sentence. If
`.claude/scripts/reflow-md.py` exists in the repo, run it to strip any hard breaks
that slipped in — it leaves code blocks, tables, blockquotes, headings, and the
managed link block untouched:

```bash
python3 .claude/scripts/reflow-md.py docs   # or .cloaked/docs; --dry-run to preview
```

**Then link.** Status/next-step docs cross-reference decisions by ID (`DEC-111`,
`G75`, `M-100`, `D-015`). Turn those bare IDs into deep links so a reader can jump
straight to the source. If `.claude/scripts/link-doc-refs.py` exists in the repo, run
it — it's deterministic and idempotent:

```bash
python3 .claude/scripts/link-doc-refs.py docs   # or .cloaked/docs
```

It writes **reference-style** links — inline prose stays clean (`[DEC-111]`) with
the URLs in an auto-generated block at the file bottom — pointing at the ledger's
real **heading slugs** (what VS Code preview and GitHub navigate to; hidden `<a id>`
anchors do not work in VS Code). Resolved decisions link to their heading; open
checklist-item decisions link to their section. If the script isn't present, skip
this step.

### Step 2 — Commit and push

Stage ALL changes (code + docs) into a single commit, then push. Never commit code first and docs separately — that defeats the purpose of atomic closeout.

## Next Steps Protocol

`NEXT_STEPS.md` is the **starting point when resuming**. It's a narrative handoff — not a task list.

### What to Include
- What we were working on and where we stopped
- What to do first when picking back up
- Any decisions that need to be made before continuing
- Blockers or dependencies to be aware of
- Context that would be lost between sessions (e.g., "we tried X and it didn't work because Y")

### Format

```markdown
# Next Steps
**Last updated:** YYYY-MM-DD HH:MM TZ  <!-- 24-hour + timezone (e.g. 2026-07-14 19:46 MST). Two devs across zones (Arizona MST, Pakistan PKT) — the TZ is what makes the stamp unambiguous. -->

## Where We Left Off
[1-2 sentences on what was in progress]

## Pick Up Here
[Plain English: what to do first, second, third. Narrative, not checkboxes.]

## Decisions Needed
[Any open questions that should be answered before continuing. Omit section if none.]

## Watch Out For
[Gotchas, blockers, or context that matters. Omit section if none.]
```

### Key Distinctions
- **NEXT_STEPS.md** = "start here next session" (replaces itself each closeout)
- **TODOS.md** = full backlog of tasks across the project (accumulates)
- **CURRENT_STATUS.md** = high-level project state (snapshot)

## Lessons Learned Protocol

### When to Capture

Flag and offer to record when:
- A solution required multiple attempts (what finally worked and why)
- User corrects an assumption or approach
- A "gotcha" is discovered (platform quirk, API behavior, edge case)
- An abandoned approach proves definitively wrong
- A technique works significantly better than expected

### Capture Prompt

> "Lesson spotted: [brief description]. Add to docs/LESSONS_LEARNED.md?"

If confirmed, append to appropriate file:
- **Project-specific** → `./docs/LESSONS_LEARNED.md` (current project)
- **Cross-project** → `~/.claude/docs/LESSONS_LEARNED.md` (global)

### Bugs vs. lessons

A lesson explains something learned; a **bug** is a place the code does the wrong thing under a reachable input. If the session surfaced a real, verifiable defect (you saw it in the source), log it in the project's bug ledger (`docs/BUG-REPORT.md`, or `.cloaked/docs/BUG-REPORT.md`) following `bug-report-style.md`, or run `/bug`. If you have only an **unconfirmed suspicion** that needs research, file it as a `SUSP-` (suspected) entry via `/bug` instead — that lane exists precisely so a hunch is neither lost nor mislabeled as proven. Optional either way; never blocks the checkpoint.

### Entry Structure

```markdown
### [Category]: [Title]
**Date:** YYYY-MM-DD HH:MM TZ
**Context:** What we were trying to do
**Problem:** What went wrong or was discovered
**Solution:** What works
**Why:** Root cause or reasoning
```

### Categories
- `Workflow` - process improvements
- `Code` - language/framework patterns
- `Tool` - CLI, API, platform behaviors
- `Architecture` - design decisions
- `Integration` - third-party service gotchas
