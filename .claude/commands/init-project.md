---
name: init-project
description: Initialize a coding project with standard folder structure and starter files.
---

You are initializing a new coding project in the **current working directory**.

The user may optionally provide a project name as $1 and a short description as $2. If not provided, infer the project name from the current directory name and ask for a one-line description before proceeding.

## Step 1: Confirm Details

Before creating anything, confirm with the user:
- **Project name**: $1 or inferred from directory name
- **Description**: $2 or ask
- **Current directory**: Show it so user can verify they're in the right place

## Step 2: Locate the doc root, then create the directory structure

**Doc routing** (per the global `.cloaked/` convention — path-based and authoritative):

- Project under **`/mnt/k/_Sites/`** (client sites) → docs go in **`.cloaked/docs/`**
- Project under **`/mnt/k/Code/`** → docs go in **`docs/`** at the project root
- Anywhere else → use whichever is already in play; if neither, default to `docs/` and say so

Below, **`<DOCS>`** means whichever of the two you resolved. Never infer a project's *stack* from its path — but this routing rule is path-based.

Create the following directories (skip any that already exist):

```
.claude/
.claude/skills/
<DOCS>/
<DOCS>/history/
<DOCS>/sessions/
<DOCS>/sessions/history/
<DOCS>/sessions/history/backups/
<DOCS>/intake/
<DOCS>/intake/meetings/
<DOCS>/intake/prs/
plans/
```

| directory | who writes it | tracked? |
|---|---|---|
| `<DOCS>/history/` | `/checkpoint` — closed next-steps, rolled-down stamp blocks | ✅ tracked |
| `<DOCS>/sessions/` | the `session-desk` skill — the live desk | ⛔ **excluded**, see below |
| `<DOCS>/sessions/history/` | archived desks; `backups/` holds ad-hoc copies and is **not** listed by `/resume-session-desk` | ⛔ excluded |
| `<DOCS>/intake/meetings/` | `meeting-reconcile` — meeting reconciliation notes | ⛔ **gitignored** |
| `<DOCS>/intake/prs/` | `pr-reconcile` — PR harvest notes, kept indefinitely as a corpus | ⛔ **gitignored** |

⚠️ **`<DOCS>/intake/` itself stays TRACKED.** Only the two subfolders are ignored. Getting this backwards is a live trap: in `fran-dash`, `docs/intake/` holds committed material (`intake-dashboard.md`, `intake-migration.md`, CSV exports) alongside the ignored subfolders, so an `intake/`-wide ignore rule would strand real project files.

⚠️ **Two different exclusion mechanisms, deliberately — do not "simplify" them into one.**

- **`<DOCS>/sessions/` → `.git/info/exclude`**, which is **per-clone and does not travel**. Ruled 2026-08-02: a desk in git invites *"it's committed, so I needn't land it"*, which is exactly the failure the desk's *Durability check* exists to catch. Because it cannot travel, the `session-desk` skill re-checks it on **every** desk creation.
- **`<DOCS>/intake/{meetings,prs}/` → `.gitignore`**, which **is committed and therefore travels**. That is the point: it stops a *teammate* committing a note too.

The mechanisms differ because the goals differ — one keeps a rule local, the other propagates it.

## Step 3: Create Starter Files

Create the following files **only if they don't already exist** (never overwrite). Use the templates below.

**The standard doc set — every project gets these five in `<DOCS>/`, plus a root `README.md`:**

| File | Purpose |
|------|---------|
| `<DOCS>/CURRENT_STATUS.md` | Session-by-session record of what happened |
| `<DOCS>/NEXT_STEPS.md` | **The starting point for any new session** |
| `<DOCS>/TODOS.md` | Drop-box for later items |
| `<DOCS>/DECISIONS.md` | Dated decision ledger — the project's constitution |
| `<DOCS>/LESSONS_LEARNED.md` | Gotchas and insights discovered while working |
| `README.md` (root) | What the project is, for a human |

Plus one companion file — the overflow for closed next-steps, so `NEXT_STEPS.md` stays lean:

| File | Purpose |
|------|---------|
| `<DOCS>/history/NEXT_STEPS-archive.md` | Where done/abandoned/superseded next-step items go; append-only, grows freely |

These are the **standard doc set**. They have a **single writer**, who maintains them via `/checkpoint`; `/wrap-up` and `/ship` commit code and deliberately do not write them. If a project already uses a variant name (`TODO.md`, `todos/`), **use what's there — do not create a second one.**

**Timestamp convention:** every `{TODAY'S DATE + TIME}` below is `YYYY-MM-DD HH:MM TZ` — 24-hour time plus a timezone abbreviation (e.g. `2026-07-14 19:46 MST`, `2026-07-15 07:46 PKT`). Always stamp the **time and timezone**, not just the date — with two developers committing on the same day from different zones (Arizona MST, Pakistan PKT), the date alone can't order or attribute updates, and a bare time can't be compared across zones.

### CLAUDE.md

```markdown
# CLAUDE.md

This file provides guidance to Claude Code when working in this repository.

## Project Overview

**Project**: {PROJECT_NAME}
**Purpose**: {DESCRIPTION}
**Created**: {TODAY'S DATE + TIME}

## Tech Stack

<!-- Update with actual stack -->
- **Language:**
- **Framework:**
- **Database:**
- **Hosting:**

## Key Commands

<!-- Add project-specific commands -->
```bash
# Dev server
# Tests
# Build
# Deploy
```

## Code Style & Conventions

<!-- Add project-specific conventions -->

## Architecture Notes

<!-- High-level architecture description -->

## Related

<!-- Links to external resources, Notion pages, etc. -->
```

### README.md

```markdown
# {PROJECT_NAME}

{DESCRIPTION}

## Getting Started

<!-- Setup instructions -->

## Development

<!-- Dev workflow -->

## License

<!-- License info -->
```

### CHANGELOG.md

```markdown
# Changelog

| Date | Time | Change |
|------|------|--------|
```

### <DOCS>/LESSONS_LEARNED.md

```markdown
# Lessons Learned — {PROJECT_NAME}

<!-- Capture gotchas, solutions, and insights as you work. -->
<!-- Format: -->
<!--
## Category: Title
**Date:** YYYY-MM-DD HH:MM TZ
**Context:** What you were trying to do
**Problem:** What went wrong or was discovered
**Solution:** What works
**Why:** Root cause or reasoning
-->
<!-- Categories: Workflow, Code, Tool, Architecture, Integration, Data, Deployment -->
```

### <DOCS>/TODOS.md

```markdown
# TODOs — {PROJECT_NAME}

## Active

<!-- Current tasks -->

## Backlog

<!-- Future tasks -->

## Completed

<!-- Move finished items here with date, time and timezone (YYYY-MM-DD HH:MM TZ) -->
```

### <DOCS>/CURRENT_STATUS.md

```markdown
# Current Status — {PROJECT_NAME}

**Last Updated:** {TODAY'S DATE + TIME}

## What's Done

- Project initialized

## In Progress

<!-- Current work -->

## Blockers

<!-- Anything blocking progress -->
```

> Note: "what to do next" lives in `NEXT_STEPS.md`, **not** here. Two places for the
> same answer is how they drift apart and start disagreeing.

### <DOCS>/NEXT_STEPS.md

```markdown
# Next Steps — {PROJECT_NAME}

**Last updated:** {TODAY'S DATE + TIME}

> Start every session here. This file answers "what do I pick up right now?" —
> details live in `CURRENT_STATUS.md`, decisions in `DECISIONS.md`.

## Pick Up Here

1. <!-- The single next action. Be specific enough to act on cold. -->

## Queued (unblocked, not yet scheduled)

<!-- Ready to go, just not now -->

## Blocked / Waiting On

<!-- What's stuck, and on whom or what -->
```

> **Keep this file to live work only.** When an item is done, abandoned, or superseded,
> **move it out** to `<DOCS>/history/NEXT_STEPS-archive.md` (never silently delete it).
> This file should never grow a "Recently Closed" pile; the archive is where closed items
> live. The doc set has a **single writer** who does this move as part of their session
> save — if that isn't you, note the closed item in your PR body instead of editing here.

### <DOCS>/history/NEXT_STEPS-archive.md

```markdown
# Next Steps — Archive — {PROJECT_NAME}

> Closed, abandoned, and superseded next-step items, moved out of `NEXT_STEPS.md` so it
> holds only live work. **Append-only, newest first — never rewritten.** Doc-reconciliation
> tooling skips this file (it lives under `history/`), so it can grow without limit.

<!-- Format — group by the closeout stamp (YYYY-MM-DD HH:MM TZ), newest block on top:

## YYYY-MM-DD HH:MM TZ
- ✅ **Done** — <item> — <one-line outcome>
- 🚫 **Abandoned** — <item> — <why it was dropped>
- ↪️ **Superseded** — <item> — overridden by <DEC-NNN / decision>
-->
```

### <DOCS>/DECISIONS.md

```markdown
# Decisions — {PROJECT_NAME}

**Last updated:** {TODAY'S DATE + TIME}

> The decision ledger. When any other doc disagrees with this file, **this file wins.**
> New decisions get appended here, dated. Record the *why*, not just the *what* — a
> decision without its reasoning gets re-litigated in six months.

## Open

<!-- Questions not yet settled. Move to Resolved when answered. -->

## Resolved

<!-- Format — keep this shape. It is what the index generator parses:
     an `### DEC-NNN <title>` heading, then a `- **Status:**` bullet.

### DEC-001 <short title>
- **Status:** ✅ RESOLVED (YYYY-MM-DD HH:MM TZ)
- **Question:** What was actually being asked?
- **Answer:** What was decided, and by whom.
- **Why:** The reasoning. This is the part that matters later.
- **Build impact:** What this changes about how we build.

Status vocabulary — use these exact words so entries stay machine-readable:
  🚧 OPEN · 📋 PROPOSED · ⏸ DEFERRED · ✅ RESOLVED (or CLOSED) · ⛔ SUPERSEDED by [DEC-NNN]

When an entry is superseded, ALSO add `- **Decided:** YYYY-MM-DD` to it. Its
Status line then carries the date it was overturned, and that bullet preserves
the date it was actually decided.

NO INDEX YET, DELIBERATELY. A long ledger earns an "Index" table at the top; a
new one does not — an index over three entries costs more to maintain than it
saves, and a stale one actively misleads. Add it when scanning the file starts
costing more than a table would. The ledger's single writer generates it with
the toolkit's `scripts/gen-dec-index.py` (which reads the shape above) — never
by hand, because hand-patched rows drift out of true.
-->
```

### .claude/skills/README.md

```markdown
# Project Skills — {PROJECT_NAME}

This directory holds project-specific Claude Code skills. Unlike global skills (in ~/.claude/skills/), these are scoped to this project and auto-load when Claude works in this directory.

## When to Create a Project Skill

Create a skill when Claude repeatedly needs domain-specific knowledge that isn't obvious from the code:
- **Business rules** (billing logic, validation rules, approval workflows)
- **Data schemas** (field mappings, required formats, API contracts)
- **Domain conventions** (naming standards, architectural patterns specific to this project)

## How to Create a Skill

```bash
mkdir -p .claude/skills/{skill-name}/guides
```

Create `.claude/skills/{skill-name}/SKILL.md`:

```yaml
---
name: skill-name description: One-line description of what this skill encodes triggers:
  - "trigger phrase that activates this skill"
---

Skill prompt content here. Encode the domain knowledge Claude needs.
```

Optionally add reference docs in the `guides/` subdirectory.

## Examples of Good Project Skills

- `billing-logic` — Encodes daily-to-monthly conversion rules so Claude never gets billing wrong
- `api-schema` — Documents the exact request/response format for all endpoints
- `parser-conventions` — Enforces field naming standards across all parser nodes
```

### .gitignore — two parts, with different conditions

⚠️ **Read this split before editing either half.** The baseline below is scaffolding for a *new* project. The intake rules are **not** — they must be ensured on **every** run, including in a project that already has a `.gitignore`, because the folders they protect are created by skills that get adopted long after `init-project` ran. Gating them on *"only if no .gitignore exists"* means the one repo that most needs them — an established one — never gets them.

#### Part A — always ensure these, appending if absent (idempotent)

```bash
# PR harvest + meeting reconciliation notes stay LOCAL. They cite line numbers
# and paths that resolve only on the single writer's machine, and their
# substance is required to reach DECISIONS.md / TODOS.md / LESSONS_LEARNED.md
# anyway — so the note itself is staging, never a record.
grep -q '^/docs/intake/meetings/$' .gitignore 2>/dev/null || printf '\n# Meeting reconciliation intake notes — LOCAL ONLY.\n/docs/intake/meetings/\n' >> .gitignore
grep -q '^/docs/intake/prs/$'      .gitignore 2>/dev/null || printf '\n# PR harvest intake notes — LOCAL ONLY, same reasoning.\n/docs/intake/prs/\n' >> .gitignore

# Session desks: per-clone, NOT .gitignore — see the note in Step 2.
grep -q '^docs/sessions/$' .git/info/exclude 2>/dev/null || printf '\n# Session Desks — visible and editor-watched, but scratch.\ndocs/sessions/\n' >> .git/info/exclude
```

**On a client site under `/mnt/k/_Sites/`**, `<DOCS>` is `.cloaked/docs/`, which is already ignored wholesale — so adjust the paths to match `<DOCS>` and skip any rule that would be redundant. **Never leave the paths as literal `/docs/…` when `<DOCS>` resolved to `.cloaked/docs/`**; a rule that matches nothing is worse than no rule, because it reads as protection.

#### Part B — baseline, only if no `.gitignore` exists

Create a sensible .gitignore. Ask the user what language/framework they're using, then generate appropriate ignore patterns. Always include these baseline entries:

```
# Environment
.env
.env.local
.env.*.local

# IDE
.vscode/
.idea/
*.sublime-workspace
*.sublime-project

# OS
.DS_Store
Thumbs.db

# Dependencies (add framework-specific)
node_modules/
vendor/
__pycache__/
*.pyc
```

## Step 4: Initialize Git

If the directory is NOT already a git repo, ask the user if they want to initialize one. If yes:
1. `git init`
2. `git add -A`
3. Create initial commit: "Initial project scaffold"

If it IS already a git repo, skip this step and mention that git is already initialized.

## Step 5: Summary

Show a tree-style summary of what was created:

```
{PROJECT_NAME}/
├── .claude/
│   └── skills/
│       └── README.md
├── .gitignore
├── CHANGELOG.md
├── CLAUDE.md
├── README.md
├── <DOCS>/                  # docs/ under /mnt/k/Code — .cloaked/docs/ on client sites
│   ├── CURRENT_STATUS.md
│   ├── NEXT_STEPS.md        # ← the session entry point (live work only)
│   ├── TODOS.md
│   ├── DECISIONS.md
│   ├── LESSONS_LEARNED.md
│   ├── history/
│   │   └── NEXT_STEPS-archive.md   # closed next-steps overflow
│   ├── sessions/                   # ⛔ .git/info/exclude — per-clone, never travels
│   │   ├── SESSION-DESK.md         # the live desk
│   │   └── history/                # archived desks + backups/
│   └── intake/                     # ✅ TRACKED — only the two subfolders below are not
│       ├── meetings/               # ⛔ .gitignore — meeting reconciliation notes
│       └── prs/                    # ⛔ .gitignore — PR harvest notes, kept as a corpus
└── plans/
```

**Note which files were created vs skipped (already existed).** Safe to run on an existing project: nothing is ever overwritten. If a doc already exists under a variant name, say so rather than creating a duplicate alongside it.

These five docs + the root README are the **standard doc set**. The per-file contract — what each one must contain and what "reconciled" means for it — lives in the `doc-reconcile` skill under *"The standard doc set"*. This command scaffolds that set; the project's single doc writer maintains it. `/wrap-up` and `/ship` deliberately do
**not** write these files.

Remind the user of these companion commands:
- `/resume` — Pick up where you left off next session
- `/ship` — Commit and push the current branch
- `/wrap-up` — Same, with the diff summarized back to you first
- `/bug` or `/file-bug` — File a defect into the ledger, or as a GitHub Issue

> Some toolkit commands (`/checkpoint`, `/audit-claude-md`, `/cross-search`) are not
> distributed to project repos — don't offer them unless they resolve in this session.
