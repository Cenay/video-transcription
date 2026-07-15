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
plans/
```

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

These are the same files `/wrap-up` reconciles on every commit. If a project already
uses a variant name (`TODO.md`, `todos/`), **use what's there — do not create a second one.**

**Timestamp convention:** every `{TODAY'S DATE + TIME}` below is `YYYY-MM-DD HH:MM TZ` — 24-hour
time plus a timezone abbreviation (e.g. `2026-07-14 19:46 MST`, `2026-07-15 07:46 PKT`). Always
stamp the **time and timezone**, not just the date — with two developers committing on the same
day from different zones (Arizona MST, Pakistan PKT), the date alone can't order or attribute
updates, and a bare time can't be compared across zones.

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

## Recently Closed

<!-- Items moved off "Pick Up Here" — with the date, time and timezone (YYYY-MM-DD HH:MM TZ) and the outcome.
     Never silently delete a closed item; a reader needs to know it was decided. -->
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

<!-- Format:
### DEC-001 <short title>
- **Status:** RESOLVED (YYYY-MM-DD HH:MM TZ)
- **Question:** What was actually being asked?
- **Answer:** What was decided, and by whom.
- **Why:** The reasoning. This is the part that matters later.
- **Build impact:** What this changes about how we build.
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
name: skill-name
description: One-line description of what this skill encodes
triggers:
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

### .gitignore (only if no .gitignore exists)

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
│   ├── NEXT_STEPS.md        # ← the session entry point
│   ├── TODOS.md
│   ├── DECISIONS.md
│   └── LESSONS_LEARNED.md
└── plans/
```

**Note which files were created vs skipped (already existed).** Safe to run on an
existing project: nothing is ever overwritten. If a doc already exists under a variant
name, say so rather than creating a duplicate alongside it.

These five docs + the root README are exactly what `/wrap-up` reconciles on every
commit — that's the contract between the two commands.

Remind the user of these companion commands:
- `/audit-claude-md` — Improve your CLAUDE.md as the project takes shape
- `/checkpoint` — Save session state before closing out
- `/resume` — Pick up where you left off next session
- `/cross-search` — Find implementations across all your projects
