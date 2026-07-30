---
name: wrap-up
description: "End-of-task shipping routine — survey the diff, then commit and push to origin with session-traceability trailers and a richer report than /ship. Does not write or reconcile documentation. Use when finishing a unit of work. Triggers: 'wrap up', 'wrap this up', 'commit now', 'finish and ship', 'commit and push'."
---

# Wrap Up

Closes out a unit of work: survey what actually changed, then commit and push with
a fuller account of it than `/ship` gives.

**This skill does not touch documentation.** It reads the repo, commits, and pushes.

**Invoking this skill IS the go-ahead to commit and push.** Do not stop to ask
permission again. Survey, commit, push, then report.

## When to Use

- Finishing a feature, fix, or research finding and you're ready to land it
- "Wrap this up", "commit now", "finish and ship", "commit and push"

**Never use for:**
- Writing or reconciling project documentation — this skill deliberately does not do
  that. The standard doc set has a **single writer**; if that is not you, record what
  you decided in your **PR body**, where it gets harvested from.
- A minimal, fastest-possible commit → `/ship` (same result, terser report)

## Why there is no doc step

Earlier versions of this skill made a full documentation reconciliation mandatory
before it would commit anything. That was removed deliberately on 2026-07-29.

`/wrap-up` is a **reflex command** — it gets typed from muscle memory at the end of
a task. Under the single-writer model, `DECISIONS.md`, `CURRENT_STATUS.md`,
`NEXT_STEPS.md`, `TODOS.md` and `LESSONS_LEARNED.md` have exactly one writer, so a
reflex command that mandates writing them invites a second writer into those files
every time anyone finishes a task. The protection had to be a property of the
command, not a flag someone could route around.

**The doc discipline was relocated, not deleted.** The per-file contract — what
"reconciled" means for each doc — now lives in the **`doc-reconcile`** skill under
*"The standard doc set"*, and `/checkpoint` is the tool that writes it. Nothing was
lost; it simply belongs to the writer now.

## Process

### 1. Survey what changed

```bash
git status
git diff --stat
git diff            # route through ctx_execute if large
```

Note the branch and whether `origin` exists. Summarize in 2–4 bullets: what changed
and why it matters.

### 2. Locate the doc root — for reporting only

If the diff touched documentation, name where those docs live so the report is
specific. Per the global `.cloaked/` convention:

- Project under **`/mnt/k/_Sites/`** (client sites) → docs live in **`.cloaked/docs/`**
- Project under **`/mnt/k/Code/`** → docs live in **`docs/`** at the project root
- Path matches neither → whichever is already in play.

**Do not create, edit, or reconcile anything here** — this step identifies the folder,
nothing more. Never infer a project's *stack* from its path; this doc-routing
convention is path-based and authoritative, but it is the only thing the path settles.

### 3. Commit

Stage what's relevant — **by name**, never `git add -A` or `git add .`. Write a
**conventional commit** (`feat:`, `fix:`, `docs:`, `chore:`…) with a body saying what
changed and why, **matching the style already in this repo's history** — check the
`git log` from step 1 for scope prefixes (`fix(skills):` vs bare `fix:`), body
conventions, and subject length. This is the one place the old "match existing format"
rule still applies: the commit message is the only thing this skill authors.

**Every commit carries the session-traceability trailers** (global rule — a doc records
*what* was decided; the trailer is the only way to recover *why*):

```
Co-Authored-By: Claude <model-that-actually-ran> <noreply@anthropic.com>
Claude-Transcript: ~/.claude/projects/<project-slug>/<session-id>.jsonl
Claude-Session: <claude.ai/code session URL>
```

- `Co-Authored-By` names **the model actually running this session** (with its context
  variant if it has one) — substitute it, never copy a version from this template.
- `Claude-Transcript` is **mandatory** — always available (the session's `.jsonl`
  filename). Derive the slug from the cwd: `/mnt/k/Code/TRFA/fran-dash` →
  `-mnt-k-Code-TRFA-fran-dash`.
- `Claude-Session` is **best-effort** — include when known, omit the line when not.
- **Never invent or guess either value.** A wrong pointer is worse than no pointer.

If on the default branch and the work is risky, confirm before committing directly.
Routine work on `main` is fine where that's the project's convention.

### 4. Push

```bash
git push origin HEAD
```

No `origin` remote → say so and stop. Don't invent one.

### 5. Report

1–3 lines: commit hash + subject, branch, and a one-line account of what landed.

If the diff **did** include doc changes someone made by hand, say which files — but
report them, don't audit them. And if the work settled a decision that is nowhere in
the ledger, say so in the report rather than writing it yourself: that is the single
writer's call, and flagging it is how it reaches them.

## Guardrails

- **Do not write documentation.** No reconciliation, no status blocks, no ledger
  entries, no stamps — not even "while I'm in there". If a doc genuinely needs to
  change, that belongs to `/checkpoint` and its single writer. Commit the hand-made
  doc edits that are already in the diff; author none of your own.
- **Stage by name, never `git add -A` or `git add .`** — an unrelated file riding
  along in a commit labelled as this work is how work gets lost.
- **Refuse anything resembling a secret** — a key, token, `.env`, or credential in the
  diff stops the commit. Say what you found and let the user resolve it.
- **Don't fabricate** commit-message claims for things not in the diff. The message
  describes the diff, nothing more.
- **Dates in the commit body are absolute, with time and timezone** (`YYYY-MM-DD HH:MM TZ`,
  e.g. `2026-07-14 19:46 MST`), taken verbatim from `date "+%Y-%m-%d %H:%M %Z"` — never
  guessed. Two developers work across zones (Arizona MST, Pakistan PKT) and ship the same
  day; a bare date can't tell two updates apart, and a bare time can't be compared across
  zones. The TZ is what orders them.
- **Archive, never `rm -rf`** — per global file-deletion safety.
- **Never move DB migrations into `archive/` yourself** — that move is the user's "I
  applied it" signal. Leave new migrations at top level and flag them pending. Verify
  live DB state before claiming a migration is in effect.

See `guides/WRAP_UP_GUIDE.md` for examples and tips.
