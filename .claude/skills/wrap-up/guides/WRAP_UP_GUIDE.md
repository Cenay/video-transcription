# Wrap Up — Usage Guide

Here's what it does and how to use it.

## What It Does

Wrap Up closes out a finished unit of work by:

- Reviewing the working-tree diff and summarizing what changed
- Committing with a conventional commit message plus the co-author **and
  session-traceability** trailers (`Claude-Transcript` is mandatory)
- Pushing to `origin` on the current branch
- Reporting the hash, branch, and what landed

**It does not touch documentation.** Invoking wrap-up is the go-ahead: it surveys,
commits, pushes, and reports. Nothing else.

## Why there's no doc pass any more

Until 2026-07-29 this skill forced a full documentation reconciliation before it would
commit anything — "doc sync is mandatory, not a prompt".

That was removed on purpose. Wrap-up is a **reflex command**: it gets typed from muscle
memory the moment a task ends. The project's standard doc set —`DECISIONS.md`,
`CURRENT_STATUS.md`, `NEXT_STEPS.md`, `TODOS.md`, `LESSONS_LEARNED.md` — has **exactly
one writer**, so a reflex command that mandated writing those files pulled a second
writer into them every single time anyone finished a task. Making that safe had to be a
property of the command rather than a flag someone could route around, so the doc path
is gone rather than optional.

**The doc discipline moved; it wasn't dropped.** The per-file contract now lives in the
`doc-reconcile` skill under *"The standard doc set"*, and `/checkpoint` is what writes
it. Both belong to the single writer.

**If you are not that writer:** record what you decided in your **PR body**. That is
where decisions get harvested from, and it is the whole reason the PR body matters on
this workflow. Don't hand-edit the doc set to route around this — the docs are
review-gated on `main` via CODEOWNERS, so a hand edit surfaces as a blocked merge
rather than a silent overwrite.

## Installation

1. The skill is symlinked to `~/.claude/skills/wrap-up`
2. Available immediately in all Claude Code sessions
3. Source of truth: `/mnt/k/Code/claude-personal-toolkit/skills/wrap-up/`

In a TRFA repo you may instead be using the managed copy at
`.claude/skills/wrap-up/` — same content, synced from the toolkit. Never edit that
copy; edit the toolkit and re-run `scripts/sync-shared.sh`.

## When to Use

Use wrap-up when:

- You've finished a feature/fix/QOL change and want it landed
- You want the diff summarized back to you before it's committed
- You want the session-traceability trailers applied without thinking about them

**Never use for:**

- Writing or reconciling the project docs — it deliberately won't. Put it in the PR body.
- A minimal, terse commit → `/ship` does the same job with a shorter report.

## How It Differs From `/ship`

| Tool | Summarizes diff | Writes docs | Commits | Pushes | Report |
|------|:---:|:---:|:---:|:---:|---|
| `/ship` | — | — | ✓ | ✓ | terse |
| **wrap-up** | ✓ | — | ✓ | ✓ | fuller |

They are near-synonyms now, and that is intentional — it is what makes reaching for
either one from muscle memory safe. Pick wrap-up when you want the diff narrated back;
pick `/ship` when you already know what you changed.

## Example Conversations

### Wrapping up a flag addition

```
You: "wrap this up"

Claude: [Loads wrap-up skill]
Surveys the diff — you added a --no-cleanup flag to transcribe-this.sh and a
second Nautilus right-click entry.

Stages both files by name, commits:
  feat: add --no-cleanup flag and keep-folder Nautilus entry
  (+ Co-Authored-By and Claude-Transcript trailers)
and pushes to origin/main.

Reports: hash, branch, and "note: this changed the documented command surface —
CLAUDE.md still describes the old flag set. Flagging for the doc writer; I did
not edit it."
```

That last line is the shape to expect. Wrap-up **notices** doc drift and **says so**.
It does not fix it.

### Wrapping up a one-liner

```
You: "finish and ship this"

Claude: [Loads wrap-up skill]
Diff is a one-line typo fix in a comment. Commits as
`docs: fix typo in pipeline comment` and pushes. Two-line report.
```

## What's Inside

**SKILL.md** — the routine:

- Survey the diff
- Locate the doc root — **for reporting only** (`.cloaked/docs/` on client sites,
  `docs/` under `/mnt/k/Code`)
- Commit (conventional message + co-author **and session-traceability** trailers)
- Push to origin, then report

**guides/WRAP_UP_GUIDE.md** — this file.

## Tips for Best Results

1. **Run it when the work is truly done** — it's a closing routine, not a save point.
2. **Read the report's doc-drift note.** If it says CLAUDE.md or the ledger is now
   out of date, that note is the handoff — carry it into your PR body.
3. **Keep the commit message factual** — it describes the diff, not your intentions.
4. **Branch first for risky work** — wrap-up confirms before committing directly to
   `main`, but branching is the cheaper habit.

## The Skill Will Help You Avoid

❌ Committing with no traceability trailer, losing the *why* behind the change
❌ `git add -A` sweeping an unrelated file into a commit labelled as this work
❌ Pushing a credential or `.env` that wandered into the diff
❌ A second writer quietly overwriting the single-writer doc set
✓ Clean, conventional commit messages with a recoverable transcript pointer

## Next Steps

1. Finish a unit of work in any repo
2. Say "wrap this up"
3. Read the report — especially any doc-drift note — and let it commit + push

Happy shipping. 🚀
