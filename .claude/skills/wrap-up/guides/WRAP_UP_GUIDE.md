# Wrap Up - Usage Guide

Here's what it does and how to use it.

## What It Does

Wrap Up closes out a finished unit of work by:
- Reviewing the working-tree diff and summarizing what changed
- **Reconciling the standard doc set every time** — NEXT_STEPS, CURRENT_STATUS, TODOS,
  DECISIONS, LESSONS_LEARNED, and the root README are all reviewed on every invocation
  and created if missing (the same set `/init-project` scaffolds). CHANGELOG and
  CLAUDE.md are updated when they exist and the change warrants it
- Sweeping for stale references — anything closed in one doc must not still read as
  open in another
- Committing with a conventional commit message + the co-author **and
  session-traceability** trailers (`Claude-Transcript` is mandatory)
- Pushing to `origin` on the current branch

**Doc sync is not a prompt.** Invoking wrap-up is the go-ahead; it does the doc work,
commits, pushes, and reports. If you want a fast commit with no doc pass, use `/ship`.

## Installation

1. The skill is symlinked to `~/.claude/skills/wrap-up`
2. Available immediately in all Claude Code sessions
3. Source of truth: `/mnt/k/Code/claude-personal-toolkit/skills/wrap-up/`

## When to Use

Use wrap-up when:
- You've finished a feature/fix/QOL change and want it landed with docs in sync
- You want a guided "did I forget to update the docs?" pass before committing
- You'd otherwise commit and immediately realize CLAUDE.md or the changelog is stale

**Never use for:**
- Mid-task progress saves — use `/checkpoint`
- A fast commit where you intentionally skip docs — use `/ship`

## How It Differs From `/ship` and `/checkpoint`

| Tool | Reviews diff | Updates docs | Commits | Pushes |
|------|:---:|:---:|:---:|:---:|
| `/checkpoint` | — | status/todos only | — | — |
| `/ship` | — | — | ✓ | ✓ |
| **wrap-up** | ✓ | ✓ (mandatory) | ✓ | ✓ |

## Example Conversations

### Wrapping up a flag addition

```
You: "wrap this up"

Claude: [Loads wrap-up skill]
Reviews the diff — you added a --no-cleanup flag to transcribe-this.sh and a
second Nautilus right-click entry.

Proposes:
  1. CLAUDE.md — document --no-cleanup in the commands list
  2. docs/planning.md — add a dated changelog entry
  3. todos/qol-improvements.md — (no shipped items, skip)

You: "1 and 2"

Claude: Updates both files, then commits:
  feat: add --no-cleanup flag and keep-folder Nautilus entry
and pushes to origin/master. Reports the commit hash and files touched.
```

### Wrapping up with nothing to document

```
You: "finish and ship this"

Claude: [Loads wrap-up skill]
Diff is a one-line typo fix in a comment. No docs need updating — confirms with you,
commits as `docs: fix typo in pipeline comment`, and pushes.
```

## What's Inside

**SKILL.md** — the routine:
- Survey the diff
- Locate the doc root (`.cloaked/docs/` on client sites, `docs/` under `/mnt/k/Code`)
- Reconcile the standard set — all five docs + root README, reviewed every time and
  created if missing (same set `/init-project` scaffolds)
- Sweep for stale references before staging
- Commit (conventional message + co-author **and session-traceability** trailers)
- Push to origin, then report

**guides/WRAP_UP_GUIDE.md** — this file.

## Tips for Best Results

1. **Run it when the work is truly done** — it's a closing routine, not a save point.
2. **Trust the doc checklist, but trim it** — say "just 1 and 3" if some proposals don't apply.
3. **Keep changelog entries factual** — they're generated from the diff, not invented.
4. **Branch first for risky work** — wrap-up will confirm before committing directly to main/master.

## The Skill Will Help You Avoid

❌ Shipping code while CLAUDE.md still describes the old behavior
❌ Forgetting to log a change in the changelog
❌ Leaving shipped TODO items marked open
✓ Docs that stay in lockstep with the code
✓ Clean, conventional commit messages with the right co-author trailer

## Next Steps

1. Finish a unit of work in any repo
2. Say "wrap this up"
3. Approve the doc checklist and let it commit + push

Happy shipping. 🚀
