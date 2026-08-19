---
name: resume
description: Start-of-session briefing — load project context and pick up where you left off.
---

You are resuming work on the current project. Give the user a fast, actionable briefing so they can jump back in immediately.

## Step 1: Check freshness FIRST — before reading a single doc

**The docs you are about to read have a single writer, and that writer's latest work may only exist on `origin`.** Briefing off stale local docs is worse than not briefing: it reports last week's state with today's confidence.

```bash
git fetch --quiet origin 2>/dev/null
git status -sb                                    # ahead/behind on the tracked branch
git log --oneline HEAD..@{upstream} 2>/dev/null    # what you don't have yet
```

- **Behind `origin`?** Say so **at the top of the briefing, prominently** — how many commits, and whether any of them touch the doc set. Then **recommend** `git pull`:

  > ⚠️ **You are 4 commits behind `origin/main`, and 2 of them touch `docs/`. Run `git pull` before starting — this briefing is from your local copy.**

- **Do NOT pull automatically.** A command that mutates the working tree on invocation is a surprise, and an auto-pull during a dirty tree or a half-finished rebase fails badly. Report and recommend; let the user run it.
- **No upstream, no `origin`, or fetch fails?** Note it in one line and continue. A freshness check that can't run must never block the briefing.

## Step 1b: Check whether anyone is waiting on your review

A blocked PR is a **teammate stuck**, not a tidy-up task. On repos with branch protection, a pull request that needs your approval sits there doing nothing until you look — and the person who opened it can't merge past it. Surface that at the same moment the user is choosing what to work on.

```bash
gh pr status 2>/dev/null      # "Requesting a code review from you" section, this repo
```

- **Any PRs awaiting this user's review?** Say so in the briefing, with the number, title and author — and put it **above** the next-steps list. Someone else's blocked work usually outranks your own backlog.
- **Widen the net only if asked** — `gh search prs --review-requested=@me --state=open` covers every repo, which is useful at the start of a day but noisy inside a single project's briefing.
- **`gh` missing, not authenticated, no network, or the call errors?** One line, or stay silent, and continue. Same rule as the freshness check: **this must never block the briefing**, and it must never be the reason a `/resume` feels slow. It is a courtesy check, not a gate.
- **Nothing pending?** Say nothing at all. An empty review queue is not news.

## Step 1c: Check whether any MERGED PR is still unharvested

⚠️ **Step 1b cannot see these, by construction** — it reads *"Requesting a code review from you"*, which is **open** PRs. A teammate's decisions arrive in PRs that are already **merged**, and under the single-writer rule those decisions are unrecorded until someone harvests them. ✅ **Measured 2026-08-14: five merged PRs went unharvested for up to twelve days**, because the only channel surfacing them was GitHub notification mail — and email here is a deliberately once-a-day channel, so it cannot carry anything time-sensitive.

**Which PRs are unharvested is derived — there is no state file.** Harvest notes are kept indefinitely at `<docs-root>/intake/prs/YYYY-MM-DD-<repo>-pr<n>-reconciliation.md`, so their filenames *are* the record:

```bash
# the SET of harvested numbers (not a high-water mark — backlogs are sparse)
ls docs/intake/prs/*-pr*-reconciliation.md 2>/dev/null | sed -E 's/.*-pr([0-9]+)-.*/\1/' | sort -n
# merged PRs in each sibling named by .claude/ledger-siblings, above the floor
gh pr list --repo <owner/name> --state merged --json number,title,author,mergedAt
```

- **Scope it to the siblings named in `.claude/ledger-siblings`** — the repos that share a decision series. ⛔ **Not every repo**; that is the noise Step 1b rightly avoids.
- ⛔ **A floor is required.** Without one the query reaches back years — ✅ an unfloored run returned 20 PRs back to 2023. The floor is the earliest PR worth harvesting (currently `TRFA-API#20`).
- **Exclude the user's own PRs.** She lands her results in the ledgers before pushing, so they are harvested by construction.
- **Report the count with titles**, above the next-steps list, and recommend `/pr-reconcile --since`. **Show the titles** so the judgement stays with the reader — a raw number alone eventually trains her to ignore it.
- ⛔ **Warn, never auto-run.** This command only reads.
- **Nothing unharvested, or `gh` unavailable?** One line, or silence, and continue. Never a gate.

## Step 2: Load Context

Read the following files (skip any that don't exist). `<DOCS>` is `docs/` for projects under `/mnt/k/Code/`, or `.cloaked/docs/` for client sites under `/mnt/k/_Sites/`:

1. `CLAUDE.md` — project overview and conventions
2. `<DOCS>/CURRENT_STATUS.md` — where we left off
3. `<DOCS>/NEXT_STEPS.md` — the intended pick-up point, if it exists
4. `<DOCS>/TODO(S).md` — outstanding tasks
5. `<DOCS>/LESSONS_LEARNED.md` — scan for recent entries (last 5)
6. `plans/*.md` — check for any plans with `status: in-progress` or `status: approved`
7. `git log --oneline -10` — recent commits

## Step 3: Deliver Briefing

Present a dense, actionable briefing. No fluff. Lead with the freshness warning from step 1 if there was one, then the review queue from step 1b if it wasn't empty. Use this format:

---

**{PROJECT_NAME}** | Last checkpoint: {copy the `**Last updated:**` line's stamp from CURRENT_STATUS.md verbatim — the full `YYYY-MM-DD HH:MM TZ`, not just the date}

**⏳ Waiting on your review:** {From step 1b — `#<number> <title> (@<author>)`, one line each. **Omit this line entirely when the queue is empty.**}

**Where we left off:** {1-2 sentences from CURRENT_STATUS.md "In Progress" and "Session Summary"}

**Next steps:** {Numbered list from CURRENT_STATUS.md "Next Steps", prioritized}

**Open TODO(s):** {Count from docs/TODO(S).md Active section, list top 3}

**Active plans:** {Any plans with status in-progress or approved — name + one-liner summary, or "None"}

**Recent lessons:** {Any entries from last 7 days — one-liners only}

**Blockers:** {From CURRENT_STATUS.md, or "None"}

---

## Step 4: Ask Direction

End with exactly one question:

> "Ready when you are."

Do NOT summarize the entire CLAUDE.md. Do NOT explain the tech stack unless asked. The user already knows their project — they just need to remember where they were.

## Why the "Last checkpoint" stamp can look old — and why that is correct

`**Last updated:**` in `CURRENT_STATUS.md` is written by the doc set's **single writer**, during their session save. `/ship` and `/wrap-up` deliberately do not touch it. So after a stretch of commit-and-push work, the stamp will trail the newest commit — sometimes by days.

**That is the intended semantics, not a bug.** The stamp answers "when did the writer last record where things stand?", not "when was this repo last touched." Report it verbatim and don't editorialize. If the gap is large and looks material, the honest move is one line noting the stamp is older than the recent commits — never to guess at a newer state or to write the header yourself.

## This command only reads

It does not pull, commit, write, or stamp anything. Its entire job is to load context and brief. If the briefing reveals that a doc is stale or wrong, **say so** — that flag goes to the single writer (or into your PR body); it is not yours to fix here.
