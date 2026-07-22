---
name: doc-reconcile
description: "Find and fix stale cross-references in project docs — places where a status doc still calls a decision open that the ledger closed, or cites a section that moved. Run BEFORE /checkpoint so the session's save writes onto reconciled docs. Use when wrapping up, after a decision closes, after a meeting reconciliation, or when a doc's claim about a decision smells out of date."
---

# doc-reconcile

Reconciles what the **docs claim** against what the **ledger says**. It does not
summarize the session, does not write a status block, and does not commit —
`/checkpoint` does all of that. This runs first, so checkpoint appends to docs
that are already true.

## Why this exists

`/checkpoint` cross-references decisions by ID, but only ever **adds** links. It
never checks whether a sentence *around* an ID still matches that decision's
status. So a doc can say "DEC-109 is live and unresolved" indefinitely while the
ledger says CLOSED, and every checkpoint will faithfully preserve it.

That is not hypothetical. On 2026-07-21, fran-dash's `CLAUDE.md` still carried a
paragraph saying a database question had "never been put to Khurram" — two days
after the ledger recorded it asked verbatim and closed. Because `CLAUDE.md` loads
into every session, three sessions inherited a false open question, and a subagent
escalated it to the user as a live contradiction requiring a ruling. The cost of
stale docs is not tidiness; it is that **future sessions act on them**.

## When to Use

- Immediately **before** `/checkpoint` (the main slot)
- After closing, deferring, or superseding any decision
- After reconciling a meeting transcript into the docs
- After a folder move or doc restructure (pairs with `folder-move-safety`)
- Any time a doc's claim about a decision feels out of date

## How to Run

### Step 1 — Generate candidates

```bash
# Use whichever copy exists — the repo copy (synced into shared repos, so a teammate
# without the toolkit still has it) or the user-level one. Never hard-code the ~/ path:
# it does not exist on a machine that has this skill only as a repo copy.
python3 "$(ls .claude/skills/doc-reconcile/scripts/check-doc-refs.py \
             ~/.claude/skills/doc-reconcile/scripts/check-doc-refs.py 2>/dev/null | head -1)"
```

Auto-detects the ledger (`docs/DECISIONS.md`, `.cloaked/docs/DECISIONS.md`, or
`DECISIONS.md`) and scans `docs/`, `specs/`, `plans/`, plus `CLAUDE.md` and
`README.md`. Useful flags:

| Flag | Effect |
|---|---|
| `--ledger PATH` | Point at a non-standard ledger |
| `--root DIR` | Add a docs root (repeatable) |
| `--also F1 F2` | Add loose files to scan |
| `--all` | Include low-severity (historical/past-tense/§N) findings |
| `--unknown` | Also report IDs with no ledger heading (noisy) |
| `--json` | Machine-readable |

Exit 0 = clean, 1 = findings, 2 = setup problem. It never edits anything.

**What it flags:**

- **STALE_OPEN** — doc says open/unasked/blocked; ledger says CLOSED or SUPERSEDED
- **STALE_CLOSED** — doc says settled; ledger says OPEN or DEFERRED
- **SELF_CONTRADICTION** — one file calls the same ID both open and closed
- **SPLIT_BACKLOG** — a docs root holds **both** `TODO.md` and `TODOS.md`. Two backlog files both look authoritative and items get lost between them. Canonical is `TODOS.md`; merge and delete the other. **Never create the second one** — a skill told to update a TODO file in a repo that already has one must use what's there.
- **MISSING_DOC** — a doc references `docs/…md` / `specs/…md` / `plans/…md` that isn't on disk. Usually a rename that didn't propagate.
- **BARE_SECTION_CITATION** *(low)* — bare `§N`; section numbers drift silently
- **UNKNOWN_ID** *(opt-in)* — referenced ID with no ledger heading

### Step 2 — Verify every candidate before touching anything

**The script is a candidate generator, not an oracle.** It matches phrasing, so it
cannot tell rot from correct history. Read each quote in context and drop it if:

- It is **narrating the past** — "DEC-109 *had gone* unasked for four meetings" is
  accurate history in a lessons doc. The script downgrades obvious past-tense and
  historical files, but not all of them.
- The status word belongs to a **different clause** — a sentence mentioning
  `[G77]` and the word "superseded" about something else entirely.
- The doc is **deliberately** preserving a superseded statement with a
  "superseded-by" banner already attached.

Expect roughly a third of high-severity hits to be legitimate. That is fine — the
alternative is scanning 30 docs by hand.

### Step 3 — Report, then propose. Do not auto-fix.

Present survivors grouped by file, each with: the quote, what the ledger says, and
a proposed replacement. **Wait for a ruling on the wording before editing.**

This is not optional caution. Fixing stale text means restating a decision, and
restating it wrong is worse than leaving it stale — a confidently-worded wrong
entry gets trusted. Two rules that come up constantly:

- **Never invent a DEC number.** If a fix wants to cite a decision that exists only
  as a proposal, cite the source doc by **path + section heading** instead. A link
  to a nonexistent ledger entry is the same rot class you are removing.
- **Watch the framing, not just the facts.** "Confirmed, not amended" and
  "amended" mean different things to the next reader; an exception erodes, a
  confirmation holds. Ask which one is meant.

### Step 4 — Apply, stamp, hand off

After the ruling, apply the edits and stamp every touched doc verbatim from
`date "+%Y-%m-%d HH:%M %Z"`:

```markdown
_Last updated YYYY-MM-DD HH:MM TZ by an AI session · transcript: `<session-id>`_
```

The transcript ID is the session's `.jsonl` filename under
`~/.claude/projects/<project-slug>/`. **Never invent it** — a wrong pointer is
worse than none. If the scratchpad path for the session is known, its UUID segment
is the session ID; otherwise the newest `.jsonl` in that folder is a reasonable
inference, and say so rather than asserting it.

Then hand off to `/checkpoint` for the session summary, TODOs, lessons, and commit.

## What this skill does NOT do

- Write session summaries or status blocks → `/checkpoint`
- Commit or push → `/checkpoint`
- Capture lessons → `/checkpoint`
- Fix broken **file paths** after a move → `folder-move-safety`
- Add deep links to bare IDs → `link-doc-refs.py`, run by `/checkpoint`

## Extending the heuristics

The two phrase lists in `scripts/check-doc-refs.py` (`OPEN_PAT`, `CLOSED_PAT`) and
the two guards (`PAST_TENSE_RE`, `HISTORICAL_FILE_RE`) are where accuracy lives.
When a real miss shows up, add the phrase rather than working around it — a missed
stale line is the failure mode that costs sessions; a false positive costs seconds.
