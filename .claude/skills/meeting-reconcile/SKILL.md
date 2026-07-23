---
name: meeting-reconcile
description: "Pull a meeting note (Notion, Google Docs, or a pasted transcript) into a project's docs — write a structured reconciliation intake note (NEW / RESOLVED / CONTRADICTION / NO-OP, each with Said / Belongs in / Proposed entry), then apply approved updates to CURRENT_STATUS, DECISIONS, and TODOS. Flags every contradiction instead of overwriting; applies only after you approve. Use after a meeting when the notes touch a project you track. Triggers: 'reconcile this meeting', 'pull these meeting notes into the project', 'reconcile <notion/docs link>'."
---

# meeting-reconcile

Brings the outside world — a meeting — into a project's docs. You hand it a note link (usually Notion, sometimes Google Docs, sometimes a pasted transcript); it reads that note plus the project's own docs and writes a **reconciliation intake note**: a persisted document that classifies every candidate change as NEW, RESOLVED, CONTRADICTION, or NO-OP, and stages each as verbatim text under **Said / Belongs in / Proposed entry**. It **proposes, it does not silently write**, and it **flags contradictions rather than resolving them** — because a meeting note is a claim about reality, and the project docs are the record you have already vetted. Only you decide which wins.

The intake note follows a fixed house format — read `guides/reconciliation-note-format.md` before writing one, and match the two fran-dash exemplars it cites (`docs/intake/2026-07-21-art-meeting-reconciliation.md`, `docs/intake/2026-07-22-nik-intro-reconciliation.md`).

This is the mirror image of `doc-reconcile`. That skill reconciles the docs against each other (does the status doc still match the ledger?). This one reconciles the docs against an **external source**. Run this first when a meeting happened; then `doc-reconcile` catches any internal drift the new writes introduced; then `/checkpoint` saves and commits.

## When to Use

- After a meeting whose notes touch a project you track (the common case).
- When you have a Notion / Google Docs link with a transcript, summary, decisions, or action items.
- When someone pasted raw meeting notes and you want them folded into the project record.
- Before `/checkpoint`, so the session's save writes onto docs that already reflect the meeting.

## What you provide

At minimum, the **source**: a Notion URL, a Google Docs / Drive URL, or pasted text. Optionally the **project** — if you don't name one, the skill uses the current working directory's project. If the meeting spans several projects, run it once per project; the source can be the same link each time.

The input can also be an **existing intake note** — a path to a `<docs-root>/intake/*-reconciliation.md` this skill wrote in an earlier session. That is the **resume mode**: instead of fetching and re-classifying, the skill picks up a half-finished reconciliation where it left off (see Step 1). This is the normal path when Step 1–4 happened one day and you come back to apply it another — expected, given one-task-per-session discipline.

## How to Run

### Step 0 — Resolve the project and its doc set

Identify the project root and where its standard docs live. Respect the `.cloaked/` convention: a project's docs may be at `docs/`, `.cloaked/docs/`, or the repo root. The standard set this skill touches:

| Doc | Role |
|---|---|
| `CURRENT_STATUS.md` | Where things stand — updated with new status the meeting revealed |
| `DECISIONS.md` | The decision ledger — new decisions get a fresh `DEC-` ID; reversals mark the old one superseded |
| `TODOS.md` | The backlog — action items from the meeting become entries (canonical name is `TODOS.md`, never create `TODO.md` alongside it) |

If any of these don't exist yet, note that in the plan and offer to create them — don't assume their absence means "skip."

### Step 1 — Fetch and read the source (or resume from an intake note)

**First, decide which mode you're in.** If the input is a path to an existing `intake/*-reconciliation.md` (or the user says "resume" / "finish" / "apply" against one), this is **resume mode**:

1. Read that intake note. Its "Scope" line tells you how far it got — still "a proposal only, nothing written" means Steps 3–4 are done and Step 5 is pending.
2. Skip fetching and re-classifying. Re-read the current project docs (they may have moved on since the note was written) and re-check each `Proposed entry` still applies — flag any that a later change has overtaken.
3. Jump to **Step 4** if contradictions are still unruled, or straight to **Step 5** if section 3 is fully ruled and only the write-in remains.

Otherwise this is **fresh mode** — continue below.

Read the note through whatever tool fits the link. **Never invent the contents** — if a fetch fails or access is denied, stop and say so; do not reconstruct the meeting from memory.

- **Notion** — use the Notion MCP tools (`notion-fetch`, `notion-query-meeting-notes`, `API-retrieve-page-markdown`) to pull the page and its blocks as markdown.
- **Google Docs / Drive** — use the Google Drive MCP (`read_file_content` / `download_file_content`) on the file ID from the URL.
- **A public web page** — route through `ctx_fetch_and_index(url, source)` then `ctx_search`, per the context-mode rules. Do not use WebFetch or `curl`.
- **Pasted text** — use it directly.

Large transcripts are exactly the "flood the context window" case: fetch into the sandbox / knowledge base and pull out only what you need — decisions, action items, status changes, open questions — rather than dragging the whole transcript into context.

### Step 2 — Distill and classify

From the source, extract the substance — decisions, action items (with owner where stated), status changes, open questions — and read it **against the project docs**. Sort every candidate into one of four buckets, which become the note's four sections:

- **NEW** — the meeting introduced something no doc records yet.
- **RESOLVED** — the meeting settles or advances an item the docs already track (an open `DEC-` now answered, a caveat now closed, a prior reconciliation's item resolved).
- **CONTRADICTION** — the meeting collides with what a doc currently claims.
- **NO-OP** — already captured; list briefly so the record shows it was considered.

Large transcripts stay in the sandbox / knowledge base (Step 1) — pull out only these classified items, not the whole transcript.

### Step 3 — Write the reconciliation intake note. Do NOT touch project docs yet.

Write the note to `<docs-root>/intake/YYYY-MM-DD-<slug>-reconciliation.md` (create `intake/` if needed; use the meeting's own date). This note **is** the proposal — its "Scope" line states plainly that nothing has been written into the project docs yet, and that must stay true until Step 5.

Follow `guides/reconciliation-note-format.md` exactly: the header (source page + id, attendees, duration, recording link, what was read, an optional cross-link to a prior reconciliation, and the suggested `DEC-` starting number continuing the ledger), then the four numbered sections. Every item gets an ID (`N1`, `R1`, `C1`, …) and — for N and R items — the three lines:

```markdown
**Said:** <quote where wording matters, else faithful paraphrase; name the speaker>
**Belongs in:** `docs/<file>.md` (new DEC / TODO / status) → "exact section heading"
**Proposed entry:** *<the verbatim text to write>*
```

Two rules carried from `doc-reconcile`, because they bite here too:

- **Never invent a DEC number.** Suggested IDs continue the ledger's last real entry and must account for numbers already claimed as *unwritten proposals* by an earlier intake file. A number that won't exist after adoption is the same rot you're trying to prevent.
- **Watch the framing, not just the facts.** "Confirmed" vs "amended", "universal pricing" vs "prices differ per location" — the exact words drive the ruling. Preserve them.

When the note is written, tell the user where it is and end with the explicit handle: **"Section 3 has N contradictions to rule. When they're settled, say `apply` to write these into the project docs."** Nothing lands in the real docs until that word — `apply` (or an unambiguous "write them in" / "approved") is the single trigger that moves to Step 5.

### Step 4 — Walk the contradictions; get a ruling

Present section 3 and get a decision on each `C<k>`. **Never auto-resolve a contradiction.** Record each ruling inline with its marker (`✅ RULED`, `✅ NOT a contradiction`, `🅿️ DEFERRED`, `⚠️` still open). A reversed decision supersedes the old `DEC-` entry (old text preserved, marked *superseded by DEC-NNN*) — never deleted. Update the note's header stamp to record that section 3 now holds rulings, e.g. `_Contradictions ruled by <name> YYYY-MM-DD HH:MM TZ._`

### Step 5 — Apply (the completing step), stamp, hand off

This is the step that **completes the skill** — it is where findings actually land in the project docs. It runs only when the user gives the apply trigger (`apply` / "write them in" / "approved"), and only once **every** section-3 contradiction carries a ruling (a lone `⚠️` still-open item blocks the write — resolve or defer it first).

For each approved N/R item and each ruled C item, take its `Proposed entry` verbatim and write it into the doc named in its `Belongs in` line — a `DEC-` sentence into `DECISIONS.md` (renumbering suggested IDs to the ledger's real next number), a line into `TODOS.md`, a status update into `CURRENT_STATUS.md`. Stamp every touched doc verbatim from `date "+%Y-%m-%d %H:%M %Z"` — **date AND 24-hour time AND timezone**, never guessed:

```markdown
_Last updated YYYY-MM-DD HH:MM TZ by an AI session · reconciled from meeting YYYY-MM-DD <topic> · transcript: `<session-id>`_
```

The transcript ID is the session's `.jsonl` filename under `~/.claude/projects/<project-slug>/`. Never invent it; if unknown, say so rather than assert it. Flip the intake note's "Scope" line to reflect what was written.

Then hand off:

1. `doc-reconcile` — catches any internal drift the new writes introduced (e.g. a status doc that now contradicts the ledger entry you just added).
2. `/checkpoint` — writes the session summary, TODOs, lessons, and commit. This skill does **not** commit.

## What this skill does NOT do

- Invent, summarize-away, or reconstruct meeting contents it couldn't fetch → it stops and reports instead.
- Resolve conflicts on its own → you rule on every contradiction.
- Reconcile the docs against each other → `doc-reconcile`.
- Write the session summary or commit → `/checkpoint`.
- Fix broken file paths after a move → `folder-move-safety`.
