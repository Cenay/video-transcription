---
name: wrap-up
description: "End-of-task shipping routine — reconcile the standard doc set (creating what's missing), then commit and push to origin. Doc sync is mandatory, not a prompt. Use when finishing a unit of work. Triggers: 'wrap up', 'wrap this up', 'commit now', 'finish and ship', 'commit and push', 'update docs and push'."
---

# Wrap Up

Closes out a unit of work: reconcile the project's standard documentation set
against what actually changed, then commit and push. It is a documentation-aware
superset of `/ship` — `/ship` just commits and pushes; `wrap-up` guarantees the
docs are true first.

**Invoking this skill IS the go-ahead to commit and push.** Do not stop to ask
permission again. Do the doc work, commit, push, then report.

## When to Use

- Finishing a feature, fix, decision, or research finding and you're ready to land it
- "Wrap this up", "commit now", "finish and ship", "commit and push"

**Never use for:**
- Mid-task saves where you're not ready to commit → `/checkpoint`
- A deliberate fast, no-docs commit → `/ship`

## The core rule

**Docs are reconciled on every invocation, not when the diff seems to warrant it.**

A diff-driven heuristic ("did this change touch something CLAUDE.md documents?")
is what lets NEXT_STEPS.md go stale: a commit can touch exactly one file and still
invalidate three others. If you closed a question, NEXT_STEPS must stop asking it.
If you made a decision, the decision log must contain it.

**A stale doc is worse than a missing one.** A missing NEXT_STEPS tells the reader
nothing; a stale one actively sends them to do work that's already done — or, on a
shared codebase, to do something that is now unsafe.

## Process

### 0. Reconcile the docs (MANDATORY, before anything else)

**Invoke the `doc-reconcile` skill and let it finish before surveying or editing anything.**

Same reason as the core rule above: this skill reconciles the standard doc *set*, but stale **cross-references between** docs — a status doc still calling a decision open that the ledger closed — survive a per-file review because no single file looks wrong. Run it first and steps 1–4 operate on reconciled docs. Step 4's stale-reference sweep then catches only what *this* session changed, which is what it was designed for.

- **Already ran it this session?** Say so in one line and continue to step 1.
- **Skill unavailable** (a project outside the shared set): don't block the ship. Do step 4's sweep by hand and note in your report that reconciliation was manual.

### 1. Survey what changed

```bash
git status
git diff --stat
git diff            # route through ctx_execute if large
```

Note the branch and whether `origin` exists. Summarize in 2–4 bullets: what changed
and why it matters.

### 2. Locate the doc root

Per the global `.cloaked/` convention:

- Project under **`/mnt/k/_Sites/`** (client sites) → docs live in **`.cloaked/docs/`**
- Project under **`/mnt/k/Code/`** → docs live in **`docs/`** at the project root
- Path matches neither → use whichever is already in play; if the project has no docs
  at all, create `docs/` at root and say so.

Never infer a project's *stack* from its path — but this doc-routing convention is
path-based and authoritative.

### 3. Reconcile the standard set

**Every one of these is reviewed on every invocation, and created if missing.** This is
the same set `/init-project` scaffolds — the two commands share one contract. Reviewing
means *reading it and confirming it is still true*, not just appending to it.

**Formatting the docs you touch:** follow the documentation style reference —
`guides/doc-style-reference.md` (toolkit) or `.claude/guides/doc-style-reference.md`
(shared repo). It keeps entries consistent, but it is **guidance, not a gate**: a
style imperfection never blocks the commit. Reconcile the content, apply the style
where it's cheap, ship.

| File | What "reconciled" means |
|------|-------------------------|
| **`<DOCS>/NEXT_STEPS.md`** | The pick-up point. Any "Pick Up Here" item this work **closed or abandoned** moves to **Recently Closed** with today's absolute date, time **and timezone** (`YYYY-MM-DD HH:MM TZ`, e.g. `2026-07-14 19:46 MST`) and the outcome. Add follow-ups the work spawned. **Never silently delete an item.** |
| **`<DOCS>/CURRENT_STATUS.md`** | Append/extend the session record: what happened, what was decided, what is now true that wasn't before. |
| **`<DOCS>/TODOS.md`** | Mark shipped items `~~struck~~ ✅ DONE <YYYY-MM-DD HH:MM TZ>`; annotate abandoned ones. A shipped feature must not stay invisible in the tracking docs. |
| **`<DOCS>/DECISIONS.md`** | If this work settled, reversed, or scoped a decision, append it dated — with the **why**, not just the what. Where a project treats this as its constitution it outranks every other doc; match its existing entry format exactly. |
| **`<DOCS>/LESSONS_LEARNED.md`** | Did this work surface a gotcha the next person would otherwise rediscover the hard way? Record it. Silence here is only correct if nothing was learned. |
| **`README.md`** (root) | Update when what the project *is*, or how you run it, changed. |

**Variant names:** if a project already uses `TODO.md` or a `todos/` folder, **use what's
there — never create a second one alongside it.**

**Also update when relevant (do not create uninvited):**

- **`<DOCS>/BUG-REPORT.md`** — if this work uncovered a real, verifiable defect (or
  fixed one), record it per `bug-report-style.md` — newest-first, with a verified
  file:line and symbol. A fix flips an existing entry to **Status: Fixed** with the
  commit SHA rather than deleting it. An *unconfirmed* suspicion that needs research
  goes in the ledger's `## Suspected / Needs Investigation` section as a `SUSP-` entry,
  not as a hedged bug. (Or run `/bug`.) Opt-in per real find — not a mandatory section.
- **`CHANGELOG.md`** — dated entry, if the project keeps one.
- **`CLAUDE.md`** — only when the project's *shape* changed: commands, architecture, env
  vars, workflows, ground rules. A stale CLAUDE.md misleads every future session.

**If a doc genuinely needs no change, that's a valid outcome — but you must have read it
to know that.** "I didn't touch it" and "I checked it and it's still true" are different
states, and only the second one is reconciliation.

### 4. Sweep for stale references — BEFORE staging

The failure this skill exists to prevent is closing something in one doc and leaving
another still asking for it. Grep for it:

```bash
# For each item you just closed or changed, confirm no doc still lists it as open.
grep -rni "<the closed thing>" docs/ *.md
```

Every hit must read as closed or superseded, not live. Fix any that don't.

### 4b. Deep-link decision references (DEC / G / M / D)

**First, un-wrap hard-wrapped prose.** Markdown prose and list items must each be
ONE continuous physical line (editors soft-wrap) — hard line breaks mid-sentence
render as broken text and make diffs noisy. Do NOT hand-wrap prose while writing,
and enforce it mechanically before linking. **If `.claude/scripts/reflow-md.py`
exists in this repo, run it** — it joins wrapped prose/list items while leaving code
blocks, tables, blockquotes, headings, frontmatter, and the managed link block
untouched:

```bash
python3 .claude/scripts/reflow-md.py <DOCS>   # e.g. docs  or  .cloaked/docs
# add --dry-run first to preview
```

Run reflow **before** the linker below, so the link block is regenerated against the
un-wrapped prose.

Long status docs cross-reference decisions by ID — `DEC-111`, `G75`, `M-100`,
`D-015`. A reader who isn't steeped in the project can't tell what those mean or
where to read them. After the docs are reconciled, turn every bare ID into a deep
link to its entry in the ledger.

**If `.claude/scripts/link-doc-refs.py` exists in this repo, run it — it does the
whole job deterministically and idempotently:**

```bash
python3 .claude/scripts/link-doc-refs.py <DOCS>   # e.g. docs  or  .cloaked/docs
# add --dry-run first to preview
```

How it works:
- Links are **reference-style** — inline prose stays clean (`[DEC-111]`), and every
  URL collects in an auto-generated `<!-- link-doc-refs -->` block at the file
  bottom. The block regenerates each run, so slugs self-heal if a heading is
  reworded; the inline text never changes.
- Targets are the ledger's real **heading slugs** (`DECISIONS.md#dec-111-…`), which
  is what VS Code's markdown preview *and* GitHub navigate to. (Hidden `<a id>`
  anchors do **not** work in VS Code preview — don't use them.)
- Definitions come only from the ledger / frozen records (`DECISIONS.md`, `intake/`,
  `discovery/`, `archive/`); those files are targets and are never rewritten.
  Narrative docs are where links get added.
- A **resolved** decision links to its own `### DEC-NNN` heading. An **open** one
  (a `- [ ] DEC-NNN` checklist row with no heading) links to the section it sits
  under. An ID with no ledger home at all is left as plain text and reported.

**If the script is absent** (a project outside the shared set), apply the same
convention by hand *only where it's cheap*: reference-link the IDs in the docs you
just edited to the target heading's slug. Don't hand-link a whole backlog.

### 5. Commit

Stage what's relevant. Write a **conventional commit** (`feat:`, `fix:`, `docs:`,
`chore:`…) with a body saying what changed and why.

**Every commit carries the session-traceability trailers** (global rule — a doc records
*what* was decided; the trailer is the only way to recover *why*):

```
Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
Claude-Transcript: ~/.claude/projects/<project-slug>/<session-id>.jsonl
Claude-Session: <claude.ai/code session URL>
```

- `Claude-Transcript` is **mandatory** — always available (the session's `.jsonl`
  filename). Derive the slug from the cwd: `/mnt/k/Code/TRFA/fran-dash` →
  `-mnt-k-Code-TRFA-fran-dash`.
- `Claude-Session` is **best-effort** — include when known, omit the line when not.
- **Never invent or guess either value.** A wrong pointer is worse than no pointer.

If on the default branch and the work is risky, confirm before committing directly.
Routine work on `main` is fine where that's the project's convention.

### 6. Push

```bash
git push origin HEAD
```

No `origin` remote → say so and stop. Don't invent one.

### 7. Report

1–3 lines: commit hash + subject, branch, and which docs were updated or created.

## Guardrails

- **Doc sync is not optional and not a prompt.** Review the whole standard set every
  time. If you are about to `git add` without having *read* NEXT_STEPS, stop.
- **Read before editing** — match existing tone, structure, and entry format.
- **Don't fabricate** changelog or decision entries for things not in the diff.
- **Convert relative dates to absolute, and always include the time and timezone** (`YYYY-MM-DD HH:MM TZ`, e.g. `2026-07-14 19:46 MST`). Two developers work across zones (Arizona MST, Pakistan PKT) and ship the same day — a bare date can't tell two updates apart, and a bare time can't be compared across zones. The TZ is what orders them.
  **This rule OVERRIDES "match existing entry format."** When the surrounding doc uses date-only stamps, add the time and TZ to *new and touched* entries anyway — do not retrofit historical entries. It applies to every stamp this skill writes: `**Last updated:**` header lines, `_Last updated … by an AI session_` traceability lines, session-summary headings in CURRENT_STATUS, new DECISIONS entries and dated blocks, LESSONS_LEARNED entry headings, Recently Closed items, and TODO strike-throughs. Get the time from the machine's own clock with `date "+%Y-%m-%d %H:%M %Z"` — this stamps the writer's local zone (`MST` in Arizona, `PKT` in Pakistan), never guess it. If the exact event time isn't known, stamp the wrap-up time rather than inventing one.
- **Archive, never `rm -rf`** — per global file-deletion safety.
- **Never move DB migrations into `archive/` yourself** — that move is the user's "I
  applied it" signal. Leave new migrations at top level and flag them pending. Verify
  live DB state before claiming a migration is in effect.
- **Don't rewrite historical or archive docs** (`intake/`, `discovery/`, meeting briefs,
  frozen predecessor records). Superseding them is correct; editing them to match the
  present falsifies the record. Add a dated banner pointing forward instead.

See `guides/WRAP_UP_GUIDE.md` for examples and tips.
