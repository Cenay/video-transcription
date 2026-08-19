# Bug Report Style Guide

_Last updated 2026-07-17 12:09 MST by an AI session · transcript: `4088ac65-e4b4-4ce7-a94b-8bae61562311`_

This guide defines the format for entries in a project's bug ledger (by default `docs/BUG-REPORT.md`, or `.cloaked/docs/BUG-REPORT.md` where the project uses the `.cloaked/` convention). Every bug logged by an AI session — or by a human — uses this shape, so that any developer can find, verify, and act on a reported issue without talking to whoever filed it.

It is a companion to `doc-style-reference.md`, which governs documentation formatting generally. Where this guide is silent, that one applies.

## The core principle

> **A report that misrepresents its own certainty is worse than no report at all.**

The goal is that a reader can open the named file at the named line, see the problem, and reproduce it — without trusting the reporter. Every rule below serves that, and a fully verified entry is the ideal.

**But verification is the aim, not the entry fee.** An earlier phrasing of this rule said an entry that *cannot be independently verified* is worse than nothing. That is too strong, and in practice it suppressed exactly the reports worth keeping: the half-formed suspicion noticed in passing, by someone who didn't have time to chase it. A hunch that is *labelled* a hunch costs a reader nothing and may save them a day. What actually damages a ledger is **false confidence** — an invented line number, "this throws" when nobody ran it, a guess wearing confirmed-bug clothes.

So: file it. Mark what you verified and what you didn't. Never invent a specific.

## Two lanes: confirmed vs. suspected

The ledger holds two kinds of entry, in two sections:

- **Confirmed bugs** (`BUG-…`) — seen in the source and reproduced. These are the default and the bulk of the ledger. Everything above applies to them literally: a verified file:line, a real reproduce step.
- **Suspected / needs-investigation** (`SUSP-…`) — "I think this is broken, but confirming it needs research I can't do right now." A hunch you don't want to lose, but that has **not** been verified.

The core principle holds for both — because a *suspected* entry's job is not to prove a defect but to be **honestly labeled as unproven and say what would settle it**. A suspicion filed as if it were confirmed breaks the ledger; a suspicion filed *as a suspicion* is a first-class, trustworthy entry. What is never allowed is dressing a hunch up in confirmed-bug clothes (an invented line number, "this throws" when you never ran it).

**When in doubt, file it as Suspected rather than not at all.** The Suspected lane exists precisely so that uncertainty has somewhere to go. A "how to confirm or refute" note is strongly wanted and makes the entry far more useful — but its absence is a reason to file a thinner entry, never a reason to stay silent. See **Suspected entries** below.

## Where bugs go

> **File:** the project's bug ledger — `docs/BUG-REPORT.md` (Code projects) or `.cloaked/docs/BUG-REPORT.md` (projects using the `.cloaked/` convention). Use whichever docs root the project already uses; create the file if it's missing.
>
> **Order:** Newest at the top, immediately under the file's intro. Never append to the bottom.

**Layout — confirmed first, suspected below.** Confirmed `BUG-…` entries sit at the top of the file (newest-first), exactly as before. A single `## Suspected / Needs Investigation` section lives at the **bottom** of the ledger and holds the `SUSP-…` entries (also newest-first). Keeping suspicions in their own section is what keeps the confirmed list signal-dense — a reader scanning for real defects never has to wade through unproven ones.

Where a project keeps per-component or per-class narrative docs, summarize the bug there too (under a **Gotchas & Important Notes** or **Troubleshooting** section) and cross-link the two: the bug ledger is the record, the component doc is the narrative. If the project has no such docs, the ledger entry stands alone.

## Entry format

Each bug is an H2 heading, followed by a blockquote of key facts, followed by prose sections.

```md
## BUG-YYYY-MM-DD-NNN — Short imperative summary of the defect

> **Status:** Open · **Lane:** Confirmed · **Severity:** High
> **File:** `path/to/SomeFile.ext:118` (relative to repo root)
> **Symbol:** `someFunction()` / `SomeClass::someMethod()`
> **Found by:** Claude Opus 5 (Terminal) · 2026-07-14 14:30 ET
> **Transcript:** `a54150ad-e72d-412b-8b27-c756543bb277`
> **Source:** IMPACT — harvested from `TRFA-API#20` by `/pr-reconcile`, 2026-08-14
> **Trigger:** fires when the drop-column migration runs here — not yet run
> **Related doc:** `docs/SomeComponent.md` (Issue 3) — omit if none

**Symptom:** What a developer or user actually observes. Externally visible behaviour, not internals.

**Cause:**

```
if (! $details && $redirect) {   // line 118
```

Why the code is wrong. Quote the offending line(s) with a `// line N` comment.

**Reproduce:**

> `GET /some/route/999999/false`   (or the exact command / input that triggers it)

**Expected:** … **Actual:** …

**Suggested fix:** What to change, and what to check before changing it.
```

### ⛔ This is the SUPERSET template — every field, marked with where it applies

**One canonical field list, ruled 2026-08-14.** `/file-bug`'s heredoc pastes the same shape into `gh issue create`, and ⚠️ **the two had already drifted three ways before anyone noticed** — `Lane:` existed only in the command, `Related doc:` only here, and the same field was called `Symbol:` here and `Function:` there, so a grep for either silently missed every entry filed the other way. **Omit a field that does not apply; never rename one.**

| field | applies to |
|---|---|
| `Status:` `Severity:` `File:` `Symbol:` `Found by:` `Transcript:` | **both paths, required** |
| `Lane:` | **GitHub Issue only.** In this ledger the lane is already carried by the ID prefix (`BUG-` / `SUSP-`) and by which section the entry sits in. ⚠️ **If it appears here anyway, and it disagrees with the prefix, the PREFIX WINS.** |
| `Source:` `Trigger:` | **IMPACT findings only** — entries produced by `/pr-reconcile` from another repo's merged PR. See below. |
| `Related doc:` | omit if none |

**`Source:` and `Trigger:` — for a defect found by harvesting another repo's PR.** `Source:` records the provenance (*which PR, which tool, when*); `Trigger:` records what makes the breakage fire and whether it has fired here yet.

★ **`Trigger:` is what keeps such an entry honest.** An IMPACT entry often sits in the confirmed lane with a Reproduce step that does not *currently* reproduce — because the change has not been deployed here. Without the trigger that reads as a bad entry; with it, it reads as **fix this before you migrate**, which is the decision the reader actually faces.

## Field rules

### ID — `BUG-YYYY-MM-DD-NNN`

Date the bug was **found**, plus a zero-padded counter that restarts each day. `BUG-2026-07-14-001`, `-002`, and so on. IDs are permanent: never renumber, never reuse an ID after a bug is closed.

### Status

One of: **Open** · **Fixed** · **Won't Fix** · **Cannot Reproduce**.

When a bug is fixed, do **not** delete the entry. Change the status to `Fixed`, and add a `**Resolution:**` line naming the commit SHA. The ledger is a history, not a to-do list.

### Severity

| Severity | Meaning |
|----------|---------|
| **Critical** | Data loss, security hole, or silent corruption of production/customer data. |
| **High** | A documented feature is broken, or an error is silently swallowed so failures go unnoticed. |
| **Medium** | Wrong behaviour in an edge case, or a maintenance trap likely to cause a future bug. |
| **Low** | Cosmetic, dead code, or a nuisance with an easy workaround. |

Severity is about **consequence**, not about how easy the fix is.

### File and line

> **Always** `path/to/File.ext:LINE`, relative to the repo root.

- **Verify the line number against the actual file before writing it.** A stale or invented line number sends the next developer to the wrong place and destroys trust in the whole ledger.
- If a bug spans several sites, list them all: `SomeFile.ext:59, :103, :214`.
- Line numbers drift as code changes. Always name the **symbol** (function / method / class) as well — that survives edits.

### Found by

Name the model that actually ran, plus its surface — e.g. `Claude Opus 5 (Terminal)`. **Name the running model; never copy a version out of this guide.** The examples here are illustrative and go stale; a version copied from them attributes the find to a model that never ran. If a human found it, name them. If provenance is unknown, write `unknown` — **never guess**. Follow the name with the date **and time** the bug was found (see below).

### Date and time — always include a timezone

> **Format:** `YYYY-MM-DD HH:MM ZZ` — e.g. `2026-07-16 14:30 ET`.

- Every bug records **when** it was found, not just the day. Use a 24-hour clock and **always** name the timezone (`ET`, `CT`, `PT`, `MST`, …) — a timestamp with no zone is ambiguous and forbidden here.
- The time belongs on the **Found by** line, immediately after the date: `**Found by:** Claude Opus 5 (Terminal) · 2026-07-16 14:30 ET`.
- The **ID** still uses the date only (`BUG-YYYY-MM-DD-NNN`); the counter already disambiguates multiple bugs found the same day.
- Use the timezone the work actually happened in. Get it from the machine's own clock (`date "+%Y-%m-%d %H:%M %Z"`) — do not guess or convert. If you don't know the zone, ask rather than inventing one.

### Transcript

The `.jsonl` session ID under `~/.claude/projects/`, so the reasoning behind the find is recoverable. Omit the line if unavailable. Never invent one.

## Suspected / Needs Investigation entries

Use a `SUSP-…` entry when you believe something is wrong but **have not verified it** — the confirmation needs research, a reproduction you can't run now, a log you don't have, or a person to ask. This is how a real concern gets tracked without pretending it's a confirmed defect.

These live under the ledger's `## Suspected / Needs Investigation` section (H2, at the bottom of the file). Each entry is an **H3** heading, so it nests visibly under that section and stays distinct from the H2 confirmed bugs above.

```md
### SUSP-YYYY-MM-DD-NNN — Short summary of the suspected problem

> **Status:** Suspected · **Certainty:** Low
> **Suspected severity (if real):** High
> **Suspected location:** `path/to/File.ext:~120` (unverified — best guess) · `someFunction()`
> **Raised by:** Claude Opus 5 (Terminal) · 2026-07-17 12:30 MST
> **Transcript:** `<session-id>`   <!-- omit if unavailable -->

**Why I suspect it:** The signal that prompted this — an error seen in the wild, a smell in the code, a user report. Be explicit about what is *observed* vs *inferred*.

**How to confirm or refute:** REQUIRED. The concrete step(s) that would settle it — the file to read, the input to try, the log to pull, the person to ask. If you cannot name what would confirm it, it is not a report yet — it's a feeling. Don't file it.

**If confirmed:** the likely impact, and roughly where a full `BUG-` entry would point.
```

### SUSP field rules

- **ID — `SUSP-YYYY-MM-DD-NNN`.** Same date-plus-daily-counter scheme as bugs, own namespace. `SUSP` and `BUG` counters are independent.
- **Status** — one of **Suspected** · **Investigating** · **Promoted** · **Refuted**. (Never **Fixed** — a suspicion is confirmed into a bug *first*, then fixed.)
- **Certainty — Low or Medium.** If you'd call it High, you can probably verify it now — do that and file a `BUG-` instead.
- **Suspected location** — mark it `unverified` and use a `~` before any line number, *or* give only the symbol. **Never write a bare exact line number on a SUSP entry** — that reads as verified. The symbol is the durable anchor.
- **How to confirm or refute — strongly wanted, not a gate.** It's the field that turns a suspicion into something actionable, so write it whenever you can name even a rough check ("open X and see whether Y is ever set"). If you genuinely can't, file the entry anyway with `_(not yet known)_` rather than dropping it. An entry is never *rejected* for lacking it — a thin suspicion beats a lost one.

### Promotion path — every SUSP entry resolves one of two ways

A suspicion is a temporary state. When the research happens:

- **Confirmed →** write a full `BUG-…` entry (verified file:line, real reproduce) in the confirmed section at the top. Then set the SUSP entry's `**Status:** Promoted → BUG-YYYY-MM-DD-NNN` and leave it in place. Don't delete it — the trail from hunch to confirmed defect is worth keeping.
- **Refuted →** set `**Status:** Refuted` and add a `**Finding:**` line stating what the research actually showed. This is the point of the lane: a documented dead end stops the next person (or the next AI session) from re-chasing the same false lead.

A SUSP entry that has sat at **Suspected** for a long time is a signal in itself — either nobody has done the research, or it wasn't worth doing. Surfacing stale ones is fair game for `/next-step`.

## Honesty rules

These are not optional. An AI logging a bug is making a claim about the codebase that a human will act on.

- **Never invent a line number, error message, table name, column name, or config key.** Read the file and confirm it.
- **Never report a bug you have not seen in the source.** "This pattern usually causes X" is not a bug report.
- Don't dress up a suspicion as a confirmed defect. Two honest homes for uncertainty: if you **saw the defect but can't reproduce it now**, file a `BUG-` and mark **Status: Cannot Reproduce**; if you **haven't confirmed it's a defect at all** and it needs research, file a `SUSP-` entry (see **Suspected entries**) — not a hedged `BUG-`.
- If you are inferring consequence rather than observing it, say which. Write "this *would* throw when `$request` is unassigned" — not "this throws" — unless you ran it.
- Quote exact error strings only if you have actually seen them. Otherwise describe the behaviour and note that the literal message is unverified.

> **⚠️ Warning:** A confidently-worded wrong bug report costs more developer time than the bug it describes. When in doubt, downgrade your certainty in the text rather than omitting the entry.

## What is *not* a bug

Keep the ledger signal-dense. These belong in a component doc's **Recommended Improvements** section (or a TODO/backlog), not here:

- Style and formatting preferences
- Missing tests
- "This could be refactored" with no defect behind it
- Framework boilerplate that is merely outdated but works

A bug is a place where the code **does the wrong thing**, or will do the wrong thing under a reachable input.

## Project-Specific Notes

Everything above — including this section's template — is stack-neutral and identical in every repo. The `.claude/guides/` copy in each shared repo is **managed**: it is md5-synced from the toolkit and drift-guarded, so a repo must **never edit its copy** to fill these in. A filled-in copy either blocks every future sync (the drift guard skips a modified file forever) or loses its notes the next time the sync runs with `--force`.

Instead, the repo records its own answers in a **repo-owned** file and points its bug ledger at it: keep a short `docs/guides/BugReportStyleGuide.md` stub holding only the project-specific notes below, and cite it from the header of `docs/BUG-REPORT.md`. The managed guide stays pristine; the stub is where the repo's specifics live. (fran-dash demonstrates this shape — a Project-Specific-Notes stub that defers to the managed `.claude/guides/bug-report-style.md` for everything else.)

The template below is the checklist of what that stub should answer:

- **Bug ledger path:** `docs/BUG-REPORT.md` _(or `.cloaked/docs/BUG-REPORT.md` where the project uses the `.cloaked/` convention)_
- **Code root prefix:** _(e.g. `src/`, `app/`, `site/app/` — whatever `file:line` references are relative to)_
- **Per-module docs:** _(where narrative module/class docs live, if any — else "none")_
- **Default timezone:** _(the zone the team usually works in, e.g. `MST`, `PKT`)_
- **Primary language(s):** _(swap the `<language>` fence tag in the templates accordingly)_
- **GitHub Issues:** _(if this repo files bugs as native Issues, name the Issue Form and say whether the Markdown ledger is still mirrored — see the toolkit's `docs/bug-workflow-rollout.md`)_
