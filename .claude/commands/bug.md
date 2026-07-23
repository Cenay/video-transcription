---
name: bug
description: File a bug into the project's ledger (docs/BUG-REPORT.md) — a verified BUG- entry, or a labeled SUSP- (suspected / needs-investigation) entry for hunches that need research. Follows the bug-report style guide.
---

File a bug report for the current project. The argument is a short description of the defect: `$ARGUMENTS`.

> **Sibling command — `/file-bug`.** This command (`/bug`) is the **ledger-only** lane: it writes the entry into `docs/BUG-REPORT.md` and stops. Its sibling `/file-bug` is the **GitHub lane** — it files the same entry as a native GitHub Issue and mirrors it into the ledger. The two run in parallel by design ([DEC-126]); the format below is identical for both. If this repo files bugs as Issues, prefer `/file-bug`; use `/bug` when you only want the local ledger.

## The one rule that matters

**A report that misrepresents its own certainty is worse than no report at all.** A *confirmed* bug must be independently verifiable — open the file at the line you're about to cite and confirm the problem is real; never invent a line number, error message, symbol name, table/column name, or config key. A *suspected* issue must be labeled as unproven and say what would confirm it. What's never allowed is filing a hunch dressed as a confirmed defect.

**Verification is the aim, not the entry fee.** A developer runs this because they judged something worth documenting — capture it, don't audit whether it earned its place. Deduce what you can, ask at most one or two short questions, and file with the gaps marked. Never refuse to file and never send someone away to gather evidence first: an unverifiable hunch belongs in the Suspected lane, not in the bin.

## Step 0 — Which lane: confirmed or suspected?

Decide before anything else:

- **Confirmed** — you can see the defect in the source and verify/reproduce it **right now** → follow Steps 1–5 as a `BUG-` entry.
- **Suspected** — you believe something's wrong but confirming it needs research, a reproduction you can't run now, a log you don't have, or someone to ask → jump to **Suspected lane** below and file a `SUSP-` entry instead.

If the argument makes the lane obvious, proceed. If it's genuinely ambiguous whether you can verify it now, ask the user one short question rather than guessing. **Never verify-by-assumption** — if you didn't actually confirm it, it's Suspected.

## Step 1 — Read the style guide

Read the bug-report style guide and follow it exactly:
- In this toolkit: `guides/bug-report-style.md`
- In a shared TRFA repo: `.claude/guides/bug-report-style.md`

If neither exists, apply the format below from memory.

## Step 2 — Locate (or create) the ledger

Per the project's doc-routing convention:
- Project under `/mnt/k/Code/` → `docs/BUG-REPORT.md`
- Project using the `.cloaked/` convention (e.g. `/mnt/k/_Sites/`) → `.cloaked/docs/BUG-REPORT.md`
- Otherwise → use whichever docs root the project already has; create the file with a one-line intro if it's missing.

## Step 3 — Verify the facts

- Open the cited file(s) and confirm the **line number** against the actual source. Line numbers drift, so also name the **symbol** (function / method / class) — that survives edits.
- Get the timestamp from the machine: `date "+%Y-%m-%d %H:%M %Z"`. Never guess the time or the zone.
- Compute the ID: `BUG-YYYY-MM-DD-NNN` using **today's** date and the next zero-padded counter for today (scan the ledger for existing `BUG-<today>-*` entries; start at `001`).
- Provenance: `Found by: <model + surface>` (or a human's name; `unknown` if truly unknown — never guess). Transcript = this session's `.jsonl` ID under `~/.claude/projects/`; omit the line if unavailable.

## Step 4 — Write the entry at the TOP of the ledger

Insert immediately under the file's intro (newest first — never append to the bottom):

```md
## BUG-YYYY-MM-DD-NNN — Short imperative summary of the defect

> **Status:** Open · **Severity:** Critical | High | Medium | Low
> **File:** `path/to/File.ext:LINE` (relative to repo root)
> **Symbol:** `someFunction()`
> **Found by:** <model + surface> · YYYY-MM-DD HH:MM TZ
> **Transcript:** `<session-id>`   <!-- omit if unavailable -->

**Symptom:** What a developer or user actually observes — externally visible behaviour, not internals.

**Cause:**

​```
<offending line(s), each with a // line N comment>
​```

Why the code is wrong.

**Reproduce:**

> `<exact command / route / input that triggers it>`

**Expected:** … **Actual:** …

**Suggested fix:** What to change, and what to check before changing it.
```

**Severity is about consequence, not fix difficulty:** Critical = data loss/security/corruption; High = a documented feature broken or an error silently swallowed; Medium = wrong behaviour in an edge case or a maintenance trap; Low = cosmetic/dead code/easy workaround.

## Step 5 — Report, don't commit

Show the entry you added and its ID. **Do not commit or push** — that's the user's call (`/ship` or `/wrap-up` when ready). If the same defect warrants a note in a component/class doc, mention it, but only add it if asked.

If you were inferring consequence rather than observing it, say which in the text ("this *would* throw when…" vs "this throws"). A confidently-worded wrong bug report costs more than the bug it describes.

## Suspected lane (unverified reports)

For a hunch you can't confirm right now. This tracks a real concern without pretending it's proven.

1. **Locate the ledger** (same routing as Step 2). Ensure it has a `## Suspected / Needs Investigation` section at the **bottom** — create it if missing.
2. **Compute the ID:** `SUSP-YYYY-MM-DD-NNN` — today's date, next zero-padded counter among today's `SUSP-` entries (its own namespace, independent of `BUG-`). Timestamp from `date "+%Y-%m-%d %H:%M %Z"`.
3. **Write an H3 entry** at the top of that section (newest-first):

```md
### SUSP-YYYY-MM-DD-NNN — Short summary of the suspected problem

> **Status:** Suspected · **Certainty:** Low | Medium
> **Suspected severity (if real):** Critical | High | Medium | Low
> **Suspected location:** `path/to/File.ext:~120` (unverified) · `someFunction()`
> **Raised by:** <model + surface or human> · YYYY-MM-DD HH:MM TZ
> **Transcript:** `<session-id>`   <!-- omit if unavailable -->

**Why I suspect it:** The signal — an error seen in the wild, a code smell, a user report. Separate observed from inferred.

**How to confirm or refute:** The concrete step(s) that would settle it. REQUIRED — if you can't name what would confirm it, don't file it.

**If confirmed:** likely impact and roughly where a `BUG-` entry would point.
```

**Rules that make it honest:** Certainty is Low or Medium only (if it's High you can probably verify now — do that, file a `BUG-`). Never write a bare exact line number — use `~N (unverified)` or just the symbol. Write "How to confirm or refute" whenever you can name even a rough check; if you genuinely can't, put `_(not yet known)_` and file anyway — it is not a gate.

4. **Report, don't commit.** Show the entry and its ID. Note that it resolves later one of two ways: **Promoted** → a real `BUG-` entry once confirmed (set the SUSP status to `Promoted → BUG-…`), or **Refuted** → status `Refuted` with a `**Finding:**` line recording what the research showed. Don't delete either way — the trail is the value.
