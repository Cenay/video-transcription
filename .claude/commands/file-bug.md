---
name: file-bug
description: File a bug as a native GitHub Issue via `gh` in the Bug Report Style Guide format — Confirmed or Suspected lane — and mirror it into docs/BUG-REPORT.md while both flows run in parallel.
---

You are filing a bug. It goes to **two places** during the pilot: a native GitHub
Issue (the future system of record) and `docs/BUG-REPORT.md` (the legacy ledger, kept
until the Issue flow is proven). Follow the **Bug Report Style Guide** exactly:
`.claude/guides/bug-report-style.md` — read it if unsure. Its core rule governs
everything here:

> **A report that misrepresents its own certainty is worse than no report at all.**

**File it even when you can't verify it.** A developer only runs this command because they
decided something is worth documenting — respect that judgement. Your job is to capture it
accurately, not to audit whether it earned its place. Deduce what you can from the codebase
and the conversation, ask **at most one or two** short questions for things you genuinely
cannot infer, and file with the gaps marked. Never refuse to file, never send someone away
to gather evidence first, and never interrogate.

## Step 0 — Which lane: Confirmed or Suspected?

Decide this first; it changes the labels and the wording of everything below. It mirrors
the `BUG-` / `SUSP-` split in the Markdown ledger and the "Confirmed or Suspected"
dropdown in `.github/ISSUE_TEMPLATE/bug_report.yml`.

| Lane | When | Labels |
|------|------|--------|
| **Confirmed** | You have **seen** the defect in the source or at runtime. | `bug` + `severity:*` |
| **Suspected** | A hunch needing research — a pattern that usually causes trouble, an inferred consequence you have not observed, or a line you could not fully verify. | `bug` + `severity:*` + `status:suspected` |

**Filing a hunch is fine and useful. Filing a hunch dressed as a confirmed defect is
not.** When in doubt, choose Suspected — nothing is lost by it, and a confidently-worded
wrong report costs more than the bug. If the user hands you a hunch, do not silently
promote it to Confirmed because it looks plausible; ask, or file it Suspected.

## Honesty gate — do this BEFORE writing anything

- **Never invent** a line number, error string, symbol, table/column name, or config key.
  This is the floor, and it does not move.
- **Verify what you reasonably can** — open the file at the line you're about to cite.
  But **an unverified fact is not a reason to drop the report.** Mark it instead:
  `File.ext:~120 (unverified)`, "error message approximate", or name the function and
  skip the line number entirely. Never write a bare exact line number you haven't
  checked — that reads as verified.
- **If filing Confirmed, you must have actually seen it** in the source or at runtime.
  If you are inferring consequence rather than observing it, that is the Suspected lane —
  and say so in words in Cause ("this *would* throw when…").
- **When the bar and the report conflict, file it as Suspected — don't discard it.** A
  hunch recorded honestly is useful. A hunch lost because it couldn't clear the
  Confirmed bar helps nobody. The rule was only ever aimed at a guess *presented as* a
  confirmed fact.
- If you cannot reproduce it, still file it but say so plainly (it maps to a
  `cannot-reproduce` label / status).

## Step 1: Gather + format the fields

Collect these from the user / your investigation and format per the style guide. **Only
Summary, Lane, Severity, Found by, When found and Symptom are needed to file** — that
matches the required set on the web Issue Form. Everything else is filled in when known
and omitted (or marked unverified) when not. Infer what you reasonably can: severity from
consequence, file/function from the code you just read, found-by from the current model +
surface, when-found from `date`. Ask only about what you truly cannot deduce.

| Field | Notes |
|-------|-------|
| **Summary** | Short imperative line → becomes the issue title, prefixed `[Bug]: `. |
| **Lane** | Confirmed or Suspected, from Step 0. Goes in the body blockquote **and** drives the `status:suspected` label. |
| **Severity** | Critical / High / Medium / Low — about *consequence*, not fix difficulty. |
| **File and line** | `path/to/File.ext:LINE`, repo-root-relative, line **verified**. `N/A (…)` for infra. |
| **Function / method** | Survives line drift. `N/A` for config/infra. |
| **Found by** | Model+surface (`Claude Opus 4.8 (Terminal)`) or a person. `unknown` if truly unknown. |
| **When found** | `YYYY-MM-DD HH:MM ZZ` — get it from the machine: `date "+%Y-%m-%d %H:%M %Z"`. TZ is mandatory. |
| **Symptom / Cause / Reproduce / Expected / Actual / Suggested fix** | Prose sections; Cause quotes the offending lines with `// line N`. |
| **Transcript** | The `.jsonl` session ID under `~/.claude/projects/`. Omit if unavailable — never invent. |

## Step 2: Create the GitHub Issue

Verify `gh` is authenticated (`gh auth status`) and you are in the repo. Build the body
as a HEREDOC in the **exact style-guide shape** (blockquote of key facts, then the prose
sections), then create the issue with the `bug` label, a severity label, and — **for the
Suspected lane only** — `status:suspected`:

```bash
gh issue create \
  --title "[Bug]: <summary>" \
  --label "bug" \
  --label "severity:<critical|high|medium|low>" \
  --label "status:suspected" `# SUSPECTED LANE ONLY — omit this line for Confirmed` \
  --body "$(cat <<'EOF'
> **Status:** Open · **Lane:** <Confirmed|Suspected> · **Severity:** <High>
> **File:** `path/to/File.ext:118`
> **Function:** `someFunction()`
> **Found by:** <Claude Opus 4.8 (Terminal)> · <2026-07-16 18:27 MST>
> **Transcript:** `<session-id>`

**Symptom:** …

**Cause:**

```<language>
if (!details && redirect) {   // line 118
```

**Reproduce:**

> `<exact command / input>`

**Expected:** … **Actual:** …

**Suggested fix:** …
EOF
)"
```

Notes:
- The **issue number is the permanent ID** — it replaces `BUG-YYYY-MM-DD-NNN`. Capture it
  from the URL `gh issue create` prints.
- **Always write the lane into the body**, even though you also apply the label. The web
  Issue Form *cannot* apply a label conditionally on its dropdown, so for issues filed
  through the browser the body is the only record of the lane. Keeping both paths writing
  the body means one place is always authoritative. Never read "no `status:suspected`
  label" as "confirmed" — read the body.
- If the labels don't exist yet in the repo, create the canonical set once (colors and
  descriptions in `docs/bug-workflow-rollout.md` → "The canonical label set"):
  `severity:critical|high|medium|low` and `status:suspected`. Falling back to body-only is
  acceptable if you can't create labels — the body line is the record either way.
- Prefer this direct `gh issue create` path (it lets you file non-interactively). The
  `.github/ISSUE_TEMPLATE/bug_report.yml` form is the equivalent path for the web UI /
  teammates who don't use Claude Code.

## Step 3: Mirror into docs/BUG-REPORT.md  (LAYERED — remove once proven)

<!-- TODO(bug-workflow): DELETE this Step 3 once GitHub Issues are the proven system of
     record. At that point BUG-REPORT.md is retired (see docs/bug-workflow-rollout.md,
     "Retiring the Markdown ledger"). Until then, keep the ledger true so nothing is lost
     if we roll back. -->

Add the same bug to `docs/BUG-REPORT.md` following the style guide: **newest at the top**,
immediately under the intro — never append to the bottom, never delete a closed entry.
Use the ledger's own ID here — `BUG-YYYY-MM-DD-NNN` for the Confirmed lane,
`SUSP-YYYY-MM-DD-NNN` in the Suspected section for the Suspected lane (the ledger keeps
its own IDs and its own two-lane split, matching `/bug`). Add a pointer to the GitHub
issue:

```md
> **GitHub issue:** #<n>  (native issue is becoming the system of record)
```

If the repo has no `docs/BUG-REPORT.md` yet, create it from the header used by other TRFA
repos (intro blockquote citing `.claude/guides/bug-report-style.md`).

## Step 4: Report back

Show: the issue number + URL, the lane, the labels applied, and confirm the ledger entry
was added (with its `BUG-` / `SUSP-` ID). One or two lines.
