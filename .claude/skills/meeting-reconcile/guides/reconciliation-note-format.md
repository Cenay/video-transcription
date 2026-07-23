# Reconciliation intake note — format & method

The reconciliation note is a **persisted intake document**, not a chat summary. It lives at `<docs-root>/intake/YYYY-MM-DD-<slug>-reconciliation.md` and is the single artifact the skill produces before any project doc is touched. It is itself the proposal: it says out loud that **nothing has been written yet**, and it stages every candidate change as verbatim text for you to approve.

Reference exemplars (fran-dash): `docs/intake/2026-07-21-art-meeting-reconciliation.md` and `docs/intake/2026-07-22-nik-intro-reconciliation.md`. Match their shape.

## Header

```markdown
# Reconciliation — <short meeting label>, YYYY-MM-DD ("<exact source page title>")

_Generated YYYY-MM-DD HH:MM TZ by an AI session · transcript: `<session-id>`_

**Source:** <Notion/Docs page name> (`<page-id>`), <duration>, <attendees>. <Video/recording link if any>. <What was read — e.g. "Full Fireflies distillation (Overview / Summary / Notes / Action Items / Key Decisions); 44 KB verbatim transcript present but not quoted line-by-line.">

<Optional one-paragraph cross-link to a prior reconciliation this meeting continues or resolves — e.g. "This is the meeting the 2026-07-21 reconciliation predicted (→ N6). The vendor's name is Nik, not Nick.">

**Scope of this file:** a proposal only. Nothing here has been written into `docs/DECISIONS.md`, `docs/CURRENT_STATUS.md`, `docs/TODOS.md` <+ any other targets>. DEC numbers below are **suggested** and start at **DEC-NNN** — the ledger currently ends at **DEC-<last>**, and DEC-<x>–<y> are already claimed as unwritten proposals by <prior intake file, if any>. Renumber on adoption.

---
```

Stamp rules from the global convention apply verbatim: date **and** 24-hour time **and** timezone from `date "+%Y-%m-%d %H:%M %Z"`, never guessed; transcript ID is the session's real `.jsonl` filename, never invented (write `(session id: unknown)` if genuinely unknown).

## Body — four numbered sections

Every candidate is classified into exactly one section and given an ID (`N1`, `R1`, `C1`, …).

### `## 1. NEW items`
Information the meeting introduced that no doc yet records. Each `### N<k> — <title>`:

```markdown
**Said:** <the evidence — a direct quote where the wording matters, else a faithful paraphrase; name who said it>
**Belongs in:** `docs/<file>.md` (new DEC / TODO / status line) <+ `→ "exact section heading"` where it lands>
**Proposed entry:** *<the exact text to write — a full DEC-NNN sentence, a TODO line, a status update>*
```

New decisions propose a fresh `DEC-` ID continuing the ledger. Action items name an owner where stated. Cite the target section by its **heading text** (`→ "The founding rule — Miami is the master"`), not a line number.

### `## 2. RESOLVED items` (a.k.a. "RESOLVED / ADVANCES an existing open item")
Things the meeting **settles or moves forward** on an item the docs already track — an open `DEC-` now answered, a caveat now closed, a prior reconciliation's item now resolved. Each `### R<k>` cites the existing ID/doc it advances and states the new standing.

### `## 3. CONTRADICTIONS`
Where the meeting collides with what a doc currently claims. **This is the section that needs a ruling — never auto-resolve it.** Each `### C<k>` shows both sides and, once ruled, carries a marker:

- `✅ NOT a contradiction` — closer reading shows they agree
- `✅ RULED` — you decided; the resolution is recorded inline
- `🅿️ DEFERRED` — parked for later
- `⚠️` — live, still needs your call

A reversed decision supersedes the old `DEC-` entry (old text preserved, marked *superseded by DEC-NNN*) — it is never deleted. When ruled, update the header stamp to note it, e.g. `_Contradictions ruled by <name> YYYY-MM-DD HH:MM TZ — section 3 now records rulings, not open questions._`

### `## 4. NO-OP / already captured`
Items mentioned that the docs already reflect — listed briefly so the record shows they were considered, not missed.

## Method notes

- **The note is written before any project doc changes.** Its "Scope" line asserting nothing is written yet must stay true until you approve.
- **DEC numbering accounts for unwritten proposals.** If a prior intake file already claimed DEC-129–131 as proposals, start at DEC-132 and say so. Never assume a suggested number already exists in the ledger.
- **Quote where framing matters.** "Confirmed" vs "amended", "universal pricing" vs "prices differ per location" — the exact words drive the ruling. Preserve them.
- **Only after approval** are the approved N/R items and ruled C items written into the real docs (each stamped), the intake note's header updated to record the rulings, and control handed to `doc-reconcile` then `/checkpoint`.
