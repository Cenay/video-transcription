# Decisions — video-transcription

**Last updated:** 2026-07-31 02:15 MST

> The decision ledger. When any other doc disagrees with this file, **this file wins.**
> New decisions get appended here, dated. Record the *why*, not just the *what* — a
> decision without its reasoning gets re-litigated in six months.

**How to read this file.** DEC- entries are in flat numeric order and **each one is self-contained**: fold later amendments, reversals, and closures into the entry itself, dated, so an entry's position on the page never determines what is currently true. **Read the entry's `**Status:**` line and stop.** Status vocabulary: `🚧 OPEN`, `⏸ DEFERRED`, `✅ CLOSED`, `⛔ SUPERSEDED by [DEC-NNN]`, `📋 PROPOSED`.

---

## 📇 Index — all decisions at a glance

_Regenerated 2026-07-31 02:15 MST from the entries below · **1 DEC entry**: **0 open** · **1 closed**._

> **This table is derived, not authoritative** — the entry below is. Regenerate it after
> adding or closing a decision rather than hand-editing; if the two disagree, the entry wins.
> Anchors are heading slugs produced by `.claude/scripts/link-doc-refs.py`, so they resolve
> in both VS Code's preview and GitHub.

| ID | Decision | Status | Date |
|---|---|---|---|
| [DEC-001](#dec-001-retire-todos-consolidate-on-docstodosmd) | Retire `todos/`, consolidate on `docs/TODOS.md` | ✅ CLOSED | 2026-07-31 |

---

## Open

<!-- Questions not yet settled. Move to Resolved when answered. -->

## Resolved

<!-- Format:
### DEC-001 <short title>
- **Status:** RESOLVED (YYYY-MM-DD HH:MM TZ)
- **Question:** What was actually being asked?
- **Answer:** What was decided, and by whom.
- **Why:** The reasoning. This is the part that matters later.
- **Build impact:** What this changes about how we build.
-->

### DEC-001 Retire `todos/`, consolidate on `docs/TODOS.md`
- **Status:** RESOLVED (2026-07-31 02:15 MST)
- **Question:** The project kept its task drop-box at `todos/qol-improvements.md` while the
  rest of the standard doc set lives in `docs/`. Keep the variant or consolidate?
- **Answer:** Consolidate. Cenay called it during project init — content moved verbatim into
  `docs/TODOS.md` (open item → Active, the two shipped items → Completed), and `todos/`
  was archived to `.archived/2026-07-31/todos/`.
- **Why:** One doc set, one location. A second task file outside `docs/` is the kind of thing
  that drifts — a later session updates whichever one it happens to find first, and the two
  start disagreeing about what's still open.
- **Build impact:** `docs/TODOS.md` is the only task drop-box. Inbound links from
  `docs/planning.md` and `docs/NEXT_STEPS.md` were repointed. Nothing in `scripts/` or
  `transcribe-this.sh` referenced the old path.
