---
name: meeting-reconcile
description: Reconcile a meeting note (Notion, Google Docs, or a transcript path/URL) into the current project's docs — or resume/apply an existing intake note. Wrapper over the meeting-reconcile skill.
---

Invoke the **`meeting-reconcile` skill** and run its full workflow against the argument below. Do not reimplement the steps here — load the skill (its SKILL.md and `guides/reconciliation-note-format.md`) and follow it exactly.

**Argument (`$1`):** `$ARGUMENTS`

Interpret it as follows:

- **A meeting source** — a Notion URL, a Google Docs / Drive URL, a local file path to a transcript, or (if `$1` is empty) pasted text the user is about to provide → run in **fresh mode**: fetch, distill, classify, and write the reconciliation intake note to `<docs-root>/intake/`, then stop for rulings.
- **An existing intake note** — a path ending in `intake/*-reconciliation.md` → run in **resume mode**: pick up where that note left off (walk any unruled contradictions, then apply on the `apply` trigger).

The project is the current working directory unless the user names another. If `$1` is empty, ask for the source (or confirm they'll paste it) before proceeding.

Nothing lands in the project docs until the user gives the explicit **`apply`** trigger, and never before every section-3 contradiction is ruled — this is the skill's Step 5, the completing step. After applying, hand off to `doc-reconcile` then `/checkpoint`, per the skill.
