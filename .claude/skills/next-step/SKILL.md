---
name: next-step
description: "Analyze project state and recommend the best next step to work on. Reviews PRDs, TODOs, plans, specs, git history, and open issues to determine the highest-ROI task. Use when starting a session, finishing a task, or deciding what to tackle next."
---

# Next Step

Reads the project's current state — documentation, plans, code, and git history — then recommends the single best next step based on strategic value, dependencies, and momentum.

## When to Use

- Starting a new session ("what should I work on?")
- Just finished a task and need direction
- Feeling stuck or unsure what matters most
- Want to validate your own instinct about priorities

## How It Works

### Step 1 — Gather Project State

Read ALL of the following that exist in the current project directory. Skip any that don't exist — not every project has all of these.

1. **PRD.md** (or any PRD file) — master plan, phases, priorities
2. **`docs/TODOS.md`** (a few repos use `TODO.md` — read whichever exists) — active task lists
3. **docs/CURRENT_STATUS.md** — where things stand
4. **docs/NEXT_STEPS.md** — prior session's handoff notes
5. **specs/*.md** — active plans and specs (skip archived)
6. **CLAUDE.md** — project context and implementation status
7. **docs/CHANGELOG.md** — recent completed work (last 10 entries)
8. `git log --oneline -10` — what was committed recently
9. `git status` — any in-progress uncommitted work
10. `git diff --stat` — scope of uncommitted changes

### Step 2 — Analyze and Score

Evaluate every candidate task against these criteria. Weight them in this order:

1. **Unblocks other work** (highest weight) — Does completing this open up 2+ downstream tasks? Blockers get priority over isolated features.
2. **Momentum** — Is this a natural continuation of recent work? Switching contexts costs time. Finishing what's in-flight beats starting something new.
3. **ROI** — Effort vs. impact. A 30-minute fix that affects every page beats a 3-session feature that affects one page.
4. **Dependencies satisfied** — Are all prerequisites done? Don't recommend something that requires unfinished work.
5. **User-facing value** — Does the user see/feel the improvement? Infrastructure work matters but visible progress builds confidence.
6. **Technical debt reduction** — Does this prevent future problems or simplify future work?

### Step 3 — Present Recommendation

Output EXACTLY this format:

```
## Recommended Next Step

**[Task name]** — [one-line description]

### Why This, Why Now
[2-3 sentences explaining why this is the best use of time right now, referencing the scoring criteria above. Be specific — "this unblocks X and Y" not "this is important".]

### What It Involves
[Bullet list of concrete subtasks — files to create/modify, patterns to follow, estimated scope]

### What It Unblocks
[What becomes possible after this is done]

### Alternatives Considered
| Task | Why Not Now |
|------|------------|
| [Other candidate 1] | [Specific reason — dependency, lower ROI, etc.] |
| [Other candidate 2] | [Specific reason] |
| [Other candidate 3] | [Specific reason] |

### Ready to Start?
[One sentence: "Say 'go' to begin, or tell me if you'd prefer one of the alternatives."]
```

## Rules

- **Always recommend exactly ONE task.** Not a list. Not "either A or B." Pick one.
- **Be opinionated.** The user is asking for your judgment, not a menu.
- **Reference specific files and line counts** when describing scope. "About 200 lines across 3 files" not "medium effort."
- **Never recommend busywork.** Reorganizing files, adding comments, or writing docs are not next steps unless they directly unblock real work.
- **If the project is in great shape** and there's no clear winner, say so: "The project is in good shape. Here are your options by priority..." and list 3 ranked choices briefly.
- **Account for what just happened.** If the last commit was a big feature, maybe a quick-win cleanup is the right palette cleanser before the next big push.
