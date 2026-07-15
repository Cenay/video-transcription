---
name: resume
description: Start-of-session briefing — load project context and pick up where you left off.
---

You are resuming work on the current project. Give the user a fast, actionable briefing so they can jump back in immediately.

## Step 1: Load Context

Read the following files (skip any that don't exist):

1. `CLAUDE.md` — project overview and conventions
2. `docs/CURRENT_STATUS.md` — where we left off
3. `docs/TODOS.md` — outstanding tasks
4. `docs/LESSONS_LEARNED.md` — scan for recent entries (last 5)
5. `plans/*.md` — check for any plans with `status: in-progress` or `status: approved`
6. `git log --oneline -10` — recent commits

## Step 2: Deliver Briefing

Present a dense, actionable briefing. No fluff. Use this format:

---

**{PROJECT_NAME}** | Last checkpoint: {date + time from current-status.md — show the full `YYYY-MM-DD HH:MM TZ`, not just the date}

**Where we left off:** {1-2 sentences from CURRENT_STATUS.md "In Progress" and "Session Summary"}

**Next steps:**
{Numbered list from CURRENT_STATUS.md "Next Steps", prioritized}

**Open TODOs:** {Count from docs/TODOS.md Active section, list top 3}

**Active plans:** {Any plans with status in-progress or approved — name + one-liner summary, or "None"}

**Recent lessons:** {Any entries from last 7 days — one-liners only}

**Blockers:** {From CURRENT_STATUS.md, or "None"}

---

## Step 3: Ask Direction

End with exactly one question:

> "Pick a next step, or tell me what you want to work on."

Do NOT summarize the entire CLAUDE.md. Do NOT explain the tech stack unless asked. The user already knows their project — they just need to remember where they were.
