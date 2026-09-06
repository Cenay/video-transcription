---
name: build-timer-note
description: Summarize this session's work as a client-readable bulleted list, printed and copied to the clipboard for a time-tracking entry.
---

Produce a short bulleted summary of what was accomplished in **this conversation**, suitable for pasting into a time-tracking note. An optional request to include all "conversations" for a given date (or range) can also be requested. When in doubt, refer to the GitHub repo(s) requested. 

## Source of truth

1. **Primary — this conversation.** Summarize the work actually done in the current session: tasks completed, things built, problems solved, decisions made.
2. **Fallback — git history.** Only if this conversation has little or no substantive work to summarize (e.g. it was just started, or context was cleared), pull the recent commit one-liners for the current repo:
   ```
   git log --oneline --no-merges --since="6am" --author="$(git config user.name)"
   ```
   If that returns nothing, widen to `-15` most recent commits and use judgement about where the last session began. When the fallback is used, print this line **above** the list:
   `_(summarized from commit history — this conversation had no session work to draw on)_`

Never invent work. If you cannot determine what was done, say so plainly and stop — do not pad the list.

## Output rules

- **Granularity: one bullet per task.** Distinct tasks stay separate even when they touch the same feature. Do not roll a whole feature into one line, and do not split a single task across multiple lines. Typically 10–20 bullets; fewer is fine for a short session.
- **Length: 4–20 words per bullet.** Enough to identify the work, no more.
- **Voice: client-readable.** Plain business English describing the *outcome*. No filenames, function names, flags, commit SHAs, or tool jargon. A non-technical reader should understand every line.
  - Good: `- Added a one-command way to file a bug report (claude-personal-toolkit)`
  - Bad: `- Created commands/bug.md with frontmatter and symlinked it`
- **Every bullet ends with its repo's short name in parentheses.** The short name is the repo folder's basename — `fran-dash`, `trfaapi.com`, `claude-personal-toolkit` — never a path, never an owner prefix, never the display title. It is the one piece of tool vocabulary the rule above allows, because time gets billed per project and a list of outcomes with no project against them cannot be entered.

  ```
  - Primary site navigation now pulled from the database instead of hardcoded links (fran-dash)
  ```

  **Which repo goes on a bullet:** the one the work actually landed in, decided **per bullet, not per session**. A session run from one repo routinely changes another — a meeting reconciliation writes into several, a toolkit change lands in the repos it is synced to — and a bullet attributed to the session's own folder in those cases is billed to the wrong project. So attribute from where the change went, not from where you were sitting.

  ⚠️ **The one exception: work drawn from a session note** — a Session Desk, a checkpoint, a status or decision doc — **takes the repo the session is running in**, since a note records what was discussed rather than where anything landed:

  ```bash
  basename "$(git rev-parse --show-toplevel)"
  ```

  Use that same value for anything else you genuinely cannot attribute — and if a bullet's repo is a guess rather than something you saw in the work, say so in one line under the list rather than guessing silently.
- **Format: plain dashes**, sentence case, no trailing periods, no nesting, no sub-bullets. The repo name in parentheses is the last thing on the line.
- **Bullets only.** No header, no date, no time estimate, no closing summary, no commentary — except the fallback notice above when it applies.
- Order the bullets roughly chronologically.

## Delivery

1. Print the list in the chat as a plain code-free markdown list.
2. Copy the same text to the clipboard:
   ```
   printf '%s\n' '<the bullet list>' | xclip -selection clipboard
   ```
   Use a heredoc if quoting gets awkward. If `xclip` is unavailable, print the list anyway and note in one line that the clipboard copy failed.

Confirm in a single short line that it was copied. Nothing else.
