---
name: build-timer-note
description: Summarize this session's work as a client-readable bulleted list, printed and copied to the clipboard for a time-tracking entry.
---

Produce a short bulleted summary of work done, suitable for pasting into a time-tracking note. **With no date, it covers this conversation. With a date or two, it covers every work repo for that window** — see *Dated mode* below.

## Dated mode — a date, or two dates

When the request names a date (`9/23`, `2026-09-23`, `yesterday`) or a range (`9/22 - 9/23`), gather the commits with the script. ⛔ **Do not compute the window or pick the repos yourself** — the rules below are in the script, tested, and easy to get subtly wrong by hand:

```bash
S=.claude/scripts/timer-commits.py; [ -f "$S" ] || S=/mnt/k/Code/claude-personal-toolkit/scripts/timer-commits.py
python3 "$S" 9/23                          # one date
python3 "$S" 9/22 9/23                     # a range
python3 "$S" 9/23 --include-personal       # only when the user says "include personal"
python3 "$S" 9/23 --repo fran-dash         # only when the user names specific repos
```

What the script decides (✅ ruled by Cenay 2026-09-24):

- **One date** → 00:00 to 23:59 of that date. **Two dates** → 00:00 of the first to 23:59 of the second.
- **Rollover:** if it is now the day *after* the end date and **before 6:00am**, the window runs to *now*, because you were still working past midnight. At 9/24 04:30, `9/23` means 9/23 00:00 → 9/24 04:30. ⓘ The 6am cutoff exists because "after 11:59pm" is true forever: without one, asking for 9/23 on 9/26 would bill three days to one date.
- **Repos:** by default, **every work repo** — each git repo under `/mnt/k/Code` and `/mnt/k/_Sites` — **except the personal one, `Code/System`**, and anything in `.archived/`. ⚠️ **`claude-personal-toolkit` is work, never personal.** Only the words *"include personal"* add `System`; only named repos narrow the scope.
- Your commits only (by `git config user.name`), from all branches, merges excluded, **deduplicated by hash**, since `Cenay/N8N` is cloned twice.

Turn the commits into bullets under the same *Output rules* below, attributing each bullet to the repo the script lists it under. If this conversation's own work falls inside the window and isn't committed yet, include it too. **Relay every `⚠️` line the script prints as one line under the list** — roots not found, repos not searched — so a thin list is never mistaken for a quiet day. ⓘ Overlap between two notes is fine: a note explains one timer entry, and the timer, not the note, is what bills. ⓘ On a machine without those roots (a teammate's), the script reports that and searches the current repo only.

## Source of truth (no date given)

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
- **Bullets only.** No header, no date, no time estimate, no closing summary, no commentary — except the fallback notice above and dated mode's `⚠️` lines, when they apply.
- Order the bullets roughly chronologically.

## Delivery

1. Print the list in the chat as a plain code-free markdown list.
2. Copy the same text to the clipboard:
   ```
   printf '%s\n' '<the bullet list>' | xclip -selection clipboard
   ```
   Use a heredoc if quoting gets awkward. If `xclip` is unavailable, print the list anyway and note in one line that the clipboard copy failed.

Confirm in a single short line that it was copied. Nothing else.
