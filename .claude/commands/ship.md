---
name: ship
description: Commit the relevant changes and push to the present branch. Generates a conventional commit message from the diff, with session-traceability trailers. Does not write documentation.
---

You are committing and pushing the current changes. Follow these steps exactly.

**This command does not write documentation.** No status blocks, no ledger entries, no
stamps. The standard doc set has a single writer who maintains it separately; if the work
settled something that belongs in the docs, **say so in your report** (step 4) rather than
writing it yourself. `/wrap-up` is the same routine with the diff summarized back to you
first — neither one touches docs.

## Step 1: Assess

Run these commands in parallel:
- `git status` (never use `-uall`)
- `git diff --stat` (staged + unstaged)
- `git log --oneline -5` (recent commit style)

If there are no changes to commit, say so and stop.

## Step 2: Stage + Commit

1. Review the diff to understand what changed
2. Draft a commit message following the project's existing style (check git log output). Use conventional commit format if the project uses it (`feat:`, `fix:`, `docs:`, `refactor:`, `chore:`, `content:`)
3. Stage relevant files by name — do NOT use `git add -A` or `git add .`
4. Do NOT stage files that look like secrets (`.env`, credentials, keys)
5. Append the **session-traceability trailers** (see below) to the message body
6. Create the commit using a HEREDOC:

```bash
git commit -m "$(cat <<'EOF'
{commit message}

Co-Authored-By: Claude <model-that-actually-ran> <noreply@anthropic.com>
Claude-Transcript: ~/.claude/projects/<project-slug>/<session-id>.jsonl
Claude-Session: <claude.ai/code session URL>
EOF
)"
```

### The trailers — every commit carries them

This is a global rule, not a nicety. A doc records *what* was decided; the trailer is the
only way to recover *why*. It exists because on 2026-07-11 a folder-consolidation decision
was recoverable only because one commit carried one — at the time, 5 of 24 commits in that
repo had one.

- **`Co-Authored-By`** names **the model actually running this session** (with its context
  variant if it has one) — substitute it, **never copy a version from this template**.
- **`Claude-Transcript` is mandatory.** It is always available: the session's `.jsonl`
  filename under `~/.claude/projects/<project-slug>/`. Derive the slug from the cwd —
  `/mnt/k/Code/TRFA/fran-dash` → `-mnt-k-Code-TRFA-fran-dash`.
- **`Claude-Session` is best-effort** — include the line when the URL is known, **omit the
  line entirely** when it isn't.
- **Never invent or guess either value.** A wrong pointer is worse than no pointer.

## Step 3: Push

Push to the current branch immediately:

```bash
git push origin HEAD
```

## Step 4: Confirm

Show the user:
- The commit hash and message
- Which branch was pushed
- Number of files changed

Then, in one line if either applies:
- **Doc drift** — the work made something in the docs untrue (a command, a status, a
  closed question still listed as open). Name the file. Do not fix it.
- **An unrecorded decision** — the work settled something that isn't in `DECISIONS.md`.
  Say what was settled.

Both belong to the doc set's single writer. Flagging them is how they reach that person —
and, if you are not that person, what to carry into your **PR body**.

## No `origin` remote?

Say so and stop. Do not invent one, and do not add a remote.
