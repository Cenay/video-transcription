---
name: ship
description: Commit all changes and push to present branch. Generates a conventional commit message from the diff.
---

You are committing and pushing the current changes. Follow these steps exactly.

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
5. Create the commit using a HEREDOC:

```bash
git commit -m "$(cat <<'EOF'
{commit message}
EOF
)"
```

## Step 3: Push

Push to cuttent branch immediately:

```bash
git push origin HEAD
```

## Step 4: Confirm

Show the user:
- The commit hash and message
- Which branch was pushed
- Number of files changed
