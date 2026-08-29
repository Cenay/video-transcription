#!/usr/bin/env bash
# Stop hook — warn when a session commits work but never updates docs/CURRENT_STATUS.md.
#
# Why this exists: on 2026-07-27 a session changed the live Report Generator, wrote
# 173 lines of session notes, and updated no status doc. The gap surfaced a week later
# and had to be reconstructed from a commit message. The rule "update CURRENT_STATUS"
# lived only in a human's memory, so it broke. Mechanical invariant -> script + hook.
#
# Exit 0  = nothing to say (no commits this session, or CURRENT_STATUS was updated)
# Exit 2  = blocks the stop once, feeding the message back so the docs get written
#
# Fires AT MOST ONCE per session. A gate that nags every turn gets disabled by reflex.

set -uo pipefail

cd "${CLAUDE_PROJECT_DIR:-.}" 2>/dev/null || exit 0
git rev-parse --git-dir >/dev/null 2>&1 || exit 0

SID="${CLAUDE_SESSION_ID:-unknown}"
BASE_FILE="$(git rev-parse --git-dir)/claude-session-head-${SID}"
FIRED_FILE="$(git rev-parse --git-dir)/claude-status-warned-${SID}"
# ⛔ Resolve the docs root — NEVER hard-code `docs/`. Projects under
# /mnt/k/_Sites keep their doc set in `.cloaked/docs/`, and the hard-coded
# path made this hook FIRE ON HEALTHY WORK there: it looked for
# `docs/CURRENT_STATUS.md`, never saw the `.cloaked/docs/` one the session had
# just written, and warned anyway. ✅ Measured 2026-08-29 with a mutation
# control — pre-fix exits 2 on a `.cloaked` repo whose session DID update its
# status doc; post-fix exits 0, and still exits 2 when it genuinely did not.
# ⚠️ NOT a silent no-op, which is what this comment claimed until the control
# was actually run: a false NEGATIVE would merely be useless, but a guard that
# fires on correct work is the kind people learn to bypass.
# doc_root.py is the single source of the path rule — it checks what EXISTS
# first (by doc-set markers), then falls back to the convention.
DOC_ROOT="$(python3 "$(dirname "$0")/doc_root.py" . 2>/dev/null)"
[ -n "$DOC_ROOT" ] || DOC_ROOT="docs"        # never break the session on a bad resolve
STATUS_DOC="$DOC_ROOT/CURRENT_STATUS.md"
# Escape dots for the grep below — `.cloaked/docs` is a real value and an
# unescaped `.` matches any character.
DOC_ROOT_RE="$(printf '%s' "$DOC_ROOT" | sed 's/[.[\*^$]/\\&/g')"

# Already warned this session — say nothing further.
[ -f "$FIRED_FILE" ] && exit 0

HEAD_NOW="$(git rev-parse HEAD 2>/dev/null)" || exit 0

# No recorded baseline yet: record it and stay quiet. First stop of a session
# establishes the mark; we can only judge a session that has one.
if [ ! -f "$BASE_FILE" ]; then
    printf '%s\n' "$HEAD_NOW" > "$BASE_FILE"
    exit 0
fi

BASE="$(cat "$BASE_FILE")"
[ "$BASE" = "$HEAD_NOW" ] && exit 0          # no commits since the mark

# Commits happened. Did any of them touch the status doc?
if git diff --name-only "$BASE..$HEAD_NOW" -- "$STATUS_DOC" | grep -q .; then
    printf '%s\n' "$HEAD_NOW" > "$BASE_FILE"
    exit 0
fi

# Ignore doc-only sessions that legitimately touched other docs but not this one
# (e.g. a pure runbook edit). Only warn when something outside docs/ changed.
if ! git diff --name-only "$BASE..$HEAD_NOW" | grep -qv "^${DOC_ROOT_RE}/"; then
    printf '%s\n' "$HEAD_NOW" > "$BASE_FILE"
    exit 0
fi

COUNT="$(git rev-list --count "$BASE..$HEAD_NOW")"
touch "$FIRED_FILE"

cat >&2 <<EOF
${STATUS_DOC} was not updated, but this session made ${COUNT} commit(s)
touching files outside ${DOC_ROOT}/.

Commits:
$(git log --format='  %h %s' "$BASE..$HEAD_NOW")

This is the exact gap that lost the 2026-07-27 Report Generator session — see
CLAUDE.md, "The rule that this project kept breaking". Add a session record to
${STATUS_DOC} (or run /checkpoint), then stop.

If a status entry genuinely does not apply here, say so and stop again — this
warning fires only once per session.
EOF
exit 2
