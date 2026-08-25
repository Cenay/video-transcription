#!/usr/bin/env python3
"""Append-only guard for permanent-record files.

★ THE INVARIANT: a file under docs/history/ may GROW without limit; no line may
ever LEAVE it. That is [DEC-052], ruled by Cenay 2026-08-21 after a stamp-history
file was found holding 4 entries where git had seen 20.

⛔ WHY THIS EXISTS RATHER THAN A NARROWER CHECK. On 2026-08-24 a measurement found
392 unique lines missing from 13 history files across three repos, earliest
2026-08-11. Every guard in place at the time passed, because each was narrower
than the claim it appeared to make:

  * `stamp-doc.py --check` audits STAMPS, keyed on date+time+TZ. The lost lines
    were `link-doc-refs` definitions. No stamp was lost, so it printed ✓.
  * `stamp-doc.py --restore` rebuilt each file as `head + stamp rows`, dropping
    everything else -- and printed "✓ nothing to restore" while deleting 16 lines.
  * The pre-commit stamp check filtered staged paths with `^docs/[A-Z_-]+\\.md$`,
    which does not match `docs/history/<DOC>-stamp-history.md` at all.

★ So this check knows nothing about stamps, links, or any other record type. It
compares LINE SETS. Any writer that removes a line fails it -- including tools
nobody has written yet. That is the whole point: it does not need to know where
the next bug will be.

Set semantics, deliberately: reordering passes (the history files are sorted
newest-first and get re-sorted), and removing a DUPLICATE copy passes (a line is
"present" if it appears at least once). Only genuine disappearance fails.

Usage:
    check-append-only.py --staged              # every staged docs/history/ file
    check-append-only.py FILE [FILE...]        # named files, worktree vs HEAD
    ALLOW_HISTORY_SHRINK=1 ... --staged        # deliberate removal, see below

⚠️ THE ESCAPE HATCH IS DELIBERATE AND MUST STAY LOUD. A merge that collapses two
renderings of one record genuinely needs to remove a line. Making that impossible
would push the work outside the guard entirely, which is worse. So it is possible,
never accidental, and it prints what it let through.
"""
import os
import subprocess
import sys
from pathlib import Path

HISTORY_DIR = "docs/history/"
OVERRIDE = "ALLOW_HISTORY_SHRINK"


def git(args, cwd=None):
    r = subprocess.run(["git"] + args, cwd=cwd, capture_output=True, text=True)
    return r.stdout if r.returncode == 0 else None


def repo_root():
    out = git(["rev-parse", "--show-toplevel"])
    return Path(out.strip()) if out else None


def lines_of(text):
    """Non-blank, stripped. Blank-line churn is formatting, not content."""
    return {l.strip() for l in text.split("\n") if l.strip()}


def staged_files(root):
    out = git(["diff", "--cached", "--name-only", "--diff-filter=ACM"], cwd=root) or ""
    return [f for f in out.split("\n") if f.startswith(HISTORY_DIR) and f.endswith(".md")]


def content_head(path, root):
    return git(["show", f"HEAD:{path}"], cwd=root)


def content_staged(path, root):
    return git(["show", f":{path}"], cwd=root)


def check(path, root, staged):
    """Returns (ok, lost_lines, note). note is set when the check could NOT run."""
    before = content_head(path, root)
    if before is None:
        return True, [], "new file (no HEAD version) — nothing to compare"
    if staged:
        after = content_staged(path, root)
        if after is None:
            return True, [], "not staged — skipped"
    else:
        p = root / path
        if not p.exists():
            return False, [], None  # deletion of a history file is always a failure
        after = p.read_text(encoding="utf-8")
    lost = sorted(lines_of(before) - lines_of(after))
    return (not lost), lost, None


def main():
    argv = sys.argv[1:]
    use_staged = "--staged" in argv
    named = [a for a in argv if not a.startswith("-")]

    root = repo_root()
    if root is None:
        # ⛔ Never fail open with a clean tick. Say the check did not run.
        print("⚠ check-append-only: not a git repo — CHECK DID NOT RUN")
        return 0

    targets = staged_files(root) if use_staged else named
    if not targets:
        print(f"✓ check-append-only: no {HISTORY_DIR} files to check")
        return 0

    override = os.environ.get(OVERRIDE) == "1"
    failures, notes, checked = [], [], []

    for t in targets:
        ok, lost, note = check(t, root, use_staged)
        if note:
            notes.append(f"{t}: {note}")
            continue
        checked.append(t)
        if not ok:
            failures.append((t, lost))

    for n in notes:
        print(f"ⓘ {n}")

    if not failures:
        print(f"✓ check-append-only: {len(checked)} history file(s) checked, no line removed")
        # ★ A checker must name what it did NOT check.
        if notes:
            print(f"  ⚠ {len(notes)} file(s) NOT compared — see the ⓘ lines above")
        return 0

    total = sum(len(l) for _, l in failures)
    verb = "ALLOWED" if override else "BLOCKED"
    print("")
    print(f"{'⚠' if override else '✗'} APPEND-ONLY VIOLATION — {total} line(s) would leave "
          f"{len(failures)} history file(s) [{verb}]")
    for t, lost in failures:
        print(f"\n  {t} — {len(lost)} line(s) removed:")
        for l in lost[:5]:
            print(f"    - {l[:150]}")
        if len(lost) > 5:
            print(f"    … and {len(lost) - 5} more")
    print("")
    if override:
        print(f"  {OVERRIDE}=1 was set, so this is being allowed through deliberately.")
        return 0
    print("  A history file is append-only: it may grow without limit, but nothing")
    print("  may leave it. Recover the lines with:")
    print(f"    git show HEAD:<file>")
    print("")
    print(f"  If the removal IS deliberate (e.g. merging two renderings of one")
    print(f"  record), re-run with {OVERRIDE}=1 and say so in the commit message.")
    print("")
    return 1


if __name__ == "__main__":
    sys.exit(main())
