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

Two renderings of one line are also not a disappearance — see `_words()`: a line
survives if some single new line carries every word of it, ignoring punctuation
and the stamp label (`_Last updated` -> `- _Prior:` is a demotion, not a loss).

Deleting a history file, or moving one out of docs/history/, is every line of it
leaving at once and fails in BOTH modes — see `staged_files()` for the day the
staged scan could not see either.

Usage:
    check-append-only.py --staged              # every staged docs/history/ file
    check-append-only.py FILE [FILE...]        # named files, worktree vs HEAD
    ALLOW_HISTORY_SHRINK=1 ... --staged        # deliberate removal, see below

⚠️ THE ESCAPE HATCH IS DELIBERATE AND MUST STAY LOUD. A merge that collapses two
renderings of one record genuinely needs to remove a line. Making that impossible
would push the work outside the guard entirely, which is worse. So it is possible,
never accidental, and it prints what it let through.
"""
import collections
import re
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
    """Every staged path under docs/history/, INCLUDING deletions.

    ⛔ This used to filter on `--diff-filter=ACM` — added, copied, modified — so a
    deleted history file, or one renamed out of the folder, was simply not in the
    list, and the guard printed "nothing staged under docs/history/" over it.
    Measured 2026-09-10: `git rm docs/history/X-stamp-history.md`, staged, exit 0
    with that message; and a `git mv` out of docs/history/ went through the same
    gap the same day. Named-file mode has always called a missing file a failure
    ("deletion of a history file is always a failure"), so the two modes disagreed.

    ★ `--no-renames`, deliberately: a rename is reported as a deletion plus an
    addition, so a file LEAVING docs/history/ is a deletion of a history file —
    which is what it is. The override is the loud, deliberate path for a move.
    """
    out = git(["diff", "--cached", "--name-only", "--no-renames"], cwd=root) or ""
    return [f for f in out.split("\n") if f.startswith(HISTORY_DIR) and f.endswith(".md")]


def content_head(path, root):
    return git(["show", f"HEAD:{path}"], cwd=root)


def content_staged(path, root):
    return git(["show", f":{path}"], cwd=root)


# ★ A STAMP'S LABEL IS MARKUP, NOT CONTENT. `_Last updated X_` and `- _Prior: X_`
# are one record in two renderings: stamp-doc.py demotes the current stamp to a
# `_Prior:` row every time it stamps a doc, and that is the ONLY text change it
# ever makes to a stamp. Measured 2026-09-10 in LSP/Staff_Form: stamping
# `docs/history/NEXT_STEPS-archive.md` (an archive that lives under history/ and
# so is guarded) tripped this check on exactly the two words "Last updated", and
# the commit went through on ALLOW_HISTORY_SHRINK=1. An override taken for
# healthy work is how an override becomes a reflex, so the label is stripped
# before the words are counted. Everything after the label — timestamp, zone,
# transcript, summary — is still content and still may not leave.
STAMP_LABEL_RE = re.compile(r"^-?\s*_(?:Last updated|Prior:)\s+")


def _words(line):
    """Word multiset of a line, for the content comparison in check().

    Punctuation and markup are ignored on purpose: `DEC-132` and `[DEC-132]`
    must compare equal, because bracketing an id adds no words and removes none.
    The stamp label (`_Last updated` / `_Prior:`) is markup too — see above.
    """
    return collections.Counter(re.findall(r"[A-Za-z0-9]+", STAMP_LABEL_RE.sub("", line.strip())))


DELETED = "DELETED"


def check(path, root, staged):
    """Returns (ok, lost_lines, note). note is set when the check could NOT run,
    or is DELETED when the whole file is gone — every line of it is then lost."""
    before = content_head(path, root)
    if before is None:
        return True, [], "new file (no HEAD version) — nothing to compare"
    if staged:
        after = content_staged(path, root)
        if after is None:
            # The path is in the staged list but has no index blob: a staged
            # deletion (or the old half of a rename, under --no-renames).
            return False, sorted(lines_of(before)), DELETED
    else:
        p = root / path
        if not p.exists():
            return False, sorted(lines_of(before)), DELETED
        after = p.read_text(encoding="utf-8")
    gone = lines_of(before) - lines_of(after)
    if not gone:
        return True, [], None

    # ⛔ A LINE MAY BE ENRICHED IN PLACE. CONTENT MAY NEVER LEAVE.
    #
    # The invariant was "no line leaves", compared as line SETS. That is too
    # strict, and it collided with a tool doing its job: link-doc-refs.py
    # brackets a bare id in prose (`DEC-132` -> `[DEC-132]`), which rewrites the
    # line. To a set comparison that is one line leaving and one arriving, so
    # every legitimate linking run looked like a violation.
    #
    # ✅ Measured 2026-08-25: three such lines across two history files, and a
    # word-level check showed ZERO words lost -- the only change was the
    # brackets. Ruled by Cenay: compare CONTENT.
    #
    # ★ A removed line is forgiven only if some SINGLE surviving line contains
    # every word of it. Not "the words are somewhere in the file" -- that would
    # let a line be shredded across others and call it survival.
    #
    # ⚠️ What this deliberately still catches: a line whose text was shortened,
    # reworded, or dropped outright. Losing one word is losing content.
    arrived = lines_of(after) - lines_of(before)
    arrived_words = [(a, _words(a)) for a in arrived]
    lost = []
    for g in sorted(gone):
        gw = _words(g)
        if not any(not (gw - aw) for _, aw in arrived_words):
            lost.append(g)
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
        # ⛔ THREE DIFFERENT SITUATIONS, AND THEY MUST NOT SHARE A MESSAGE.
        # This printed "no docs/history/ files to check" for all of them, so a
        # run that checked NOTHING AT ALL -- because no target was given --
        # looked identical to a clean run. Measured 2026-08-25: that message
        # was misread twice in one session, once by the author of this comment,
        # who concluded the guard "degrades gracefully" from a no-op invocation.
        # ★ A checker must say what it did NOT check.
        if not use_staged:
            print("⚠ check-append-only: no files named and --staged not given "
                  "— CHECK DID NOT RUN. Pass --staged (what the hook does), "
                  "or name files to check.")
            return 0
        if not (root / HISTORY_DIR).is_dir():
            print(f"✓ check-append-only: this repo has no {HISTORY_DIR} — "
                  f"nothing for this guard to protect")
            return 0
        print(f"✓ check-append-only: nothing staged under {HISTORY_DIR} "
              f"— no history file is being changed by this commit")
        return 0

    override = os.environ.get(OVERRIDE) == "1"
    failures, notes, checked = [], [], []

    deleted = set()
    for t in targets:
        ok, lost, note = check(t, root, use_staged)
        if note == DELETED:
            deleted.add(t)
        elif note:
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
        if t in deleted:
            print(f"\n  {t} — the WHOLE FILE is deleted or moved out of "
                  f"{HISTORY_DIR} ({len(lost)} line(s)):")
        else:
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
    print(f"  If the removal IS deliberate (merging two renderings of one record,")
    print(f"  or moving a living doc OUT of {HISTORY_DIR} because it is not a")
    print(f"  record), re-run with {OVERRIDE}=1 and say so in the commit message.")
    print("")
    return 1


if __name__ == "__main__":
    sys.exit(main())
