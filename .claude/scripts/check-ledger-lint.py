#!/usr/bin/env python3
"""Block a commit that makes the decision ledger lint WORSE than HEAD's.

Run:  python3 scripts/check-ledger-lint.py --staged [--ledger PATH]
Exit: 0 no regression (or nothing to check) · 1 a new failure · 2 setup problem

⛔ WHY NO-REGRESSION AND NOT "MUST LINT CLEAN". Measured 2026-08-28 across
every repo carrying a ledger: fran-dash, claude-personal-toolkit,
trfaapi.com and video-transcription all lint clean — and **Staff_Form does
not**, on a real pre-existing defect (its DEC-015 carries a `**Status:**`
line that sits outside the parsed field run). An absolute check would block
every commit in that repo until an unrelated ledger is repaired.

★ That is the exact failure this project has already paid for: a guard was
installed into 9 repos, proved green, and only then measured — it would have
blocked 37 healthy docs. The ruling that followed was carve-out first, then
port. *A guard that fires on healthy work teaches you to bypass guards.*

★ So the invariant is a DELTA, the same shape as check-append-only.py: the
staged ledger may not fail any check that HEAD's version passes. A ledger
that was already broken stays committable; breaking a passing check does not.
A pre-existing failure is REPORTED on every run, so it stays visible rather
than becoming the new normal.

⚠️ NOT CHECKED, and it matters: whether the ledger is CORRECT. This compares
two lint runs. A ledger that lints clean can still say false things, and a
carried failure is not thereby acceptable — only not newly caused here.
"""

import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
FAIL_RE = re.compile(r"^\s*✗\s*(\S+)")


def failing_checks(lint, ledger):
    """The set of check ids that FAIL for `ledger`, or None if lint can't run."""
    r = subprocess.run([sys.executable, str(lint), "--ledger", str(ledger)],
                       capture_output=True, text=True)
    out = r.stdout + r.stderr
    if r.returncode not in (0, 1):
        return None, out
    return {m.group(1) for m in (FAIL_RE.match(l) for l in out.split("\n")) if m}, out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--staged", action="store_true",
                    help="compare the staged ledger against HEAD's version")
    ap.add_argument("--ledger", default="docs/DECISIONS.md")
    args = ap.parse_args()

    rel = args.ledger
    staged = subprocess.run(["git", "diff", "--cached", "--name-only",
                             "--diff-filter=ACM"], capture_output=True, text=True).stdout.split()
    if args.staged and rel not in staged:
        return 0                                   # nothing to check, silently

    lint = HERE / "ledger-lint.py"
    if not lint.is_file():
        # ⛔ Never fail open with a clean tick — say the check did not run.
        print(f"⚠ ledger-lint.py not found beside {HERE} — the ledger was NOT checked")
        return 2

    with tempfile.TemporaryDirectory() as d:
        # HEAD's version. A ledger that is NEW in this commit has no baseline,
        # so every failure in it is newly introduced — an empty baseline is the
        # correct reading, not a reason to skip.
        head = subprocess.run(["git", "show", f"HEAD:{rel}"],
                              capture_output=True, text=True)
        base = Path(d) / "head.md"
        base.write_text(head.stdout if head.returncode == 0 else "", encoding="utf-8")

        # The STAGED content, not the working tree's — they differ whenever
        # something is left unstaged, and the commit is made of the index.
        blob = subprocess.run(["git", "show", f":{rel}"], capture_output=True, text=True)
        if blob.returncode != 0:
            print(f"⚠ cannot read the staged {rel} — the ledger was NOT checked")
            return 2
        cur = Path(d) / "staged.md"
        cur.write_text(blob.stdout, encoding="utf-8")

        was, _ = failing_checks(lint, base)
        now, now_out = failing_checks(lint, cur)

    if now is None:
        print(f"⚠ ledger-lint.py errored on the staged {rel} — NOT checked")
        return 2
    if was is None:
        was = set()

    new = sorted(now - was)
    carried = sorted(now & was)

    if carried:
        print(f"note: {rel} already fails {', '.join(carried)} in HEAD — carried, not caused here")
    if not new:
        if not carried:
            print(f"✓ ledger-lint: {rel} introduces no new failure")
        return 0

    print(f"⛔ {rel} newly fails {len(new)} check(s) that HEAD passes: {', '.join(new)}")
    for line in now_out.split("\n"):
        m = FAIL_RE.match(line)
        if m and m.group(1) in new:
            print(f"  {line.strip()}")
    print("  Fix the ledger, or restore it, before committing.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
