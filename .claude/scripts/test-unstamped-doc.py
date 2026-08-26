#!/usr/bin/env python3
"""Suite for the unstamped-doc carve-out in `stamp-doc.py --check` ([DEC-054]).

Run:  python3 scripts/test-unstamped-doc.py           (exit 0 = pass)
      python3 scripts/test-unstamped-doc.py --prove    (the run that counts)

★ THE RULE: a doc that never carried a stamp is not a broken doc. `--check`
hunts a CORRUPTED or LOST chain; a doc with no chain has none to break.

⛔ THE HOLE TO WATCH: forgiving a stamp that was just DELETED. ★ It is `lost`
— the git-backed audit — that closes it, NOT the `not ever` half of the
condition. Both deletion cases here stay caught when `not ever` is mutated
away, and `--prove` reports that rather than hiding it.

⚠️ THAT CORRECTION CAME FROM THE MUTANT, NOT FROM REVIEW. Two earlier drafts of
this suite asserted `not ever` was load-bearing and were wrong; each time the
case written to prove it stayed green under the naive mutant. The deletion
cases below are kept as genuine regression cover — they must never go quiet —
but they do not measure that condition, and the suite now says so.

✅ Why the carve-out was needed at all, measured 2026-08-25: ported verbatim
into the fleet's pre-commit hooks, this check would have blocked commits
touching 37 tracked docs across 6 repos, and every one of the 37 failed for
"no stamp at all" — not one corruption, not one loss.
"""

import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
STAMP = HERE / "stamp-doc.py"

S1 = ("_Last updated 2026-08-20 10:00 MST by an AI session · transcript: "
      "`dddddddd-0000-0000-0000-000000000001` — the current one._")


def sh(cwd, *a):
    return subprocess.run(a, cwd=cwd, capture_output=True, text=True)


def repo(body, history=None):
    d = Path(tempfile.mkdtemp())
    (d / "docs" / "history").mkdir(parents=True)
    doc = d / "docs" / "README.md"
    doc.write_text(body, encoding="utf-8")
    if history is not None:
        (d / "docs" / "history" / "README-stamp-history.md").write_text(
            "# Stamp history — README.md\n\nNewest first.\n\n---\n\n" + history + "\n",
            encoding="utf-8")
    sh(d, "git", "init", "-q", ".")
    sh(d, "git", "config", "user.email", "t@t"); sh(d, "git", "config", "user.name", "t")
    sh(d, "git", "add", "-A"); sh(d, "git", "commit", "-qm", "first")
    return d, doc


def check(doc):
    r = subprocess.run([sys.executable, str(STAMP), str(doc), "--check"],
                       capture_output=True, text=True)
    return r.returncode, r.stdout + r.stderr


PRIOR = ("_Prior: 2026-08-19 10:00 MST by an AI session · transcript: "
         "`dddddddd-0000-0000-0000-000000000002` — an older one._")

PLAIN = "# Readme\n\nOrdinary documentation. No stamp, never had one.\n"
STAMPED = f"# Readme\n\n{S1}\n\nBody.\n"

CASES = []


def case(fn):
    CASES.append(fn); return fn


@case
def an_unstamped_doc_passes():
    _, doc = repo(PLAIN)
    rc, out = check(doc)
    assert rc == 0, f"blocked an ordinary unstamped doc: {out}"


@case
def and_it_says_so_rather_than_claiming_well_formed():
    """⚠️ A pass must not read as 'the stamp block is fine' — there isn't one."""
    _, doc = repo(PLAIN)
    _, out = check(doc)
    assert "NOT STAMPED" in out, f"did not name what it skipped: {out}"
    assert "well-formed" not in out, f"claimed a stamp block it never saw: {out}"


@case
def a_deleted_stamp_is_still_caught():
    """⛔ THE ONE THAT MATTERS. Commit a stamp, then remove it."""
    d, doc = repo(STAMPED)
    doc.write_text(PLAIN, encoding="utf-8")
    rc, out = check(doc)
    assert rc != 0, f"a DELETED stamp sailed through — the carve-out is a hole: {out}"
    assert "ABSENT" in out or "2026-08-20" in out, f"caught, but not as a loss: {out}"


@case
def a_stamped_doc_is_still_checked():
    _, doc = repo(STAMPED)
    rc, out = check(doc)
    assert rc == 0, f"a healthy stamped doc was refused: {out}"
    assert "well-formed" in out, f"did not audit the stamp block: {out}"


@case
def a_malformed_stamp_is_still_caught():
    """A doc that HAS a stamp and mangles it must still fail.

    ⚠️ The fixture is the LEGACY BLOB — two stamps concatenated onto one line,
    the shape of the 5,556-character line this tool was written to end. An
    earlier draft of this case merely stripped the stamp's trailing `_` and
    PASSED: the parser has a delimiter repair for exactly that, so it was
    testing a self-healing case and calling it malformed.
    """
    bad = f"# Readme\n\n{S1} {PRIOR}\n\nBody.\n"
    _, doc = repo(bad)
    rc, out = check(doc)
    assert rc != 0, f"a malformed stamp passed: {out}"


@case
def a_stamp_deleted_from_the_doc_is_caught_when_history_still_has_it():
    """Delete the doc's own stamp while its history file still carries an older one.

    ⚠️ Written believing `lost` would be empty here and that only `not ever`
    would catch it. Measured: WRONG. `ever` is built from git across BOTH the
    doc and its history file, so the doc's committed stamp is missing from
    `now` and lands in `lost` anyway. Kept as regression cover for a real
    scenario; it does not discriminate the condition it was written for.
    """
    _, doc = repo(STAMPED, history=f"- {PRIOR}")
    # the doc keeps its history file; only the doc's own stamp goes
    doc.write_text(PLAIN, encoding="utf-8")
    rc, out = check(doc)
    assert rc != 0, f"a stamp deleted from the doc passed silently: {out}"


def prove():
    """Revert the guard-rail half and assert the deletion case goes red."""
    src = STAMP.read_text(encoding="utf-8")

    # THE NAIVE CARVE-OUT: forgive whenever the doc has no stamp right now,
    # without asking git whether one was ever committed.
    naive = src.replace(
        "        never_stamped = not ever and not AUDIT_DATE_RE.search(original)",
        "        never_stamped = not AUDIT_DATE_RE.search(original)")
    assert naive != src, "could not locate the never_stamped condition"

    # THE CARVE-OUT REMOVED ENTIRELY: back to blocking every unstamped doc.
    gone = src.replace(
        '            problems = [p for p in problems if not p.startswith("no `_Last updated")]',
        '            pass')
    assert gone != src, "could not locate the forgiveness line"

    expected = {
        "carve-out removed": (gone, {"an_unstamped_doc_passes",
                                     "and_it_says_so_rather_than_claiming_well_formed"}),
    }
    problems = []
    try:
        for name, (mutant, must_fail) in expected.items():
            STAMP.write_text(mutant, encoding="utf-8")
            reds = set()
            for fn in CASES:
                try:
                    fn()
                except Exception:
                    reds.add(fn.__name__)
            print(f"  {name}: {len(reds)} red — {sorted(reds)}")
            for m in sorted(must_fail - reds):
                problems.append(f"    ⛔ {m} stayed GREEN under '{name}'")
            if "a_stamped_doc_is_still_checked" in reds:
                problems.append(f"    ⛔ the control went red under '{name}'")
        # ⚠️ Reported, never asserted. `not ever` is belt-and-braces: `lost`
        # catches every deletion on its own, so no case here can go red when
        # this half is mutated away. Silence would read as coverage.
        STAMP.write_text(naive, encoding="utf-8")
        reds = {fn.__name__ for fn in CASES if _red(fn)}
        print(f"  naive carve-out (no git consultation): {len(reds)} red — {sorted(reds)}")
        if not reds:
            print("    ⓘ NOT COVERED, and correctly so — every deletion case stays "
                  "caught by `lost`, the git-backed audit. `not ever` is a second, "
                  "redundant condition; nothing here depends on it.")
        else:
            problems.append("    ⛔ a case went red under the naive carve-out — "
                            "`not ever` IS load-bearing after all, and both the "
                            "source comment and this note must be corrected")
    finally:
        STAMP.write_text(src, encoding="utf-8")

    if problems:
        print("\n".join(problems)); return 1
    print("✅ removing the carve-out blocks healthy docs; the deletion cases stay caught")
    return 0


def _red(fn):
    try:
        fn(); return False
    except Exception:
        return True


def main():
    if "--prove" in sys.argv:
        return prove()
    failed = 0
    for fn in CASES:
        try:
            fn(); print(f"  ✓ {fn.__name__}")
        except Exception as exc:
            failed += 1; print(f"  ✗ {fn.__name__}: {exc}")
    if failed:
        print(f"\n{failed} of {len(CASES)} failed"); return 1
    print(f"\nall {len(CASES)} passed — ⚠️ now run --prove")
    return 0


if __name__ == "__main__":
    sys.exit(main())
