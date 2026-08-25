#!/usr/bin/env python3
"""Regression suite for `stamp-doc.py --check`'s history audit and `--restore`.

Run:  python3 scripts/test-stamp-audit.py           (exit 0 = pass)
      python3 scripts/test-stamp-audit.py --prove    (the run that counts)

⛔ WHAT THIS GUARDS, AND WHY NOTHING ELSE COULD. Every other check in
`stamp-doc.py` is INTERNAL CONSISTENCY — is the fold well-formed, is the current
stamp closed, is there a shadow chain. A stamp that was silently dropped leaves
a perfectly valid file behind, so a clean lint says nothing about it.

✅ Measured 2026-08-21 across the fleet: **130 stamps had been lost** — 39 in the
toolkit, 83 in fran-dash, 8 in trfaapi.com — at a rate of roughly one per commit
through 2026-07, the era before this script, when a session hand-prepended its
stamp and REPLACED the previous one instead of accumulating it. Every affected
file linted clean the whole time.

★ git is the only witness, so the audit asks git.

⚠️ THE SECOND DEFECT, found by running the restore rather than reasoning about
it: 8 of the 130 lived in the LEGACY ONE-LINE BLOB — several stamps concatenated
onto one line. A line-anchored regex finds one and misses the rest. The audit
saw them (it scans for dates anywhere) while `--restore` could not recover them,
so the tool reported a loss it was unable to fix. Fixed by reusing `split_blob`,
the parser that already knows that shape.
"""

import re
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
STAMP = HERE / "stamp-doc.py"

DOC = """# Current Status

_Last updated 2026-08-20 10:00 MST by an AI session · transcript: `aaaaaaaa-0000-0000-0000-000000000001` — the third._

<details>
<summary>📜 <strong>Stamp history</strong> — the 3 previous updates (older ones: <code>history/CURRENT_STATUS-stamp-history.md</code>)</summary>

- _Prior: 2026-08-19 10:00 MST by an AI session · transcript: `aaaaaaaa-0000-0000-0000-000000000002` — the second._

</details>

Body text.
"""

# A stamp that only ever existed in an earlier commit.
OLD = ("_Last updated 2026-08-18 10:00 MST by an AI session · transcript: "
       "`aaaaaaaa-0000-0000-0000-000000000003` — the first._")

# Two stamps on ONE line — the legacy blob that a line-anchored regex mangles.
BLOB = (OLD + " _Prior: 2026-08-17 09:00 MST by an AI session · transcript: "
        "`aaaaaaaa-0000-0000-0000-000000000004` — the zeroth._")


def repo(first_body):
    """A throwaway git repo whose docs/CURRENT_STATUS.md has two revisions."""
    d = Path(tempfile.mkdtemp())
    (d / "docs").mkdir()
    doc = d / "docs" / "CURRENT_STATUS.md"
    run = lambda *a: subprocess.run(["git"] + list(a), cwd=d, capture_output=True)
    run("init", "-q")
    run("config", "user.email", "t@t"); run("config", "user.name", "t")
    doc.write_text(first_body, encoding="utf-8")
    run("add", "-A"); run("commit", "-qm", "first")
    doc.write_text(DOC, encoding="utf-8")          # the old stamp disappears here
    run("add", "-A"); run("commit", "-qm", "second")
    return d, doc


def check(doc):
    r = subprocess.run([sys.executable, str(STAMP), str(doc), "--check"],
                       capture_output=True, text=True)
    return r.returncode, r.stdout + r.stderr


CASES = []


def case(fn):
    CASES.append(fn)
    return fn


@case
def a_lost_stamp_is_detected():
    """⛔ THE POINT. A stamp in git but not in the files must FAIL --check."""
    _, doc = repo(DOC.replace("Body text.", OLD + "\n\nBody text."))
    rc, out = check(doc)
    assert rc == 1, f"a lost stamp did not fail --check:\n{out}"
    assert "2026-08-18 10:00 MST" in out, f"the lost stamp was not named:\n{out}"


@case
def a_lost_stamp_inside_a_legacy_blob_is_detected():
    """⚠️ Two stamps on one line — the shape that defeated the first --restore."""
    _, doc = repo(DOC.replace("Body text.", BLOB + "\n\nBody text."))
    rc, out = check(doc)
    assert rc == 1, f"a blob-era loss did not fail:\n{out}"
    assert "2026-08-17 09:00 MST" in out, f"the blob's second stamp was missed:\n{out}"


@case
def restore_recovers_a_lost_stamp():
    _, doc = repo(DOC.replace("Body text.", OLD + "\n\nBody text."))
    subprocess.run([sys.executable, str(STAMP), str(doc), "--restore"],
                   capture_output=True, text=True)
    rc, out = check(doc)
    assert rc == 0, f"--check still fails after --restore:\n{out}"
    hist = doc.parent / "history" / "CURRENT_STATUS-stamp-history.md"
    assert "2026-08-18 10:00 MST" in hist.read_text(encoding="utf-8")


@case
def restore_recovers_from_a_legacy_blob():
    """★ A blob must be recovered as SEPARATE stamps, not re-imported whole.

    ⛔ THIS ASSERTION WAS WRONG WHEN FIRST WRITTEN, and the mutation run caught
    it. It only checked that `--check` passes afterwards — which it does even
    with the blob handling removed, because the whole blob line gets copied
    wholesale into the history file and the second stamp rides along INSIDE it.
    The audit scans for dates anywhere, so it is satisfied; meanwhile the file
    has re-acquired the one-line blob shape this entire script exists to
    eliminate. A green test over a re-created 5,556-character line.

    ★ The property is one stamp per line, which is what makes the file
    readable, greppable and rollable at all."""
    _, doc = repo(DOC.replace("Body text.", BLOB + "\n\nBody text."))
    subprocess.run([sys.executable, str(STAMP), str(doc), "--restore"],
                   capture_output=True, text=True)
    rc, out = check(doc)
    assert rc == 0, f"a blob-era loss was reported but not recoverable:\n{out}"

    hist = (doc.parent / "history" / "CURRENT_STATUS-stamp-history.md")
    lines = [l for l in hist.read_text(encoding="utf-8").split("\n")
             if l.lstrip().startswith("- _Prior:")]
    for l in lines:
        n = len(re.findall(r"\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}\s+[A-Z]{2,5}", l))
        assert n == 1, (f"a restored line carries {n} stamps — the blob was "
                        f"re-imported whole instead of split:\n  {l[:140]}")
    got = {m for l in lines
           for m in re.findall(r"(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}\s+[A-Z]{2,5})", l)}
    assert "2026-08-17 09:00 MST" in got, f"the blob's second stamp is missing: {got}"


@case
def restore_preserves_the_stamp_text_verbatim():
    """Recovery is relocation, not rewriting — only the prefix may change."""
    _, doc = repo(DOC.replace("Body text.", OLD + "\n\nBody text."))
    subprocess.run([sys.executable, str(STAMP), str(doc), "--restore"],
                   capture_output=True, text=True)
    hist = (doc.parent / "history" / "CURRENT_STATUS-stamp-history.md").read_text(encoding="utf-8")
    tail = OLD.split("MST", 1)[1].rstrip("_")
    assert tail.strip() in hist, "the stamp's text was altered during recovery"


@case
def restore_does_not_duplicate_a_record():
    """⛔ A DEFECT I SHIPPED. Splitting a blob can yield a stamp that already
    exists as its own row in slightly different wording — the same moment as
    `_Prior: <date> · transcript: X_` and as `_Prior: <date> by an AI session ·
    transcript: X_`. Appending both turns a repair into a duplication.

    ✅ Measured: the first restore introduced **57 duplicate rows** across four
    fran-dash history files. Every date-count check passed the whole time,
    because a duplicate loses nothing — it was only visible by reading the file,
    which is how Cenay found it."""
    _, doc = repo(DOC.replace("Body text.", BLOB + "\n\nBody text."))
    hist = doc.parent / "history"
    hist.mkdir(exist_ok=True)
    # the same record already present, in the terser wording
    (hist / "CURRENT_STATUS-stamp-history.md").write_text(
        "# CURRENT_STATUS.md — stamp history\n\n"
        "- _Prior: 2026-08-17 09:00 MST · transcript: "
        "`aaaaaaaa-0000-0000-0000-000000000004`_\n", encoding="utf-8")
    subprocess.run([sys.executable, str(STAMP), str(doc), "--restore"],
                   capture_output=True, text=True)
    rows = [l for l in (hist / "CURRENT_STATUS-stamp-history.md")
            .read_text(encoding="utf-8").split("\n") if l.lstrip().startswith("- _Prior:")]
    seen = {}
    for l in rows:
        d = re.search(r"(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}\s+[A-Z]{2,5})", l)
        t = re.search(r"transcript:\s*`([^`]+)`", l)
        k = (d.group(1), t.group(1) if t else l)
        assert k not in seen, (f"duplicate record {k}:\n  {seen.get(k,'')[:90]}\n  {l[:90]}")
        seen[k] = l
    kept = [l for l in rows if "2026-08-17 09:00" in l][0]
    assert "by an AI session" in kept, (
        f"dedupe kept the TERSER copy — information was dropped:\n  {kept}")


# ── the guard must not go slack ──────────────────────────────────────────────

@case
def a_healthy_doc_passes():
    """⛔ THE CONTROL. Without it, "always fail" would score green above."""
    _, doc = repo(DOC)
    rc, out = check(doc)
    assert rc == 0, f"a doc that never lost anything was failed:\n{out}"
    assert "history audit" in out and "still present" in out, out


@case
def an_unauditable_doc_says_so():
    """⚠️ A checker must NAME WHAT IT DID NOT CHECK.

    Outside a git repo the audit cannot run. It must say that, not print a
    clean bill of health — otherwise the tool built to catch a silent loss
    becomes another way to have one."""
    d = Path(tempfile.mkdtemp())
    (d / "docs").mkdir()
    doc = d / "docs" / "CURRENT_STATUS.md"
    doc.write_text(DOC, encoding="utf-8")
    rc, out = check(doc)
    assert rc == 0, out
    assert "NOT run" in out, f"an unauditable doc reported as clean:\n{out}"


def prove():
    """Revert the audit and assert the right cases go red."""
    src = STAMP.read_text(encoding="utf-8")
    broken = src.replace("        problems += lost\n", "")
    assert broken != src, "could not locate the audit wiring to revert"
    # ⚠️ The blob handling is a SECOND, independent part of the fix; reverting
    # only the wiring would leave `restore_recovers_from_a_legacy_blob` green
    # and bless a --restore that cannot fix what --check reports.
    noblob = src.replace("            if len(PRIOR_SPLIT.findall(line)) > 0:",
                         "            if False:")
    assert noblob != src, "could not locate the blob handling to revert"

    expected = {
        "audit unwired": (broken, {"a_lost_stamp_is_detected",
                                   "a_lost_stamp_inside_a_legacy_blob_is_detected"}),
        "blob handling removed": (noblob, {"restore_recovers_from_a_legacy_blob"}),
        "dedupe removed": (
            src.replace("    rows = list(best.values())\n", ""),
            {"restore_does_not_duplicate_a_record"}),
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
            if "a_healthy_doc_passes" in reds:
                problems.append(f"    ⛔ the control went red under '{name}'")
    finally:
        STAMP.write_text(src, encoding="utf-8")

    if problems:
        print("\n".join(problems))
        return 1
    print("✅ both reverts turn exactly the right cases red, and the control holds")
    return 0


def main():
    if "--prove" in sys.argv:
        return prove()
    failed = 0
    for fn in CASES:
        try:
            fn()
            print(f"  ✓ {fn.__name__}")
        except Exception as exc:
            failed += 1
            print(f"  ✗ {fn.__name__}: {exc}")
    if failed:
        print(f"\n{failed} of {len(CASES)} failed")
        return 1
    print(f"\nall {len(CASES)} passed — ⚠️ now run --prove")
    return 0


if __name__ == "__main__":
    sys.exit(main())
