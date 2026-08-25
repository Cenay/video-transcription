#!/usr/bin/env python3
"""Regression suite for the snapshot/stamp label split ([DEC-265], 2026-08-25).

The defect these lock down, reproduced 2026-08-25: CURRENT_STATUS.md carried a
one-line snapshot labelled `**Last updated:**` ABOVE its real `_Last updated ...`
stamp. stamp-doc.py accepted BOTH labels as the stamp, so it had to break a tie,
and it did so by asking whether the line contained the string `transcript:`.
A snapshot whose prose merely contained that word won the tie, was consumed and
rewritten as `- _Prior:` inside the stamp fold, and the real stamp was orphaned
below </details> -- while --check reported "stamp block is well-formed", exit 0.

Run:  python3 .claude/scripts/test-snapshot-label.py
"""
import re, subprocess, sys, tempfile, pathlib

SCRIPT = str(pathlib.Path(__file__).with_name("stamp-doc.py"))
STAMP_A = ("_Last updated 2026-08-25 08:00 MST by an AI session · transcript: "
           "`aaaa1111-2222-3333-4444-555566667777` — the real stamp._")
NEW = ("_Last updated 2026-08-25 10:00 MST by an AI session · transcript: "
       "`bbbb1111-2222-3333-4444-555566667777` — new._")

def run(body, *args):
    d = pathlib.Path(tempfile.mkdtemp())
    (d / "docs").mkdir()
    f = d / "docs" / "T.md"
    f.write_text(body, encoding="utf-8")
    r = subprocess.run([sys.executable, SCRIPT, str(f), *args],
                       capture_output=True, text=True, cwd=d)
    return r, f

fails = []
def check(name, cond, detail=""):
    print(("  ok   " if cond else "  FAIL ") + name + (f"  [{detail}]" if not cond and detail else ""))
    if not cond:
        fails.append(name)

print("1. a SNAPSHOT containing the word `transcript:` must never be taken as the stamp")
body = ("# T\n\n**Snapshot:** 2026-08-25 09:00 MST (session 99 — reviewed the "
        "transcript: nothing changed)\n\n" + STAMP_A + "\n\nbody\n")
r, f = run(body, "--stamp", NEW)
out = f.read_text(encoding="utf-8")
check("snapshot line survives verbatim in place", "**Snapshot:** 2026-08-25 09:00 MST" in out)
check("snapshot NEVER appears as a `- _Prior:` row",
      not re.search(r"^- _Prior: 2026-08-25 09:00", out, re.M))
check("the real 08:00 stamp is what got folded",
      re.search(r"^- _Prior: 2026-08-25 08:00", out, re.M) is not None)
check("exactly one `_Last updated` line remains",
      len(re.findall(r"^_Last updated ", out, re.M)) == 1,
      f"found {len(re.findall(r'^_Last updated ', out, re.M))}")

print("2. TWO `_Last updated` lines must be REFUSED, never silently disambiguated")
body = ("# T\n\n" + NEW + "\n\nmiddle\n\n" + STAMP_A + "\n")
r, _ = run(body, "--check")
check("--check exits non-zero", r.returncode != 0, f"exit={r.returncode}")
check("the message names the ambiguity", "lines found" in (r.stdout + r.stderr))
check("it refuses rather than guesses", "Refusing to guess" in (r.stdout + r.stderr))

print("3. the retired `**Last updated:**` snapshot label is reported, not silently ignored")
body = ("# T\n\n**Last updated:** 2026-08-25 09:00 MST (session 99)\n\n" + STAMP_A + "\n")
r, _ = run(body, "--check")
check("--check exits non-zero", r.returncode != 0, f"exit={r.returncode}")
check("the message says how to fix it", "**Snapshot:**" in (r.stdout + r.stderr))

print("4. a `_Prior status:` line (the DECISIONS shape) is not mistaken for a stamp")
body = ("# T\n\n" + STAMP_A + "\n\n## DEC-001\n**Status:** CLOSED\n"
        "_Prior status: PROPOSED 2026-07-20 00:01 MST — \"nothing agreed\"_\n")
r, f = run(body, "--stamp", NEW)
out = f.read_text(encoding="utf-8")
check("the `_Prior status:` line is untouched",
      '_Prior status: PROPOSED 2026-07-20 00:01 MST — "nothing agreed"_' in out)
check("it did not become a stamp row",
      not re.search(r"^- _Prior: 2026-07-20", out, re.M))

print("4b. per-SECTION stamps are a legitimate convention, not an error")
# LSP/Staff_Form's CURRENT_STATUS.md carries a header stamp plus one per
# `## Session record` block -- five in all, every one a real record. A blanket
# "more than one `_Last updated` is an error" refused that whole repo.
body = ("# T\n\n" + STAMP_A + "\n\nintro\n\n"
        "## Session record — 2026-08-04\n"
        "_Last updated 2026-08-04 00:40 MST by an AI session · transcript: `cccc`_\n\n"
        "notes\n\n"
        "## Session record — 2026-08-03\n"
        "_Last updated 2026-08-03 12:59 MST by an AI session · transcript: `dddd`_\n")
r, f = run(body, "--stamp", NEW)
out = f.read_text(encoding="utf-8")
check("the write succeeds", r.returncode == 0, f"exit={r.returncode} {(r.stderr or '')[:90]}")
check("the HEADER stamp is the one folded",
      re.search(r"^- _Prior: 2026-08-25 08:00", out, re.M) is not None)
check("per-section stamps are untouched",
      "2026-08-04 00:40 MST" in out and "2026-08-03 12:59 MST" in out)
check("they did NOT become prior rows",
      not re.search(r"^- _Prior: 2026-08-0[34]", out, re.M))

print("4c. but TWO header stamps is still a hard refusal (the orphan case)")
body = ("# T\n\n" + NEW + "\n\nmiddle\n\n" + STAMP_A + "\n\n## A section\n\ntext\n")
r, _ = run(body, "--check")
check("--check still exits non-zero", r.returncode != 0, f"exit={r.returncode}")
check("still refuses rather than guesses", "Refusing to guess" in (r.stdout + r.stderr))

print("4d. ALL stamp history goes to <docs-root>/history/, wherever the doc lives")
# Was `doc.parent / "history"`, so stamping a ROOT-level CLAUDE.md created a
# stray `history/` beside the source tree, split from the files in docs/history/.
# Ruled 2026-08-25: all history for all files goes to the docs root.
import os as _os
d = pathlib.Path(tempfile.mkdtemp()); (d / "docs").mkdir()
subprocess.run(["git", "init", "-q", "."], cwd=d, capture_output=True)
(d / "CLAUDE.md").write_text(f"# T\n\n{STAMP_A}\n\nbody\n", encoding="utf-8")
for i in range(4):
    subprocess.run([sys.executable, SCRIPT, str(d / "CLAUDE.md"), "--stamp",
                    f"_Last updated 2026-08-25 1{i}:00 MST by an AI session · transcript: `c{i}`_"],
                   cwd=d, capture_output=True)
check("a ROOT doc creates NO stray history/ at the repo root", not (d / "history").exists())
check("its history lands in docs/history/", (d / "docs" / "history" / "CLAUDE-stamp-history.md").exists())
fold = [l for l in (d / "CLAUDE.md").read_text(encoding="utf-8").split("\n") if "<code>" in l]
check("and the fold names that path from the doc",
      bool(fold) and "docs/history/" in fold[0], fold[0][:80] if fold else "no fold")
(d / "docs" / "X.md").write_text(f"# X\n\n{STAMP_A}\n\nbody\n", encoding="utf-8")
for i in range(4):
    subprocess.run([sys.executable, SCRIPT, str(d / "docs" / "X.md"), "--stamp",
                    f"_Last updated 2026-08-26 1{i}:00 MST by an AI session · transcript: `d{i}`_"],
                   cwd=d, capture_output=True)
check("a doc already in docs/ is unaffected",
      (d / "docs" / "history" / "X-stamp-history.md").exists())

print("5. control — a well-formed doc still passes and still stamps")
body = ("# T\n\n**Snapshot:** 2026-08-25 09:00 MST (session 99)\n\n" + STAMP_A + "\n")
r, f = run(body, "--stamp", NEW)
out = f.read_text(encoding="utf-8")
check("new stamp is now current", "2026-08-25 10:00 MST" in out)
check("previous stamp folded", re.search(r"^- _Prior: 2026-08-25 08:00", out, re.M) is not None)
r, _ = run(out, "--check")
check("--check passes on the result", r.returncode == 0, f"exit={r.returncode}")

print()
if fails:
    print(f"✗ {len(fails)} failing: " + ", ".join(fails))
    sys.exit(1)
print("✓ all snapshot/stamp label checks pass")
