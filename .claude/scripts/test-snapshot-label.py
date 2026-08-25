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
