#!/usr/bin/env python3
"""Regression suite for link-doc-refs.py's destructive-run guards (T20, 2026-08-25).

The defect, reproduced on a throwaway copy of fran-dash/docs before the fix:

    link-doc-refs.py docs/history
      -> "ids resolvable: 0 (DEC/G/M/D across 0 ledger/record file(s))"
      -> "linked: 12 narrative doc(s)"        <- the word LINKED
      -> DECISIONS-stamp-history.md  173 -> 72 lines, 98 link definitions -> 0
      -> 12 of 13 files damaged, exit 0, no warning

The script regenerates each doc's managed block FROM the ID map, so a run that
cannot see the ledger regenerates every block to empty. It had already PRINTED
that it found nothing, and carried on -- the same "found nothing therefore there
is nothing" defect that made stamp-doc.py --restore delete 367 lines while
reporting success.

Run:  python3 .claude/scripts/test-link-guards.py
"""
import os, pathlib, re, shutil, subprocess, sys, tempfile

SCRIPT = str(pathlib.Path(__file__).with_name("link-doc-refs.py"))
DEFS = re.compile(r"^\[[^\]]+\]:\s*\S+", re.M)

LEDGER = """# Decisions

### DEC-001 The first decision

- **Status:** RESOLVED

### DEC-002 The second decision

- **Status:** RESOLVED
"""
NARRATIVE = """# Status

We applied [DEC-001] and then [DEC-002].
"""

def build():
    d = pathlib.Path(tempfile.mkdtemp())
    (d / "docs" / "history").mkdir(parents=True)
    (d / "docs" / "DECISIONS.md").write_text(LEDGER, encoding="utf-8")
    (d / "docs" / "CURRENT_STATUS.md").write_text(NARRATIVE, encoding="utf-8")
    subprocess.run([sys.executable, SCRIPT, str(d / "docs")], capture_output=True, text=True)
    # move the now-linked doc into history/ so it has a populated managed block there
    shutil.copy2(d / "docs" / "CURRENT_STATUS.md", d / "docs" / "history" / "OLD-STATUS.md")
    return d

def run(d, target, **env):
    e = dict(os.environ, **env)
    return subprocess.run([sys.executable, SCRIPT, str(d / target)],
                          capture_output=True, text=True, env=e)

fails = []
def check(name, cond, detail=""):
    print(("  ok   " if cond else "  FAIL ") + name + (f"  [{detail}]" if not cond and detail else ""))
    if not cond: fails.append(name)

print("1. a run that cannot see the ledger must REFUSE, and write nothing")
d = build()
hist = d / "docs" / "history" / "OLD-STATUS.md"
before = hist.read_text(encoding="utf-8")
n_before = len(DEFS.findall(before))
check("fixture actually has definitions to lose", n_before > 0, f"{n_before}")
r = run(d, "docs/history")
check("exits non-zero", r.returncode != 0, f"exit={r.returncode}")
check("says why", "refusing to run" in (r.stdout + r.stderr).lower())
check("names the likely cause", "SUBDIR" in (r.stdout + r.stderr).upper())
check("file is byte-identical", hist.read_text(encoding="utf-8") == before)

print("2. the correct invocation is unaffected")
d = build()
r = run(d, "docs")
check("exits zero", r.returncode == 0, f"exit={r.returncode}")
check("resolves the ledger", "ids resolvable: 2" in r.stdout, r.stdout.strip()[:60])

print("3. a block that would SHRINK is refused, and nothing is written")
d = build()
cs = d / "docs" / "CURRENT_STATUS.md"
t = cs.read_text(encoding="utf-8")
body, sep, blk = t.partition("<!-- link-doc-refs:start")
# ⚠️ Removing the BRACKETS is not enough -- the tool re-brackets a resolvable
# bare ID, so the block never shrinks and there is nothing to refuse. The ID has
# to leave the prose entirely. (This suite's first draft got that wrong and
# reported a passing guard that had never been exercised.)
cs.write_text(body.replace("and then [DEC-002]", "and nothing else") + sep + blk,
              encoding="utf-8")
n_before = len(DEFS.findall(cs.read_text(encoding="utf-8")))
r = run(d, "docs")
check("reports the refusal", "REFUSED" in (r.stdout + r.stderr))
check("names the counts", "→" in (r.stdout + r.stderr) or "->" in (r.stdout + r.stderr))
check("block is preserved", len(DEFS.findall(cs.read_text(encoding="utf-8"))) == n_before)

print("4. the override permits a deliberate shrink")
r = run(d, "docs", ALLOW_LINK_BLOCK_SHRINK="1")
check("block shrinks when asked", len(DEFS.findall(cs.read_text(encoding="utf-8"))) < n_before)

print()
if fails:
    print(f"✗ {len(fails)} failing: " + ", ".join(fails)); sys.exit(1)
print("✓ all link-guard checks pass")
