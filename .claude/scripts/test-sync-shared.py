#!/usr/bin/env python3
"""Suite for sync-shared.sh — the distribution mechanism for the shared assets.

⛔ WHY THIS EXISTS. Every one of the 12 repos gets its doc tooling from this
script, and it had no test. Two of its properties are load-bearing and neither
was verified anywhere:

  * it must SKIP a locally-edited file rather than clobber it
  * --dry-run must write NOTHING

★ Everything here runs against a throwaway repo via `--repo`, never a real one,
and prefers --dry-run. The one non-dry run writes only into that temp dir.

⚠️ KNOWN LIMIT, stated rather than hidden: this exercises the copy/skip/report
path. It does NOT exercise --push, which commits and pushes to real remotes.

Run:  python3 scripts/test-sync-shared.py
"""
import os, pathlib, subprocess, sys, tempfile

SYNC = str(pathlib.Path(__file__).with_name("sync-shared.sh"))
SCRIPTS = pathlib.Path(__file__).resolve().parent

def sh(cwd, *a, **kw):
    return subprocess.run(a, cwd=cwd, capture_output=True, text=True, **kw)

def target():
    """A throwaway repo shaped like a sync target."""
    d = pathlib.Path(tempfile.mkdtemp()) / "faux-repo"
    (d/".claude"/"scripts").mkdir(parents=True)
    (d/".claude"/"commands").mkdir(parents=True)
    (d/"docs").mkdir()
    sh(d, "git", "init", "-q", ".")
    sh(d, "git", "config", "user.email", "t@t"); sh(d, "git", "config", "user.name", "t")
    (d/"README.md").write_text("# faux\n", encoding="utf-8")
    sh(d, "git", "add", "-A"); sh(d, "git", "commit", "-qm", "init")
    return d

def run(d, *args):
    return sh(SCRIPTS.parent, "bash", SYNC, "--repo", str(d), *args)

fails = []
def check(name, cond, detail=""):
    print(("  ok   " if cond else "  FAIL ") + name + (f"  [{detail}]" if not cond and detail else ""))
    if not cond: fails.append(name)

print("1. a repo missing the assets is reported as needing them")
d = target()
r = run(d, "--dry-run")
out = r.stdout + r.stderr
check("names the repo", "faux-repo" in out, out[-160:])
check("says it would change", "updated" in out or "would change" in out, out[-160:])

print("2. ⭐ --dry-run writes NOTHING")
before = sorted(p.name for p in (d/".claude"/"scripts").iterdir())
run(d, "--dry-run")
after = sorted(p.name for p in (d/".claude"/"scripts").iterdir())
check("the target is untouched", before == after, f"{before} -> {after}")

print("3. a real run copies the assets in")
r = run(d)
names = sorted(p.name for p in (d/".claude"/"scripts").iterdir())
check("stamp-doc.py arrived", "stamp-doc.py" in names, str(names[:6]))
check("doc_root.py arrived (a dependency of stamp-doc)", "doc_root.py" in names, str(names[:6]))

print("4. a second run reports it as already in sync")
r = run(d, "--dry-run")
check("says in sync", "in sync" in (r.stdout + r.stderr), (r.stdout + r.stderr)[-160:])

print("5. ⭐ a LOCALLY EDITED asset is skipped, never clobbered")
victim = d/".claude"/"scripts"/"stamp-doc.py"
mine = "# a local edit that must survive\n" + victim.read_text(encoding="utf-8")
victim.write_text(mine, encoding="utf-8")
r = run(d)
out = r.stdout + r.stderr
kept = victim.read_text(encoding="utf-8").startswith("# a local edit that must survive")
check("the local edit survives", kept, "CLOBBERED")
check("and the skip is reported", "skip" in out.lower(), out[-200:])

print("6. an unknown repo path fails loudly rather than silently doing nothing")
r = sh(SCRIPTS.parent, "bash", SYNC, "--repo", "/nonexistent/nope", "--dry-run")
check("non-zero or an explicit complaint",
      r.returncode != 0 or "no target" in (r.stdout + r.stderr).lower()
      or "not" in (r.stdout + r.stderr).lower(), f"exit={r.returncode} {(r.stdout+r.stderr)[-120:]}")

print()
if fails:
    print(f"✗ {len(fails)} failing: " + ", ".join(fails)); sys.exit(1)
print("✓ all sync-shared checks pass")
