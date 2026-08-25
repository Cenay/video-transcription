#!/usr/bin/env python3
"""Suite for check-append-only.py — the append-only guard on docs/history/.

⛔ WHY THIS EXISTS. The guard runs on every commit, unattended, and had no test.
Three separate checkers were caught GREEN WHILE INERT on 2026-08-25 -- an
over-escaped regex that matched no lines, a fixture the tool silently undid, and
a sweep grepping for a word that only appears in the PASS branch. A guard nobody
is watching is exactly where that failure hides: it goes quiet and every commit
sails through.

★ So every check here asserts the guard FAILS on deliberately broken input.
"Exits 0 on a good repo" proves nothing on its own.

Run:  python3 scripts/test-append-only.py
"""
import os, pathlib, subprocess, sys, tempfile

GUARD = str(pathlib.Path(__file__).with_name("check-append-only.py"))

def sh(cwd, *a, **kw):
    return subprocess.run(a, cwd=cwd, capture_output=True, text=True, **kw)

def repo(with_history=True):
    d = pathlib.Path(tempfile.mkdtemp())
    sh(d, "git", "init", "-q", ".")
    sh(d, "git", "config", "user.email", "t@t"); sh(d, "git", "config", "user.name", "t")
    (d/"docs").mkdir()
    if with_history:
        (d/"docs"/"history").mkdir()
        (d/"docs"/"history"/"X-stamp-history.md").write_text(
            "# X — stamp history\n\n"
            "- _Prior: 2026-08-01 10:00 MST · transcript: `aaaa`_\n"
            "- _Prior: 2026-07-31 09:00 MST · transcript: `bbbb`_\n"
            "- _Prior: 2026-07-30 08:00 MST · transcript: `cccc`_\n", encoding="utf-8")
    (d/"README.md").write_text("# t\n", encoding="utf-8")
    sh(d, "git", "add", "-A"); sh(d, "git", "commit", "-qm", "init")
    return d

def run(d, **env):
    return sh(d, sys.executable, GUARD, "--staged", env=dict(os.environ, **env))

fails = []
def check(name, cond, detail=""):
    print(("  ok   " if cond else "  FAIL ") + name + (f"  [{detail}]" if not cond and detail else ""))
    if not cond: fails.append(name)

print("1. removing a line from docs/history/ must BLOCK the commit")
d = repo()
h = d/"docs"/"history"/"X-stamp-history.md"
lines = h.read_text(encoding="utf-8").split("\n")
gone = lines.pop(3)                      # drop a stamp row
h.write_text("\n".join(lines), encoding="utf-8")
sh(d, "git", "add", "-A")
r = run(d)
check("exits non-zero", r.returncode != 0, f"exit={r.returncode}")
out = r.stdout + r.stderr
check("names the file", "X-stamp-history.md" in out)
check("shows the removed line", gone.strip()[:40] in out, out[:120])
check("calls it a violation", "VIOLATION" in out.upper())

print("2. ADDING lines is always allowed")
d = repo()
h = d/"docs"/"history"/"X-stamp-history.md"
h.write_text(h.read_text(encoding="utf-8") + "- _Prior: 2026-07-29 07:00 MST · transcript: `dddd`_\n", encoding="utf-8")
sh(d, "git", "add", "-A")
r = run(d)
check("exits zero", r.returncode == 0, f"exit={r.returncode} {(r.stdout+r.stderr)[:100]}")

print("3. EDITING a line in place is a removal, and must block")
d = repo()
h = d/"docs"/"history"/"X-stamp-history.md"
h.write_text(h.read_text(encoding="utf-8").replace("transcript: `aaaa`", "transcript: `zzzz`"), encoding="utf-8")
sh(d, "git", "add", "-A")
r = run(d)
check("exits non-zero", r.returncode != 0, f"exit={r.returncode}")

print("4. ALLOW_HISTORY_SHRINK=1 permits it, and says so out loud")
d = repo()
h = d/"docs"/"history"/"X-stamp-history.md"
lines = h.read_text(encoding="utf-8").split("\n"); lines.pop(3)
h.write_text("\n".join(lines), encoding="utf-8")
sh(d, "git", "add", "-A")
r = run(d, ALLOW_HISTORY_SHRINK="1")
check("exits zero", r.returncode == 0, f"exit={r.returncode}")
check("still PRINTS the violation", "VIOLATION" in (r.stdout + r.stderr).upper())
check("says it was allowed deliberately", "ALLOWED" in (r.stdout + r.stderr).upper())

print("5. a repo with no docs/history/ passes, and says which")
d = repo(with_history=False)
(d/"docs"/"other.md").write_text("x\n", encoding="utf-8"); sh(d, "git", "add", "-A")
r = run(d)
check("exits zero", r.returncode == 0, f"exit={r.returncode}")
check("says there was nothing to check", "no docs/history" in (r.stdout + r.stderr).lower())

print("6. a BRAND NEW history file is reported as not-compared, never silently passed")
d = repo()
(d/"docs"/"history"/"NEW-stamp-history.md").write_text("# new\n\n- _Prior: 2026-08-02 11:00 MST_\n", encoding="utf-8")
sh(d, "git", "add", "-A")
r = run(d)
out = r.stdout + r.stderr
check("exits zero (nothing was lost)", r.returncode == 0, f"exit={r.returncode}")
check("names it as not compared", "NEW-stamp-history.md" in out and
      ("no HEAD version" in out or "NOT compared" in out), out[:140])

print("6b. the three no-target cases must be DISTINGUISHABLE, not one message")
# ⛔ They shared one message, so a run that checked nothing looked like a clean
# run. That was misread twice on 2026-08-25.
d = repo()
r = sh(d, sys.executable, GUARD)                        # no --staged, no names
out = (r.stdout + r.stderr)
check("no target at all says CHECK DID NOT RUN", "DID NOT RUN" in out.upper(), out[:120])
check("...and does not claim a clean result", "✓" not in out, out[:120])

d2 = repo(with_history=False)
(d2/"docs"/"z.md").write_text("z\n", encoding="utf-8"); sh(d2, "git", "add", "-A")
out2 = "".join(run(d2)[i] for i in (0,)) if False else (lambda r: r.stdout + r.stderr)(run(d2))
check("a repo with no history dir says exactly that",
      "no docs/history" in out2 and "protect" in out2, out2[:120])

d3 = repo()
(d3/"README.md").write_text("# t\n\nx\n", encoding="utf-8"); sh(d3, "git", "add", "README.md")
out3 = (lambda r: r.stdout + r.stderr)(run(d3))
check("history exists but nothing staged says THAT instead",
      "nothing staged" in out3, out3[:120])
check("the three messages are all different",
      len({out.strip(), out2.strip(), out3.strip()}) == 3)

print("7. an unstaged removal is NOT the guard's business")
d = repo()
h = d/"docs"/"history"/"X-stamp-history.md"
lines = h.read_text(encoding="utf-8").split("\n"); lines.pop(3)
h.write_text("\n".join(lines), encoding="utf-8")      # changed but NOT staged
(d/"README.md").write_text("# t\n\nunrelated\n", encoding="utf-8")
sh(d, "git", "add", "README.md")                      # something IS staged, just not the history file
r = run(d)
check("exits zero", r.returncode == 0, f"exit={r.returncode}")

print()
if fails:
    print(f"✗ {len(fails)} failing: " + ", ".join(fails)); sys.exit(1)
print("✓ all append-only guard checks pass")
