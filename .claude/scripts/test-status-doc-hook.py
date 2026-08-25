#!/usr/bin/env python3
"""Suite for check-status-doc-updated.sh — the Stop hook that catches a session
committing work while never updating docs/CURRENT_STATUS.md.

⛔ WHY THIS EXISTS. The hook records its own origin: a session changed the live
Report Generator, wrote 173 lines of notes, updated no status doc, and the gap
surfaced a week later. It has run unattended in one repo ever since with nothing
verifying it still fires -- and on 2026-08-25 it had never been observed firing
at all.

★ The load-bearing check is #3: it must WARN. A hook that has gone quiet looks
exactly like a hook with nothing to report.

Run:  python3 scripts/test-status-doc-hook.py
"""
import os, pathlib, subprocess, sys, tempfile

HOOK = str(pathlib.Path(__file__).with_name("check-status-doc-updated.sh"))
SID = "test-session"

def sh(cwd, *a, **kw):
    return subprocess.run(a, cwd=cwd, capture_output=True, text=True, **kw)

def repo():
    d = pathlib.Path(tempfile.mkdtemp())
    sh(d, "git", "init", "-q", ".")
    sh(d, "git", "config", "user.email", "t@t"); sh(d, "git", "config", "user.name", "t")
    (d/"docs").mkdir()
    (d/"docs"/"CURRENT_STATUS.md").write_text("# Current Status\n\nstuff\n", encoding="utf-8")
    (d/"app.py").write_text("x = 1\n", encoding="utf-8")
    sh(d, "git", "add", "-A"); sh(d, "git", "commit", "-qm", "init")
    return d

def fire(d, sid=SID):
    return sh(d, "bash", HOOK, env=dict(os.environ, CLAUDE_PROJECT_DIR=str(d), CLAUDE_SESSION_ID=sid))

def commit(d, path, text, msg):
    (d/path).parent.mkdir(parents=True, exist_ok=True)
    (d/path).write_text(text, encoding="utf-8")
    sh(d, "git", "add", "-A"); sh(d, "git", "commit", "-qm", msg)

fails = []
def check(name, cond, detail=""):
    print(("  ok   " if cond else "  FAIL ") + name + (f"  [{detail}]" if not cond and detail else ""))
    if not cond: fails.append(name)

print("1. the first run only sets the baseline — it must not warn")
d = repo()
r = fire(d)
check("exits zero", r.returncode == 0, f"exit={r.returncode}")
check("says nothing", not r.stderr.strip(), r.stderr[:80])

print("2. commits touching ONLY docs/ do not warn")
d = repo(); fire(d)
commit(d, "docs/NOTES.md", "notes\n", "docs: notes")
r = fire(d)
check("stays quiet", not r.stderr.strip(), r.stderr[:80])

print("3. ⭐ code committed, CURRENT_STATUS untouched — it MUST warn")
d = repo(); fire(d)
commit(d, "app.py", "x = 2\n", "feat: change the app")
r = fire(d)
err = r.stderr
check("warns on stderr", bool(err.strip()), "SILENT — the hook has gone quiet")
check("names the status doc", "CURRENT_STATUS.md" in err, err[:100])
check("counts the commits", "1 commit" in err, err[:100])

print("4. it fires ONCE per session, not on every stop")
r2 = fire(d)
check("second run is quiet", not r2.stderr.strip(), r2.stderr[:80])
# ⚠️ A different session does NOT warn on its first run -- it has no baseline
# yet, so that run only sets its own mark. This is correct: a session is
# accountable for the commits IT made, not for what it walked in on.
check("a different session's FIRST run only sets its baseline",
      not fire(d, sid="other").stderr.strip())
commit(d, "app.py", "x = 9\n", "feat: more")
check("...and it warns on the next stop, after its own commit",
      bool(fire(d, sid="other").stderr.strip()))

print("5. updating CURRENT_STATUS in the same span clears it")
d = repo(); fire(d)
commit(d, "app.py", "x = 3\n", "feat: change")
(d/"docs"/"CURRENT_STATUS.md").write_text("# Current Status\n\nupdated\n", encoding="utf-8")
sh(d, "git", "add", "-A"); sh(d, "git", "commit", "-qm", "docs: status")
r = fire(d)
check("stays quiet", not r.stderr.strip(), r.stderr[:80])

print("6. outside a git repo it exits quietly rather than erroring")
d = pathlib.Path(tempfile.mkdtemp())
r = sh(d, "bash", HOOK, env=dict(os.environ, CLAUDE_PROJECT_DIR=str(d), CLAUDE_SESSION_ID=SID))
check("exits zero", r.returncode == 0, f"exit={r.returncode}")

print()
if fails:
    print(f"✗ {len(fails)} failing: " + ", ".join(fails)); sys.exit(1)
print("✓ all status-doc hook checks pass")
