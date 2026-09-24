#!/usr/bin/env python3
"""
timer-commits.py — gather YOUR commits across the work repos for a date window,
for /build-timer-note. Reports; never writes.

WINDOW RULES (ruled 2026-09-24 by Cenay):
  one date   D        -> D 00:00 .. D 23:59:59
  two dates  D1 D2    -> D1 00:00 .. D2 23:59:59
  ROLLOVER: if "now" is on the day AFTER the end date and before the cutoff
  (06:00), the window ends at NOW instead — you were still working past midnight.
  Example: at 9/24 04:30, "9/23" means 9/23 00:00 .. 9/24 04:30.
  Why a cutoff: "after 11:59pm of the date" is true forever, so without one a
  request for 9/23 made on 9/26 would bill three days of work to one date.

REPO SCOPE (ruled 2026-09-24):
  default  -> every git repo under the roots (/mnt/k/Code, /mnt/k/_Sites),
              EXCEPT the personal repos (Code/System) and anything in .archived/.
              claude-personal-toolkit is WORK, never personal.
  --include-personal  -> also the personal repos.
  --repo NAME         -> only these repo basenames (repeatable).
  A root that does not exist (a teammate's machine) is reported, and the current
  repo is used instead — never a silent empty result.

Commits are yours (--author = git config user.name), from ALL refs, merges
excluded, and DEDUPED BY HASH: Cenay/N8N is cloned twice, and without the
dedupe every commit in it would be billed twice.

Usage:
  timer-commits.py 9/23
  timer-commits.py 2026-09-22 9/23 --include-personal
  timer-commits.py yesterday --repo fran-dash
  timer-commits.py 9/23 --now "2026-09-24 04:30"     # testing the rollover
"""

import argparse
import os
import re
import subprocess
import sys
from datetime import date, datetime, time, timedelta

ROOTS = ["/mnt/k/Code", "/mnt/k/_Sites"]
PERSONAL = {"/mnt/k/Code/System"}
ROLLOVER_CUTOFF = time(6, 0)
SKIP_DIRS = {".archived", "node_modules", "vendor", ".git"}
MAX_DEPTH = 3


def parse_date(s, now):
    s = s.strip().lower()
    if s == "today":
        return now.date()
    if s == "yesterday":
        return now.date() - timedelta(days=1)
    m = re.fullmatch(r"(\d{4})-(\d{1,2})-(\d{1,2})", s)
    if m:
        return date(int(m[1]), int(m[2]), int(m[3]))
    m = re.fullmatch(r"(\d{1,2})/(\d{1,2})(?:/(\d{2,4}))?", s)
    if m:
        if m[3]:
            y = int(m[3]) + (2000 if len(m[3]) == 2 else 0)
            return date(y, int(m[1]), int(m[2]))
        d = date(now.year, int(m[1]), int(m[2]))
        # No year given and that date is still ahead -> they mean last year's.
        return d if d <= now.date() else date(now.year - 1, d.month, d.day)
    raise ValueError(f"unrecognized date: {s!r} (use 9/23, 2026-09-23, today, yesterday)")


def window(d1, d2, now):
    start = datetime.combine(d1, time(0, 0))
    end = datetime.combine(d2, time(23, 59, 59))
    rolled = False
    if now.date() == d2 + timedelta(days=1) and now.time() < ROLLOVER_CUTOFF:
        end, rolled = now, True
    elif now < end:
        end = now          # today's window cannot extend into the future
    return start, end, rolled


def find_repos(roots):
    repos, missing = [], []
    for root in roots:
        if not os.path.isdir(root):
            missing.append(root)
            continue
        base = root.rstrip(os.sep).count(os.sep)
        for dirpath, dirnames, _ in os.walk(root):
            if ".git" in dirnames or os.path.isfile(os.path.join(dirpath, ".git")):
                repos.append(dirpath)
            depth = dirpath.count(os.sep) - base
            dirnames[:] = [] if depth >= MAX_DEPTH else [d for d in dirnames if d not in SKIP_DIRS]
    return sorted(repos), missing


def git(repo, *args):
    r = subprocess.run(["git", "-C", repo, *args], capture_output=True, text=True)
    return r.stdout if r.returncode == 0 else None


def main(argv):
    ap = argparse.ArgumentParser(description="Your commits across the work repos for a date window.")
    ap.add_argument("dates", nargs="+", help="one date, or a start and end date")
    ap.add_argument("--include-personal", action="store_true")
    ap.add_argument("--repo", action="append", default=[], help="limit to these repo basenames")
    ap.add_argument("--now", help="override the current time (testing): 'YYYY-MM-DD HH:MM'")
    ap.add_argument("--roots", nargs="+", default=ROOTS)
    a = ap.parse_args(argv)

    now = datetime.strptime(a.now, "%Y-%m-%d %H:%M") if a.now else datetime.now().replace(microsecond=0)
    if len(a.dates) > 2:
        ap.error("give one date or two")
    try:
        d1 = parse_date(a.dates[0], now)
        d2 = parse_date(a.dates[-1], now)
    except ValueError as e:
        ap.error(str(e))
    if d2 < d1:
        ap.error(f"end date {d2} is before start date {d1}")
    start, end, rolled = window(d1, d2, now)

    repos, missing = find_repos(a.roots)
    notes = []
    if missing:
        notes.append(f"roots not found, NOT searched: {', '.join(missing)}")
    if not repos:
        top = git(os.getcwd(), "rev-parse", "--show-toplevel")
        if top:
            repos = [top.strip()]
            notes.append("no repos under the roots — fell back to the current repo only")
    excluded = []
    if not a.include_personal:
        excluded = [r for r in repos if r in PERSONAL]
        repos = [r for r in repos if r not in PERSONAL]
    if a.repo:
        want = set(a.repo)
        repos = [r for r in repos if os.path.basename(r) in want]
        unknown = want - {os.path.basename(r) for r in repos}
        if unknown:
            notes.append(f"--repo not found: {', '.join(sorted(unknown))}")

    author = (git(os.getcwd(), "config", "user.name") or "").strip()
    since, until = start.strftime("%Y-%m-%d %H:%M:%S"), end.strftime("%Y-%m-%d %H:%M:%S")
    seen, by_repo, failed, head_only = {}, {}, [], []
    log_args = ["--no-merges", f"--author={author}", f"--since={since}", f"--until={until}",
                "--format=%H%x09%ad%x09%s", "--date=format-local:%Y-%m-%d %H:%M"]
    for repo in repos:
        out = git(repo, "log", "--all", *log_args)
        if out is None:
            # --all dies on ONE broken ref (trfa-doco carries a 2020 Dropbox
            # "conflicted copy" branch ref). Fall back to HEAD and say so.
            out = git(repo, "log", "HEAD", *log_args)
            if out is not None:
                head_only.append(repo)
        if out is None:
            failed.append(repo)
            continue
        for line in out.splitlines():
            sha, when, subj = line.split("\t", 2)
            if sha in seen:
                continue            # same commit in a second clone
            seen[sha] = repo
            by_repo.setdefault(os.path.basename(repo), []).append((when, subj))

    print(f"window: {start:%Y-%m-%d %H:%M} .. {end:%Y-%m-%d %H:%M}"
          + ("  (rolled over: still working past midnight)" if rolled else ""))
    print(f"author: {author}   repos searched: {len(repos) - len(failed)}"
          + (f"   personal excluded: {', '.join(os.path.basename(r) for r in excluded)}" if excluded else ""))
    for name in sorted(by_repo, key=lambda n: min(by_repo[n])[0]):
        print(f"\n[{name}]  {len(by_repo[name])} commit(s)")
        for when, subj in sorted(by_repo[name]):
            print(f"  {when}  {subj}")
    if not by_repo:
        print("\nno commits by this author in the window")
    if head_only:
        notes.append(f"a broken ref blocked --all; searched the current branch only: {', '.join(head_only)}")
    if failed:
        notes.append(f"git log failed, NOT searched: {', '.join(failed)}")
    for n in notes:
        print(f"⚠️  {n}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
