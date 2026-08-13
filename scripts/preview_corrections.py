#!/usr/bin/env python3
"""Preview term corrections against cached transcripts — changes nothing.

Read-only. It never writes to the cache, never calls an API, and never costs
anything. Use it to see exactly what the term list would do to real meetings
before wiring corrections into the pipeline.

USAGE
-----
  # One transcript, showing every line that would change
  ./venv/bin/python scripts/preview_corrections.py trfa-catch-up-after-tampa

  # Every cached transcript, ranked by how much each would change
  ./venv/bin/python scripts/preview_corrections.py --all

  # Every transcript, with the changed lines shown too (long)
  ./venv/bin/python scripts/preview_corrections.py --all --lines

  # What a term list change would do: check one word everywhere
  ./venv/bin/python scripts/preview_corrections.py --all --grep nick

A name fragment is enough — "tampa" matches the Tampa catch-up. Paths work too.
"""

import argparse
import glob
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from terms import apply_corrections, load_terms  # noqa: E402

CACHE_DIR = ROOT / "temp" / "transcribe-cache"

BOLD, DIM, RED, GREEN, YELLOW, RESET = (
    "\033[1m", "\033[2m", "\033[31m", "\033[32m", "\033[33m", "\033[0m"
)


def resolve(name: str) -> list[Path]:
    """Accept a path, a filename, or any fragment of one."""
    p = Path(name)
    if p.exists():
        return [p]
    hits = sorted(CACHE_DIR.glob(f"*{name}*.json"))
    return hits


def read_transcript(path: Path) -> list[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    utts = data.get("utterances") or []
    if utts:
        return [u.get("text", "") for u in utts]
    raw = data.get("raw_text") or data.get("text") or ""
    return raw.splitlines()


def highlight(before: str, after: str) -> tuple[str, str]:
    """Colour just the words that differ, so the change is obvious."""
    bw, aw = before.split(), after.split()
    if len(bw) == len(aw):
        b = " ".join(f"{RED}{w}{RESET}" if w != aw[i] else w for i, w in enumerate(bw))
        a = " ".join(f"{GREEN}{w}{RESET}" if w != bw[i] else w for i, w in enumerate(aw))
        return b, a
    return before, after


def preview_one(path: Path, terms, show_lines=True, grep=None) -> tuple[int, int]:
    """Show what the term list would change in one transcript.

    With --grep, the job is different and deliberately so: you are vetting a
    CANDIDATE word that is usually NOT in the term list yet, so filtering to
    lines that already change would show you nothing. Grep mode therefore shows
    every line containing the word, whether or not it is corrected today.
    """
    lines = read_transcript(path)
    changed, total, matched = 0, 0, 0
    out_lines = []

    for i, line in enumerate(lines, 1):
        fixed, changes = apply_corrections(line, terms)

        if grep:
            if grep.lower() not in line.lower():
                continue
            matched += 1
            if fixed != line:
                changed += 1
                total += sum(c.count for c in changes)
            if show_lines:
                out_lines.append(f"  {DIM}line {i}{RESET}")
                if fixed == line:
                    # Not corrected today — highlight the candidate in context.
                    shown = re.sub(f"({re.escape(grep)})",
                                   f"{YELLOW}\\1{RESET}", line, flags=re.I)
                    out_lines.append(f"    {DIM}={RESET} {shown}")
                else:
                    b, a = highlight(line, fixed)
                    out_lines.append(f"    {RED}- {RESET}{b}")
                    out_lines.append(f"    {GREEN}+ {RESET}{a}")
            continue

        if fixed == line:
            continue
        changed += 1
        total += sum(c.count for c in changes)
        if show_lines:
            b, a = highlight(line, fixed)
            out_lines.append(f"  {DIM}line {i}{RESET}")
            out_lines.append(f"    {RED}- {RESET}{b}")
            out_lines.append(f"    {GREEN}+ {RESET}{a}")

    if grep:
        if matched:
            print(f"\n{BOLD}{path.name}{RESET}")
            print(f"  {YELLOW}{matched} line(s) contain {grep!r}{RESET}"
                  f" · {changed} already corrected by the current list")
            for l in out_lines:
                print(l)
        return matched, total

    print(f"\n{BOLD}{path.name}{RESET}")
    print(f"  {len(lines)} utterances · {YELLOW}{total} correction(s) "
          f"on {changed} line(s){RESET}")
    for l in out_lines:
        print(l)
    return changed, total


def main():
    ap = argparse.ArgumentParser(description="Preview term corrections (read-only).")
    ap.add_argument("name", nargs="?", help="transcript name, fragment, or path")
    ap.add_argument("--all", action="store_true", help="every cached transcript")
    ap.add_argument("--lines", action="store_true",
                    help="with --all, also print the changed lines")
    ap.add_argument("--grep", metavar="WORD",
                    help="only show lines containing WORD (before correction)")
    args = ap.parse_args()

    terms = load_terms()
    print(f"{BOLD}Term list:{RESET} {len(terms)} terms — "
          + ", ".join(t.correct for t in terms))
    print(f"{DIM}Read-only preview. Nothing is written.{RESET}")

    if args.all:
        files = sorted(CACHE_DIR.glob("*.json"))

        # --- grep mode: vetting a candidate word across every meeting --------
        if args.grep:
            matched_lines, corrected_lines, files_hit = 0, 0, 0
            for f in files:
                try:
                    m, _ = preview_one(f, terms, show_lines=True, grep=args.grep)
                except Exception as e:  # noqa: BLE001
                    print(f"  {RED}skipped {f.name}: {e}{RESET}")
                    continue
                if m:
                    files_hit += 1
                    matched_lines += m
            print(f"\n  {BOLD}{matched_lines} line(s) containing {args.grep!r} "
                  f"across {files_hit} transcript(s){RESET}")
            print(f"  {DIM}Read them with one question: is this word EVER used "
                  f"innocently?{RESET}")
            print(f"  {DIM}If never, it is safe to add — and safe to force: "
                  f"if the classifier refuses it.{RESET}")
            return 0

        # --- normal mode: what the current list would change ------------------
        rows = []
        for f in files:
            try:
                if args.lines:
                    _, t = preview_one(f, terms, show_lines=True)
                else:
                    text_lines = read_transcript(f)
                    t = 0
                    for line in text_lines:
                        _, ch = apply_corrections(line, terms)
                        t += sum(x.count for x in ch)
            except Exception as e:  # noqa: BLE001
                print(f"  {RED}skipped {f.name}: {e}{RESET}")
                continue
            rows.append((t, f.name))

        if not args.lines:
            rows.sort(reverse=True)
            print(f"\n{BOLD}Corrections per transcript (highest first){RESET}")
            for t, name in rows:
                if t:
                    print(f"  {YELLOW}{t:5d}{RESET}  {name}")
            unchanged = sum(1 for t, _ in rows if not t)
            print(f"\n  {BOLD}{sum(t for t, _ in rows)} total corrections "
                  f"across {len(rows)} transcripts{RESET}"
                  f"  {DIM}({unchanged} unchanged){RESET}")
            print(f"\n  {DIM}See the actual lines for one file:{RESET}")
            top = rows[0][1].replace("-raw-transcript.json", "") if rows else "NAME"
            print(f"  ./venv/bin/python scripts/preview_corrections.py {top}")
        return 0

    if not args.name:
        ap.error("give a transcript name/fragment, or --all")

    hits = resolve(args.name)
    if not hits:
        print(f"{RED}No transcript matching {args.name!r} in {CACHE_DIR}{RESET}")
        print("Try:  ls temp/transcribe-cache/ | head")
        return 1
    if len(hits) > 1:
        print(f"{YELLOW}{len(hits)} matches — showing all:{RESET}")
    for h in hits:
        preview_one(h, terms, show_lines=True, grep=args.grep)
    return 0


if __name__ == "__main__":
    sys.exit(main())
