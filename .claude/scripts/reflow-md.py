#!/usr/bin/env python3
"""
reflow-md.py — unwrap hard-wrapped Markdown prose so each paragraph / list item
is ONE continuous physical line, and let the editor soft-wrap it.

WHY: hard line breaks inserted mid-sentence (wrapping prose at ~80 cols) render as
broken-up text and make diffs noisy. The house rule is: no hard breaks in Markdown
prose — hard breaks are only meaningful in structural blocks. This script enforces
that rule mechanically instead of relying on anyone remembering it.

WHAT IS LEFT UNTOUCHED (never joined):
  - YAML frontmatter (the leading --- ... --- block)
  - Fenced code blocks (``` or ~~~), verbatim
  - Tables (any line beginning with |)
  - Blockquotes (lines beginning with >)   — house rule allows hard breaks here
  - Headings (#), horizontal rules (---, ***, ___)
  - HTML comments and the auto-generated link-doc-refs managed block
  - Link reference definitions ([id]: url)
  - Blank lines (paragraph separators are preserved exactly)

WHAT IS JOINED:
  - Consecutive non-blank prose lines -> one line (single space between).
  - A wrapped list item and its indented continuation lines -> one line, with the
    list marker and its original indentation preserved.

LOGICAL-LINE BOUNDARIES (start a new output line even with no blank line between):
  - a list marker:  -  *  +  1.  1)
  - a bold label at line start:  **Status:**  **Last updated:**  (protects the
    stacked metadata fields that head these docs from being merged into one line)

Usage:
    python3 reflow-md.py <path-or-dir> [more ...]   # apply in place
    python3 reflow-md.py <path-or-dir> --dry-run     # report only, write nothing
    python3 reflow-md.py <path-or-dir> --quiet       # apply, summary only
"""

import os
import re
import sys

FENCE_RE = re.compile(r"^\s*(```|~~~)")
HR_RE = re.compile(r"^\s*([-*_])(\s*\1){2,}\s*$")
HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s")
LIST_MARKER_RE = re.compile(r"^(\s*)([-*+]|\d+[.)])\s+")
BOLD_LABEL_RE = re.compile(r"^\*\*[^*].*?:\*\*")
LINK_DEF_RE = re.compile(r"^\[[^\]]+\]:\s")
COMMENT_RE = re.compile(r"^\s*<!--")


def is_passthrough(stripped, raw):
    """A line that must be emitted verbatim and never joined to a neighbour."""
    if stripped == "":
        return True
    if raw.lstrip().startswith(("|", ">")):
        return True
    if HEADING_RE.match(raw) or HR_RE.match(raw):
        return True
    if COMMENT_RE.match(raw) or LINK_DEF_RE.match(raw.lstrip()):
        return True
    return False


def starts_logical_line(raw):
    """True if this reflowable line begins a NEW logical line (vs. a continuation)."""
    if LIST_MARKER_RE.match(raw):
        return True
    if BOLD_LABEL_RE.match(raw.strip()):
        return True
    return False


def reflow(text):
    lines = text.split("\n")
    out = []
    buf = None          # current logical line being assembled, or None
    in_code = False
    i = 0
    n = len(lines)

    def flush():
        nonlocal buf
        if buf is not None:
            out.append(buf.rstrip())
            buf = None

    # Preserve a leading YAML frontmatter block verbatim.
    if lines and lines[0].strip() == "---":
        out.append(lines[0])
        i = 1
        while i < n:
            out.append(lines[i])
            if lines[i].strip() == "---":
                i += 1
                break
            i += 1

    while i < n:
        raw = lines[i]
        stripped = raw.strip()

        if FENCE_RE.match(raw):
            flush()
            out.append(raw)
            in_code = not in_code
            i += 1
            continue

        if in_code:
            out.append(raw)
            i += 1
            continue

        # Preserve the managed link-doc-refs block verbatim.
        if "link-doc-refs:start" in raw:
            flush()
            while i < n:
                out.append(lines[i])
                if "link-doc-refs:end" in lines[i]:
                    i += 1
                    break
                i += 1
            continue

        if is_passthrough(stripped, raw):
            flush()
            out.append(raw)
            i += 1
            continue

        # Reflowable line.
        if buf is None or starts_logical_line(raw):
            flush()
            buf = raw.rstrip()
        else:
            buf = buf + " " + stripped
        i += 1

    flush()
    return "\n".join(out)


def process_file(path, dry_run, quiet):
    with open(path, "r", encoding="utf-8") as f:
        original = f.read()
    trailing_nl = original.endswith("\n")
    new = reflow(original)
    if trailing_nl and not new.endswith("\n"):
        new += "\n"
    if new == original:
        return 0
    before = original.count("\n")
    after = new.count("\n")
    collapsed = before - after
    if dry_run:
        print(f"  would reflow {os.path.relpath(path)}  (-{collapsed} line breaks)")
    else:
        with open(path, "w", encoding="utf-8") as f:
            f.write(new)
        if not quiet:
            print(f"  reflowed {os.path.relpath(path)}  (-{collapsed} line breaks)")
    return 1


def gather(paths):
    files = []
    for p in paths:
        if os.path.isdir(p):
            for root, _dirs, names in os.walk(p):
                for name in names:
                    if name.endswith(".md"):
                        files.append(os.path.join(root, name))
        elif p.endswith(".md"):
            files.append(p)
    return sorted(files)


def main(argv):
    dry_run = "--dry-run" in argv
    quiet = "--quiet" in argv
    paths = [a for a in argv if not a.startswith("--")]
    if not paths:
        print("usage: reflow-md.py <path-or-dir> [...] [--dry-run] [--quiet]")
        return 2
    files = gather(paths)
    changed = sum(process_file(f, dry_run, quiet) for f in files)
    verb = "would change" if dry_run else "changed"
    print(f"reflow-md: {verb} {changed} of {len(files)} file(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
