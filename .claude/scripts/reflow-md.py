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
  - HTML comments, single- AND multi-line, and the link-doc-refs managed block
  - Block-level HTML tags (<details>, <summary>, <table>, ...) — one per line
  - Labeled field lines, with or without a leading marker glyph
    (`**Status:** ...`, `⚠️ **Time-bounded:** ...`, `★ **Build impact:** ...`)
  - Link reference definitions ([id]: url)
  - Blank lines (paragraph separators are preserved exactly)

WHAT IS JOINED:
  - Consecutive non-blank prose lines -> one line (single space between).
  - A wrapped list item and its indented continuation lines -> one line, with the
    list marker and its original indentation preserved.

LOGICAL-LINE BOUNDARIES (start a new output line even with no blank line between):
  - a list marker:  -  *  +  1.  1)
  - ANY line starting with bold:  **Status:**  **Instructions**  **Why ...**
    (protects both the stacked metadata fields that head these docs AND bold
    pseudo-headings used between list blocks)

SCOPE — by default a directory argument means "the .md files I have actually
changed", not "every .md under here". Uncommitted files only (staged, unstaged,
or untracked), resolved via git. Pass --all to walk the whole tree. Explicit
file paths are always processed as given.

OPT-OUT — a file containing `<!-- reflow-md: ignore -->` anywhere is skipped
entirely. For documents whose structure cannot be inferred: prose converted out
of .docx, where headings arrive as bare unmarked lines. Prefer this over bending
the heuristics, which costs correctness on every other file to fix one.

Usage:
    python3 reflow-md.py <path-or-dir> [more ...]   # changed files only
    python3 reflow-md.py <path-or-dir> --all        # every .md under the path
    python3 reflow-md.py <path-or-dir> --dry-run    # report only, write nothing
    python3 reflow-md.py <path-or-dir> --quiet      # apply, summary only
"""

import os
import re
import subprocess
import sys

FENCE_RE = re.compile(r"^\s*(```|~~~)")
HR_RE = re.compile(r"^\s*([-*_])(\s*\1){2,}\s*$")
HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s")
LIST_MARKER_RE = re.compile(r"^(\s*)([-*+]|\d+[.)])\s+")
# Any line that OPENS with bold starts a new logical line — not only the
# `**Label:**` form. A bold run with no colon is how these docs write pseudo-
# headings between list blocks (`**Instructions**`, `**Optional Toppings**`) and
# how the decision ledger writes un-colonned fields (`**Why ... .**`). The old
# colon-anchored pattern treated those as prose continuations and joined them
# onto the preceding list item, which changes the RENDERED output.
# Deliberate trade-off: over-splitting merely leaves a hard break that a human
# can join; over-joining silently destroys structure. Prefer the former.
# A field line may open with a marker glyph before the bold run:
#   ⚠️ **Time-bounded:** …    ★ **Build impact:** …    ✅ **Ruled:** …
# Anchoring `**` at column 0 made those invisible, so they were joined onto the
# field above -- exactly the "two records on one line" case that CLAUDE.md's
# one-line-per-RECORD rule forbids, and that every ledger parser then cannot see.
# Deliberately narrow: the prefix must be non-word, non-space, non-markup, and be
# followed by whitespace then `**`, so a line opening with a word, a backtick, a
# link, or a list marker is unaffected. Same trade-off as above -- an extra split
# is repairable by hand, a silent merge is not.
MARKER_PREFIX = r"(?:[^\w\s`*_\[\](){}<>#|-]{1,4}\s+)?"
BOLD_LABEL_RE = re.compile(rf"^{MARKER_PREFIX}\*\*(?!\*)\S")
ITALIC_LABEL_RE = re.compile(r"^_(?!_)\S")
LINK_DEF_RE = re.compile(r"^\[[^\]]+\]:\s")
COMMENT_RE = re.compile(r"^\s*<!--")
COMMENT_END_RE = re.compile(r"-->")
# Block-level HTML is structural: each tag belongs on its own line. The case that
# forced this is the stamp fold that `stamp-doc.py` builds -- `<details>` and
# `<summary>` were being joined onto one line. The `- _Prior:_` bullets inside it
# already survive via LIST_MARKER_RE; only the tag lines were being eaten.
HTML_BLOCK_RE = re.compile(
    r"^\s*</?(?:details|summary|table|thead|tbody|tfoot|tr|td|th"
    r"|div|section|p|br|hr|ul|ol|li|blockquote|img)\b",
    re.IGNORECASE,
)
# File-level opt-out. The marker must be ALONE on its line and outside any code
# fence — not merely present somewhere in the text. Matching it anywhere means
# any document that *describes* the escape (a skill, a guide, this file's own
# docstring) silently exempts itself. That bit immediately on the first run.
IGNORE_RE = re.compile(r"^\s*<!--\s*reflow-md:\s*ignore\s*-->\s*$", re.IGNORECASE)


def has_ignore_marker(text):
    """True when the opt-out appears on its own line outside a code fence."""
    in_fence = False
    for line in text.splitlines():
        if FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        if not in_fence and IGNORE_RE.match(line):
            return True
    return False


def is_bold_only_line(stripped):
    """
    True when the ENTIRE line is a single bold run — `**Instructions**`,
    `**Bake**`, `**Option 1: Stretch by Hand**`.

    These are pseudo-headings, not labels introducing inline content, and they
    are structural: the lines after them are separate blocks. Treated as
    passthrough so nothing joins TO them and nothing joins ONTO them. A line
    like `**Status:** value here` is NOT bold-only and keeps label behaviour.
    """
    if not (stripped.startswith("**") and stripped.endswith("**") and len(stripped) > 4):
        return False
    return "**" not in stripped[2:-2]


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
    if HTML_BLOCK_RE.match(raw):
        return True
    if is_bold_only_line(stripped):
        return True
    return False


def starts_logical_line(raw):
    """True if this reflowable line begins a NEW logical line (vs. a continuation)."""
    if LIST_MARKER_RE.match(raw):
        return True
    if BOLD_LABEL_RE.match(raw.strip()):
        return True
    # An italic run opening the line is the other field form these docs use —
    # `_Last updated ..._`, `_Prior status: ..._` — stacked directly under a
    # **Status:** line. Without this they get folded into the field above.
    if ITALIC_LABEL_RE.match(raw.strip()):
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

        # Preserve an HTML comment verbatim, including a MULTI-LINE one.
        # COMMENT_RE alone matches only the opening line, so the continuation
        # lines of a block comment were treated as reflowable prose and joined --
        # despite the docstring promising HTML comments are passthrough. The
        # comment that documents this script's own opt-out is six lines long, so
        # this is what made `--dry-run` on a governed ledger report changes that
        # could never be reconciled away.
        if COMMENT_RE.match(raw):
            flush()
            while i < n:
                out.append(lines[i])
                if COMMENT_END_RE.search(lines[i]):
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

    # Explicit opt-out, for files whose structure a formatter genuinely cannot
    # infer — the classic case is prose converted out of .docx, where headings
    # arrive as plain bare lines with no marker of any kind. Without an escape
    # those files are reported on every run forever, and the standing temptation
    # is to keep widening the heuristics until they misfire on real prose.
    # Same convention as doc-reconcile's `<!-- doc-reconcile: ignore -->`.
    if has_ignore_marker(original):
        if not quiet:
            print(f"  skipped {path}  (reflow-md: ignore)")
        return 0

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


def walk_md(path):
    found = []
    for root, _dirs, names in os.walk(path):
        for name in names:
            if name.endswith(".md"):
                found.append(os.path.join(root, name))
    return found


def git_changed_md(path):
    """
    .md files under `path` with UNCOMMITTED changes — staged, unstaged, or
    untracked. Returns None if `path` is not inside a git work tree, so the
    caller can decide what to do rather than silently reformatting everything.
    """
    try:
        top = subprocess.run(
            ["git", "-C", path, "rev-parse", "--show-toplevel"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", top, "status", "--porcelain"],
            capture_output=True, text=True, check=True,
        ).stdout
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return None

    want = os.path.abspath(path)
    files = []
    for line in status.splitlines():
        if len(line) < 4:
            continue
        entry = line[3:]
        if " -> " in entry:            # rename: take the destination
            entry = entry.split(" -> ", 1)[1]
        entry = entry.strip().strip('"')
        if not entry.endswith(".md"):
            continue
        full = os.path.abspath(os.path.join(top, entry))
        if full == want or full.startswith(want.rstrip(os.sep) + os.sep):
            if os.path.isfile(full):
                files.append(full)
    return files


def gather(paths, use_all):
    files, skipped_dirs = [], []
    for p in paths:
        if os.path.isdir(p):
            if use_all:
                files.extend(walk_md(p))
                continue
            changed = git_changed_md(p)
            if changed is None:
                # Not a git work tree — cannot tell what was edited. Refuse to
                # touch the whole tree by accident; require an explicit --all.
                skipped_dirs.append(p)
                continue
            files.extend(changed)
        elif p.endswith(".md"):
            files.append(os.path.abspath(p))
    return sorted(set(files)), skipped_dirs


def main(argv):
    dry_run = "--dry-run" in argv
    quiet = "--quiet" in argv
    use_all = "--all" in argv
    paths = [a for a in argv if not a.startswith("--")]
    if not paths:
        print("usage: reflow-md.py <path-or-dir> [...] [--all] [--dry-run] [--quiet]")
        return 2

    files, skipped = gather(paths, use_all)
    for p in skipped:
        print(f"reflow-md: {p} is not in a git work tree — skipped. "
              f"Re-run with --all to reformat every .md under it.")

    changed = sum(process_file(f, dry_run, quiet) for f in files)
    verb = "would change" if dry_run else "changed"
    scope = "all .md" if use_all else "changed .md"
    print(f"reflow-md: {verb} {changed} of {len(files)} {scope} file(s)")
    if not use_all and not files and not skipped:
        print("reflow-md: nothing uncommitted to reflow "
              "(this is the default scope — pass --all to widen it).")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
