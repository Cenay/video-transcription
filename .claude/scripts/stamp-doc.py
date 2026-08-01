#!/usr/bin/env python3
"""Prepend a session-traceability stamp to a doc, folding and rolling down the priors.

Spec: fran-dash/plans/stamp-block-format.md

The stamp chain at the top of every long-lived doc used to grow as a single
unbounded line (5,556 chars in the worst measured case) because each session
prepended into the *same paragraph*. This script replaces that prepend with
prepend-then-fold-then-roll:

    _Last updated <current stamp>_        <- stays inline, bare, one line

    <details>
    <summary>...</summary>

    - _Prior: <stamp 1>_                  <- the --keep most recent, one per line
    - _Prior: <stamp 2>_
    - _Prior: <stamp 3>_

    </details>

Everything older moves verbatim, newest-first, to history/<DOC>-stamp-history.md.
Nothing is ever trimmed or deleted -- the chain is the only way back to the
session that produced a change.

Docs still in the old one-line blob format are converted automatically on the
first run; no separate migration pass is needed.

Usage:
    stamp-doc.py DOC --stamp "_Last updated 2026-07-31 15:50 MST by ... _"
    stamp-doc.py DOC --stamp-file new-stamp.txt      # long stamps; - reads stdin
    stamp-doc.py DOC --convert-only                  # reshape, add no new stamp
    stamp-doc.py DOC --check                         # lint only, never writes
    stamp-doc.py DOC --stamp "..." --dry-run         # print both files, write neither
"""

import argparse
import re
import sys
from pathlib import Path

# A stamp delimiter is "Prior:" followed by a date -- never the bare word, which
# appears in stamp prose ("...as noted in a prior session"). The underscore is
# optional because of the malformed-delimiter case (spec section 5); the
# lookbehind stops a well-formed `_Prior:` from also matching one character in,
# which would split a single stamp into a stray `_` plus its body.
PRIOR_SPLIT = re.compile(r"(?=(?<!_)_?Prior: \d{4}-\d{2}-\d{2})")
PRIOR_LINT = re.compile(r"^- _Prior: \d{4}-\d{2}-\d{2} \d{2}:\d{2} [A-Z]{2,5}(?![A-Z])")
# Two header forms exist in the wild -- the italic `_Last updated ...` one and a
# bold `**Last updated:** ...` variant. Both are accepted and each doc keeps the
# form it already uses; resume.md and every existing grep target this line.
CURRENT_RE = re.compile(
    r"^(_Last updated |\*\*Last updated:\*\* )\d{4}-\d{2}-\d{2} \d{2}:\d{2} [A-Z]{2,5}(?![A-Z])"
)
FOLD_START = "<details>"
FOLD_END = "</details>"
SUMMARY_RE = re.compile(r"^<summary>📜 <strong>Stamp history</strong>")
MANAGED_BLOCK = "<!-- link-doc-refs:start"


class StampError(Exception):
    pass


class NoStampError(StampError):
    """The doc has no stamp yet -- a first stamp is inserted rather than an error."""


def _close(text):
    """Ensure a stamp segment is delimited by a trailing underscore."""
    text = text.rstrip()
    return text if text.endswith("_") else text + "_"


def _close_current(text):
    """Close the current stamp only if it uses the italic form."""
    text = text.rstrip()
    return _close(text) if text.startswith("_") else text


def demote(current):
    """Rewrite an outgoing current stamp into a prior. The only text edit made."""
    body = CURRENT_RE.match(current).group(1)
    return _close("_Prior: " + current[len(body):].strip())


def split_blob(line):
    """Split a legacy one-line stamp blob into (current, [priors]), repaired.

    The repair (spec section 5): a prior that lost its leading `_` is
    concatenated invisibly onto the end of the preceding stamp. Splitting on an
    *optional* underscore finds it; we then close the predecessor and open the
    orphan. This is the only edit ever made to stamp text.
    """
    segments = [s.strip() for s in PRIOR_SPLIT.split(line.strip()) if s.strip()]
    if not segments:
        raise StampError("stamp line is empty after splitting")

    current = _close_current(segments[0])
    priors, repairs = [], []
    for seg in segments[1:]:
        if not seg.startswith("_"):
            # The orphan lost its opening delimiter and was swallowed by the
            # preceding stamp. Close that one, open this one. Text is untouched.
            repairs.append(seg[:60])
            if priors:
                priors[-1] = _close(priors[-1])
            seg = "_" + seg
        priors.append(_close(seg))
    return current, priors, repairs


def parse_doc(text):
    """Return (pre, current, priors, post, had_fold).

    `pre` is everything above the stamp line, `post` everything below the stamp
    block. Priors are collected from the legacy blob first, then the fold, which
    preserves newest-first ordering in the mixed case.
    """
    lines = text.split("\n")
    # A doc can hold more than one "Last updated" line -- CURRENT_STATUS.md has a
    # bold session-summary line above its real stamp. The traceability stamp is
    # the one carrying a `transcript:` pointer; prefer it, then the italic form.
    cands = [i for i, l in enumerate(lines) if CURRENT_RE.match(l)]
    if not cands:
        raise NoStampError("no `_Last updated YYYY-MM-DD HH:MM TZ ...` line found")
    idx = next(
        (i for i in cands if "transcript:" in lines[i]),
        next((i for i in cands if lines[i].startswith("_")), cands[0]),
    )

    current, priors, repairs = split_blob(lines[idx])

    # A fold may follow, separated by a blank line.
    end = idx + 1
    had_fold = False
    scan = end
    while scan < len(lines) and lines[scan].strip() == "":
        scan += 1
    if scan < len(lines) and lines[scan].strip() == FOLD_START:
        had_fold = True
        close = next(
            (i for i in range(scan, len(lines)) if lines[i].strip() == FOLD_END), None
        )
        if close is None:
            raise StampError("stamp fold opens with <details> but never closes")
        for l in lines[scan:close]:
            if l.strip().startswith("- _Prior:"):
                priors.append(_close(l.strip()[2:]))
        end = close + 1

    return lines[:idx], current, priors, lines[end:], had_fold, repairs


def render_fold(priors, history_name):
    if not priors:
        return []
    n = len(priors)
    return [
        "",
        FOLD_START,
        f"<summary>📜 <strong>Stamp history</strong> — the {n} previous "
        f"update{'s' if n != 1 else ''} (older ones: "
        f"<code>history/{history_name}</code>)</summary>",
        "",
        *[f"- {p}" for p in priors],
        "",
        FOLD_END,
    ]


def history_header(doc_name, stamp):
    return "\n".join([
        f"# {doc_name} — stamp history",
        "",
        f"_Rolled out of `../{doc_name}`'s header{stamp}. **Verbatim, newest first "
        "— moved, never edited.** The parent file keeps the current stamp plus the "
        "most recent few; everything older lives here. This is the traceability "
        "chain (`Claude-Session` / `Claude-Transcript` pointers back to the session "
        "that produced each change), so entries are **appended and never trimmed**._",
        "",
        "---",
        "",
        "",
    ])


def merge_history(existing, rolled, doc_name):
    """Prepend rolled-down priors to the history file, newest-first.

    Insertion point is immediately after the `---` separator that follows the
    header, which keeps any trailing managed block (link-doc-refs) untouched.
    """
    if not rolled:
        return existing
    block = "\n".join(f"- {p}" for p in rolled)
    if existing is None:
        return history_header(doc_name, "") + block + "\n"

    lines = existing.split("\n")
    sep = next((i for i, l in enumerate(lines) if l.strip() == "---"), None)
    if sep is None:
        raise StampError(
            "history file has no `---` separator; cannot find the insertion point"
        )
    at = sep + 1
    while at < len(lines) and lines[at].strip() == "":
        at += 1
    return "\n".join(lines[:at] + block.split("\n") + [""] + lines[at:])


def lint(text, path):
    problems = []
    try:
        _, current, priors, _, had_fold, repairs = parse_doc(text)
    except StampError as e:
        return [str(e)]
    if current.startswith("_") and not current.endswith("_"):
        problems.append("current stamp is not closed with `_`")
    if PRIOR_SPLIT.search(current):
        problems.append("current stamp still contains an inline `Prior:` — not folded")
    if repairs:
        problems.append(f"{len(repairs)} prior(s) missing the leading `_` delimiter")
    if priors and not had_fold:
        problems.append(f"{len(priors)} prior(s) are inline, not in a <details> fold")
    for p in priors:
        if not PRIOR_LINT.match("- " + p):
            problems.append(f"malformed prior: {p[:70]}")
    return problems


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("doc", type=Path)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--stamp", help="the new `_Last updated ..._` line, verbatim")
    g.add_argument("--stamp-file", help="read the new stamp from a file ('-' = stdin)")
    ap.add_argument("--keep", type=int, default=3,
                    help="priors kept in the fold (default 3)")
    ap.add_argument("--convert-only", action="store_true",
                    help="reshape the existing chain, add no new stamp")
    ap.add_argument("--check", action="store_true", help="lint only, never write")
    ap.add_argument("--dry-run", action="store_true",
                    help="print what would be written, write nothing")
    args = ap.parse_args()

    if not args.doc.is_file():
        sys.exit(f"error: {args.doc} not found")
    original = args.doc.read_text(encoding="utf-8")

    if args.check:
        problems = lint(original, args.doc)
        for p in problems:
            print(f"✗ {args.doc}: {p}")
        if not problems:
            print(f"✓ {args.doc}: stamp block is well-formed")
        sys.exit(1 if problems else 0)

    new_stamp = args.stamp
    if args.stamp_file:
        new_stamp = (sys.stdin.read() if args.stamp_file == "-"
                     else Path(args.stamp_file).read_text(encoding="utf-8"))
    if new_stamp:
        new_stamp = _close_current(new_stamp.strip())
        if not CURRENT_RE.match(new_stamp):
            sys.exit("error: --stamp must start `_Last updated YYYY-MM-DD HH:MM TZ ` "
                     "(or the bold `**Last updated:** ...` variant)")
    elif not args.convert_only:
        sys.exit("error: pass --stamp/--stamp-file, or --convert-only")

    try:
        pre, current, priors, post, _, repairs = parse_doc(original)
    except NoStampError:
        # A doc being stamped for the first time (fresh from /init-project, or one
        # a session has only now materially rewritten). Insert below the H1.
        if not new_stamp:
            sys.exit(f"error: {args.doc} has no stamp yet and nothing to convert")
        lines = original.split("\n")
        at = next((i + 1 for i, l in enumerate(lines) if l.startswith("# ")), 0)
        while at < len(lines) and lines[at].strip() == "":
            at += 1
        seeded = "\n".join(lines[:at] + [new_stamp, ""] + lines[at:])
        if not args.dry_run:
            args.doc.write_text(seeded, encoding="utf-8")
        print(f"{'would write' if args.dry_run else 'wrote'} {args.doc}: "
              "first stamp inserted, no priors")
        return
    except StampError as e:
        sys.exit(f"error: {args.doc}: {e}")

    old_priors = list(priors)
    if new_stamp:
        # The outgoing current stamp becomes prior #1 -- the `Last updated`
        # -> `Prior:` rewrite is the only text change ever made to a stamp.
        priors = [demote(current)] + priors
        current = new_stamp

    keep, roll = priors[: args.keep], priors[args.keep:]

    hist_dir = args.doc.parent / "history"
    hist_path = hist_dir / f"{args.doc.stem}-stamp-history.md"
    existing_hist = hist_path.read_text(encoding="utf-8") if hist_path.is_file() else None
    new_hist = merge_history(existing_hist, roll, args.doc.name)

    new_doc = "\n".join(pre + [current] + render_fold(keep, hist_path.name) + post)

    # Verification (spec section 4): every prior that existed before this run must
    # appear exactly once across {parent, history}. Move, never delete.
    haystack = new_doc + (new_hist or "")
    missing = [p for p in old_priors if p not in haystack]
    dupes = [p for p in old_priors if haystack.count(p) != 1]
    if missing or dupes:
        sys.exit(f"error: refusing to write — {len(missing)} prior(s) would be lost, "
                 f"{len(dupes)} duplicated. No files changed.")

    if args.dry_run:
        print(f"--- {args.doc} (current + {len(keep)} folded) ---")
        print("\n".join(new_doc.split("\n")[: len(pre) + 1 + len(render_fold(keep, hist_path.name))]))
        print(f"\n--- {hist_path} ({len(roll)} rolled down) ---")
        print("\n".join((new_hist or "").split("\n")[:12]))
    else:
        if roll or new_hist != existing_hist:
            hist_dir.mkdir(parents=True, exist_ok=True)
            hist_path.write_text(new_hist, encoding="utf-8")  # history first, always
        args.doc.write_text(new_doc, encoding="utf-8")

    verb = "would write" if args.dry_run else "wrote"
    print(f"{verb} {args.doc}: current + {len(keep)} folded, {len(roll)} rolled to "
          f"{hist_path}" + (f" · {len(repairs)} delimiter repair(s)" if repairs else ""))


if __name__ == "__main__":
    main()
