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
    problems.extend(lint_shadow_chain(text))
    return problems


# A doc may legitimately carry a bold `**Last updated:**` summary line ABOVE the
# real traceability stamp (see parse_doc) -- CURRENT_STATUS.md does. That line is
# a one-line snapshot, and it is NOT a stamp chain: it has no fold, no roll-down
# and no cap, so anything prepended to it grows forever.
#
# Measured 2026-08-11 in fran-dash: two `_Prior:_` lines had accumulated beneath
# that bold line -- 1,960 characters of header, every byte of it already present
# in both the stamp fold and the session block it summarised. That is the same
# shape as the 5,556-character stamp line this script was written to prevent,
# starting over in a file the script was not looking at.
#
# parse_doc deliberately skips these lines (it locks onto the `transcript:`
# stamp), so without this check they are invisible to every existing guard.
SHADOW_PRIOR_RE = re.compile(r"^_Prior:_\s")


def lint_shadow_chain(text):
    """Standalone `_Prior:_` lines forming a second, unbounded stamp chain."""
    hits = [l for l in text.split("\n") if SHADOW_PRIOR_RE.match(l)]
    if not hits:
        return []
    chars = sum(len(l) for l in hits)
    return [
        f"{len(hits)} standalone `_Prior:_` line(s) ({chars} chars) form a second, "
        f"unbounded stamp chain outside the fold — the `**Last updated:**` snapshot "
        f"takes NO prior chain; move the history to the fold or delete it as redundant"
    ]


# ⛔ THE ONE THING THIS FILE CANNOT SEE FROM THE FILE ITSELF: a stamp that used
# to exist and no longer does. Every other check here is internal consistency —
# is the fold well-formed, is the current stamp closed, is there a shadow chain.
# A stamp that was silently dropped leaves a perfectly valid file behind.
#
# ✅ Measured 2026-08-21 on this repo: 24 distinct stamps had existed across the
# history of `docs/CURRENT_STATUS.md`; **8 were present.** Sixteen were gone,
# lost at a rate of exactly ONE PER COMMIT between 2026-07-14 and 2026-07-31 —
# the era before this script, when a session hand-prepended its stamp and
# REPLACED the previous one instead of accumulating it. Nothing has been lost
# since; the fold and roll-down work. But nothing noticed the old loss either,
# and the file that is supposed to be the permanent record read as healthy.
#
# ★ git is the only witness. The audit asks it what stamps this doc has ever
# carried and fails if any of them is absent from the doc plus its history file.
AUDIT_DATE_RE = re.compile(
    r"_(?:Last updated|Prior:)\s+(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}\s+[A-Z]{2,5})")


def _git(args, cwd):
    import subprocess
    r = subprocess.run(["git"] + args, cwd=cwd, capture_output=True, text=True)
    return r.stdout if r.returncode == 0 else ""


def audit_history(doc: Path):
    """Every stamp git has ever seen for this doc must still be present.

    Returns (problems, checked, ever). `checked` is False when the audit could
    not run — not a git repo, or git unavailable — and the caller MUST say so
    rather than printing a clean result. A checker that silently skips is the
    failure it exists to catch.
    """
    root = doc.resolve().parent
    while root != root.parent and not (root / ".git").exists():
        root = root.parent
    if not (root / ".git").exists():
        return [], False, set()

    hist = doc.parent / "history" / f"{doc.stem}-stamp-history.md"
    rels = []
    for f in (doc, hist):
        try:
            rels.append(str(f.resolve().relative_to(root)))
        except ValueError:
            return [], False, set()

    ever = set()
    revs = _git(["log", "--format=%H", "--", rels[0]], root).split()
    if not revs:
        return [], False, set()
    for rev in revs:
        for rel in rels:
            for m in AUDIT_DATE_RE.finditer(_git(["show", f"{rev}:{rel}"], root)):
                ever.add(m.group(1).strip())

    now = set()
    for f in (doc, hist):
        if f.exists():
            for m in AUDIT_DATE_RE.finditer(f.read_text(encoding="utf-8")):
                now.add(m.group(1).strip())

    lost = sorted(ever - now)
    if not lost:
        return [], True, ever
    shown = ", ".join(lost[:6]) + (f" … and {len(lost) - 6} more" if len(lost) > 6 else "")
    return ([f"{len(lost)} stamp(s) that exist in git history are ABSENT from this "
             f"doc and its stamp-history file — {shown}. Recover them with "
             f"`git log -p -- {rels[0]}`; this file is append-only and nothing "
             f"may leave it."], True, ever)


def restore_history(doc: Path, apply=False):
    """Recover stamps that exist in git but not in the files, VERBATIM.

    ⛔ Recovery is append-only and text-preserving by construction: each stamp is
    taken byte-for-byte from the revision where it was current, only its
    `_Last updated ` prefix rewritten to `_Prior: ` so it reads as history. No
    stamp already present is touched, and none is ever removed.

    ⚠️ Restoring by hand does not scale and is how MORE get lost — 130 were
    missing across three repos when this was written.
    """
    root = doc.resolve().parent
    while root != root.parent and not (root / ".git").exists():
        root = root.parent
    if not (root / ".git").exists():
        return [], "not a git repo"

    hist = doc.parent / "history" / f"{doc.stem}-stamp-history.md"
    rel = str(doc.resolve().relative_to(root))
    hrel = str(hist.resolve().relative_to(root))

    line_re = re.compile(
        r"^\s*(?:[-*]\s+)?_(?:Last updated|Prior:)\s+"
        r"(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}\s+[A-Z]{2,5}).*?_\s*$", re.M)
    # ⛔ Historical revisions include the LEGACY ONE-LINE BLOB era, where several
    # stamps were concatenated onto a single line — the 5,556-character line this
    # script was written to end. A line-anchored regex sees one stamp there and
    # misses the rest: 8 of the 130 losses were invisible to the first version of
    # this function for exactly that reason.
    #
    # ★ `split_blob` is the parser that already knows that shape, including the
    # repair for a prior that lost its leading `_`. Reusing it is the point —
    # a second private parser here is how one file starts giving two answers.
    def stamps_in(text):
        out = {}
        for line in text.split("\n"):
            if not AUDIT_DATE_RE.search(line):
                continue
            candidates = [line]
            if len(PRIOR_SPLIT.findall(line)) > 0:
                try:
                    cur, priors, _ = split_blob(line.lstrip("-* ").strip())
                    candidates = [cur] + priors
                except StampError:
                    pass
            for c in candidates:
                m = AUDIT_DATE_RE.search(c)
                if m:
                    out.setdefault(m.group(1).strip(), c.strip())
        return out

    found = {}
    for rev in _git(["log", "--format=%H", "--", rel], root).split():
        for r in (rel, hrel):
            for d, t in stamps_in(_git(["show", f"{rev}:{r}"], root)).items():
                found.setdefault(d, t)

    present = set()
    for f in (doc, hist):
        if f.exists():
            for m in AUDIT_DATE_RE.finditer(f.read_text(encoding="utf-8")):
                present.add(m.group(1).strip())

    missing = {d: t for d, t in found.items() if d not in present}

    # ⚠️ A rewrite is also warranted with NOTHING missing: the history file may
    # already hold blob lines that need splitting. Returning early on `missing`
    # alone left 19 of them in place and reported "nothing to restore" — a clean
    # message over a file still carrying the shape this script exists to remove.
    blobs = 0
    if hist.exists():
        for m in re.finditer(r"^- _Prior:.*$", hist.read_text(encoding="utf-8"), re.M):
            if len(AUDIT_DATE_RE.findall(m.group(0))) > 1:
                blobs += 1
    if (not missing and not blobs) or not apply:
        return sorted(missing), None

    rows = []
    if hist.exists():
        body = hist.read_text(encoding="utf-8")
        head, _, _ = body.partition("- _Prior:")
        # ⛔ EXISTING rows are split too, not just recovered ones. A history file
        # can already CONTAIN blob lines — 19 of them in fran-dash — either
        # rolled down from the blob era or re-imported whole by a restore that
        # predated the split. Leaving them is leaving the exact shape this
        # script exists to eliminate, in the file that is meant to be the
        # permanent record. Splitting here makes `--restore` self-healing and
        # idempotent: a line already carrying one stamp passes through untouched.
        for m in re.finditer(r"^- _Prior:.*$", body, re.M):
            line = m.group(0)
            for t in stamps_in(line).values():
                d = AUDIT_DATE_RE.search(t)
                if d:
                    t = re.sub(r"^_Last updated ", "_Prior: ", t.lstrip("-* ").strip())
                    rows.append((d.group(1).strip(), "- " + t))
    else:
        hist.parent.mkdir(parents=True, exist_ok=True)
        head = history_header(doc.name, "")
    for d, txt in missing.items():
        t = txt.lstrip("-* ").strip()
        t = re.sub(r"^_Last updated ", "_Prior: ", t)
        rows.append((d, "- " + t))

    rows.sort(key=lambda r: r[0], reverse=True)   # newest first, as the file states
    hist.write_text(head.rstrip("\n") + "\n\n" + "\n".join(t for _, t in rows) + "\n",
                    encoding="utf-8")
    return sorted(missing), None


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
    ap.add_argument("--restore", action="store_true",
                    help="recover stamps present in git but absent from the files")
    ap.add_argument("--dry-run", action="store_true",
                    help="print what would be written, write nothing")
    args = ap.parse_args()

    if not args.doc.is_file():
        sys.exit(f"error: {args.doc} not found")
    original = args.doc.read_text(encoding="utf-8")

    if args.restore:
        missing, why = restore_history(args.doc, apply=not args.dry_run)
        if why:
            print(f"⚠ {args.doc}: cannot restore — {why}")
            sys.exit(2)
        if not missing:
            print(f"✓ {args.doc}: nothing to restore")
        else:
            verb = "would restore" if args.dry_run else "restored"
            print(f"✓ {args.doc}: {verb} {len(missing)} stamp(s) — "
                  f"{missing[0]} … {missing[-1]}")
        sys.exit(0)

    if args.check:
        problems = lint(original, args.doc)
        # ⛔ The audit runs as part of --check, not behind its own flag. A guard
        # that has to be remembered is the one that was missing here for five
        # weeks while sixteen stamps went out of the file.
        lost, checked, ever = audit_history(args.doc)
        problems += lost
        for p in problems:
            print(f"✗ {args.doc}: {p}")
        # ⚠️ A checker must NAME WHAT IT DID NOT CHECK. Silence has to mean
        # "nothing there", never "I could not look" -- otherwise the tool built
        # to catch a silent loss becomes another way to have one.
        if not checked:
            print(f"⚠ {args.doc}: history audit NOT run (no git repo, or the doc "
                  f"has no commits) — loss of an old stamp cannot be detected here")
        elif not lost:
            print(f"✓ {args.doc}: history audit — all {len(ever)} stamp(s) ever "
                  f"committed are still present")
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
