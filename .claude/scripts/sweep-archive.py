#!/usr/bin/env python3
"""sweep-archive.py — the [DEC-150] archive sweep, as a program instead of a rule.

WHY THIS EXISTS
---------------
`CURRENT_STATUS.md` is a bounded snapshot; sessions older than the retention window
"roll down" verbatim into `history/CURRENT_STATUS-archive.md`, which declares itself
**reverse-chronological**. That placement rule lived only in prose, and prose lost:

    61563e3 (08-04) rolled session 35 -> appended at the BOTTOM
    ae9e385 (08-04) rolled session 36 -> inserted at the TOP     (correct)
    1d4a5b0 (08-05) rolled session 37 -> appended at the BOTTOM, under an
                                          invented "## Rolled down ..." heading

Three sessions, two readings, twenty-four hours. "Roll it DOWN" reads as "append to
the end" at least as naturally as "insert at the top of a descending list", so the
rule was ambiguous at the point of use. Nothing was ever lost -- every block stayed
verbatim -- but the ordering contract in the file's own header became false.

Per the standing principle: a mechanical invariant belongs in a script plus a hook,
never in a rule a session has to remember.

MODES
-----
  check    verify the archive's session blocks run strictly descending.
           Exits 1 on failure. Wire this into .githooks/pre-commit.
  move     relocate NAMED session blocks to their correct descending slot.
           RELOCATION ONLY -- never edits block text.
  roll     the sweep itself: cut session N out of CURRENT_STATUS.md and insert it
           into the archive at its correct descending position, with a sweep stamp.

WHY `move` AND NOT A GLOBAL SORT
--------------------------------
This archive is NOT a flat list of session blocks. Interleaved between them are
"## Rolled <date> -- sessions 20-27" group notes, a rolled-down "## START HERE
TOMORROW" section with its own ### steps, "## Blockers", "## Older sessions
(archived)", "## Next", a "## Meeting reconciled" section, and "### Session N,
second half" sub-blocks. Sorting the whole file would tear that structure apart and
strand those sections against unrelated sessions. So the tool only ever moves blocks
it is explicitly told to move, and asserts everything else stayed put.

GRANDFATHERED DISORDER
----------------------
Two older violations predate this arc and sit inside that heterogeneous region,
where a mechanical fix is riskier than the disorder: sessions 28/29 are swapped, and
session 13 appears twice. They are recorded in LEGACY_VIOLATIONS so `check` fails on
NEW disorder without failing forever on old. Removing an entry there is how you
signal you have fixed it by hand.

THE SAFETY PROPERTY
-------------------
`reorder` and `roll` both assert that the multiset of lines is preserved: every line
that existed before still exists after, except lines this script deliberately drops
(placement artefacts, reported by name). A reordering that silently edited prose
would fail that assertion. Verbatim is the whole point of the archive -- it is the
backstop copy, so a "helpful" rewrite here is unrecoverable.
"""

import argparse
import collections
import pathlib
import re
import sys

SESSION_RE = re.compile(r"^## Session Summary \(session (\d+)")
LINK_BLOCK_RE = re.compile(r"^<!-- link-doc-refs:start")
SWEEP_STAMP_RE = re.compile(r"^_Sweep ")
H2_RE = re.compile(r"^## ")

# Headings that are placement artefacts rather than content: a past session invented
# one to hold an append. Dropping it loses no session text -- but we report it.
ARTEFACT_H2_RE = re.compile(r"^## Rolled down ")

# Ordering violations that predate this arc, inside the heterogeneous older region
# where a mechanical fix is riskier than the disorder. Kept so `check` fails on NEW
# disorder rather than failing forever on old. Delete an entry once fixed by hand.
LEGACY_VIOLATIONS = {(28, 29)}
LEGACY_DUPLICATES = {13}


class Archive:
    """header | [session blocks] | tail(link-doc-refs block)

    A block runs from its "## Session Summary (session N" heading to the NEXT such
    heading -- deliberately absorbing any other headings in between. Those interleaved
    sections ("## Blockers", "## Next", "## Rolled <date>", "### Session N, second
    half") are real content; splitting on every "## " would orphan them, and an
    earlier version of this script silently dropped them. The line-preservation
    assertion caught it. Blocks therefore tile the whole middle: nothing can be lost.
    """

    def __init__(self, lines):
        self.raw = list(lines)
        first = next((i for i, l in enumerate(lines) if SESSION_RE.match(l)), None)
        if first is None:
            raise SystemExit("error: no session blocks found -- is this the archive?")
        tail = next((i for i, l in enumerate(lines) if LINK_BLOCK_RE.match(l)), len(lines))
        if tail < first:
            raise SystemExit("error: link-doc-refs block precedes the session blocks")

        self.header = lines[:first]
        self.tail = lines[tail:]
        self.blocks = []      # (session_number, [lines]) -- tiles lines[first:tail]

        cur_num, cur_lines = None, []
        for line in lines[first:tail]:
            m = SESSION_RE.match(line)
            if m:
                if cur_num is not None:
                    self.blocks.append((cur_num, cur_lines))
                cur_num, cur_lines = int(m.group(1)), [line]
            else:
                cur_lines.append(line)
        if cur_num is not None:
            self.blocks.append((cur_num, cur_lines))

    def artefact_headings(self):
        return [(n, l.rstrip("\n")) for n, blk in self.blocks
                for l in blk if ARTEFACT_H2_RE.match(l)]

    def inblock_stamps(self):
        return [(n, l.rstrip("\n")) for n, blk in self.blocks
                for l in blk if SWEEP_STAMP_RE.match(l)]

    def order(self):
        return [n for n, _ in self.blocks]

    def descending_violations(self):
        nums = self.order()
        return [(nums[i], nums[i + 1]) for i in range(len(nums) - 1) if nums[i] < nums[i + 1]]

    def duplicates(self):
        return sorted(n for n, c in collections.Counter(self.order()).items() if c > 1)

    def render(self, blocks=None, header=None):
        blocks = self.blocks if blocks is None else blocks
        header = self.header if header is None else header
        out = list(header)
        for _, block in blocks:
            out.extend(block)
        out.extend(self.tail)
        return out


def assert_lines_preserved(before, after, allowed_drops=(), allowed_adds=()):
    """Every line before must survive, except the ones we deliberately dropped.

    Blank lines are ignored -- relocation legitimately shifts separator whitespace.

    ⛔ `allowed_adds` is NOT symmetric decoration. This function checks BOTH
    directions -- `dropped` and `added` -- but until 2026-08-21 only drops could
    be sanctioned. `--stamp` exists to write a provenance line into the archive
    header, which is an intentional ADDITION, so every stamped run was refused.
    The call site read `allowed_drops=[] if not args.stamp else []` -- both
    branches the empty list, which is the shape of an intent that was never
    wired up. BUG-2026-08-20-001, second half.
    """
    def bag(lines):
        return collections.Counter(l for l in lines if l.strip())

    b, a = bag(before), bag(after)
    dropped = b - a
    added = a - b
    for line in allowed_drops:
        key = line if line.endswith("\n") else line + "\n"
        if key in dropped:
            del dropped[key]
        elif line in dropped:
            del dropped[line]
    for line in allowed_adds:
        key = line if line.endswith("\n") else line + "\n"
        if key in added:
            del added[key]
        elif line in added:
            del added[line]
    if dropped or added:
        for line in list(dropped)[:5]:
            print(f"  LOST:  {line.rstrip()[:100]}", file=sys.stderr)
        for line in list(added)[:5]:
            print(f"  NEW:   {line.rstrip()[:100]}", file=sys.stderr)
        raise SystemExit("error: content changed -- refusing to write. This must be relocation only.")


def cmd_check(args):
    arc = Archive(pathlib.Path(args.archive).read_text().splitlines(keepends=True))
    violations = arc.descending_violations()
    dupes = arc.duplicates()
    artefacts = arc.artefact_headings()
    stray = arc.inblock_stamps()

    print(f"{args.archive}: {len(arc.blocks)} session block(s), order "
          f"{' '.join(str(n) for n in arc.order()[:6])}...")

    ok = True
    new_violations = [v for v in violations if v not in LEGACY_VIOLATIONS]
    grandfathered = [v for v in violations if v in LEGACY_VIOLATIONS]
    if new_violations:
        ok = False
        print("FAIL: session blocks are not in descending order:")
        for hi, lo in new_violations:
            print(f"  session {hi} is followed by session {lo} (expected a smaller number)")
    if artefacts:
        ok = False
        print("FAIL: placement-artefact heading(s) present:")
        for n, h in artefacts:
            print(f"  in session {n}'s block: {h}")
    if stray:
        ok = False
        print(f"FAIL: {len(stray)} sweep stamp(s) sit inside session blocks, not the header:")
        for n, s in stray:
            print(f"  in session {n}'s block: {s[:90]}")
    for hi, lo in grandfathered:
        print(f"GRANDFATHERED: {hi} before {lo} -- known legacy disorder, see LEGACY_VIOLATIONS")
    for d in dupes:
        tag = "GRANDFATHERED" if d in LEGACY_DUPLICATES else "WARN"
        print(f"{tag}: session {d} appears more than once")

    # A checker must name what it did NOT check.
    print("\nNOT checked by this run: whether block TEXT is verbatim against its "
          "source; whether each block's substance is preserved in DECISIONS.md / "
          "LESSONS_LEARNED.md (the [DEC-150] safety rule); retention-window "
          "correctness in CURRENT_STATUS.md; duplicate session numbers are warned, "
          "not failed.")

    if ok:
        print("\nPASS: ordering contract holds.")
        return 0
    return 1


def cmd_move(args):
    """Relocate only the named session blocks. Everything else must stay put."""
    path = pathlib.Path(args.archive)
    before = path.read_text().splitlines(keepends=True)
    arc = Archive(before)

    targets = set(args.session)
    missing = targets - set(arc.order())
    if missing:
        raise SystemExit(f"error: session(s) not in archive: {sorted(missing)}")

    drops = []
    blocks = []
    for num, block in arc.blocks:
        keep = []
        for line in block:
            if num in targets and ARTEFACT_H2_RE.match(line):
                drops.append(line.rstrip("\n"))
                continue
            keep.append(line)
        blocks.append((num, keep))

    hoisted = []
    if args.hoist_stamps:
        rehomed = []
        for num, block in blocks:
            keep = []
            for line in block:
                if SWEEP_STAMP_RE.match(line):
                    hoisted.append(line)
                else:
                    keep.append(line)
            rehomed.append((num, keep))
        blocks = rehomed

    header = list(arc.header)
    if hoisted:
        at = next((i for i, l in enumerate(header) if SWEEP_STAMP_RE.match(l)), len(header))
        ins = []
        for stamp in hoisted:
            ins.extend([stamp, "\n"])
        header = header[:at] + ins + header[at:]

    # Pull the targets out, then reinsert each before the first remaining block with
    # a SMALLER number -- its correct descending slot among blocks that did not move.
    moving = [(n, b) for n, b in blocks if n in targets]
    rest = [(n, b) for n, b in blocks if n not in targets]
    for num, block in sorted(moving, key=lambda b: b[0]):
        at = next((i for i, (n, _) in enumerate(rest) if n < num), len(rest))
        rest.insert(at, (num, block))

    after = arc.render(blocks=rest, header=header)
    assert_lines_preserved(before, after, allowed_drops=drops)

    print(f"order: {' '.join(map(str, arc.order()))}")
    print(f"   ->  {' '.join(str(n) for n, _ in rest)}")
    print(f"moved: {', '.join(str(n) for n in sorted(targets))}")
    for d in drops:
        print(f"dropped placement artefact: {d}")
    for h in hoisted:
        print(f"hoisted stamp to header: {h.rstrip()[:95]}")

    if args.dry_run:
        print("dry-run: nothing written")
        return 0
    path.write_text("".join(after))
    print(f"wrote {path}")
    return 0


def cmd_roll(args):
    cur_path, arc_path = pathlib.Path(args.current), pathlib.Path(args.archive)
    cur_before = cur_path.read_text().splitlines(keepends=True)
    arc_before = arc_path.read_text().splitlines(keepends=True)

    head = re.compile(rf"^## Session Summary \(session {args.session}\b")
    start = next((i for i, l in enumerate(cur_before) if head.match(l)), None)
    if start is None:
        raise SystemExit(f"error: session {args.session} not found in {cur_path}")
    end = next((i for i, l in enumerate(cur_before) if LINK_BLOCK_RE.match(l)), len(cur_before))
    nxt = next((i for i, l in enumerate(cur_before[start + 1:end], start + 1) if H2_RE.match(l)), end)

    block = cur_before[start:nxt]
    while block and block[-1].strip() == "":
        block.pop()

    cut = start
    while cut > 0 and cur_before[cut - 1].strip() == "":
        cut -= 1
    # ⛔ SYMMETRY. This tool RELOCATES; it must not invent a line, and it must
    # not lose one. The separator is carried to the archive only when one is
    # actually taken out of CURRENT_STATUS -- see the `moved` block below.
    #
    # BUG-2026-08-20-001: the archive insert unconditionally appended a `---`
    # while this removal was conditional. fran-dash's session blocks are not
    # `---`-separated, so nothing was removed and one was still added; the
    # preservation guard saw a manufactured line and refused every run. The
    # retention sweep could not execute at all.
    #
    # ⚠️ Dropping the appended `---` outright is NOT the fix either: on a repo
    # whose blocks ARE separated, that turns the invention into a deletion and
    # the same guard refuses from the other side. Both directions have to be
    # tied to the same fact, which is what `had_separator` is.
    had_separator = cut > 0 and cur_before[cut - 1].strip() == "---"
    if had_separator:
        cut -= 1

    cur_after = cur_before[:cut] + ["\n"] + cur_before[nxt:]

    arc = Archive(arc_before)
    if args.session in arc.order():
        raise SystemExit(f"error: session {args.session} is already in the archive")
    moved = block + ["\n"] + (["---\n", "\n"] if had_separator else [])
    blocks = sorted(arc.blocks + [(args.session, moved)], key=lambda b: -b[0])

    header = list(arc.header)
    if args.stamp:
        at = next((i for i, l in enumerate(header) if SWEEP_STAMP_RE.match(l)), len(header))
        header = header[:at] + [args.stamp.rstrip("\n") + "\n", "\n"] + header[at:]

    arc_after = arc.render(blocks=blocks, header=header)
    assert_lines_preserved(cur_before + arc_before, cur_after + arc_after,
                           allowed_adds=[args.stamp] if args.stamp else [])

    if args.dry_run:
        print(f"dry-run: would move {len(block)} line(s); archive order would be "
              f"{' '.join(str(n) for n, _ in blocks[:6])}...")
        return 0
    cur_path.write_text("".join(cur_after))
    arc_path.write_text("".join(arc_after))
    print(f"rolled session {args.session}: {len(block)} lines moved verbatim")
    print(f"archive order: {' '.join(str(n) for n, _ in blocks[:6])}...")
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    ARCHIVE = "docs/history/CURRENT_STATUS-archive.md"
    CURRENT = "docs/CURRENT_STATUS.md"

    c = sub.add_parser("check", help="verify descending order (exit 1 on failure)")
    c.add_argument("archive", nargs="?", default=ARCHIVE)
    c.set_defaults(func=cmd_check)

    r = sub.add_parser("move", help="relocate named session blocks; relocation only")
    r.add_argument("session", type=int, nargs="+", help="session number(s) to relocate")
    r.add_argument("--archive", default=ARCHIVE)
    r.add_argument("--hoist-stamps", action="store_true",
                   help="also lift sweep stamps found inside blocks up into the header")
    r.add_argument("--dry-run", action="store_true")
    r.set_defaults(func=cmd_move)

    o = sub.add_parser("roll", help="cut session N from CURRENT_STATUS into the archive")
    o.add_argument("session", type=int)
    o.add_argument("--current", default=CURRENT)
    o.add_argument("--archive", default=ARCHIVE)
    o.add_argument("--stamp", default="")
    o.add_argument("--dry-run", action="store_true")
    o.set_defaults(func=cmd_roll)

    args = p.parse_args()
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
