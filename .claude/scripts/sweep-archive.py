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


# ── THE UNRESOLVED-ITEM GATE ────────────────────────────────────────────────────
#
# ⛔ RULED 2026-08-30 by Cenay: "Something should NOT roll when it carries an
# unresolved item, ever."
#
# WHY THIS EXISTS. The [DEC-150] retention window is an AGE test -- "older than the
# last ~2 days or 4 sessions". Age is a bad proxy for done. In a week of heavy
# documentation work, a block three sessions old can still be the only place an open
# question is written down, and rolling it moves that question out of the file that
# /resume actually loads. The pre-existing safety rule asks "is this block's SUBSTANCE
# preserved somewhere?" -- which is a different and weaker question than "is anything
# in here still OPEN?". A block can satisfy the first and still bury the second.
#
# Measured on 2026-08-30, the run that prompted the ruling: five blocks were rolled
# under the age rule; sessions 60 and 61 carried live unresolved items (the three
# CHALLENGED Art decisions plus [DEC-258] 🚧 OPEN; the rename ruling owed on
# [DEC-071]/[DEC-083]/[DEC-107]/[DEC-117]). Both were recoverable only because a human
# read them. Nothing in the tooling looked.
#
# HARD markers BLOCK the roll and have no override -- "ever" was the ruling. The way
# to unblock a block is to resolve the item or re-home it, which is the correct
# incentive. SOFT markers are ADVISORY: a good checkpoint always says what it left
# undone, so blocking on that phrase would freeze the file permanently. They are
# printed by name so that silence never means "I did not look".
# RULED 2026-08-30 by Cenay, and this is the whole invariant:
#   "nothing can be removed out of the file if it's open, outstanding, work in
#    process or otherwise not complete."
#
# THE GATE FAILS TOWARD HOLDING, ON PURPOSE. A false positive costs a block that
# stays in a file it was already in. A false negative loses work. Those are not
# symmetric, so every judgement call below resolves toward refusing -- which is why
# a bare glyph inside quoted prose still counts, and why nothing here was narrowed
# to cut noise. Noise is the cheap failure.
#
# THERE ARE NO SOFT MARKERS ANY MORE. The earlier split had "left undone",
# "carried forward", "deferred" and "parked" as advisory, reasoning that every good
# checkpoint says what it left undone, so blocking on the phrase would freeze the
# file. That optimized for the file getting shorter, which is not the requirement.

# ── THE GATE IS PER-REPO OPT-IN ────────────────────────────────────────────────
#
# ⛔ RULED 2026-08-31 by Cenay: the [DEC-267] no-override gate is FRAN-DASH ONLY.
#
# WHY THIS SEAM EXISTS. This file is a SHARED_SCRIPTS asset — `sync-shared.sh`
# delivers it to eleven repos. The gate was ruled for fran-dash, whose
# CURRENT_STATUS.md is a working ledger with live open items in it. Shipping it
# on-by-default would silently impose that ruling on ten repos that never made
# it, including ones (`video-transcription`, `Staff_Form`) that are not
# doc-ledger projects at all. ✅ Measured before choosing the default: run against
# `dashboard`'s real CURRENT_STATUS.md, the gate would hold 7 of its 11 session
# blocks — so "on by default" is not a theoretical imposition.
#
# ★ A MARKER FILE, NOT A FLAG, AND THAT IS THE WHOLE POINT. A `--gate` flag would
# be an override by another name: any by-hand `roll` that omitted it would sweep
# past the gate, which is exactly what "no override, ever" forbids. The marker is
# a property of the REPO, so every invocation in fran-dash is gated and no
# invocation anywhere else is. Same shape as `.claude/ledger-siblings`.
#
# ⚠️ THE RESIDUAL RISK, STATED: deleting the marker turns the gate off silently.
# It is committed, so the deletion shows up in a diff — but nothing refuses it.
# That is a real (small) hole in "no override" and it is named here rather than
# papered over.
GATE_MARKER = ".claude/sweep-gate"


def gate_enabled(start=None):
    """Is the unresolved-item gate switched on for THIS repo?

    Returns (bool, note). The note is printed by callers so that a disabled gate
    is always visible — ⛔ silence must never be the difference between "checked
    and clean" and "did not look", which is the failure this whole file exists
    to prevent.
    """
    here = pathlib.Path(start or ".").resolve()
    for d in (here, *here.parents):
        if (d / GATE_MARKER).exists():
            return True, ""
        if (d / ".git").exists():
            break
    return False, (f"gate NOT enabled here (no {GATE_MARKER}) — "
                   "unresolved-item checks were SKIPPED, not passed")


HARD_MARKERS = [
    ("open status glyph", re.compile("[\U0001F6A7⏰⏳⏸\U0001F7E1]")),
    ("unchecked task box", re.compile(r"^\s*[-*]\s\[ \]")),
    ("ruling owed", re.compile(
        r"UNRULED|UNDECIDED|UNRESOLVED|needs? a ruling|need a ruling from|awaiting a ruling", re.I)),
    ("challenged decision", re.compile(r"CHALLENGED")),
    ("open in caps", re.compile(r"\bOPEN\b")),
    ("work in process", re.compile(
        r"\bWIP\b|work in progress|work in process|\bin progress\b|mid-entry", re.I)),
    ("not complete", re.compile(
        r"not complete|incomplete|unfinished|not finished|not started|never started"
        r"|not yet|yet to be|still owed|\bowed\b|left undone|carried forward", re.I)),
    ("to do / to be determined", re.compile(r"\bTODO\b|\bTBD\b")),
    ("outstanding", re.compile(r"\boutstanding\b", re.I)),
    ("deferred / parked / revisit", re.compile(
        r"\b(?:deferred|parked|revisit|follow[- ]up)\b", re.I)),
    ("blocked / waiting", re.compile(r"blocked on|waiting on|\bawaiting\b|\bpending\b", re.I)),
    ("section headed as open", re.compile(
        r"^#{2,4} .*\b(open|unresolved|outstanding|owed|blocked|pending|carried forward"
        r"|to be decided|in progress|next steps?)\b", re.I)),
]

# Kept as an empty list rather than deleted: the reporting path still distinguishes
# blocking from advisory, and a future ruling may re-introduce one. An empty list is
# a stated position; a deleted code path is an accident waiting to be re-added.
SOFT_MARKERS = []

# Both heading forms. The archive is uniformly "## Session Summary (session N",
# which is what SESSION_RE above parses; CURRENT_STATUS.md switched to
# "## Session N — ..." around session 63, and cmd_roll was blind to the new form --
# it raised "session N not found", so it failed loudly rather than silently, but a
# sweep of sessions 63+ was impossible.
CURRENT_SESSION_RE = re.compile(r"^## (?:Session Summary \(session (\d+)|Session (\d+)\b)")


def current_session_number(line):
    m = CURRENT_SESSION_RE.match(line)
    if not m:
        return None
    return int(m.group(1) or m.group(2))


def open_decision_ids(ledger_path):
    """IDs whose ledger Status line carries the 🚧 OPEN marker.

    Returns (ids, note). `note` is non-empty when the ledger could not be read --
    the caller must surface it rather than treating an empty set as "nothing open".
    """
    path = pathlib.Path(ledger_path)
    if not path.exists():
        return set(), f"ledger not found at {ledger_path} -- open-decision citations NOT checked"
    ids, cur = set(), None
    for line in path.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^#+\s+(DEC-\d+|G\d+)\b", line)
        if m:
            cur = m.group(1)
            continue
        if cur and re.match(r"^[-*]?\s*\*\*Status:\*\*", line):
            if "🚧" in line:
                ids.add(cur)
            cur = None
    return ids, ""


def unresolved_findings(block, open_ids):
    """(hard, soft) findings for one session block.

    `block` is a list of lines. The managed link-definition block is excluded: it is
    generated, and its [DEC-NNN]: lines are not citations by the session's author.
    """
    body, in_links = [], False
    for line in block:
        if LINK_BLOCK_RE.match(line):
            in_links = True
        if not in_links:
            body.append(line)
        if line.startswith("<!-- link-doc-refs:end"):
            in_links = False

    hard, soft = [], []
    for i, line in enumerate(body):
        for label, pat in HARD_MARKERS:
            if pat.search(line):
                hard.append((label, i, line.strip()))
        for label, pat in SOFT_MARKERS:
            if pat.search(line):
                soft.append((label, i, line.strip()))

    text = "".join(body)
    for did in sorted(set(re.findall(r"\[(DEC-\d+)\]", text)) & open_ids):
        hard.append((f"discusses {did}, still OPEN in the ledger -- the entry itself never moves, but this block describes unfinished work", -1, ""))
    return hard, soft


def report_unresolved(session, hard, soft, note, stream=sys.stderr):
    if note:
        print(f"  \u26a0\ufe0f  {note}", file=stream)
    if hard:
        print(f"\u26d4 REFUSED session {session}: it carries {len(hard)} unresolved "
              f"item(s). Nothing moved.", file=stream)
        for label, ln, txt in hard:
            where = f"line +{ln}: " if ln >= 0 else ""
            print(f"     - {label} -- {where}{txt[:110]}", file=stream)
        print("     Resolve the item, or re-home it into TODOS.md / DECISIONS.md / "
              "NEXT_STEPS.md, then roll.\n     There is no override: ruled 2026-08-30 "
              "by Cenay -- a block carrying an unresolved item never rolls.", file=stream)
    if soft:
        print(f"  \u26a0\ufe0f  session {session}: {len(soft)} ADVISORY marker(s) -- "
              f"not blocking, read them:", file=stream)
        for label, ln, txt in soft:
            print(f"     - {label} -- line +{ln}: {txt[:110]}", file=stream)
    if not hard and not soft:
        print(f"  \u2705 session {session}: no unresolved markers, no open-decision "
              f"citations.", file=stream)


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


def split_current_blocks(lines):
    """CURRENT_STATUS.md -> {session_number: [lines]}. Sub-headings travel with
    their parent block, which is why the scan is for session headings only."""
    starts = [i for i, l in enumerate(lines) if current_session_number(l) is not None]
    end = next((i for i, l in enumerate(lines) if LINK_BLOCK_RE.match(l)), len(lines))
    out = {}
    for a, b in zip(starts, starts[1:] + [end]):
        out[current_session_number(lines[a])] = lines[a:b]
    return out


def cmd_guard_removal(args):
    """⛔ THE INVARIANT, ENFORCED AGAINST HAND EDITS -- not just against this tool.

    Ruled 2026-08-30 by Cenay: nothing leaves CURRENT_STATUS.md while it is open,
    outstanding, work in process or otherwise not complete.

    `roll` refusing is not enough: it guards ONE code path. A session block can be
    deleted by an editor, a bad merge, a script, or by me. This compares the staged
    file against HEAD and refuses the COMMIT, which is the only place every path
    converges. Same shape as Check 4b's append-only guard on docs/history/.
    """
    import subprocess
    # ⚠️ Per-repo opt-in since 2026-08-31 (fran-dash only). ⛔ Exits 0 so a repo
    # without the marker commits exactly as it did before this tool gained a
    # gate -- but it SAYS SO, because a guard that is off and silent is
    # indistinguishable from a guard that ran and found nothing.
    on, why = gate_enabled(pathlib.Path(args.current).parent)
    if not on:
        print(f"  note: {why}", file=sys.stderr)
        return 0
    path = args.current
    try:
        head = subprocess.run(["git", "show", f"HEAD:{path}"], capture_output=True,
                              text=True, check=True).stdout.splitlines(keepends=True)
    except subprocess.CalledProcessError:
        print(f"  note: {path} has no HEAD version -- nothing to compare, skipping",
              file=sys.stderr)
        return 0
    staged = subprocess.run(["git", "show", f":{path}"], capture_output=True,
                            text=True).stdout.splitlines(keepends=True)
    if not staged:
        staged = pathlib.Path(path).read_text(encoding="utf-8").splitlines(keepends=True)

    before, after = split_current_blocks(head), split_current_blocks(staged)
    open_ids, note = open_decision_ids(args.ledger)
    if note:
        print(f"  ⚠️  {note}", file=sys.stderr)

    gone, shrunk, failed = [], [], False
    for num, block in sorted(before.items(), reverse=True):
        hard, _ = unresolved_findings(block, open_ids)
        if num not in after:
            if hard:
                gone.append((num, hard))
                failed = True
        elif hard and len(after[num]) < len(before[num]):
            shrunk.append((num, len(before[num]) - len(after[num]), len(hard)))

    for num, hard in gone:
        print(f"⛔ REFUSED: session {num} was REMOVED from {path} while carrying "
              f"{len(hard)} unresolved item(s):", file=sys.stderr)
        for label, ln, txt in hard[:6]:
            where = f"line +{ln}: " if ln >= 0 else ""
            print(f"     - {label} -- {where}{txt[:100]}", file=sys.stderr)
        if len(hard) > 6:
            print(f"     ... and {len(hard) - 6} more", file=sys.stderr)
    if failed:
        print("   Nothing open, outstanding or in progress may leave this file "
              "(ruled 2026-08-30). Restore the block, or resolve/re-home the items "
              "first. There is no override.", file=sys.stderr)
        return 1

    for num, lost, nhard in shrunk:
        print(f"  ⚠️  session {num} SHRANK by {lost} line(s) and still carries "
              f"{nhard} unresolved marker(s) -- allowed, because resolving an item "
              f"legitimately shortens a block. READ THE DIFF.", file=sys.stderr)

    checked = sum(1 for b in before.values() if unresolved_findings(b, open_ids)[0])
    print(f"  ✅ no unresolved session block left {path} "
          f"({checked} of {len(before)} block(s) carry unresolved items and are held)")
    print("     NOT checked: whether text was deleted from INSIDE a surviving block "
          "-- that is reported as a shrink warning above, never blocked, because it "
          "is indistinguishable from resolving an item in place.", file=sys.stderr)
    return 0


def cmd_roll(args):
    cur_path, arc_path = pathlib.Path(args.current), pathlib.Path(args.archive)
    cur_before = cur_path.read_text().splitlines(keepends=True)
    arc_before = arc_path.read_text().splitlines(keepends=True)

    start = next((i for i, l in enumerate(cur_before)
                  if current_session_number(l) == args.session), None)
    if start is None:
        raise SystemExit(f"error: session {args.session} not found in {cur_path}")
    end = next((i for i, l in enumerate(cur_before) if LINK_BLOCK_RE.match(l)), len(cur_before))
    nxt = next((i for i, l in enumerate(cur_before[start + 1:end], start + 1) if H2_RE.match(l)), end)

    block = cur_before[start:nxt]
    while block and block[-1].strip() == "":
        block.pop()

    # ⛔ THE UNRESOLVED-ITEM GATE -- ruled 2026-08-30, no override. See HARD_MARKERS.
    # ⚠️ Per-repo opt-in since 2026-08-31: fran-dash only. See GATE_MARKER.
    on, why = gate_enabled(cur_path.parent)
    if not on:
        print(f"⚠️  {why}", file=sys.stderr)
    else:
        open_ids, note = open_decision_ids(args.ledger)
        hard, soft = unresolved_findings(block, open_ids)
        report_unresolved(args.session, hard, soft, note)
        if hard:
            return 1

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
    LEDGER = "docs/DECISIONS.md"

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
    o.add_argument("--ledger", default=LEDGER)

    g = sub.add_parser("guard-removal", help="refuse a commit that removes an "
                       "unresolved session block from CURRENT_STATUS.md")
    g.add_argument("--current", default=CURRENT)
    g.add_argument("--ledger", default=LEDGER)
    g.set_defaults(func=cmd_guard_removal)
    o.add_argument("--dry-run", action="store_true")
    o.set_defaults(func=cmd_roll)

    args = p.parse_args()
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
