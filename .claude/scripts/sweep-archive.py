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

    def staged_or_disk(p):
        txt = subprocess.run(["git", "show", f":{p}"], capture_output=True, text=True).stdout
        if not txt and pathlib.Path(p).exists():
            txt = pathlib.Path(p).read_text(encoding="utf-8")
        return txt

    staged_text = "".join(staged)
    archive_txt = staged_or_disk(args.archive)
    todos_txt = staged_or_disk(args.todos)
    recorded = {}
    for l in archive_txt.splitlines():
        m = AUDIT_LINE_RE.match(l)
        if m:
            recorded[m.group(2)] = m.group(1)

    def carried_out(block, hard):
        """[DEC-379]: every flagged line has a verdict in the archive, and every
        carried one is in TODOS. Returns the list of what is missing."""
        missing = []
        for label, ln, txt in hard:
            if ln < 0:
                continue
            # A flagged line still in the staged file has not left it. This matters
            # because split_current_blocks() lets a session block absorb a following
            # non-session "## " section that `roll` (which stops at the next "## ")
            # leaves behind -- without this, those lines would demand a verdict.
            if txt.strip() in staged_text:
                continue
            v = recorded.get(flag_hash(txt))
            if v is None:
                missing.append(("no audit verdict in the archive", ln, txt.strip()))
            elif v == "carried" and carried_text(txt) not in todos_txt:
                missing.append(("carried, but not found in TODOS", ln, txt.strip()))
        return missing

    gone, shrunk, failed, carried_ok = [], [], False, []
    for num, block in sorted(before.items(), reverse=True):
        hard, _ = unresolved_findings(block, open_ids)
        if num not in after:
            if hard:
                missing = carried_out(block, hard)
                if missing:
                    gone.append((num, missing))
                    failed = True
                else:
                    carried_ok.append(num)
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

    for num in carried_ok:
        print(f"  ✅ session {num} left with flagged lines, every one accounted for in the "
              f"archive's audit record ([DEC-379])", file=sys.stderr)
    for num, lost, nhard in shrunk:
        print(f"  ⚠️  session {num} SHRANK by {lost} line(s) and still carries "
              f"{nhard} unresolved marker(s) -- allowed, because resolving an item "
              f"legitimately shortens a block. READ THE DIFF.", file=sys.stderr)

    held = sum(1 for n, b in before.items() if n in after and unresolved_findings(b, open_ids)[0])
    print(f"  ✅ no unresolved session block left {path} without an audit record "
          f"({len(carried_ok)} left with every flagged line accounted for; {held} still in "
          f"the file carry flagged lines and are held)")
    print("     NOT checked: whether text was deleted from INSIDE a surviving block "
          "-- that is reported as a shrink warning above, never blocked, because it "
          "is indistinguishable from resolving an item in place.", file=sys.stderr)
    return 0


# ── CARRY: a block leaves once every flagged line is accounted for ─────────────
#
# ⛔ RULED 2026-09-29 by Cenay, fran-dash [DEC-379] (amends [DEC-267] for
# CURRENT_STATUS.md): "Seems like we're carrying a lot that has been done because
# it's in a session that has one open item." Measured that day: 54 of 54 blocks
# held, most by one to five flagged lines.
#
# THE INVARIANT, STILL NO OVERRIDE: a block with flagged lines may leave only when
# EVERY flagged line has an audit verdict --
#   carried   -> copied into TODOS.md as an unticked item (the live home for open work)
#   done      -> finished since; evidence carries a citation
#   rehomed   -> still open but already tracked in a live file; evidence names it
#   not-open  -> the marker word is descriptive; evidence says why
# The verdicts are written into the archive beside the block, keyed by a short hash
# of each flagged line, and `guard-removal` re-checks them at the commit: a removed
# block passes only if every flagged line's hash is in the staged archive, and every
# carried line's text is in the staged TODOS.md. A `--carry` run is therefore not an
# override -- it is a different, checkable way to satisfy the same "nothing open is
# lost" property. A flagged line with no verdict still refuses the whole block.
#
# ⓘ A citation of a DEC the ledger still marks OPEN is accepted as re-homed by
# definition: the ledger entry IS the live home, and it never moves.
AUDIT_VERDICTS = ("carried", "done", "rehomed", "not-open")
AUDIT_LINE_RE = re.compile(r"^- AUDIT (carried|done|rehomed|not-open) \[([0-9a-f]{10})\]")


def evidence_cites(ev):
    # CITATION_RE lives with the `items` mode further down; built lazily here.
    return re.search(CITATION_RE.pattern + r"|\b(?:BUG|SUSP|SMOKE)-\d{4}-\d\d-\d\d-\d{3}\b"
                     r"|\b[\w.-]+\.md\b", ev) is not None


def flag_hash(text):
    import hashlib
    return hashlib.sha1(text.strip().encode("utf-8")).hexdigest()[:10]


def carried_text(line):
    """The open item as it lands in TODOS: the line minus its list marker/checkbox."""
    return re.sub(r"^\s*(?:[-*]\s+)?(?:\[[ xX]\]\s+)?", "", line.strip())


def load_audit(path, session):
    import json
    data = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    out = {}
    for blk in data:
        if int(blk.get("session", -1)) != session:
            continue
        for e in blk.get("entries", []):
            out[e["line"].strip()] = (e.get("verdict", ""), (e.get("evidence") or "").strip())
    return out


def check_audit(hard, audit):
    """-> (verdicts {line: (verdict, evidence)}, problems [str]). Refuse on any problem."""
    verdicts, problems = {}, []
    for label, ln, txt in hard:
        if ln < 0:                      # an open-DEC citation: the ledger is its home
            continue
        key = txt.strip()
        if key in verdicts:
            continue
        if key not in audit:
            problems.append(f"no audit verdict for flagged line: {key[:100]}")
            continue
        verdict, ev = audit[key]
        if verdict not in AUDIT_VERDICTS:
            problems.append(f"unknown verdict {verdict!r} for: {key[:80]}")
        elif verdict in ("done", "rehomed") and not evidence_cites(ev):
            problems.append(f"{verdict} needs a citation in its evidence: {key[:80]}")
        elif verdict in ("not-open", "carried") and len(ev) < 15:
            problems.append(f"{verdict} needs a one-sentence reason: {key[:80]}")
        verdicts[key] = (verdict, ev)
    return verdicts, problems


def audit_record(session, verdicts, date):
    rec = ["\n", f"#### Sweep audit — session {session}, {date} ([DEC-379])\n", "\n"]
    for line, (verdict, ev) in verdicts.items():
        rec.append(f"- AUDIT {verdict} [{flag_hash(line)}] {ev}\n")
    return rec


def carry_into_todos(todos_lines, session, carried, date):
    """Insert a newest-first section of unticked items above the first ## heading."""
    if not carried:
        return todos_lines, []
    add = [f"## Carried from CURRENT_STATUS — session {session} (swept {date}, [DEC-379])\n", "\n"]
    for line in carried:
        add.append(f"- [ ] {carried_text(line)} _(carried from `CURRENT_STATUS.md` session "
                   f"{session}; the block is in `history/CURRENT_STATUS-archive.md`)_\n")
    add.append("\n")
    at = next((i for i, l in enumerate(todos_lines) if H2_RE.match(l)), len(todos_lines))
    return todos_lines[:at] + add + todos_lines[at:], add


def cmd_roll(args):
    cur_path, arc_path = pathlib.Path(args.current), pathlib.Path(args.archive)
    cur_before = cur_path.read_text().splitlines(keepends=True)
    arc_before = arc_path.read_text().splitlines(keepends=True)

    start = next((i for i, l in enumerate(cur_before)
                  if current_session_number(l) == args.session), None)
    if start is None:
        raise SystemExit(f"error: session {args.session} not found in {cur_path}")
    end = next((i for i, l in enumerate(cur_before) if LINK_BLOCK_RE.match(l)), len(cur_before))
    # A block ends at the next "## ", OR at a bare `---` separator, OR at a stamp line.
    # ⛔ Measured 2026-09-29: fran-dash's stamp block and two file-level notes sat
    # between session 99 and session 101 after a `---`; ending only at "## " carried
    # them into the archive inside session 99, and stamp-doc then found no chain.
    # Stopping early fails SAFE -- anything past the stop stays in the live file.
    nxt = next((i for i, l in enumerate(cur_before[start + 1:end], start + 1)
                if H2_RE.match(l) or l.strip() == "---" or l.startswith("_Last updated ")), end)

    block = cur_before[start:nxt]
    while block and block[-1].strip() == "":
        block.pop()

    # ⛔ THE UNRESOLVED-ITEM GATE -- ruled 2026-08-30, no override. See HARD_MARKERS.
    # ⚠️ Per-repo opt-in since 2026-08-31: fran-dash only. See GATE_MARKER.
    record, todo_adds = [], []
    todos_path = pathlib.Path(args.todos)
    todos_before = todos_after = None
    on, why = gate_enabled(cur_path.parent)
    if not on:
        print(f"⚠️  {why}", file=sys.stderr)
    else:
        open_ids, note = open_decision_ids(args.ledger)
        hard, soft = unresolved_findings(block, open_ids)
        if hard and args.carry:
            # [DEC-379]: the block may leave if every flagged line is accounted for.
            if not args.audit:
                raise SystemExit("error: --carry needs --audit <file>")
            if note:
                print(f"  ⚠️  {note}", file=sys.stderr)
            verdicts, problems = check_audit(hard, load_audit(args.audit, args.session))
            if problems:
                print(f"⛔ REFUSED session {args.session}: the audit does not account for "
                      f"every flagged line. Nothing moved.", file=sys.stderr)
                for p in problems:
                    print(f"     - {p}", file=sys.stderr)
                return 1
            import datetime
            today = datetime.date.today().isoformat()
            record = audit_record(args.session, verdicts, today)
            carried = [l for l, (v, _) in verdicts.items() if v == "carried"]
            todos_before = todos_path.read_text(encoding="utf-8").splitlines(keepends=True)
            todos_after, todo_adds = carry_into_todos(todos_before, args.session, carried, today)
            tally = collections.Counter(v for v, _ in verdicts.values())
            print(f"  ✅ session {args.session}: {len(verdicts)} flagged line(s) accounted for "
                  f"({', '.join(f'{n} {v}' for v, n in sorted(tally.items()))})", file=sys.stderr)
        else:
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
    moved = block + record + ["\n"] + (["---\n", "\n"] if had_separator else [])
    blocks = sorted(arc.blocks + [(args.session, moved)], key=lambda b: -b[0])

    header = list(arc.header)
    if args.stamp:
        at = next((i for i, l in enumerate(header) if SWEEP_STAMP_RE.match(l)), len(header))
        header = header[:at] + [args.stamp.rstrip("\n") + "\n", "\n"] + header[at:]

    arc_after = arc.render(blocks=blocks, header=header)
    adds = ([args.stamp] if args.stamp else []) + record + todo_adds
    tb, ta = (todos_before or []), (todos_after or [])
    assert_lines_preserved(cur_before + arc_before + tb, cur_after + arc_after + ta,
                           allowed_adds=adds)

    if args.dry_run:
        print(f"dry-run: would move {len(block)} line(s); archive order would be "
              f"{' '.join(str(n) for n, _ in blocks[:6])}..."
              + (f"; would carry {max(len(todo_adds) - 3, 0)} open item(s) into {todos_path}"
                 if todo_adds else ""))
        return 0
    cur_path.write_text("".join(cur_after))
    arc_path.write_text("".join(arc_after))
    if todos_after is not None and todo_adds:
        todos_path.write_text("".join(todos_after))
        print(f"carried {len(todo_adds) - 3} open item(s) into {todos_path}")
    print(f"rolled session {args.session}: {len(block)} lines moved verbatim")
    print(f"archive order: {' '.join(str(n) for n, _ in blocks[:6])}...")
    return 0


# ── ITEM-LEVEL SWEEP (`items`) ──────────────────────────────────────────────────
#
# ⛔ RULED 2026-09-14 by Cenay, [DEC-324]: "what's open and live remains; closed,
# deleted and resolved moves." For TODOS.md / NEXT_STEPS.md the unit is the ITEM,
# not the session block: if a block holds 1 open item and 5 ticked ones, the 5
# move and the 1 stays. `roll` is block-scoped and cannot do that, so until
# 2026-09-29 item sweeps were done by hand. This mode is that hand-sweep as a
# program, with the same line-preservation guard as `roll`.
#
# WHAT MOVES: a ticked item (`- [x]`) at the TOP level of a `## ` section, plus its
# indented continuation lines, verbatim. It lands under the SAME `## ` heading in
# the archive -- appended to that section if it exists, otherwise a new section is
# created in date order (newest on top). A session can therefore appear in the live
# file AND the archive at once; [DEC-324] says that is intended.
#
# WHAT IS HELD, and printed by name (silence never means "did not look"):
#   - a ticked item whose own continuation holds an unticked `- [ ]` (open work
#     inside it -- the [DEC-267] invariant, item-sized);
#   - a ticked item with NO citation (DEC/G id, commit hash, PR/issue #, AOC-/E-
#     id, or a repo path). That is a PROXY for the [DEC-324] safety rule "the
#     archive is the backstop, never the sole copy" -- it checks that the item
#     points somewhere, NOT that the place it points holds its substance.
#   - nested ticked items stay with their parent; they are never split off it.
#
# A section left with only its heading after the sweep loses the heading too (it
# moved with its items). A section with no items left but PROSE remaining keeps
# the prose and is reported: prose is not item-shaped and is not this mode's call.
TICK_RE = re.compile(r"^(\s*)[-*] \[[xX]\]")
BOX_RE = re.compile(r"^(\s*)[-*] \[[ xX]\]")
OPEN_BOX_RE = re.compile(r"^\s*[-*] \[ \]")
DATE_RE = re.compile(r"\b(20\d\d-\d\d-\d\d)\b")
# A tick does not always mean "all of it is finished": measured on fran-dash's
# TODOS.md 2026-09-29, 4 ticked items said "STILL OPEN" / "still open" / "not yet
# confirmed" about a remainder. Fails toward holding, like the [DEC-267] gate.
RESIDUAL_RE = re.compile(r"still open|not done|reopened|not yet confirmed", re.I)
# The residual hold lifts only when the item itself says, dated and with a citation,
# that the remainder was closed or moved somewhere live. Not a flag, not an override:
# it is written into the item, so the archive copy carries the reason it moved.
REMAINDER_SETTLED_RE = re.compile(r"\*\*Remainder (?:closed|re-homed) \(\d{4}-\d\d-\d\d[^)]*\):\*\*")
CITATION_RE = re.compile(
    r"\[?(?:DEC-\d+|G\d+|AOC-\d+|E-\d+)\]?"          # ledger ids
    r"|`[0-9a-f]{7,40}`"                               # commit hash in backticks
    r"|\b(?:PR|issue)\s*#\d+|[\w.-]+#\d+|\(#\d+\)"    # PR / issue refs
    r"|`?(?:docs|plans|site|tools|specs|\.claude)/[\w./-]+")  # a repo path


def _indent(line):
    return len(line) - len(line.lstrip())


def heading_date(heading):
    m = DATE_RE.search(heading)
    return m.group(1) if m else None


def split_sections(lines):
    """-> (head, [(heading_line, [body lines])], tail). Tail = link-doc-refs block."""
    end = next((i for i, l in enumerate(lines) if LINK_BLOCK_RE.match(l)), len(lines))
    starts = [i for i in range(end) if H2_RE.match(lines[i])]
    if not starts:
        return lines[:end], [], lines[end:]
    secs = [(lines[a], lines[a + 1:b]) for a, b in zip(starts, starts[1:] + [end])]
    return lines[:starts[0]], secs, lines[end:]


def top_level_items(body):
    """[(start, stop, is_ticked)] for items not nested inside another item."""
    out, i = [], 0
    while i < len(body):
        m = BOX_RE.match(body[i])
        if not m:
            i += 1
            continue
        ind, j, last = len(m.group(1)), i + 1, i
        while j < len(body):
            if body[j].strip() == "":
                j += 1
                continue
            if _indent(body[j]) > ind:
                last = j
                j += 1
                continue
            break
        out.append((i, last + 1, bool(TICK_RE.match(body[i]))))
        i = last + 1
    return out


def cmd_items(args):
    live_path, arc_path = pathlib.Path(args.live), pathlib.Path(args.archive)
    live_before = live_path.read_text().splitlines(keepends=True)
    arc_before = arc_path.read_text().splitlines(keepends=True)
    head, secs, tail = split_sections(live_before)

    moves, held, new_secs, prose_left = [], [], [], []   # moves: (heading, [item lines])
    for heading, body in secs:
        keep, taken = list(body), []
        for a, b, ticked in reversed(top_level_items(body)):
            if not ticked:
                continue
            item = body[a:b]
            label = body[a].strip()[:90]
            if any(OPEN_BOX_RE.match(l) for l in item[1:]):
                held.append(("open child item", heading.strip(), label))
                continue
            if RESIDUAL_RE.search("".join(item)) and not REMAINDER_SETTLED_RE.search("".join(item)):
                held.append(("says part is still open", heading.strip(), label))
                continue
            if not CITATION_RE.search("".join(item)):
                held.append(("no citation", heading.strip(), label))
                continue
            taken.insert(0, item)
            del keep[a:b]
            # collapse the blank-line pair the removal leaves behind
            if a < len(keep) and a > 0 and keep[a].strip() == "" and keep[a - 1].strip() == "":
                del keep[a]
        if not taken:
            new_secs.append((heading, body))
            continue
        moves.append((heading, taken))
        if any(l.strip() for l in keep):
            new_secs.append((heading, keep))
            if not any(BOX_RE.match(l) for l in keep):
                prose_left.append(heading.strip())
        # else: the section emptied -- its heading moves with its items

    live_after = list(head)
    for heading, body in new_secs:
        live_after += [heading] + body
    live_after += tail

    a_head, a_secs, a_tail = split_sections(arc_before)
    a_secs = [(h, list(b)) for h, b in a_secs]
    created = []
    for heading, items in moves:
        payload = []
        for it in items:
            payload += it + ([] if it[-1].strip() == "" else ["\n"])
        idx = next((k for k, (h, _) in enumerate(a_secs) if h == heading), None)
        if idx is not None:
            body = a_secs[idx][1]
            while body and body[-1].strip() == "":
                body.pop()
            a_secs[idx] = (heading, body + ["\n"] + payload)
            continue
        date = heading_date(heading)
        # newest on top: before the first section dated OLDER than this one. An
        # undated heading goes to the BOTTOM -- in practice those are the oldest,
        # pre-session-numbering sections ("## Active").
        at = len(a_secs) if date is None else next(
            (k for k, (h, _) in enumerate(a_secs) if (heading_date(h) or "9999") < date),
            len(a_secs))
        a_secs.insert(at, (heading, ["\n"] + payload))
        created.append((heading.strip(), date))

    a_head = list(a_head)
    if args.stamp:
        a_head += [args.stamp.rstrip("\n") + "\n", "\n"]
    arc_after = list(a_head)
    for heading, body in a_secs:
        arc_after += [heading] + body
    arc_after += a_tail

    # A heading that stays in the live file AND is created in the archive is the one
    # sanctioned duplicate line; so is the stamp. Everything else must be relocation.
    live_heads = {h for h, _ in new_secs}
    created_names = {h for h, _ in created}
    adds = [h for h, _ in moves if h.strip() in created_names and h in live_heads]
    # ...and a section that EMPTIED into an archive section of the same name merges
    # into it, so its heading line exists once where it existed twice.
    drops = [h for h, _ in moves if h.strip() not in created_names and h not in live_heads]
    if args.stamp:
        adds.append(args.stamp)
    assert_lines_preserved(live_before + arc_before, live_after + arc_after,
                           allowed_drops=drops, allowed_adds=adds)

    n_items = sum(len(i) for _, i in moves)
    n_lines = sum(len(x) for _, i in moves for x in i)
    print(f"{'dry-run: would move' if args.dry_run else 'moved'} {n_items} ticked item(s), "
          f"{n_lines} line(s) verbatim, from {len(moves)} section(s)")
    for h, d in created:
        print(f"  new archive section{'' if d else ' (NO DATE in heading -- placed at the bottom)'}: {h[:90]}")
    for why, h, label in held:
        print(f"  ⚠️  HELD ({why}) in {h[:50]}: {label}", file=sys.stderr)
    for h in prose_left:
        print(f"  ⚠️  no items left, prose remains -- review by hand: {h[:90]}", file=sys.stderr)
    print("NOT checked: whether a moved item's substance lives in DECISIONS / LESSONS_LEARNED "
          "(the citation test is a proxy); nested ticked items (they stay with their parent); "
          "non-checkbox bullets (never moved).", file=sys.stderr)
    if args.dry_run:
        return 0
    live_path.write_text("".join(live_after))
    arc_path.write_text("".join(arc_after))
    print(f"wrote {live_path} and {arc_path}")
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
    o.add_argument("--carry", action="store_true",
                   help="[DEC-379]: roll a block with flagged lines once --audit accounts for each")
    o.add_argument("--audit", default="", help="JSON audit verdicts, for --carry")
    o.add_argument("--todos", default="docs/TODOS.md", help="where carried open items land")

    g = sub.add_parser("guard-removal", help="refuse a commit that removes an "
                       "unresolved session block from CURRENT_STATUS.md")
    g.add_argument("--current", default=CURRENT)
    g.add_argument("--ledger", default=LEDGER)
    g.add_argument("--archive", default=ARCHIVE)
    g.add_argument("--todos", default="docs/TODOS.md")
    g.set_defaults(func=cmd_guard_removal)
    o.add_argument("--dry-run", action="store_true")
    o.set_defaults(func=cmd_roll)

    t = sub.add_parser("items", help="move TICKED items (item-level, [DEC-324]) from a "
                       "live doc into its archive, under the same ## heading")
    t.add_argument("--live", default="docs/TODOS.md")
    t.add_argument("--archive", default="docs/history/TODOS-archive.md")
    t.add_argument("--stamp", default="")
    t.add_argument("--dry-run", action="store_true")
    t.set_defaults(func=cmd_items)

    args = p.parse_args()
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
