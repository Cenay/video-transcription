"""The ledger grammar — one definition, imported by every reader and writer.

Spec: plans/ledger-tooling-contract.md §2.

WHY THIS EXISTS. Three scripts each carried their own idea of what a decision
entry looks like, and they disagreed in ways that cancelled out only by luck:

    heading      gen-dec-index ^###      check-doc-refs ^#{2,4}   link-doc-refs ^#{1,6}
    status       REQUIRES a bullet       REJECTS a bullet          --

Measured on fran-dash/docs/DECISIONS.md (2026-08-01): 169 `## DEC-` headings and
0 `### DEC-`, so gen-dec-index parsed **0 of 169**; 170 bare `**Status:**` lines
and 0 bulleted, so its STATUS_RE would have matched **0 of 170** even if the
headings had been found. The two status regexes are exact complements — copying
either one wholesale moves the break rather than fixing it.

So the rules here take the UNION of the shapes that occur in real ledgers:

  - Accept a RANGE of heading levels, never one. Capture the level so a caller
    may report on it; do not legislate it.
  - Make the leading bullet OPTIONAL wherever a `**Field:**` line is parsed.
    `(?:[-*]\\s+)?` costs nothing and absorbs both conventions.
  - Scan a small WINDOW for a field rather than requiring adjacency.

Being liberal in what we accept is deliberate. A reader that rejects a valid
ledger reports zero entries, and zero entries is indistinguishable from an empty
file — which is precisely how this broke without anyone noticing.
"""

import re
from collections import namedtuple

# Entry headings: `## DEC-163 Title`, `### [DEC-163] Title`, `#### BUG-7 Title`.
# The bracket is optional because both forms are in use; the ID prefixes are the
# families these docs actually number.
# ⚠️ The hyphen is OPTIONAL on the single-letter families. Predecessor-repo
# records are written UNHYPHENATED — `## G77 URL parity is a launch gate`,
# `[M-043]`, `[D-049]` — and a pattern that demanded `G-77` silently dropped
# G77's row when the index was regenerated: 170 rows in, 169 out. Worse, the
# shrink guard that exists to catch exactly that carried its own copy of this
# pattern, missed G77 the same way, counted 169 existing rows, and waved the
# write through. One pattern, exported, used everywhere — that is the point of
# this module, and duplicating it is how it fails.
ID_PATTERN = r"(?:(?:DEC|BUG|SUSP|LES|ADR)-\d+|[GMD]-?\d+)"

ENTRY_RE = re.compile(
    rf"^(?P<level>#{{2,4}})\s+\[?(?P<id>{ID_PATTERN})\]?\s*(?P<title>.*?)\s*$"
)

# An index row: `| [DEC-001](#anchor) | … |` or `| [G77](#anchor) | … |`.
ROW_ID_RE = re.compile(rf"^\|\s*\[?({ID_PATTERN})\]?")

FENCE_RE = re.compile(r"^\s*(```|~~~)")

# How far below a heading a field line may sit. check-doc-refs already scans a
# window; gen-dec-index assumed strict adjacency and missed any entry with a
# blank line or a lead-in sentence under the heading.
FIELD_LOOKAHEAD = 3


def field_re(name):
    """Match a labeled field line, with or without a leading bullet.

    Accepts all of:
        **Status:** ✅ CLOSED
        - **Status:** ✅ CLOSED
        Status: ✅ CLOSED
        ⚠️ **Status:** ✅ CLOSED
    """
    return re.compile(
        rf"^\s*(?:[-*]\s+)?(?:[^\w\s`*_]{{1,4}}\s+)?\*{{0,2}}{name}:?\*{{0,2}}\s*(?P<value>.+?)\s*$",
        re.IGNORECASE,
    )


STATUS_RE = field_re("Status")
DECIDED_RE = field_re("Decided")

Entry = namedtuple("Entry", "id level title line status decided slug")


def slugify(text):
    """Slugify exactly as link-doc-refs.py:74 does, so anchors agree.

    ⚠️ Do NOT "tidy" this by collapsing whitespace. When punctuation removed in
    step 2 sat between two spaces (`A — B`), GitHub and VS Code both emit a
    DOUBLE hyphen (`a--b`). A naive `\\s+`→`-` slugifier yields `a-b` and every
    em-dashed anchor silently breaks. A checker built on the tidy version
    reported 3 false "broken anchors" on a ledger whose anchors were correct.

    test-ledger-contract.py asserts this stays byte-equivalent to that copy.
    """
    s = text.strip().lower()
    s = re.sub(r"[^\w\s-]", "", s, flags=re.UNICODE)
    s = re.sub(r"\s", "-", s)
    return s


def parse_entries(text):
    """Yield an Entry per decision heading, skipping fenced code blocks.

    `status`/`decided` are None when absent — the caller decides whether that is
    an error. Reporting an entry with a missing Status is far more useful than
    dropping it, which is what strict adjacency used to do silently.
    """
    lines = text.split("\n")
    in_fence = False
    out = []
    for i, line in enumerate(lines):
        if FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        m = ENTRY_RE.match(line)
        if not m:
            continue
        status = decided = None
        for probe in lines[i + 1 : i + 1 + FIELD_LOOKAHEAD + 1]:
            if ENTRY_RE.match(probe):
                break
            if status is None:
                sm = STATUS_RE.match(probe)
                if sm:
                    status = sm.group("value")
            if decided is None:
                dm = DECIDED_RE.match(probe)
                if dm:
                    decided = dm.group("value")
        title = m.group("title")
        out.append(Entry(
            id=m.group("id"),
            level=len(m.group("level")),
            title=title,
            line=i + 1,
            status=status,
            decided=decided,
            slug=slugify(f"{m.group('id')} {title}"),
        ))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Cross-repo DEC- collision check  (fran-dash [DEC-220], 2026-08-11)
#
# The TRFA program runs ONE monotonic DEC- series across two repos
# (fran-dash [DEC-205]). Two sessions in two repos can each read their own
# ledger, see the same high-water mark, and allocate the same number. That
# failure is SILENT: duplicate ids in separate repos do not conflict in git,
# fail no test, and surface only when someone cites one and the reader opens
# the other. It came within one number of happening on 2026-08-11.
#
# The check lives HERE, beside ENTRY_RE, because the only hard part is "what is
# an entry and where does its id live" — and this module already answers that
# for both heading forms in use (`## DEC-200 Title` and
# `### DEC-218 — 2026-07-14 — Title`). ✅ Measured 2026-08-11: 220 entries
# parsed from fran-dash, 13 from the API repo, with no per-repo special-casing.

# DEC-001..011 exist in BOTH repos as different decisions — grandfathered legacy
# numbers predating unification ([DEC-205]). They are permanent, known, and
# accepted; flagging them on every commit is how a checker gets switched off.
LEGACY_FLOOR = 12

# One sibling ledger path per line; blank lines and #-comments ignored. Paths
# may be absolute or relative to the repo root. Absent file => nothing to check.
SIBLING_CONFIG = ".claude/ledger-siblings"


class SiblingUnreadable(Exception):
    """A sibling was configured but could not be read or yielded no entries."""


# A registry stub is a DELIBERATE duplicate: the number is real in the other
# repo and recorded here as a receipt ([DEC-219]). Counting one as a collision
# makes the checker fire on the very mechanism that prevents collisions.
# ⚠️ Found by running the checker against the two REAL ledgers rather than only
# the fixtures in the test file: it reported DEC-205 and DEC-218, both stubs.
# Matched in the status HEAD (before the first em-dash), because a real decision
# may legitimately be *about* registry stubs — [DEC-219] is exactly that.
def is_registry(status):
    return bool(status) and "REGISTRY" in status.split("—", 1)[0].upper()


def dec_numbers(text, include_registry=False):
    """The set of DEC- integers a ledger OWNS, ignoring the legacy floor.

    Registry stubs are excluded by default — they name numbers owned elsewhere,
    so including them would report every correctly-recorded cross-repo number as
    a collision.
    """
    out = set()
    for e in parse_entries(text):
        if not e.id.startswith("DEC-"):
            continue
        if not include_registry and is_registry(e.status):
            continue
        n = int(e.id.split("-", 1)[1])
        if n >= LEGACY_FLOOR:
            out.add(n)
    return out


def read_sibling(path):
    """Parse a sibling ledger. Raises SiblingUnreadable rather than returning
    an empty set, because an empty set is indistinguishable from "no conflicts"
    and that is precisely how this class of bug hides. `ledger_contract`'s own
    docstring records a pattern that matched 0 of 170 entries; a cross-repo
    checker that reads zero rows on the far side reports a clean run.
    """
    import os
    if not os.path.exists(path):
        raise SiblingUnreadable(f"sibling ledger not found: {path}")
    try:
        text = open(path, encoding="utf-8").read()
    except OSError as exc:
        raise SiblingUnreadable(f"sibling ledger unreadable: {path} ({exc})")
    nums = dec_numbers(text)
    if not nums:
        raise SiblingUnreadable(
            f"sibling ledger parsed to ZERO DEC- entries: {path} — refusing to "
            f"report 'no collisions', because that is what a broken parser prints"
        )
    return nums


def load_siblings(repo_root="."):
    """-> list of configured sibling ledger paths (may be empty)."""
    import os
    cfg = os.path.join(repo_root, SIBLING_CONFIG)
    if not os.path.exists(cfg):
        return []
    paths = []
    for raw in open(cfg, encoding="utf-8"):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        paths.append(line if os.path.isabs(line) else os.path.join(repo_root, line))
    return paths


def check_dec_collisions(local_ledger, sibling_paths):
    """-> (collisions, skips). `collisions` is [(n, sibling_path)] sorted;
    `skips` is [reason] for siblings that could not be checked.

    A skip is NEVER a pass. The caller must surface every skip, or an absent
    sibling silently becomes a clean run — the exact shape being defended
    against. Khurram has no fran-dash checkout, so skips are expected and
    routine; being quiet about them is what makes them dangerous.
    """
    local = dec_numbers(open(local_ledger, encoding="utf-8").read())
    collisions, skips = [], []
    for path in sibling_paths:
        try:
            theirs = read_sibling(path)
        except SiblingUnreadable as exc:
            skips.append(str(exc))
            continue
        for n in sorted(local & theirs):
            collisions.append((n, path))
    return collisions, skips


def _cli(argv):
    import argparse
    import os
    ap = argparse.ArgumentParser(
        prog="ledger_contract.py check-collisions",
        description="Fail when a DEC- number in this repo's ledger is also used "
                    "in a sibling repo's ledger.")
    ap.add_argument("ledger", help="this repo's DECISIONS.md")
    ap.add_argument("--sibling", action="append", default=[],
                    help="sibling ledger path (repeatable). Default: read "
                         f"{SIBLING_CONFIG}")
    ap.add_argument("--repo-root", default=".")
    args = ap.parse_args(argv)

    if not os.path.exists(args.ledger):
        print(f"skip: no ledger at {args.ledger}")
        return 0

    siblings = args.sibling or load_siblings(args.repo_root)
    if not siblings:
        print(f"skip: no sibling ledgers configured ({SIBLING_CONFIG} absent) — "
              f"cross-repo DEC- collisions NOT checked")
        return 0

    collisions, skips = check_dec_collisions(args.ledger, siblings)
    for s in skips:
        print(f"skip: {s}")
        print("      cross-repo DEC- collisions NOT checked against that sibling")
    if collisions:
        print("")
        for n, path in collisions:
            # Zero-pad to 3. `n` is an int (the set is built via int()), so an
            # unpadded f-string printed "DEC-12" for a heading that reads
            # `## DEC-012` -- unmatchable by grep and unfindable in the ledger.
            # It misrendered precisely the legacy range around the DEC-012 floor,
            # i.e. the numbers a reader is already most likely to be confused by.
            # Found 2026-08-11 by the API repo's adversarial pass, not by these tests.
            print(f"  DEC-{n:03d} exists in BOTH this ledger and {path}")
        print("")
        print("  One DEC- series spans both repos ([DEC-205]). Renumber this "
              "entry to a free number,")
        print("  or publish a registry stub if the number was legitimately "
              "consumed elsewhere ([DEC-219]).")
        return 1
    # Report what was ACTUALLY checked, not what was configured. Saying
    # "1 sibling checked" after skipping that sibling is the silent-pass shape
    # this whole check exists to remove -- the last line is what a reader
    # believes. Caught by running the negative case, not by reading the code.
    checked = len(siblings) - len(skips)
    if checked == 0:
        print(f"⚠ NOT CHECKED: all {len(siblings)} configured sibling(s) were "
              f"skipped — no cross-repo collision check ran")
        return 0
    print(f"ok: no cross-repo DEC- collisions ({checked} of {len(siblings)} "
          f"sibling(s) checked, floor DEC-{LEGACY_FLOOR:03d})")
    return 0


if __name__ == "__main__":
    import sys
    _argv = sys.argv[1:]
    if _argv and _argv[0] == "check-collisions":
        raise SystemExit(_cli(_argv[1:]))
    raise SystemExit("usage: ledger_contract.py check-collisions LEDGER [--sibling PATH]")
