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
