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

# ─────────────────────────────────────────────────────────────────────────────
# THE ENTRY SPEC — the strict half  (ruled 2026-08-20 by Cenay)
#
# ⛔ PARSE LIBERALLY, WRITE STRICTLY. Everything ABOVE this line reads a ledger
# and is deliberately permissive — the docstring records why: a strict reader
# once found 0 of 169 entries, and zero entries is indistinguishable from an
# empty file. Everything BELOW defines the ONE shape new entries are written in.
# The two must never be "harmonized". Loosening the writer is a style change;
# tightening the reader breaks every ledger reader on the machine.
#
# WHY THIS IS DATA AND NOT PROSE. A template already existed — an HTML comment
# inside the ledger that `commands/init-project.md` scaffolds, prescribing very
# nearly this shape. It did not hold, for three reasons worth keeping in view:
# it reached a project only at creation; it was invisible in the rendered
# document; and nothing imported it, so nothing could check it. As data it is
# imported by the writer, the checker and the template generator alike, so the
# shape cannot be stated twice and cannot drift.
#
# Cenay, 2026-08-20: "I need it not to rot, and although it's more strict it
# means my docs are consistent. Something I very much need."

ENTRY_LEVEL = 3                  # `### DEC-045 Title` — see below for why not 2

# Ordered. `Added` is first by ruling. Each is (label, required).
#
# ★ `Added` and `Decided` are NOT redundant. `Decided` is when the decision was
# made; `Added` is when the entry was written. They diverge constantly and the
# gap is exactly what a reader needs — this repo's ledger holds 16 entries
# decided across three weeks and all added in one evening.
ENTRY_FIELDS = (
    ("Added", True),
    ("Status", True),
    ("Decided", True),
    ("Question", True),
    ("Answer", True),
    ("Why", True),
    ("Build impact", True),
    ("Sources", False),
)
REQUIRED_FIELDS = tuple(name for name, req in ENTRY_FIELDS if req)

# Written when the authoring date cannot be recovered. A literal, matched
# exactly — which is why the spelling is fixed here rather than per-caller.
ADDED_BACKFILL = "(Backfilled from scripts)"

# The house stamp format, unchanged: date AND 24-hour time AND zone, taken from
# `date`, never guessed. Two developers in different zones ship into one file.
ADDED_FORMAT = "YYYY-MM-DD HH:MM TZ"

# Two literal markers for a required `Build impact` that cannot simply be filled
# in. Ruled 2026-08-20 by Cenay, after normalization stopped short on eight
# entries — which turned out to be three different problems, not one.
#
# ⛔ NEITHER IS A PLACEHOLDER TO TIDY AWAY LATER. Each states a fact:
#
#   IMPACT_NONE   the entry was written without a build impact and one cannot be
#                 added now. Writing what a decision from July "changed about how
#                 we build" today means putting new claims into a historical
#                 record, which the never-rewrite-archives rule forbids outright.
#
#   IMPACT_BELOW  the entry HAS its build impact, further down, after its
#                 narrative. Cenay ruled a marker plus a note over a forced move:
#                 in these entries the impact reads as a closing summary landing
#                 after the execution notes and the verification passes, and
#                 lifting it would put the conclusion before the evidence.
#
# Literals, matched exactly, defined once — the same reasoning as the back-fill
# marker for a date that cannot be recovered.
IMPACT_NONE = "(None recorded)"
IMPACT_BELOW = "(Recorded below, after the narrative)"

# The general form of the same idea, ruled 2026-08-20 by Cenay for the fields
# that have no entry-specific marker of their own — `Why` above all.
#
# THE MEASUREMENT THAT FORCED IT. On fran-dash: `Why` is required and missing on
# 254 of 259 entries. 192 of those have prose in the entry the reason can be
# lifted out of; **62 have nothing anywhere in the entry.** Same shape on
# `Question` (14) and `Build impact` (11) — 87 field values in total that no
# amount of reading can recover, because the reasoning was never written down.
#
# ⛔ THE THREE OPTIONS AND WHY THIS ONE. Inventing a reason is the worst possible
# output — a fabricated `Why` is indistinguishable from a real one and becomes
# the permanent record of why a decision was made. Relaxing the field to optional
# was Cenay's first instinct and was withdrawn on measurement: ✅ the writer and
# the checker read the SAME `REQUIRED_FIELDS`, so relaxing it for history relaxes
# it for every new entry too — verified by running the writer against an entry
# with no `Why` before and after the change (refused, then silently accepted).
#
# ★ So the field stays REQUIRED and the gap gets STATED. The sentinel is a claim
# — "we looked and there is nothing" — which is true, and is a different thing
# from silence. It is also greppable, so the size of the hole stays countable;
# relaxing the field would have made it invisible.
#
# ⛔ `write-entry.py` REFUSES any of these on a new entry. That is what keeps the
# rule honest: a back-fill may state a gap, an author may not. Ruled 2026-08-20
# — "I don't want to drop any we CAN rebuild. Only those that we have nothing to
# recover from."
#
# ★ The wording is Cenay's own, 2026-08-20, and is kept verbatim: it says what
# was done and when, not merely that something is absent. "On this backfill"
# time-stamps the claim — a later reader knows the gap was assessed during the
# back-fill rather than left unexamined, and that new evidence may still fill it.
FIELD_UNRECOVERABLE = "(Nothing to recover from on this backfill)"

# Every literal a BACK-FILL may write into a required field, and no author may.
# One set, so the writer's refusal and any future checker read the same list
# rather than each carrying its own idea of what a placeholder looks like.
BACKFILL_SENTINELS = (ADDED_BACKFILL, IMPACT_NONE, IMPACT_BELOW, FIELD_UNRECOVERABLE)

# ⚠️ RESOLVED only. `CLOSED` was retired 2026-08-20 — two words for one state
# across 325 status lines, and the index generator's status parser has already
# produced one wrong classification from a vocabulary it had to guess at.
STATUS_VOCAB = ("🚧 OPEN", "📋 PROPOSED", "⏸ DEFERRED", "✅ RESOLVED",
                "⛔ SUPERSEDED", "🔗 REGISTRY")

# Just the words, for checking a Status line that also carries a glyph and a
# qualifier. Derived, never typed out again — a second list would drift.
STATUS_WORDS = frozenset(s.split()[-1] for s in STATUS_VOCAB)

# `2026-08-20 09:35 MST`. The house stamp: date AND 24-hour time AND zone, taken
# from `date` and never guessed, because two developers in different zones ship
# into the same files on the same day and a bare date cannot order them.
ADDED_STAMP_RE = re.compile(r"^\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}\s+[A-Z]{2,5}$")

# Field labels seen in the wild that mean one of ours. Meeting-derived entries
# grew a whole second vocabulary; these are the same fields under other names.
FIELD_SYNONYMS = {
    "Decision": "Answer",
    "Ruling": "Answer",
    "Rationale": "Why",
    "Closed": "Status",
}

# A registry stub is a receipt, not a decision — the number was consumed by an
# entry in another repo. It takes its own minimal shape and is exempt from the
# required set and from the restatement.
REGISTRY_FIELDS = (("Status", True), ("Decided", True), ("Owner", True))

# THE THIRD ENTRY KIND. Ruled 2026-08-20 by Cenay: a build task carried into the
# ledger is not a decision, and the decision template does not fit it.
#
# ⛔ WHY THIS IS NOT A LOOPHOLE. 42 entries in fran-dash say so in their own
# text — `**Type:** Build task carried forward (confirmed work, not a decision)`
# — and carry a `Task` line instead of Question/Answer/Why. Forcing the decision
# shape onto "Run the four scope-scan greps" would manufacture a question, an
# answer and a rationale that exist to satisfy the checker rather than to inform
# anyone. That is the same fabrication the unrecoverable sentinel exists to
# avoid, arriving from the other direction.
#
# ★ The precedent is REGISTRY_FIELDS directly above: this ledger already
# tolerates a second kind of entry with its own required set, for the same
# reason — the thing being recorded is not a decision. A third kind with the
# same justification is consistency, not an exception.
#
# ⚠️ Detected from the presence of a `Task` field, the way a registry stub is
# detected from REGISTRY in its Status. Self-describing: an entry declares what
# it is by what it carries, so nothing external has to hold a list.
TASK_FIELDS = (("Added", True), ("Status", True), ("Task", True))


def entry_template(entry_id="DEC-NNN", title="<short title — no date, no ID repeated>"):
    """The canonical skeleton, generated from the spec above.

    Every human-readable copy of the shape comes from here — the file under
    `templates/`, and the comment `commands/init-project.md` scaffolds into a
    new ledger. Neither is hand-written, so neither can drift from what the
    checker enforces.
    """
    hint = {
        "Added": ADDED_FORMAT,
        "Status": "<glyph + WORD> (" + ADDED_FORMAT + ") — <optional qualifier>",
        "Decided": "YYYY-MM-DD",
        "Question": "<what was actually being asked>",
        "Answer": "<what was decided, and by whom>",
        "Why": "<the reasoning — the part that matters later>",
        "Build impact": "<what this changes about how we build>",
        "Sources": "<the meeting, pull request or chat session it came from>",
    }
    lines = [f"{'#' * ENTRY_LEVEL} {entry_id} {title}"]
    lines += [f"- **{name}:** {hint.get(name, '')}".rstrip() for name, _ in ENTRY_FIELDS]
    lines += [
        "",
        "<free-form body — extra bold-labeled paragraphs and sub-headings both allowed>",
        "",
        "> **TL;DR —** <plain English, no identifiers, no links, last element of the entry>",
    ]
    return "\n".join(lines)


def status_vocab_line():
    """The status words, for the same generated copies."""
    return " · ".join(f"`{s}`" for s in STATUS_VOCAB)


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


HEADING_RE = re.compile(r"^(#{1,6})\s")
# A horizontal rule. Some ledgers put one between entries, so it is trailing
# furniture belonging to no field — anything appended after it reads as the NEXT
# entry's opening line. ✅ Measured 2026-08-20: 31 of them in fran-dash's ledger.
HR_RE = re.compile(r"^\s*(?:-{3,}|\*{3,}|_{3,})\s*$")
# `<!-- link-doc-refs:start … -->`, `<!-- dec-index:start … -->`. A managed block
# is machine-written and belongs to no entry — but the last entry on the page is
# directly followed by one, so a reader that ignores this hands that entry the
# whole rest of the file.
MANAGED_RE = re.compile(r"^\s*<!--\s*[\w-]+:(?:start|end)\b")


def fenced_lines(lines):
    """Line indices inside (or opening/closing) a code fence.

    An entry that QUOTES a form in an example must not be read as carrying one.
    That is how a checker starts passing a file that never adopted a rule.
    """
    out, in_f = set(), False
    for i, line in enumerate(lines):
        if FENCE_RE.match(line):
            in_f = not in_f
            out.add(i)
            continue
        if in_f:
            out.add(i)
    return out


def entry_bodies(lines, entries):
    """(entry, first_body_index, body_lines) for each entry, in page order.

    The body ends at the next heading **of the entry's own level or shallower**,
    or at the start of a managed block. Not at the next DEC- heading: a ledger
    ending with a `## Notes` section, or with the generated link block, would
    give its final entry the remainder of the file.

    ⚠️ And NOT at any heading whatsoever, which is what this did until
    2026-08-20. A long entry may carry its own sub-sections — fran-dash's
    `## DEC-219` has a `### The reservation rule` inside it — and ending the body
    at that sub-heading truncates the entry at its first subsection. Measured
    there, and it was silent in the worst way: the placer and the checker shared
    the wrong boundary, so a restatement landed mid-entry and the checker then
    agreed it was last.
    """
    fences = fenced_lines(lines)
    out = []
    for e in entries:
        start = e.line          # e.line is 1-based, so this is the line AFTER it
        end = start
        while end < len(lines):
            if end not in fences:
                hm = HEADING_RE.match(lines[end])
                if (hm and len(hm.group(1)) <= e.level) or MANAGED_RE.match(lines[end]):
                    break
            end += 1
        out.append((e, start, lines[start:end]))
    return out


def body_tail(body):
    """Index one past the last line of real content in an entry body.

    Trailing blank lines and a trailing horizontal rule are furniture: they
    separate this entry from the next one, so content belongs BEFORE them. This
    is the one right answer to "where does something appended to this entry go",
    and both the placer and the checker read it from here rather than each
    deciding for itself.
    """
    n = len(body)
    while n > 0 and (not body[n - 1].strip() or HR_RE.match(body[n - 1])):
        n -= 1
    return n


# ─────────────────────────────────────────────────────────────────────────────
# Restatement grammar — the `> **TL;DR —**` line  (2026-08-19)
#
# WHY IT LIVES HERE. Two different documents now carry the same rule: a Session
# Desk question (desk rule 14) and a ledger entry (check 8 of ledger-lint). The
# grammar is identical in both; only the PLACEMENT differs — a desk item folds
# its body, so the restatement sits above the folds, while a ledger entry has no
# folds and takes it last. Placement is the consumer's business; what counts as
# a restatement is not.
#
# So the regexes and the verdicts live once, here, and both checkers import
# them. The alternative was a second hand-written copy, which is precisely the
# failure this module was created to end: three scripts, three copies of one
# grammar, three different wrong answers about the same ledger in one evening.
#
# ⚠️ Deliberately NO emoji in the required form. `🗣️` is U+1F5E3 U+FE0F, and a
# copy that drops the variation selector is byte-different but visually
# identical — a check keyed on it would fail for a reason nobody could see.
#
# The leading `>` is the blockquote wrapper (Cenay, 2026-08-19: "Can the TL;DR
# section be in a backquote structure so it stands out?"). Optional in the
# PATTERN and required by the templates: a document written before the
# blockquote form, or by another session, must not hard-fail over presentation.
# Its absence is a WARNING — see TLDR_QUOTED_RE and each consumer's check.
TLDR_RE = re.compile(r"^(?:>\s*)*\*\*[^*]*\bTL;DR\b[^*]*\*\*\s*(.*)$")

# Whether that TL;DR line was actually wrapped in a blockquote.
TLDR_QUOTED_RE = re.compile(r"^>\s")

# Bare identifiers, which are the whole reason this rule exists. Every one of
# these is a LOOKUP: the reader has to leave the record to find out what the
# token means before they can understand it. Cenay, 2026-08-19: "I spent half my
# time looking up the things you reference so I can understand the damn
# question." A gloss on first appearance does not help — by the ninth item that
# gloss is 400 lines up, which is the same trip. In a ledger it is worse: many
# entries cite ANOTHER repo's decisions, so there is no local heading to jump to
# at all.
#
# Scanned on the RAW text, never on code-stripped output: these are almost
# always written inside backticks, so stripping inline code first would make the
# check silently never fire, which reads as green.
TLDR_JARGON_RE = re.compile(r"\b(?:DEC|SUSP|BUG|G|M|D|T|Q)-?\d+\b")

# A link is a lookup with extra steps, and an anchor into the same page is the
# worst kind — it moves the reader away from the thing they were reading.
TLDR_LINK_RE = re.compile(r"\]\(|https?://")

# A proxy for "plain", not a measure of it — and a loose one on purpose.
#
# Raised to 150 by Cenay 2026-08-20: "I would rather have too many words, than
# not enough clarity on the TL;DR -- so if the cap is holding us back, lift it."
#
# ⚠️ MEASURED FIRST, and the measurement says the number was never the
# constraint: across the first 80 drafted restatements the longest ran 71 words,
# ZERO landed in the 80-90 band, and none exceeded the cap. The limiter was the
# drafter compressing by habit toward ~35-50 words, not the guard. Raising the
# number is the standing instruction for what comes next; it does not by itself
# make anything clearer, and nobody should read a bigger cap as the fix.
#
# ★ The direction of the trade is what this records: an over-long restatement
# that reads clearly costs a few seconds. A compressed one silently drops the
# amendment, the exception, or what the choice was BETWEEN — and reads perfectly
# well while doing it, which is why no checker catches it.
#
# (Was 90, set 2026-08-19: "80-100 should cover it." Was 60 before that, which
# is roughly two sentences and was squeezing plain restatements back into
# shorthand — the exact failure the rule exists to prevent, arriving through the
# guard meant to enforce it.)
#
# WARN only, never an error — at every value it has ever held. A long TL;DR that
# reads clearly is fine; the flag only says "check whether this drifted back
# into being the background".
TLDR_MAX_WORDS = 150


def tldr_findings(text):
    """Judge a restatement's text — the part after `**TL;DR —**`.

    `text` is the raw remainder captured by TLDR_RE.group(1), already stripped.
    Returns a list of (kind, severity, detail) with kind in:

        "empty"   error   detail None   — the line says TL;DR and nothing else
        "jargon"  error   detail [ids]  — bare identifiers the reader must look up
        "link"    error   detail None   — a link is a lookup with extra steps
        "long"    warn    detail words  — over TLDR_MAX_WORDS

    The CALLER formats the message, because a desk names a question (`Q7:`) and a
    ledger names an entry (`DEC-031:`). What is wrong is shared; how it is said
    is not.

    NOT JUDGED, and it is the important half: whether the restatement is
    accurate, or actually plainer than what it restates. A jargon-free sentence
    can still be incomprehensible, and nothing mechanical can tell.
    """
    if not text:
        return [("empty", "error", None)]
    out = []
    jargon = sorted({m.group(0) for m in TLDR_JARGON_RE.finditer(text)})
    if jargon:
        out.append(("jargon", "error", jargon))
    if TLDR_LINK_RE.search(text):
        out.append(("link", "error", None))
    words = len(text.split())
    if words > TLDR_MAX_WORDS:
        out.append(("long", "warn", words))
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


# A ledger that shares its DEC- series with nobody says so, in one word, on a
# line of its own. This is NOT the same state as an absent config file, and the
# difference is the whole point: absent means "nobody has said", which is
# unsafe to answer from; STANDALONE means "someone checked, and this series is
# its own". Only the second permits allocating a number.
#
# Added 2026-08-19, because next-free refused on this toolkit forever. The
# refusal was right -- it could not tell the two states apart -- but the repo
# is provably standalone: it holds DEC-001..028 while fran-dash holds
# DEC-001..220+, so the two cannot possibly be one series.
STANDALONE = "standalone"


def load_siblings(repo_root="."):
    """-> list of configured sibling ledger paths, or the string STANDALONE.

    Three distinct returns, and callers must treat them differently:
      []          no config file -- nobody has declared anything. Unsafe.
      STANDALONE  explicitly declared to share its series with no one. Safe.
      [paths...]  the siblings to read.
    """
    import os
    cfg = os.path.join(repo_root, SIBLING_CONFIG)
    if not os.path.exists(cfg):
        return []
    paths = []
    for raw in open(cfg, encoding="utf-8"):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if line.lower() == STANDALONE:
            return STANDALONE
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


def reserved_in_unapplied_intake(repo_root="."):
    """-> (reservations, scanned_count) where reservations is [(n, path)].

    An intake note that has been WRITTEN but not yet APPLIED claims DEC- numbers
    that appear in no ledger. `next-free` cannot see them, so it would hand one
    out twice. This is not hypothetical: the 2026-08-10 API session had to read
    fran-dash's intake by hand to discover DEC-215/216/217 were spoken for, and
    recorded DEC-218 as "the next genuinely free number" for exactly this reason.

    Deliberately a WARNING, never a subtraction. Parsing prose for reservations
    is fuzzy, and silently skipping numbers on a fuzzy match would produce gaps
    nobody can explain later. Name them; let the human rule.
    """
    import os
    import re
    reservations, scanned = [], 0
    intake = os.path.join(repo_root, "docs", "intake")
    if not os.path.isdir(intake):
        return reservations, scanned
    for dirpath, _dirs, files in os.walk(intake):
        for fn in files:
            if not fn.endswith("-reconciliation.md"):
                continue
            path = os.path.join(dirpath, fn)
            try:
                text = open(path, encoding="utf-8").read()
            except OSError:
                continue
            scanned += 1
            # Only notes NOT yet applied hold live reservations -- an applied
            # note's numbers are in the ledger, where dec_numbers() sees them.
            #
            # Read the STATE off the "Scope of this file:" line specifically,
            # never off the whole document. First cut matched "PENDING" anywhere
            # in the text and then harvested every DEC- number in the file; on
            # the real corpus that reported 121 reservations out of 24 notes,
            # because an applied note cites dozens of IDs in ordinary prose.
            # It "passed" only because the true answer happened to sit above
            # every number it wrongly collected.
            scope = ""
            for line in text.splitlines():
                if "Scope of this file:" in line:
                    scope = line.upper()
                    break
            if not scope or "APPLIED" in scope:
                continue
            if "PROPOSAL ONLY" not in scope and "PENDING" not in scope:
                continue
            # And take the number from the house-format declaration only, not
            # from prose. Every note this skill writes carries the line
            # "Suggested DEC IDs start at DEC-NNN".
            for m in re.finditer(r"[Ss]uggested DEC IDs? start at\D{0,12}DEC-(\d{3,})",
                                 text):
                reservations.append((int(m.group(1)), path))
    return reservations, scanned


def _cli_next_free(argv):
    """Report the next unused DEC- number across THIS ledger and every sibling.

    Exists because the hand-maintained "Next free number is DEC-NNN" line in
    NEXT_STEPS.md went stale four times in eleven days — and it is an ALLOCATOR,
    so trusting it does not merely mislead, it manufactures a collision.
    """
    import argparse
    import os
    ap = argparse.ArgumentParser(
        prog="ledger_contract.py next-free",
        description="Print the next free DEC- number across this ledger and "
                    "all sibling ledgers. Refuses to answer if any sibling "
                    "cannot be read.")
    ap.add_argument("ledger", help="this repo's DECISIONS.md")
    ap.add_argument("--sibling", action="append", default=[],
                    help=f"sibling ledger path (repeatable). Default: read {SIBLING_CONFIG}")
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--count", type=int, default=1,
                    help="report a run of N consecutive free numbers")
    args = ap.parse_args(argv)

    if not os.path.exists(args.ledger):
        print(f"REFUSING: no ledger at {args.ledger}")
        return 2

    siblings = args.sibling or load_siblings(args.repo_root)

    # A skip is NEVER a pass -- and for allocation it is worse than for
    # collision-checking. There, an unread sibling means "not checked"; here it
    # means the number printed may already be taken in the repo nobody read.
    # Measured 2026-08-12: DEC-221 was live in the API repo and absent here.
    standalone = siblings == STANDALONE
    if standalone:
        siblings = []
    elif not siblings:
        print(f"REFUSING to name a free number: no sibling ledgers configured "
              f"({SIBLING_CONFIG} absent).")
        print("  The DEC- series spans repos ([DEC-205]), so a single-ledger "
              "answer is not an answer.")
        print(f"  If this ledger genuinely shares its series with nobody, say so"
              f" explicitly: put the single word `{STANDALONE}` in"
              f" {SIBLING_CONFIG}.")
        return 1

    used = dec_numbers(open(args.ledger, encoding="utf-8").read(),
                       include_registry=True)
    # Registry stubs COUNT here, unlike in check-collisions. A stub is a receipt
    # for a number consumed by another repo ([DEC-219]) -- the number is spent,
    # so re-issuing it would create the very duplicate the stub records.
    read_ok = []
    for path in siblings:
        try:
            text = open(path, encoding="utf-8").read()
        except OSError as exc:
            print(f"REFUSING to name a free number: sibling unreadable: "
                  f"{path} ({exc})")
            return 1
        nums = dec_numbers(text, include_registry=True)
        if not nums:
            print(f"REFUSING to name a free number: sibling parsed to ZERO "
                  f"DEC- entries: {path}")
            print("  That is what a broken parser prints, and it is "
                  "indistinguishable from an empty ledger.")
            return 1
        used |= nums
        read_ok.append(path)

    if not used:
        print("REFUSING: no DEC- entries found in any ledger")
        return 1

    n = max(used) + 1
    run = list(range(n, n + max(1, args.count)))

    if len(run) == 1:
        print(f"next free: DEC-{run[0]:03d}")
    else:
        print(f"next free: DEC-{run[0]:03d}..DEC-{run[-1]:03d} "
              f"({len(run)} consecutive)")
    if standalone:
        print(f"  scanned: {args.ledger} ALONE — this ledger is declared "
              f"`{STANDALONE}` in {SIBLING_CONFIG}; "
              f"highest in use DEC-{max(used):03d} (registry stubs counted)")
        print("  NOT CHECKED: any other repo's ledger. That is correct only if "
              "the declaration is true — re-check it before trusting this "
              "number if the repo has since joined a shared series.")
    else:
        print(f"  scanned: {args.ledger} + {len(read_ok)} sibling(s); "
              f"highest in use DEC-{max(used):03d} (registry stubs counted)")

    # Name what was NOT checked, so silence cannot read as "nothing there".
    reservations, scanned = reserved_in_unapplied_intake(args.repo_root)
    clashes = sorted({r for r, _p in reservations if r >= run[0]})
    if clashes:
        print("")
        print(f"  ⚠ UNAPPLIED INTAKE claims {len(clashes)} number(s) at or above "
              f"this: " + ", ".join(f"DEC-{c:03d}" for c in clashes))
        for c in clashes:
            for r, p in reservations:
                if r == c:
                    print(f"      DEC-{c:03d} <- {os.path.relpath(p, args.repo_root)}")
                    break
        print("  These are in NO ledger yet, so the number above does not "
              "account for them. Read those notes before allocating.")
    elif scanned:
        print(f"  ✅ checked {scanned} intake reconciliation note(s) for "
              f"unapplied reservations — none at or above DEC-{run[0]:03d}")
    else:
        print("  ⚠ no intake reconciliation notes scanned (docs/intake absent) "
              "— unapplied reservations NOT checked")
    return 0


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
    # STANDALONE is a STRING, and a string is iterable -- without this branch the
    # loop below treats it as ten one-character sibling paths and reports ten
    # skips. That is exactly what happened on 2026-08-19 when the sentinel was
    # added to next-free and this second caller was missed: "all 10 configured
    # sibling(s) were skipped". A sentinel that shares a type with the normal
    # value has to be handled at EVERY consumer, not the one you were editing.
    if siblings == STANDALONE:
        print(f"skip: this ledger is declared `{STANDALONE}` in {SIBLING_CONFIG} "
              f"— it shares its DEC- series with no one, so there is nothing to "
              f"collide with")
        print("      NOT CHECKED: any other repo's ledger, by design. Correct "
              "only while that declaration is true.")
        return 0
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
    if _argv and _argv[0] == "next-free":
        raise SystemExit(_cli_next_free(_argv[1:]))
    raise SystemExit(
        "usage: ledger_contract.py check-collisions LEDGER [--sibling PATH]\n"
        "       ledger_contract.py next-free       LEDGER [--sibling PATH] [--count N]")
