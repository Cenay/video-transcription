#!/usr/bin/env python3
"""ledger-lint — check that a decision ledger says the same thing twice.

Run:  python3 scripts/ledger-lint.py [--ledger PATH] [--strict] [--quiet]
Exit: 0 all checks pass · 1 a check failed · 2 setup problem (no ledger, no parse)

Spec: plans/ledger-tooling-contract.md §3.

WHY. A ledger states each fact twice — once in the entry, once in the index —
and a reader trusts whichever they happen to read. Nothing checked that the two
agreed. The index in fran-dash was maintained BY HAND for 169 entries because
the generator could not parse the file, and the only reason anyone knew the
counts were right was that someone recounted them by hand and said so in a
warning above the table.

The nine checks below are what make the round trip trustworthy rather than
hoped-for. They are cheap; run this before finishing any session that touched
the ledger.

Checks 1-7 are about the ledger agreeing with itself. Check 8 is about it being
READABLE: every entry ends with a plain-English restatement that costs the
reader no lookups (the same rule as the Session Desk's rule 14, with its grammar
imported rather than restated). Check 9 is about it being UNIFORM: every entry
is written in the one shape ruled 2026-08-20, emitted by write-entry.py from the
same spec this check reads.

DESIGN NOTE. Every check reports the specific offending IDs or paths, never just
a count. "3 anchors are broken" sends you hunting; naming them does not. This
matters more than usual here: an earlier hand-rolled anchor checker reported
"3 broken anchors" on a ledger whose anchors were all correct — the checker had
a subtly wrong slugifier. A report you cannot verify at a glance is a report that
gets believed when it is wrong.
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from ledger_contract import (  # noqa: E402
    ADDED_BACKFILL, ADDED_STAMP_RE, ENTRY_LEVEL, FIELD_SYNONYMS,
    ID_PATTERN, REGISTRY_FIELDS, REQUIRED_FIELDS, STATUS_WORDS, TLDR_MAX_WORDS,
    TLDR_QUOTED_RE, TLDR_RE, body_tail, entry_bodies, fenced_lines,
    parse_entries, tldr_findings,
)

# ⚠️ Built from the SHARED ID_PATTERN, not hand-written. A private copy here
# required a hyphen, could not see `| [G77](#…)`, and reported G77 as an entry
# with no index row — while gen-dec-index's own private copy deleted that row and
# its shrink guard's third private copy failed to notice. Three scripts, three
# copies, three different wrong answers about the same ledger, all in one night.
ROW_RE = re.compile(rf"^\|\s*\[(?P<id>{ID_PATTERN})\]\((?P<anchor>#[^)]*)\)\s*\|")
# "**169 DEC entries**: **62 open** · **1 proposed** · ..."
#
# ⚠️ BOTH NOUNS, and this is the fix for a check that silently never ran. This
# pattern demanded "entries"; `gen-dec-index.py` writes "decisions", and every
# ledger on this machine is generator-written — measured 2026-08-20: fran-dash
# "**246 DEC decisions**", video-transcription "**9 DEC decisions**", this repo
# "**45 DEC decisions**", and NOT ONE saying "entries". So check 4 reported "no
# counts line found — skipped" on every ledger it had ever been pointed at, and
# the header arithmetic it exists to verify was never once verified.
#
# The LINTER was the wrong file to leave alone here: "entries" appears only in
# this comment and in a hand-written header nobody has produced. Accepting both
# costs nothing and cannot break a ledger that already passes.
TOTAL_RE = re.compile(r"\*\*(\d+)\s+DEC (?:entries|decisions)\*\*")
# ⚠️ The label may be a bare em-dash. `gen-dec-index` writes "**1 —**" for an
# entry whose Status it could not parse — an honest count of "one I cannot
# classify". A pattern accepting only words cannot see it, so the tallies came
# up one short and check 4 reported a header that was correct: 170+66+5+4 = 245
# against a stated 246, where the missing 1 was the em-dash entry.
TALLY_RE = re.compile(r"\*\*(\d+)\s+([a-z]+|[—–-])\*\*")
# A line that is SHAPED like a counts line — "**<n> DEC <something>**" — so a
# noun this script does not know produces a loud "I could not parse it" instead
# of the flat lie "no counts line found". That distinction is the whole lesson:
# silence has to mean "nothing there", never "I did not look".
SUSPECT_TOTAL_RE = re.compile(r"\*\*\d+\s+DEC\b[^*]*\*\*")
# A path-looking citation: docs/foo.md, plans/bar.md, scripts/baz.py
PATH_RE = re.compile(r"(?<![\w/.-])((?:[\w.-]+/)+[\w.-]+\.(?:md|py|sh|ya?ml|json|csv))")
# Opt-out for a table whose paths are relative to somewhere else (e.g. an index
# of a frozen sibling repo). Applies until the next blank-line-separated block.
BASE_RE = re.compile(r"<!--\s*ledger-lint:\s*base=(?P<base>\S+)\s*-->")
SKIP_RE = re.compile(r"<!--\s*ledger-lint:\s*ignore-paths\s*-->")
# `<!-- ledger-lint: adopting -->` — a back-fill is in progress on THIS ledger,
# so a missing restatement stays a warning until the last one lands.
#
# WHY (ruled 2026-08-20 by Cenay). The self-arming rule — warn while no entry
# has one, error once any entry does — is right for a ledger finished in one
# sitting. It is wrong at 252 entries, where the back-fill cannot honestly be
# done in one session and the first restatement would turn the file red for
# days. The marker suppresses that window explicitly, per ledger.
#
# ⚠️ It is loud, and it CANNOT outlive its purpose: every run reports the
# remaining count, and once nothing is missing the marker itself becomes an
# ERROR telling you to remove it. A suppression you can forget about is just a
# switched-off check with extra steps.
ADOPTING_RE = re.compile(r"<!--\s*ledger-lint:\s*adopting\b[^>]*-->")


class Report:
    def __init__(self, quiet):
        self.quiet, self.failed = quiet, 0

    def ok(self, name, detail=""):
        if not self.quiet:
            print(f"  ✓ {name}{(' — ' + detail) if detail else ''}")

    def fail(self, name, detail, items=()):
        self.failed += 1
        print(f"  ✗ {name} — {detail}")
        for it in list(items)[:15]:
            print(f"      {it}")
        if len(list(items)) > 15:
            print(f"      … and {len(list(items)) - 15} more")

    def warn(self, name, detail, items=()):
        print(f"  ⚠ {name} — {detail}")
        for it in list(items)[:10]:
            print(f"      {it}")


def find_ledger(explicit):
    if explicit:
        p = Path(explicit)
        return p if p.is_file() else None
    for cand in ("docs/DECISIONS.md", ".cloaked/docs/DECISIONS.md", "DECISIONS.md"):
        p = Path(cand)
        if p.is_file():
            return p
    return None


def index_rows(lines):
    """Index rows with their line numbers, in page order."""
    return [(i, m) for i, line in enumerate(lines) for m in [ROW_RE.match(line)] if m]


def gitignored(paths, root):
    """Subset of `paths` that git ignores. One call, not one per path."""
    if not paths:
        return set()
    try:
        r = subprocess.run(["git", "-C", str(root), "check-ignore", "--stdin"],
                           input="\n".join(paths), capture_output=True, text=True)
        return {l.strip() for l in r.stdout.split("\n") if l.strip()}
    except Exception:
        return set()


def cited_paths(lines):
    """Path-shaped citations, minus any under a base= or ignore-paths marker."""
    out, skip_until_blank, base = [], False, None
    in_managed = False
    for i, line in enumerate(lines):
        # The generated index block is machine-written and names the tool that
        # wrote it ("auto-generated by scripts/gen-dec-index.py"). That is a tool
        # NAME in prose, not a link anyone follows — flagging it told the reader
        # to go fix a file the generator itself had just written.
        if "dec-index:start" in line:
            in_managed = True
        elif "dec-index:end" in line:
            in_managed = False
            continue
        if in_managed:
            continue
        if SKIP_RE.search(line):
            skip_until_blank = True
            continue
        bm = BASE_RE.search(line)
        if bm:
            base, skip_until_blank = bm.group("base"), False
            continue
        if line.strip() == "":
            skip_until_blank, base = False, None
            continue
        if skip_until_blank:
            continue
        for m in PATH_RE.finditer(line):
            out.append((i + 1, m.group(1), base))
    return out


# A field line, bulleted or bare. Both dialects are in use — bulleted in two
# ledgers, bare in two — so the pattern must SEE both in order to report the
# bare one. A pattern that only matched the canonical form would report a bare
# ledger as having no fields at all, which reads as a parse failure rather than
# as the finding it is.
FIELD_LINE_RE = re.compile(r"^(?P<bullet>-\s+)?\*\*(?P<label>[^*:]{1,40}):\*\*\s*(?P<value>.*)$")
DATE_IN_TITLE_RE = re.compile(r"^\s*[—-]?\s*\d{4}-\d{2}-\d{2}")


def check_shape(r, lines, entries):
    """Check 9 — every entry is written in the one canonical shape.

    Ruled 2026-08-20 by Cenay: `###` heading, the required fields first and in
    order as `- **Label:**` bullets, restatement last. ⛔ Everything BETWEEN the
    fields and the restatement is free — bold paragraphs, sub-headings, tables,
    folds. That bound is deliberate and it is what lets the rule be strict:
    checked at the two ends of an entry only, so it can never refuse a
    well-written one. A strict rule that refuses good work gets switched off.

    The shape comes from `ledger_contract`'s ENTRY_FIELDS, imported rather than
    restated — the same module the writer emits from. So the checker and the
    writer cannot disagree about what conforming means; there is one definition
    and both read it.

    NOT CHECKED: whether a field's CONTENT is any good. An entry can carry all
    eight fields, in order, and say nothing worth reading.
    """
    adopting = any(ADOPTING_RE.search(l) for l in lines)
    required = list(REQUIRED_FIELDS)
    reg_required = [n for n, req in REGISTRY_FIELDS if req]

    findings, conforming = {}, []

    def note(reason, eid):
        findings.setdefault(reason, []).append(eid)

    for e, start, body in entry_bodies(lines, entries):
        ok = True
        if e.level != ENTRY_LEVEL:
            note(f"heading is at level {e.level}, not {ENTRY_LEVEL} "
                 f"({'#' * ENTRY_LEVEL} <ID> <title>)", e.id)
            ok = False
        if DATE_IN_TITLE_RE.match(e.title):
            note("a date in the heading — it belongs in `Added`", e.id)
            ok = False

        # The leading run of field-shaped lines. It ends at the first line that
        # is not one — which is where the free-form body begins.
        run, seen_any = [], False
        for raw in body:
            if not raw.strip():
                if seen_any:
                    break
                continue
            m = FIELD_LINE_RE.match(raw)
            if not m:
                break
            seen_any = True
            run.append(m)

        if not run:
            note("no field lines at all under the heading", e.id)
            continue

        labels = [m.group("label").strip() for m in run]
        status_val = next((m.group("value") for m in run
                           if m.group("label").strip() == "Status"), "")
        registry = "REGISTRY" in status_val.upper()
        want = reg_required if registry else required

        # ⛔ ONLY the required prefix is held to the shape. Everything after it is
        # free — the ruling allows extra bold-labeled paragraphs, and they are
        # written as bullets in this very ledger ("The asymmetry, stated plainly",
        # "Why the toolkit is standalone"). The first version of this check read
        # any unfamiliar label anywhere in the leading run as a violation and
        # reported four of them here, all of which the ruling permits. A checker
        # that flags what the rule allows is one you learn to disbelieve.
        prefix = run[:len(want)]
        if [m.group("label").strip() for m in prefix] != want:
            got = labels or ["nothing"]
            first_bad = next((i for i in range(len(want))
                              if i >= len(labels) or labels[i] != want[i]), 0)
            note(f"required fields are not first and in order — expected "
                 f"{want[first_bad]!r} at position {first_bad + 1}, found "
                 f"{(got[first_bad] if first_bad < len(got) else 'nothing')!r}", e.id)
            ok = False
        elif any(m.group("bullet") is None for m in prefix):
            # Checked on the prefix only, for the same reason: an unbulleted
            # `**Note:**` paragraph further down is body, not a malformed field.
            note("field lines are not bulleted — the shape is `- **Label:** value`", e.id)
            ok = False

        # Old vocabulary is reported wherever it appears, because renaming it is
        # part of what normalization has to do.
        renamed = [l for l in labels if l in FIELD_SYNONYMS]
        if renamed:
            pairs = ", ".join(f"{l} → {FIELD_SYNONYMS[l]}" for l in sorted(set(renamed)))
            note(f"uses a renamed field label ({pairs})", e.id)
            ok = False

        if status_val:
            words = set(re.findall(r"[A-Z]{3,}", status_val))
            if not words & STATUS_WORDS:
                extra = " — `CLOSED` was retired 2026-08-20, use `RESOLVED`" \
                    if "CLOSED" in words else ""
                note(f"`Status` names none of the vocabulary{extra}", e.id)
                ok = False

        added = next((m.group("value").strip() for m in run
                      if m.group("label").strip() == "Added"), None)
        if added is not None and added != ADDED_BACKFILL and not ADDED_STAMP_RE.match(added):
            note("`Added` is not a full stamp (date, 24-hour time, zone) nor the "
                 "back-fill marker", e.id)
            ok = False

        if ok:
            conforming.append(e.id)

    total = len(entries)
    bad = total - len(conforming)
    items = [f"{reason} — {len(ids)}: {', '.join(ids[:6])}"
             + (f" … and {len(ids) - 6} more" if len(ids) > 6 else "")
             for reason, ids in sorted(findings.items(), key=lambda kv: -len(kv[1]))]

    if not bad:
        r.ok("9 entries conform to the entry template", f"{total} checked")
        return 0
    # Same self-arming rule as check 8, and for the same reason: a ledger where
    # NOTHING conforms predates the template, so a wall of hard failures on a
    # file nobody has touched is a gate that gets overridden by reflex.
    if adopting:
        r.warn("9 entries conform to the entry template",
               f"{bad} of {total} do not — ⚠️ this ledger is marked ADOPTING, so these "
               f"are warnings on purpose while the reshaping runs", items)
    elif conforming:
        r.fail("9 entries conform to the entry template",
               f"{bad} of {total} do not, and {len(conforming)} do — so this ledger has "
               f"adopted the template", items)
    else:
        r.warn("9 entries conform to the entry template",
               f"{bad} of {total} do not — WARNING ONLY: no entry here conforms yet, so "
               f"this ledger predates the template. Reshape one and the rest become "
               f"required", items)
    return bad


def check_tldr(r, lines, entries):
    """Check 8 — every entry ends with a plain-English restatement.

    The rule and its grammar are shared with the Session Desk's rule 14 and are
    imported, not restated: see ledger_contract's restatement section. What is
    local to a ledger is PLACEMENT. A desk item folds its body, so the
    restatement has to sit above the folds to survive a collapse; a ledger entry
    has no folds, its fields are always visible, so the natural reading order is
    fields first and the plain-English summary LAST.

    Why a ledger needs this at least as much as a desk: measured 2026-08-19,
    the `**Question:**` field across these entries runs a median of 32 words and
    is dense with cross-referenced ids — many of them decisions in ANOTHER repo,
    which the reader cannot resolve by scrolling. And a ledger is read more
    often than a desk, by someone with less context, months later.

    NOT CHECKED, and it is the important half: whether a restatement is accurate
    or genuinely plainer. Nothing mechanical can tell.
    """
    fences = fenced_lines(lines)
    bodies = entry_bodies(lines, entries)

    hits_by_entry = []
    for e, start, body in bodies:
        hits = [(start + n, m.group(1).strip(), raw.strip())
                for n, raw in enumerate(body)
                if (start + n) not in fences
                for m in [TLDR_RE.match(raw.strip())] if m]
        hits_by_entry.append((e, start, body, hits))

    # The rule grandfathers itself, and the test is mechanical: a ledger where NO
    # entry carries a restatement predates the rule, so each missing one is a
    # WARNING. The moment ONE entry has it, the ledger has adopted the rule and
    # every other missing one is an ERROR.
    #
    # Not optional. Without it, check 8 opens with a hard failure on every entry
    # of a file nobody has touched — and a gate that cries wolf gets overridden
    # by reflex, which is the lesson already recorded about the mtime-based drift
    # check. Self-healing, with no dated grandfather list to maintain: writing
    # one restatement arms the rule for the whole ledger.
    adopted = any(hits for _, _, _, hits in hits_by_entry)

    missing, errs, warns = [], [], []
    for e, start, body, hits in hits_by_entry:
        if not hits:
            missing.append(f"{e.id} (line {e.line})")
            continue
        if len(hits) > 1:
            errs.append(f"{e.id}: {len(hits)} TL;DR lines — there must be exactly one "
                        f"(lines {', '.join(str(n + 1) for n, _, _ in hits)})")

        n, textv, rawv = hits[0]
        ln = n + 1

        # Presentation, so a warning and never an error — a ledger written before
        # the blockquote form stays usable.
        if not TLDR_QUOTED_RE.match(rawv):
            warns.append(f"{e.id} (line {ln}): the TL;DR is not in a blockquote — "
                         f"prefix it with `> ` so it stands out from the fields")

        # Blank lines and a trailing horizontal rule after the restatement are
        # furniture separating this entry from the next, not content — so the
        # restatement is still last. body_tail() is the single definition of
        # that, shared with the placer so the two cannot disagree.
        trailing = [start + j + 1
                    for j, raw in enumerate(body[n - start + 1:], n - start + 1)
                    if raw.strip() and j < body_tail(body)]
        if trailing:
            errs.append(f"{e.id} (line {ln}): the TL;DR is not the last element — "
                        f"{len(trailing)} more line(s) follow it before the next entry "
                        f"(first at line {trailing[0]})")

        for kind, _sev, detail in tldr_findings(textv):
            if kind == "empty":
                errs.append(f"{e.id} (line {ln}): the TL;DR line is empty")
            elif kind == "jargon":
                errs.append(f"{e.id} (line {ln}): names {', '.join(detail)} — say the THING "
                            f"the identifier refers to, not the identifier")
            elif kind == "link":
                errs.append(f"{e.id} (line {ln}): carries a link — a restatement must be "
                            f"readable where it stands, with nothing to click")
            elif kind == "long":
                warns.append(f"{e.id} (line {ln}): runs {detail} words (over "
                             f"{TLDR_MAX_WORDS}) — it may have drifted back into the "
                             f"background it replaces")

    adopting = any(ADOPTING_RE.search(l) for l in lines)

    if adopting and not missing:
        # Nothing outstanding HERE — but the marker covers check 9 too, and that
        # may still have work to do. Whether it has outlived its purpose is
        # decided once, in main(), after both checks have reported. ⚠️ Deciding
        # it here made a ledger mid-reshape fail for carrying a marker it still
        # needed; the suite caught it on the first run.
        r.ok("8 every entry carries a TL;DR",
             f"{len(entries)} checked — ⚠️ this ledger is marked ADOPTING")
    elif adopting:
        r.warn("8 every entry carries a TL;DR",
               f"{len(missing)} of {len(entries)} still without one — ⚠️ this ledger is "
               f"marked ADOPTING, so these are warnings on purpose while the back-fill "
               f"runs. Remove the marker when the count reaches zero", missing)
    elif missing and adopted:
        r.fail("8 every entry carries a TL;DR", f"{len(missing)} without one", missing)
    elif missing:
        r.warn("8 every entry carries a TL;DR",
               f"{len(missing)} without one — WARNING ONLY: no entry on this ledger has "
               f"one yet, so it predates the rule. Add one and the rest become required",
               missing)
    else:
        r.ok("8 every entry carries a TL;DR", f"{len(entries)} checked")

    if errs:
        r.fail("8a TL;DR restatements are self-contained",
               f"{len(errs)} problem(s) — no bare identifiers, no links, last in the entry",
               errs)
    elif not missing:
        r.ok("8a TL;DR restatements are self-contained")
    if warns:
        r.warn("8b TL;DR presentation", f"{len(warns)} worth a look", warns)
    return len(missing)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ledger", help="path to the ledger (default: auto-detect)")
    ap.add_argument("--strict", action="store_true",
                    help="treat unresolved path citations as failures, not warnings")
    ap.add_argument("--quiet", action="store_true", help="print failures only")
    args = ap.parse_args()

    ledger = find_ledger(args.ledger)
    if not ledger:
        print("SETUP: no ledger found (looked for docs/DECISIONS.md, "
              ".cloaked/docs/DECISIONS.md, DECISIONS.md). Pass --ledger.")
        return 2

    root = Path(subprocess.run(["git", "rev-parse", "--show-toplevel"],
                               capture_output=True, text=True).stdout.strip() or ".")
    text = ledger.read_text(encoding="utf-8")
    lines = text.split("\n")
    r = Report(args.quiet)

    if not args.quiet:
        print(f"ledger-lint {ledger}")

    # 1 — every entry parses. Zero is a setup failure, not a lint failure: an
    #     unparseable ledger and an empty one look identical from here, and that
    #     ambiguity is exactly what hid the gen-dec-index breakage for weeks.
    entries = parse_entries(text)
    if not entries:
        print(f"SETUP: parsed 0 entries from {ledger}. Expected headings shaped "
              f"'## DEC-NNN Title' (## to ####). Nothing else could be checked.")
        return 2
    r.ok("1 entries parse", f"{len(entries)} found")

    # 2 — every entry has a non-empty Status. Catches the mixed bullet/bare shape,
    #     which otherwise yields a full-looking index with a blank Status column.
    missing = [f"{e.id} (line {e.line})" for e in entries if not e.status]
    if missing:
        r.fail("2 every entry has a Status", f"{len(missing)} without one", missing)
    else:
        r.ok("2 every entry has a Status")

    rows = index_rows(lines)
    if not rows:
        r.warn("3-5 index checks", "no index table found — skipped")
    else:
        # 3 — index and entries agree, both directions. A one-way check passes
        #     happily while the index carries a row for a deleted entry.
        entry_ids = {e.id for e in entries}
        row_ids = [m.group("id") for _, m in rows]
        dupes = sorted({i for i in row_ids if row_ids.count(i) > 1})
        only_index = sorted(set(row_ids) - entry_ids)
        only_entry = sorted(entry_ids - set(row_ids))
        if only_index or only_entry or dupes:
            detail = (f"{len(only_index)} in index only, {len(only_entry)} in "
                      f"entries only, {len(dupes)} duplicated")
            r.fail("3 index ↔ entries agree", detail,
                   [f"index only: {i}" for i in only_index]
                   + [f"entries only: {i}" for i in only_entry]
                   + [f"duplicate row: {i}" for i in dupes])
        else:
            r.ok("3 index ↔ entries agree", f"{len(rows)} rows ↔ {len(entries)} entries")

        # 4 — the header's own arithmetic. It is prose, so nothing recomputes it;
        #     it drifts silently every time an entry closes.
        header = next((l for _, l in enumerate(lines)
                       if TOTAL_RE.search(l) and TALLY_RE.search(l)), None)
        if not header:
            # Two different states wearing one message until 2026-08-20. A ledger
            # with no counts line has nothing to check; a ledger WITH one this
            # script cannot parse has something to check and did not. Reporting
            # both as "no counts line found" is how check 4 sat dead for weeks
            # while printing a reassuring line.
            suspect = next((l for l in lines if SUSPECT_TOTAL_RE.search(l)
                            and TALLY_RE.search(l)), None)
            if suspect:
                r.fail("4 header counts sum to the rows",
                       "found a line SHAPED like a counts line but could not parse its "
                       "total — check 4 did NOT run. Expected '**<n> DEC entries**' or "
                       "'**<n> DEC decisions**'",
                       [suspect.strip()[:160]])
            else:
                r.warn("4 header counts sum to the rows",
                       "no counts line found — NOT CHECKED (the header's arithmetic is "
                       "unverified, not verified-clean)")
        else:
            total = int(TOTAL_RE.search(header).group(1))
            tallies = [(int(n), w) for n, w in TALLY_RE.findall(header)
                       if not w.endswith("entries")]
            # The total appears as "**169 DEC entries**" and is re-matched by
            # TALLY_RE; drop it before summing or every ledger fails this check.
            tallies = [(n, w) for n, w in tallies if n != total or w != "dec"]
            s = sum(n for n, _ in tallies)
            parts = ", ".join(f"{n} {w}" for n, w in tallies)
            # The header counts "N DEC entries" explicitly, and a ledger may carry
            # non-DEC rows beside them (fran-dash indexes the G77 launch gate as a
            # row). Comparing against ALL rows made a correct header look wrong.
            # ⚠️ REGISTRY rows are excluded, because the header excludes them.
            # A registry stub is a receipt for a number consumed by another
            # repo's decision, and the generated header counts it separately:
            # "**246 DEC decisions** … plus **5 registry stubs**". Counting all
            # DEC- rows made check 4 report 246-versus-251 on a header whose
            # arithmetic was correct — ✅ 170+66+5+4+1 = 246, and 246 + 5 stubs
            # + the G77 gate = 252 rows. This was reported to Cenay as "that
            # ledger disagrees with itself" before it was measured. It did not;
            # the checker did. A checker that invents a defect is worse than no
            # checker, and this one was one regeneration away from being blamed
            # on the file it was pointed at.
            def _is_registry(line):
                # The STATUS cell only — `| [id](#a) | title | status | date |`.
                # ⚠️ Matching the whole line excluded one row too many: DEC-219's
                # TITLE is "A registry stub is a receipt, not a decision", so a
                # line-wide match counted a real decision as a stub. 251 - 6 = 245
                # against a header saying 246 — a checker off by one in the other
                # direction, which is how a fix becomes the next bug.
                cells = line.split("|")
                return len(cells) > 3 and "REGISTRY" in cells[3].upper()

            dec_rows = len([1 for i, m in rows
                            if m.group("id").startswith("DEC-")
                            and not _is_registry(lines[i])])
            if total != dec_rows:
                r.fail("4 header total matches the rows",
                       f"header says {total} DEC entries, table has {dec_rows}")
            elif s != total:
                r.fail("4 header counts sum to the total",
                       f"{parts} = {s}, but header says {total}")
            else:
                # An em-dash tally is not a failure, but it must not be silent
                # either: it means an entry's Status could not be classified.
                unparsed = sum(n for n, w in tallies if w in "—–-")
                extra = (f" — ⚠️ {unparsed} entry(s) counted as unclassified, "
                         f"which means their Status did not parse") if unparsed else ""
                r.ok("4 header counts sum", f"{parts} = {total}{extra}")

        # 5 — every index anchor resolves to a real heading slug.
        #     ⚠️ slugify() must be link-doc-refs.py's, double hyphens and all.
        slugs = {e.slug for e in entries}
        broken = [f"{m.group('id')} → {m.group('anchor')} (line {i + 1})"
                  for i, m in rows if m.group("anchor").lstrip("#") not in slugs]
        if broken:
            r.fail("5 index anchors resolve", f"{len(broken)} broken", broken)
        else:
            r.ok("5 index anchors resolve", f"{len(rows)} checked")

    # 6 / 7 — citations. Two different failures wearing the same shape: a path
    #     that resolves for nobody (6), and one that resolves for YOU but not for
    #     a teammate because it is gitignored (7). The second is the nastier one —
    #     it looks fine on the machine that wrote it.
    cites = cited_paths(lines)
    if not cites:
        r.ok("6-7 path citations", "none found")
    else:
        uniq = sorted({p for _, p, _ in cites})
        ignored = gitignored(uniq, root)
        # A ledger that merges predecessor repos cites them by their own root
        # (`dashboard/CLAUDE.md`, `migration/plans/gap-report.md`). Those resolve
        # one directory up. Reporting them as broken buries the real breaks: the
        # first run here produced 264 "unresolved" of which the overwhelming
        # majority were simply cross-repo. A checker whose output you have to
        # filter by eye is one you stop reading.
        siblings = [d for d in root.parent.iterdir() if d.is_dir()] \
            if root.parent.exists() else []
        unresolved, cross, ambiguous = [], set(), {}
        for line_no, p, base in cites:
            if p in ignored:
                continue
            if base and (root / base / p).exists():
                continue
            if (root / p).exists():
                continue
            # Relative to the CITING DOCUMENT, which is how a Markdown link
            # actually resolves. The ledger lives in docs/, so `history/foo.md`
            # means `docs/history/foo.md`. Checking only repo-root-relative paths
            # reported 154 of these as broken when every one of them worked.
            if (ledger.parent / p).exists():
                continue
            # Prefixed with a sibling repo's name — `dashboard/CLAUDE.md`. Resolves.
            if (root.parent / p).exists():
                cross.add(p.split("/")[0])
                continue
            # UNPREFIXED but present in a sibling — `docs/SCHEMA.md` meaning
            # `migration/docs/SCHEMA.md`. The file exists, so this is not a broken
            # link; but a reader in THIS repo who follows it finds nothing, so it
            # is not clean either. Its own class: real debt, not an emergency.
            hit = next((s.name for s in siblings if (s / p).exists()), None)
            if hit:
                ambiguous.setdefault(hit, set()).add(p)
                continue
            unresolved.append(f"{p} (line {line_no})")
        if cross:
            r.ok("6a cross-repo citations resolve",
                 f"via sibling repos: {', '.join(sorted(cross))}")
        if ambiguous:
            total = sum(len(v) for v in ambiguous.values())
            r.warn("6b ambiguous cross-repo citations",
                   f"{total} path(s) exist only in a sibling repo and are written "
                   f"without its prefix",
                   [f"{repo}/ — {len(paths)} path(s), e.g. {sorted(paths)[0]}"
                    for repo, paths in sorted(ambiguous.items())])
        if unresolved:
            u = sorted(set(unresolved))
            if args.strict:
                r.fail("6 path citations resolve", f"{len(u)} unresolved", u)
            else:
                r.warn("6 path citations resolve",
                       f"{len(u)} unresolved (use --strict to fail on these)", u)
        else:
            r.ok("6 path citations resolve", f"{len(uniq)} checked")

        if ignored:
            cited_ignored = sorted({f"{p} (line {n})" for n, p, _ in cites if p in ignored})
            msg = (f"{len(ignored)} path(s) are local-only and would be dead for a "
                   f"teammate following them")
            # ⚠ by default, not ✗. A ledger legitimately DESCRIBES gitignored files
            # — fran-dash's entry on the `.claude/` sharing rule names
            # `.claude/settings.local.json` precisely to say it is per-machine and
            # not shared. That mention is correct documentation, and failing the
            # run over it is how a checker earns a reputation for being wrong.
            # Distinguishing "mentions" from "links" reliably is not something
            # this can do, so it reports and lets a human judge. --strict to fail.
            if args.strict:
                r.fail("7 no citation points at a gitignored path", msg, cited_ignored)
            else:
                r.warn("7 citations to gitignored paths",
                       msg + " (use --strict to fail on these)", cited_ignored)
        else:
            r.ok("7 no gitignored citations")

    # 8 — every entry ends with a plain-English restatement (see check_tldr).
    outstanding = check_tldr(r, lines, entries)

    # 9 — every entry is written in the one canonical shape (see check_shape).
    outstanding += check_shape(r, lines, entries)

    # The adopting marker is SHARED by checks 8 and 9, so whether it has outlived
    # its purpose can only be judged after both have run. ⚠️ Judging it inside
    # check 8 made a ledger that had finished its restatements but not its
    # reshaping fail for carrying a marker it still needed.
    if any(ADOPTING_RE.search(l) for l in lines) and not outstanding:
        r.fail("the adopting marker has outlived its purpose",
               "every entry now carries a restatement AND conforms to the template — "
               "delete the `<!-- ledger-lint: adopting -->` marker so both checks "
               "enforce again")

    print()
    if r.failed:
        print(f"{r.failed} check(s) FAILED")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
