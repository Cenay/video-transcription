#!/usr/bin/env python3
"""write-entry — emit a ledger entry in the one canonical shape.

Run:  python3 scripts/write-entry.py ENTRY.json [--ledger PATH] [--append]
                                     [--dry-run]
      python3 scripts/write-entry.py --template          # the skeleton, for scaffolding
Exit: 0 emitted (or appended) · 1 refused · 2 setup problem

Spec: plans/ledger-entry-shapes.md — ruled 2026-08-20 by Cenay.

WHY A PROGRAM AND NOT A TEMPLATE FILE. A template already existed: an HTML
comment inside the ledger that `commands/init-project.md` scaffolds, prescribing
very nearly this shape. It did not hold, because it reached a project only at
creation, was invisible in the rendered document, and nothing imported it. This
script is the answer to all three — the shape comes from `ledger_contract`'s
ENTRY_FIELDS, so it is stated once, and every copy anyone reads is generated.

★ IT IS ALSO THE NORMALIZER'S BACK END, and that is why it was built first.
Reshaping an existing ledger does not need its own reshaping logic: parse an old
entry into field values, hand them here, and assert the free-form body came
through verbatim. A normalized old entry and a brand-new one then come out of
the same code path — identical by construction, not because two implementations
happen to agree.

FIELD ORDER IS NOT VALIDATED, IT IS IMPOSED. Input is a mapping; output is
always spec order. An entry cannot be written with its fields out of order,
which removes a whole class of finding from the checker's job.

⛔ IT DOES NOT ALLOCATE NUMBERS. Pass `id`. Allocation belongs to
`ledger_contract.py next-free`, which also consults sibling ledgers and
unapplied reservations — duplicating a thinner version of that here is exactly
how two repos issue the same number.
"""

import argparse
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from ledger_contract import (  # noqa: E402
    ADDED_BACKFILL, ADDED_STAMP_RE, BACKFILL_SENTINELS, ENTRY_FIELDS,
    ENTRY_LEVEL, FIELD_SYNONYMS, ID_PATTERN, REGISTRY_FIELDS, REQUIRED_FIELDS,
    STATUS_WORDS, body_tail, entry_bodies, entry_template, parse_entries,
    status_vocab_line, tldr_findings,
)

TLDR_PREFIX = "> **TL;DR —** "
ID_RE = re.compile(rf"^{ID_PATTERN}$")
DECIDED_RE = re.compile(r"^\d{4}-\d{2}-\d{2}")
# ⚠️ The stamp pattern and the status words are IMPORTED, not defined here —
# the checker reads the same two, so a ledger cannot be written in a shape its
# own linter rejects.


def emit_entry(entry_id, title, fields, body="", tldr="", registry=False):
    """The canonical rendering. Importable — the normalizer calls this directly.

    `fields` is a mapping; order comes from the spec, never from the caller.
    """
    spec = REGISTRY_FIELDS if registry else ENTRY_FIELDS
    out = [f"{'#' * ENTRY_LEVEL} {entry_id} {title}".rstrip()]
    for name, _required in spec:
        val = (fields.get(name) or "").strip()
        if val:
            out.append(f"- **{name}:** {val}")
    body = (body or "").strip("\n")
    if body:
        out += ["", body]
    # ★ A registry stub carries a restatement too — ruled 2026-08-20. It is the
    # one line that makes a receipt useful: "this number belongs to a decision
    # about activity-log retention, recorded in the interface repo" tells the
    # reader more than the receipt itself does. So there is no exemption here,
    # and the restatement check needs no special case either.
    if tldr:
        out += ["", TLDR_PREFIX + tldr.strip()]
    return "\n".join(out)


def validate(spec_in):
    """Every reason this entry may not be written. Returns a list of problems.

    ALL of them, never the first — a caller who has to re-run once per problem
    stops reading the output and starts guessing.
    """
    p = []
    entry_id = (spec_in.get("id") or "").strip()
    if not entry_id:
        p.append("no `id` — this script does not allocate. Get one with: "
                 "python3 scripts/ledger_contract.py next-free <ledger>")
    elif not ID_RE.match(entry_id):
        p.append(f"id {entry_id!r} is not a recognized identifier shape")

    title = (spec_in.get("title") or "").strip()
    if not title:
        p.append("no `title`")
    elif entry_id and title.startswith(entry_id):
        p.append("the title repeats the id — the heading already carries it")
    elif re.match(r"^\d{4}-\d{2}-\d{2}", title):
        p.append("the title starts with a date — the date belongs in `Added` "
                 "(ruled 2026-08-20: no date in the heading)")

    registry = bool(spec_in.get("registry"))
    fields = spec_in.get("fields") or {}
    if not isinstance(fields, dict):
        return p + ["`fields` must be an object of label → value"]

    for label in fields:
        if label in FIELD_SYNONYMS:
            p.append(f"field {label!r} was renamed to {FIELD_SYNONYMS[label]!r} — "
                     f"use the canonical label")
        elif label not in {n for n, _ in ENTRY_FIELDS} | {n for n, _ in REGISTRY_FIELDS}:
            p.append(f"field {label!r} is not in the spec")

    required = ([n for n, r in REGISTRY_FIELDS if r] if registry else REQUIRED_FIELDS)
    for name in required:
        if not (fields.get(name) or "").strip():
            p.append(f"required field {name!r} is missing or empty")

    # ⛔ A BACK-FILL MAY STATE A GAP; AN AUTHOR MAY NOT. The sentinels exist so a
    # historical entry whose reasoning was never written down can conform without
    # anyone inventing one. Ruled 2026-08-20: "I don't want to drop any we CAN
    # rebuild. Only those that we have nothing to recover from."
    #
    # ★ Refusing them HERE is the whole reason the field could stay required
    # instead of being relaxed. Relaxing it would have applied to new entries
    # too — the writer and the checker read the same REQUIRED_FIELDS — so the
    # gap-statement had to be something only the back-fill can place. `Added` is
    # exempt: its sentinel is checked just below, where a real stamp is also
    # accepted, because an unrecoverable authoring date is an ordinary outcome
    # for an entry written before the field existed.
    for name, val in ((n, (fields.get(n) or "").strip()) for n in required):
        if name != "Added" and val in BACKFILL_SENTINELS:
            p.append(f"field {name!r} is the back-fill literal {val!r} — that states "
                     f"a gap in a HISTORICAL entry and may only be written by the "
                     f"back-fill. A new entry must carry a real value.")

    added = (fields.get("Added") or "").strip()
    if added and added != ADDED_BACKFILL and not ADDED_STAMP_RE.match(added):
        p.append(f"`Added` is {added!r} — expected `{ADDED_STAMP_RE.pattern}`-shaped "
                 f"(e.g. `2026-08-20 09:35 MST`, from `date`, never guessed) "
                 f"or the literal `{ADDED_BACKFILL}` when it cannot be recovered")

    status = (fields.get("Status") or "").strip()
    if status:
        words = set(re.findall(r"[A-Z]{3,}", status))
        if not words & STATUS_WORDS:
            p.append(f"`Status` names none of the vocabulary — {status_vocab_line()}"
                     + (" (⚠️ `CLOSED` was retired 2026-08-20; use `RESOLVED`)"
                        if "CLOSED" in words else ""))

    decided = (fields.get("Decided") or "").strip()
    if decided and not DECIDED_RE.match(decided):
        p.append(f"`Decided` is {decided!r} — expected a date, `YYYY-MM-DD` first")

    tldr = (spec_in.get("tldr") or "").strip()
    # No exemption for registry stubs — ruled 2026-08-20. A receipt's restatement
    # is the most useful line in it: it says what the OTHER repo decided, which
    # the stub itself deliberately does not record.
    if not tldr:
        p.append("no `tldr` — every entry ends with a plain-English restatement, "
                 "registry stubs included: say what the decision in the other repo "
                 "actually was")
    else:
        for kind, sev, detail in tldr_findings(tldr):
            if sev != "error":
                continue
            p.append({
                "empty": "the restatement is empty",
                "jargon": f"the restatement names {', '.join(detail or [])} — say the "
                          f"THING the identifier refers to, not the identifier",
                "link": "the restatement carries a link — it must be readable where it "
                        "stands, with nothing to click",
            }[kind])
    return p


def append_to(ledger, rendered):
    """Insert after the last entry, before any generated block.

    Uses the same body-boundary helpers as the checker, so "where does the last
    entry end" has one answer rather than two that agree until they do not.
    """
    text = ledger.read_text(encoding="utf-8")
    lines = text.split("\n")
    entries = parse_entries(text)
    if not entries:
        return None, f"parsed 0 entries from {ledger} — refusing to guess where to append"
    bodies = entry_bodies(lines, entries)
    e, start, body = bodies[-1]
    at = start + body_tail(body)
    out = lines[:at] + ["", rendered] + lines[at:]
    return "\n".join(out), None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("entry_json", nargs="?", help="the entry, as JSON ('-' for stdin)")
    ap.add_argument("--template", action="store_true",
                    help="print the canonical skeleton and status vocabulary, and exit")
    ap.add_argument("--ledger", default="docs/DECISIONS.md")
    ap.add_argument("--append", action="store_true", help="append it to the ledger")
    ap.add_argument("--dry-run", action="store_true", help="print, write nothing")
    args = ap.parse_args()

    if args.template:
        print(entry_template())
        print()
        print("Status vocabulary — use these exact words so entries stay machine-readable:")
        print("  " + status_vocab_line().replace("`", ""))
        return 0

    if not args.entry_json:
        ap.error("give an entry JSON file, or --template")
    try:
        raw = sys.stdin.read() if args.entry_json == "-" \
            else Path(args.entry_json).read_text(encoding="utf-8")
        spec_in = json.loads(raw)
    except Exception as exc:
        print(f"SETUP: cannot read the entry — {exc}")
        return 2

    problems = validate(spec_in)
    if problems:
        print(f"REFUSED — {len(problems)} problem(s), nothing written:")
        for p in problems:
            print(f"  ✗ {p}")
        return 1

    rendered = emit_entry(spec_in["id"], spec_in["title"], spec_in.get("fields") or {},
                          spec_in.get("body", ""), spec_in.get("tldr", ""),
                          registry=bool(spec_in.get("registry")))

    if not args.append:
        print(rendered)
        return 0

    ledger = Path(args.ledger)
    if not ledger.is_file():
        print(f"SETUP: no ledger at {ledger}")
        return 2
    if re.search(rf"^#{{2,4}}\s+\[?{re.escape(spec_in['id'])}\b", ledger.read_text(encoding="utf-8"),
                 re.M):
        print(f"REFUSED: {spec_in['id']} already exists in {ledger}")
        return 1
    new_text, err = append_to(ledger, rendered)
    if err:
        print(f"REFUSED: {err}")
        return 1
    if args.dry_run:
        print(rendered)
        print(f"\ndry run — would append the above to {ledger}")
        return 0
    ledger.write_text(new_text, encoding="utf-8")
    print(f"appended {spec_in['id']} to {ledger}")
    print("⚠️ regenerate the index and re-lint:")
    print(f"   python3 scripts/gen-dec-index.py --ledger {ledger}")
    print(f"   python3 scripts/ledger-lint.py --ledger {ledger}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
