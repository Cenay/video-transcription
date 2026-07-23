#!/usr/bin/env python3
"""Find stale decision references across a project's docs.

Reads a decision ledger (default docs/DECISIONS.md), builds an ID -> status map,
then scans every OTHER markdown file for mentions of those IDs and reports any
whose surrounding sentence disagrees with the ledger.

Reports only. Never edits a file. Exit 0 = clean, 1 = findings, 2 = setup error.

Usage:
    python3 check-doc-refs.py                     # auto-detect ledger + docs root
    python3 check-doc-refs.py --root .cloaked/docs
    python3 check-doc-refs.py --ledger docs/DECISIONS.md --also CLAUDE.md README.md
    python3 check-doc-refs.py --json              # machine-readable
"""

import argparse
import json
import os
import re
import sys

# ---------------------------------------------------------------- ID grammar

# DEC-111, G75, M-100, D-015, BUG-007, SUSP-003
ID_RE = re.compile(r'\b((?:DEC|BUG|SUSP|LES|ADR)-\d+|[GMD]-?\d+)\b')

# A ledger heading: "## DEC-027 Catalog: mirror Bookeo ..." or "### [DEC-027] ..."
HEADING_RE = re.compile(r'^#{2,4}\s+\[?((?:DEC|BUG|SUSP|LES|ADR)-\d+|[GMD]-?\d+)\]?\b(.*)$')
STATUS_RE = re.compile(r'^\s*\**Status:?\**\s*(.+)$', re.IGNORECASE)

FENCE_RE = re.compile(r'^\s*(```|~~~)')

# Sentence-ish split. Cenay's docs keep each paragraph on ONE physical line,
# so line-then-sentence gives tight, quotable units.
SENT_SPLIT_RE = re.compile(r'(?<=[.!?])\s+')

# ------------------------------------------------------------- status phrases

OPEN_PAT = re.compile(
    r'\b('
    r'still open|still (?:un)?resolved|unresolved|live and unresolved'
    r'|never (?:actually )?(?:been |yet )?(?:asked|spoken|put to|raised|answered|discussed)'
    r'|not (?:yet )?(?:been )?(?:asked|closed|resolved|answered|decided|named|settled)'
    r'|remains? open|stays? open|is open|are open'
    r'|outstanding|unanswered|awaiting|pending'
    r'|open question|open item|to be (?:decided|determined|named)|TBD'
    r'|blocked on|gated on|waiting on'
    r'|has(?:n.t| not) been (?:asked|answered|closed|decided)'
    r')\b',
    re.IGNORECASE,
)

CLOSED_PAT = re.compile(
    r'\b('
    r'closed|resolved|settled|superseded|retired|ratified'
    r'|(?:was|were|is|are|now) (?:decided|answered|agreed|ruled)'
    r'|no longer (?:open|outstanding|a question)'
    r'|✅'
    r')\b',
    re.IGNORECASE,
)

# Bare "§N" citations — the other rot class: section numbers go stale silently.
BARE_SECTION_RE = re.compile(r'§\s*\d+')

# Narration ABOUT a past state is not staleness. "DEC-109 had gone unasked for four
# meetings" is correct history; "DEC-109 is still unasked" is rot. Without this guard
# every lessons/status entry ever written trips the open/closed patterns.
PAST_TENSE_RE = re.compile(
    r'\b('
    r'had (?:gone|been|not)|was (?:recorded|described|framed|believed|thought|left)'
    r'|were (?:recorded|described|carried|left)'
    r'|at the time|used to|originally|previously|formerly|back then'
    r'|no longer|since (?:closed|resolved|settled|superseded)|turned out'
    r'|this (?:used to|once)|has since|have since'
    r')\b',
    re.IGNORECASE,
)

# Docs whose JOB is to record history — findings here are almost always legitimate
# narration, so they are reported at low severity rather than suppressed.
HISTORICAL_FILE_RE = re.compile(
    r'(LESSONS_LEARNED|CHANGELOG|-archive|/history/|/intake/|/discovery/)',
    re.IGNORECASE,
)

CLOSED_STATES = {'closed', 'superseded', 'rejected'}
OPEN_STATES = {'open', 'deferred'}

DEFAULT_EXCLUDE_DIRS = {'history', 'archive', '.archived', 'node_modules', '.git', 'vendor'}


def classify_status(raw):
    """Map a ledger **Status:** line to a coarse state."""
    s = raw.upper()
    if 'SUPERSEDED' in s:
        return 'superseded'
    if 'DEFERRED' in s:
        return 'deferred'
    if 'CLOSED' in s or 'RESOLVED' in s or 'DONE' in s:
        return 'closed'
    if 'REJECTED' in s:
        return 'rejected'
    if 'OPEN' in s or 'WIP' in s or 'DRAFT' in s:
        return 'open'
    return 'unknown'


def strip_fences(lines):
    """Yield (lineno, text) for lines outside fenced code blocks."""
    in_fence = False
    for i, line in enumerate(lines, 1):
        if FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        if not in_fence:
            yield i, line


def parse_ledger(path):
    """Return {id: {'status':..., 'state':..., 'title':..., 'line':...}}."""
    with open(path, encoding='utf-8') as fh:
        lines = fh.readlines()

    entries = {}
    kept = list(strip_fences(lines))
    for idx, (lineno, line) in enumerate(kept):
        m = HEADING_RE.match(line)
        if not m:
            continue
        dec_id, title = m.group(1), m.group(2).strip()
        status_raw = ''
        # Status usually sits on the next line or two.
        for _, follow in kept[idx + 1: idx + 4]:
            sm = STATUS_RE.match(follow)
            if sm:
                status_raw = sm.group(1).strip()
                break
        entries[dec_id] = {
            'status': status_raw,
            'state': classify_status(status_raw),
            'title': title,
            'line': lineno,
        }
    return entries


def find_docs(roots, ledger_path, extra_files, exclude_dirs):
    seen, out = set(), []
    for root in roots:
        if not os.path.isdir(root):
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in exclude_dirs]
            for fn in sorted(filenames):
                if not fn.endswith('.md'):
                    continue
                p = os.path.normpath(os.path.join(dirpath, fn))
                if p == os.path.normpath(ledger_path) or p in seen:
                    continue
                seen.add(p)
                out.append(p)
    for f in extra_files:
        p = os.path.normpath(f)
        if os.path.isfile(p) and p != os.path.normpath(ledger_path) and p not in seen:
            seen.add(p)
            out.append(p)
    return out


def scan_file(path, ledger, prefixes, report_unknown):
    """Return findings for one doc."""
    with open(path, encoding='utf-8') as fh:
        lines = fh.readlines()

    findings = []
    per_id = {}  # id -> set of 'open'/'closed' language seen in this file
    historical = bool(HISTORICAL_FILE_RE.search('/' + path.replace(os.sep, '/')))
    section_hits = []

    def emit(kind, severity, lineno, dec_id, quote, note, sentence=''):
        if sentence and PAST_TENSE_RE.search(sentence):
            severity = 'low' if severity == 'medium' else 'medium'
            note += ' ⓘ Past-tense narration detected — may be correct history, not rot.'
        if historical and severity == 'high':
            severity = 'low'
            note += ' ⓘ Historical doc — recording a past state here is expected.'
        findings.append({'kind': kind, 'severity': severity, 'file': path,
                         'line': lineno, 'id': dec_id, 'quote': quote, 'note': note})

    for lineno, line in strip_fences(lines):
        if BARE_SECTION_RE.search(line) and 'href' not in line:
            section_hits.append(lineno)

        if not ID_RE.search(line):
            continue

        for sentence in SENT_SPLIT_RE.split(line):
            ids = set(ID_RE.findall(sentence))
            if not ids:
                continue
            says_open = bool(OPEN_PAT.search(sentence))
            says_closed = bool(CLOSED_PAT.search(sentence))
            for dec_id in ids:
                if says_open:
                    per_id.setdefault(dec_id, set()).add('open')
                if says_closed:
                    per_id.setdefault(dec_id, set()).add('closed')

                entry = ledger.get(dec_id)
                if entry is None:
                    prefix = dec_id.split('-')[0] if '-' in dec_id else dec_id[0]
                    if report_unknown and prefix in prefixes:
                        emit('UNKNOWN_ID', 'low', lineno, dec_id,
                             sentence.strip()[:300],
                             f'{dec_id} is referenced but has no heading in the ledger.')
                    continue

                state = entry['state']
                status = entry['status'] or state.upper()
                if says_open and state in CLOSED_STATES:
                    emit('STALE_OPEN', 'high', lineno, dec_id, sentence.strip()[:300],
                         f'Doc treats {dec_id} as open; ledger says {status[:80]}.',
                         sentence)
                elif says_closed and state in OPEN_STATES:
                    emit('STALE_CLOSED', 'high', lineno, dec_id, sentence.strip()[:300],
                         f'Doc treats {dec_id} as settled; ledger says {status[:80]}.',
                         sentence)

    for dec_id, langs in sorted(per_id.items()):
        if len(langs) > 1:
            emit('SELF_CONTRADICTION', 'high', 0, dec_id, '',
                 f'{path} describes {dec_id} as BOTH open and closed in different '
                 f'places. One of them is stale.')

    if section_hits:
        shown = ', '.join(str(n) for n in section_hits[:12])
        more = f' (+{len(section_hits) - 12} more)' if len(section_hits) > 12 else ''
        emit('BARE_SECTION_CITATION', 'low', section_hits[0], None, '',
             f'{len(section_hits)} bare "§N" citation(s) — lines {shown}{more}. '
             f'Section numbers drift silently; cite path + section HEADING.')

    return findings


def check_doc_files(roots, docs):
    """Catch split backlogs and references to docs that don't exist."""
    findings = []

    # A repo must have exactly ONE backlog file. Two = a silently split backlog.
    for root in roots:
        singular = os.path.join(root, 'TODO.md')
        plural = os.path.join(root, 'TODOS.md')
        if os.path.isfile(singular) and os.path.isfile(plural):
            findings.append({
                'kind': 'SPLIT_BACKLOG', 'severity': 'high',
                'file': root, 'line': 0, 'id': None, 'quote': '',
                'note': f'BOTH {singular} and {plural} exist. Two backlog files both look '
                        f'authoritative and items get lost between them. Merge into one '
                        f'(canonical: TODOS.md) and delete the other.',
            })

    # References to sibling docs that aren't there.
    ref_re = re.compile(r'`([\w./-]*(?:docs|specs|plans)/[\w./-]+\.md)`')

    # SHALLOW ESCAPE HATCH for the cross-repo false positive — a doc that documents
    # ANOTHER repo's file layout (e.g. what `/bug` writes into target repos), or names a
    # file it plans to create. Two forms, both HTML comments so they render invisibly:
    #   <!-- doc-reconcile: ignore-missing docs/X.md docs/Y.md -->
    #        SCAN-GLOBAL allowlist. Declared in ANY scanned doc; those exact refs are
    #        never reported missing anywhere. Use for a path referenced across many docs.
    #   <!-- doc-reconcile: ignore -->
    #        LINE-SCOPED. Skips MISSING_DOC for refs on the line it sits on.
    # This is deliberately shallow; the deep fix is real cross-repo path awareness
    # (resolve against a repo map / skip known-repo prefixes) — tracked in the toolkit's
    # docs/TODOS.md under "doc-reconcile follow-ups". `ignore\s*-->` cannot match
    # `ignore-missing …` (the hyphen breaks it), so the two forms don't collide.
    ignore_missing_re = re.compile(r'<!--\s*doc-reconcile:\s*ignore-missing\s+([^>]*?)\s*-->')
    ignore_line_re = re.compile(r'<!--\s*doc-reconcile:\s*ignore\s*-->')

    # Pass 1: collect the scan-global allowlist declared anywhere.
    allow_missing = set()
    doc_lines = {}
    for doc in docs:
        with open(doc, encoding='utf-8') as fh:
            doc_lines[doc] = fh.readlines()
        for m in ignore_missing_re.finditer(''.join(doc_lines[doc])):
            for tok in m.group(1).split():
                allow_missing.add(os.path.normpath(tok))

    missing = {}
    for doc in docs:
        for lineno, line in strip_fences(doc_lines[doc]):
            if ignore_line_re.search(line):
                continue
            for ref in ref_re.findall(line):
                if os.path.normpath(ref) in allow_missing:
                    continue
                # Resolve against: cwd, the citing doc's folder, and the repo's PARENT
                # (sibling-repo citations like `migration/docs/x.md` are normal in a
                # monorepo-adjacent layout and must not be reported as missing).
                bases = ('.', os.path.dirname(doc), '..', '../..')
                if any(os.path.isfile(os.path.join(b, ref)) for b in bases):
                    continue
                missing.setdefault(ref, []).append((doc, lineno))

    for ref, hits in sorted(missing.items()):
        doc, lineno = hits[0]
        extra = f' (+{len(hits) - 1} more reference(s))' if len(hits) > 1 else ''
        findings.append({
            'kind': 'MISSING_DOC', 'severity': 'medium',
            'file': doc, 'line': lineno, 'id': None, 'quote': '',
            'note': f'References `{ref}`, which does not exist{extra}. '
                    f'Renamed, moved, or never written?',
        })

    return findings


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--ledger', help='Path to the decision ledger.')
    ap.add_argument('--root', action='append', default=[],
                    help='Docs root to scan (repeatable).')
    ap.add_argument('--also', nargs='*', default=[],
                    help='Extra files to scan (e.g. CLAUDE.md README.md).')
    ap.add_argument('--exclude', nargs='*', default=[],
                    help='Extra directory names to skip.')
    ap.add_argument('--unknown', action='store_true',
                    help='Also report IDs with no ledger heading (noisy).')
    ap.add_argument('--all', action='store_true',
                    help='Show low-severity findings too (default: high + medium).')
    ap.add_argument('--json', action='store_true', help='Machine-readable output.')
    args = ap.parse_args()

    ledger = args.ledger
    if not ledger:
        for cand in ('docs/DECISIONS.md', '.cloaked/docs/DECISIONS.md', 'DECISIONS.md'):
            if os.path.isfile(cand):
                ledger = cand
                break
    if not ledger or not os.path.isfile(ledger):
        print('SETUP: no decision ledger found. Pass --ledger <path>.', file=sys.stderr)
        return 2

    roots = args.root or [d for d in ('docs', '.cloaked/docs', 'specs', 'plans')
                          if os.path.isdir(d)]
    extra = args.also or [f for f in ('CLAUDE.md', 'README.md') if os.path.isfile(f)]
    exclude = DEFAULT_EXCLUDE_DIRS | set(args.exclude)

    entries = parse_ledger(ledger)
    if not entries:
        print(f'SETUP: parsed 0 entries from {ledger}. Check its heading format.',
              file=sys.stderr)
        return 2

    prefixes = {k.split('-')[0] if '-' in k else k[0] for k in entries}

    docs = find_docs(roots, ledger, extra, exclude)
    findings = []
    for d in docs:
        findings.extend(scan_file(d, entries, prefixes, args.unknown))
    findings.extend(check_doc_files(roots, docs + [ledger]))

    suppressed = 0
    if not args.all and not args.json:
        before = len(findings)
        findings = [f for f in findings if f['severity'] != 'low']
        suppressed = before - len(findings)

    order = {'high': 0, 'medium': 1, 'low': 2}
    findings.sort(key=lambda f: (order.get(f['severity'], 9), f['file'], f['line']))

    if args.json:
        print(json.dumps({
            'ledger': ledger,
            'entries': len(entries),
            'scanned': len(docs),
            'findings': findings,
        }, indent=2))
        return 1 if findings else 0

    print(f'Ledger : {ledger}  ({len(entries)} entries)')
    print(f'Scanned: {len(docs)} docs under {", ".join(roots)}'
          + (f' + {", ".join(extra)}' if extra else ''))
    print()

    if not findings:
        print('CLEAN — no reference disagreements found.'
              + (f'  ({suppressed} low-severity hidden; --all to see)' if suppressed else ''))
        return 0

    counts = {}
    for f in findings:
        counts[f['kind']] = counts.get(f['kind'], 0) + 1
    print('FINDINGS: ' + ', '.join(f'{k}={v}' for k, v in sorted(counts.items()))
          + (f'   ({suppressed} low-severity hidden; --all to see)' if suppressed else ''))
    print()

    for f in findings:
        loc = f"{f['file']}:{f['line']}" if f['line'] else f['file']
        print(f"[{f['severity'].upper()}] {f['kind']}  {loc}")
        print(f"  {f['note']}")
        if f['quote']:
            print(f"  > {f['quote']}")
        print()

    print('Reported only — nothing was edited. These are CANDIDATES, not verdicts:')
    print('read each quote in context and confirm it is rot before proposing a fix.')
    return 1


if __name__ == '__main__':
    sys.exit(main())
