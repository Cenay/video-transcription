#!/usr/bin/env python3
"""
link-doc-refs.py — turn DEC / G / M / D reference IDs in project docs into
reference-style deep links to their heading in the ledger files.

WHY heading slugs (not <a id> anchors): VS Code's markdown preview only navigates
to fragments it can match against a real `#` heading's generated slug. It does NOT
honour `<a id="...">` anchors — clicking one yields "Cannot find header to navigate
to". So links must target the actual heading slug. GitHub uses a compatible slug,
so the same link works in both.

WHY reference-style: inline `[DEC-111](DECISIONS.md#dec-111-really-long-slug)` makes
dense prose unreadable in source. Instead we write shortcut references — `[DEC-111]`
in the text — and collect every URL in one auto-generated block at the bottom of the
file. The block is regenerated on every run, so if a heading is reworded (its slug
changes) the links self-heal; the clean inline `[DEC-111]` never has to change.

What it does:
  - Builds an ID -> (ledger-file, heading-slug) map by slugifying the ledger headings
    (DEC/M/D) exactly the way VS Code / GitHub do. G items are bullets, not headings,
    so each G maps to the slug of the section heading it lives under.
  - In every *narrative* doc, brackets resolvable bare IDs into shortcut references
    and (re)writes the managed link-definition block. IDs with no ledger heading are
    left EXACTLY AS WRITTEN and reported as unresolved — never rewritten.
  - DECISIONS.md is BOTH a target and self-linked (see below). The frozen historical
    records (intake/, discovery/, archive/) are targets only — never rewritten.

SELF-LINKING THE LEDGER (added 2026-07-20). DECISIONS.md used to be target-only, on
the assumption that a chronological ledger's entries rarely point at each other. That
stopped being true when it was restructured into a flat numeric list of self-contained
entries: cross-references became the ONLY way to express supersession, and the file
accumulated hundreds of them. Bracketed refs there had no definitions, so they rendered
as literal "[DEC-122]" text. It is now rewritten like any narrative doc, with three
rules that only matter for a self-linking file:

  1. HEADING LINES ARE NEVER REWRITTEN. This is not cosmetic — bracketing an ID in
     "## DEC-122 One migrations folder…" would change the heading's own slug, breaking
     every link that targets it, and would change again on each run. Headings are
     passed through verbatim in every doc; it is merely harmless elsewhere and
     essential here.
  2. NO SELF-SECTION LINKS. Inside DEC-122's own section, a "DEC-122" mention is left
     as plain text — a link that jumps to the heading you are already reading is noise.
  3. SAME-FILE LINKS USE A BARE FRAGMENT ("#dec-122-…", not "DECISIONS.md#dec-122-…"),
     which is what both VS Code preview and GitHub want for an intra-document anchor.

Usage:
    python3 link-doc-refs.py <docs-dir>            # apply in place
    python3 link-doc-refs.py <docs-dir> --dry-run  # report only, write nothing
    python3 link-doc-refs.py <docs-dir> --quiet    # apply, summary only
"""

import os
import re
import sys

# Reference forms as they appear in prose. G has no dash ("G75").
ID_RE = re.compile(r"\b(DEC-\d+|M-\d+|D-\d+|G\d+)\b")
# A range: <id><sep><number>, e.g. DEC-110-114 / DEC-110–114 / G1–G8.
RANGE_RE = re.compile(r"\b(DEC-\d+|G\d+)\s*([-–—])\s*(G?\d+)\b")
HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")

BLOCK_START = "<!-- link-doc-refs:start (auto-generated — edit the IDs in prose, not this block) -->"
BLOCK_END = "<!-- link-doc-refs:end -->"
BLOCK_RE = re.compile(
    r"\n*" + re.escape(BLOCK_START) + r".*?" + re.escape(BLOCK_END) + r"\n*",
    re.DOTALL,
)

def rel_norm(relpath: str) -> str:
    """Normalize a relative path to forward slashes for stable comparisons."""
    return relpath.replace(os.sep, "/")


def slugify(text: str) -> str:
    """Slugify heading text exactly the way GitHub (github-slugger) and VS Code's
    markdown preview do, so a `#slug` link resolves in both:

      1. lowercase
      2. remove every char that is NOT a word char, whitespace, or hyphen
         (drops punctuation, backticks, colons, em dashes, emoji, …)
      3. replace each whitespace char with a single hyphen — NOT collapsing

    Step 3 is why we must not collapse: when punctuation removed in step 2 sat
    between two spaces (e.g. "layouts + slots" → "layouts  slots"), the renderers
    emit a DOUBLE hyphen ("layouts--slots"). Collapsing here would produce a slug
    that never matches the rendered anchor."""
    s = text.strip().lower()
    s = re.sub(r"[^\w\s-]", "", s, flags=re.UNICODE)
    s = re.sub(r"\s", "-", s)
    return s


def file_heading_slugs(lines: list) -> list:
    """Return [(line_index, level, text, unique_slug)] for every ATX heading,
    de-duplicating slugs in document order exactly like the renderers (-1, -2…)."""
    seen = {}
    out = []
    in_fence = False
    for i, line in enumerate(lines):
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        m = HEADING_RE.match(line)
        if not m:
            continue
        level = len(m.group(1))
        text = m.group(2)
        base = slugify(text)
        n = seen.get(base, 0)
        slug = base if n == 0 else f"{base}-{n}"
        seen[base] = n + 1
        out.append((i, level, text, slug))
    return out


def _section_slug_at(lines, headings):
    """Yield (line_index, current_section_slug) for every non-heading line."""
    slug_at = {h[0]: h[3] for h in headings}
    cur = None
    for i, line in enumerate(lines):
        if i in slug_at:
            cur = slug_at[i]
            continue
        yield i, line, cur


def build_id_map(docs_dir, ledger_files):
    """Build {ID: (relpath, slug)} from the ledger / frozen record files ONLY
    (never narrative docs, which merely reference IDs). Precedence, highest first:
      1. an ID's own `### DEC/M/D-NNN` heading  (precise anchor)
      2. an open-decision `- [ ] DEC-NNN` checklist row  (→ its section heading)
      3. a `- **GNN**` / bold bullet definition  (→ its section heading)
    setdefault + this pass order means the most precise target wins."""
    id_map = {}
    checkbox_re = re.compile(r"^\s*[-*]\s+\[[ xX]\]\s+\**(DEC-\d+|M-\d+|D-\d+|G\d+)\b")
    bold_re = re.compile(r"^\s*[-*]\s+\*\*(DEC-\d+|M-\d+|D-\d+|G\d+)\b")

    for abspath, relpath in ledger_files.items():
        if not os.path.isfile(abspath):
            continue
        with open(abspath, encoding="utf-8") as fh:
            lines = fh.readlines()
        headings = file_heading_slugs(lines)

        # Pass 1 — precise: an ID that has its OWN heading. Prefer the canonical
        # "DEC-<n> <space>" heading over combo headings like "DEC-110/114 …".
        # G items are usually bullets (handled in pass 3), but a G that is important
        # enough to earn its own heading — G77, the URL-parity launch gate — must be
        # matched here too. Note G carries NO dash ("G77", not "G-77").
        for _, _, text, slug in headings:
            hm = re.match(r"^(DEC|M|D)-(\d+)(?=\s|$)", text)
            if hm:
                id_map.setdefault(f"{hm.group(1)}-{hm.group(2)}", (relpath, slug))
                continue
            gm = re.match(r"^G(\d+)(?=\s|$)", text)
            if gm:
                id_map.setdefault(f"G{gm.group(1)}", (relpath, slug))

        # Pass 2 — open-decision checklist rows → the section they live under
        # (e.g. "Build tasks carried forward"). Runs before bold bullets so an
        # open DEC maps to its task list, not an earlier meeting-note mention.
        for _, line, cur in _section_slug_at(lines, headings):
            m = checkbox_re.match(line)
            if m and cur:
                id_map.setdefault(m.group(1), (relpath, cur))

        # Pass 3 — bold bullet definitions (G items, meeting-note bullets) →
        # their section heading.
        for _, line, cur in _section_slug_at(lines, headings):
            m = bold_re.match(line)
            if m and cur:
                id_map.setdefault(m.group(1), (relpath, cur))
    return id_map


def rel_href(from_path, docs_dir, target_relpath, slug):
    target_abs = os.path.join(docs_dir, target_relpath)
    # Same-file (the ledger linking to itself): emit a bare fragment. A relative
    # path to yourself ("DECISIONS.md#slug") resolves in some renderers and not
    # others; "#slug" is unambiguous everywhere.
    if os.path.abspath(target_abs) == os.path.abspath(from_path):
        return f"#{slug}"
    rel = os.path.relpath(target_abs, os.path.dirname(from_path))
    return f"{rel}#{slug}"


# Counts existing link definitions, for the shrink guard in rewrite_doc.
DEFN_LINE_RE = re.compile(r"^\[[^\]]+\]:\s*\S+", re.M)


def rewrite_doc(abspath, docs_dir, id_map, dry, self_relpath=None):
    """Bracket resolvable IDs and regenerate the definition block.

    `self_relpath` is this doc's own relpath when it is ALSO a link target (i.e.
    DECISIONS.md). It enables the no-self-section-link rule; leave it None for
    ordinary narrative docs."""
    with open(abspath, encoding="utf-8") as fh:
        before = fh.read()
    had_block = BLOCK_START in before

    # 1. Strip any existing managed block; we regenerate it from scratch.
    text = BLOCK_RE.sub("\n", before).rstrip("\n") + "\n"

    def resolvable(rid):
        return rid in id_map

    # 2. An ID this ledger cannot resolve is REPORTED, never rewritten.
    #
    # This step used to "self-heal" by stripping the brackets off any ID absent
    # from the local ledger, on the theory that it was dangling. ⛔ That theory
    # is wrong whenever a repo cites ANOTHER repo's decisions, and measured
    # 2026-08-19 in this toolkit it was wrong every single time: it rewrote
    # [DEC-150] and [DEC-157] in CURRENT_STATUS.md, [DEC-169] and [DEC-089] in
    # TODOS.md, and one more in NEXT_STEPS.md -- five real fran-dash decisions
    # turned into ordinary prose. Zero were typos.
    #
    # It is not cosmetic, which is how the old TODO described it: the brackets
    # are the ONLY signal that a token is a ledger reference at all, so removing
    # them makes a decision read as a bare string and silently breaks the search
    # that would find it.
    #
    # ⚠️ Sibling-ledger lookup is deliberately NOT the fix. "Shares a DEC- number
    # series with" (allocation) and "may cite decisions from" (reference) are
    # different relations: this toolkit is `standalone` for numbering and still
    # cites fran-dash constantly. Conflating them would re-break exactly this.
    #
    # Nothing is left dangling by leaving the text alone. Step 1 strips and
    # regenerates the whole managed block, so an unresolvable ID simply gets no
    # definition and Markdown renders `[DEC-999]` as literal text -- visible
    # brackets, not a broken link. The unwrap was never needed for correctness.
    # A genuine typo surfaces in the `unresolved` report at the end of the run,
    # which is a person's call to make, not a silent rewrite of their prose.

    # 3. Bracket bare, resolvable IDs line by line (protecting code + existing links).
    src_lines = text.splitlines(keepends=True)
    # Map line index -> the unique slug of the heading ON that line, so we can track
    # which section each line belongs to (needed for the no-self-section-link rule).
    slug_on_line = {h[0]: h[3] for h in file_heading_slugs(src_lines)}
    cur_section = None

    out_lines = []
    in_fence = False
    for idx, line in enumerate(src_lines):
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            out_lines.append(line)
            continue
        if in_fence:
            out_lines.append(line)
            continue
        # HEADINGS PASS THROUGH UNTOUCHED. Rewriting one would change its own slug
        # and break every link aimed at it (and re-break it on the next run).
        if idx in slug_on_line:
            cur_section = slug_on_line[idx]
            out_lines.append(line)
            continue

        protected = []

        def stash(m):
            protected.append(m.group(0))
            return f"\x00{len(protected) - 1}\x00"

        s = line
        # Unwrap any PRE-EXISTING shortcut that is a self-section reference, so a
        # hand-written "[DEC-122]" inside DEC-122's own entry degrades to plain text
        # instead of surviving into the definition block as a link to itself.
        if self_relpath is not None:
            def unwrap_self(m):
                rid = m.group(1)
                if rid in id_map:
                    t_rel, t_slug = id_map[rid]
                    if t_rel == self_relpath and t_slug == cur_section:
                        return rid
                return m.group(0)

            s = re.sub(r"\[(DEC-\d+|M-\d+|D-\d+|G\d+)\](?![(\[:])", unwrap_self, s)
        # Inline code FIRST — otherwise a bracket inside a code span (e.g.
        # `- [ ] DEC-NNN`) gets stashed as a "shortcut ref", then the code stash
        # wraps that placeholder, and the two nest and corrupt on restore.
        s = re.sub(r"`[^`]*`", stash, s)                # inline code
        s = re.sub(r"\[[^\]]*\]\([^)]*\)", stash, s)   # inline links
        s = re.sub(r"\[[^\]]*\]\[[^\]]*\]", stash, s)   # full reference links
        s = re.sub(r"\[[^\]]*\]", stash, s)             # shortcut refs / def-ish

        def range_sub(m):
            head, sep, tail = m.group(1), m.group(2), m.group(3)
            prefix = re.match(r"(DEC-|G)", head).group(1)
            tail_id = tail if tail.startswith(("DEC-", "G")) else (
                f"{prefix}{tail}" if prefix == "G" else f"DEC-{tail}")
            h_ok, t_ok = resolvable(head), resolvable(tail_id)
            if not h_ok and not t_ok:
                return m.group(0)
            h = f"[{head}]" if h_ok else head
            # namespaced label so a bare "[114]" can't collide with prose.
            t = f"[{tail}][{tail_id.lower()}]" if t_ok else tail
            return f"{h}{sep}{t}"

        s = RANGE_RE.sub(range_sub, s)
        # protect links the range pass just minted
        s = re.sub(r"\[[^\]]*\]\[[^\]]*\]", stash, s)
        s = re.sub(r"\[[^\]]*\]", stash, s)

        def id_sub(m):
            rid = m.group(1)
            if not resolvable(rid):
                return rid
            # No self-section links: inside DEC-122's own section, "DEC-122" stays
            # plain text rather than becoming a link to the heading above it.
            if self_relpath is not None:
                target_rel, target_slug = id_map[rid]
                if target_rel == self_relpath and target_slug == cur_section:
                    return rid
            return f"[{rid}]"

        s = ID_RE.sub(id_sub, s)

        def unstash(m):
            return protected[int(m.group(1))]

        # Iterate: a restored span may itself contain a placeholder. Bounded by
        # the number of stashed spans, so it always terminates.
        for _ in range(len(protected) + 1):
            if "\x00" not in s:
                break
            s = re.sub(r"\x00(\d+)\x00", unstash, s)
        out_lines.append(s)

    text = "".join(out_lines)

    # Collect every reference label actually present in the final text, so the
    # definition block always matches inline usage. Scanning (rather than noting
    # during substitution) is what makes re-runs idempotent: pre-existing
    # `[DEC-111]` shortcuts are counted too.
    used = {}
    for m in re.finditer(r"\[(DEC-\d+|M-\d+|D-\d+|G\d+)\](?![(\[:])", text):
        rid = m.group(1)
        if rid in id_map:
            used[rid] = id_map[rid]
    for m in re.finditer(r"\]\[dec-(\d+)\]", text):
        rid = f"DEC-{m.group(1)}"
        if rid in id_map:
            used[f"dec-{m.group(1)}"] = id_map[rid]

    # 4. Append the regenerated definition block (deduped by lowercased label).
    if used:
        seen = {}
        for label, (relpath, slug) in used.items():
            key = label.lower()
            if key in seen:
                continue
            seen[key] = (label, relpath, slug)
        defs = []
        for label, relpath, slug in sorted(seen.values(), key=lambda x: x[0].lower()):
            href = rel_href(abspath, docs_dir, relpath, slug)
            defs.append(f"[{label}]: {href}")
        block = "\n" + BLOCK_START + "\n" + "\n".join(defs) + "\n" + BLOCK_END + "\n"
        text = text.rstrip("\n") + "\n" + block

    # Don't touch a file that has no links to add and never had a managed block —
    # avoids spurious trailing-whitespace-only diffs.
    if not used and not had_block:
        return False, 0

    # ⛔ GUARD 2 — REFUSE TO SHRINK A MANAGED BLOCK.
    #
    # The general form of Guard 1, and the one that catches causes not yet
    # invented: whatever the reason, a regenerated block holding FEWER
    # definitions than the one it replaces is destroying references that a
    # previous run resolved. That is a defect until a human says otherwise.
    # Same shape as the append-only guard on docs/history/ ([DEC-260]).
    #
    # ALLOW_LINK_BLOCK_SHRINK=1 is the loud, deliberate override — correct when
    # IDs have genuinely been removed from a doc's prose.
    n_before = len(DEFN_LINE_RE.findall(before))
    n_after = len(DEFN_LINE_RE.findall(text))
    if n_after < n_before and not os.environ.get("ALLOW_LINK_BLOCK_SHRINK"):
        print(
            f"  ⛔ REFUSED {os.path.relpath(abspath, docs_dir)}: managed block would "
            f"shrink {n_before} → {n_after} definitions. Nothing written.\n"
            f"     If the IDs really were removed from the prose, re-run with "
            f"ALLOW_LINK_BLOCK_SHRINK=1.",
            file=sys.stderr,
        )
        return False, -1

    changed = text != before
    if changed and not dry:
        with open(abspath, "w", encoding="utf-8") as fh:
            fh.write(text)
    return changed, len(used)


def iter_docs(docs_dir):
    for root, _, files in os.walk(docs_dir):
        for f in files:
            if f.endswith(".md"):
                yield os.path.join(root, f)


def unresolved_in(docs_dir, ledger_paths, id_map):
    """Report referenced IDs that have no ledger heading (left untouched)."""
    missing = {}
    for doc in iter_docs(docs_dir):
        if doc in ledger_paths:
            continue
        with open(doc, encoding="utf-8") as fh:
            for rid in ID_RE.findall(fh.read()):
                if rid not in id_map:
                    missing.setdefault(rid, 0)
                    missing[rid] += 1
    return missing


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    flags = {a for a in sys.argv[1:] if a.startswith("--")}
    if not args:
        print(__doc__)
        sys.exit(1)
    docs_dir = os.path.abspath(args[0])
    dry = "--dry-run" in flags
    quiet = "--quiet" in flags
    if not os.path.isdir(docs_dir):
        print(f"error: not a directory: {docs_dir}")
        sys.exit(1)

    all_md = sorted(iter_docs(docs_dir))
    relpaths = {p: rel_norm(os.path.relpath(p, docs_dir)) for p in all_md}

    # Two different roles, and they are NOT the same set.
    #
    # TARGETS hold the ID definitions: the ledger (DECISIONS.md) plus the frozen
    # historical records (intake/, discovery/, archive/). Narrative docs merely
    # reference IDs, so they never contribute definitions.
    #
    # FROZEN are target-only and never rewritten — the archive must stay verbatim
    # (matches the wrap-up guardrail). DECISIONS.md is deliberately NOT frozen: it
    # is a target that also gets self-linked, because its entries cross-reference
    # each other constantly since the flat-numeric restructure.
    LEDGER = "DECISIONS.md"

    def is_frozen(relpath):
        return rel_norm(relpath).split("/")[0] in ("intake", "discovery", "archive")

    def is_target(relpath):
        return rel_norm(relpath) == LEDGER or is_frozen(relpath)

    frozen_paths = {p for p, rel in relpaths.items() if is_frozen(rel)}
    target_files = {p: relpaths[p] for p, rel in relpaths.items() if is_target(rel)}
    id_map = build_id_map(docs_dir, target_files)

    # ⛔ GUARD 1 — AN EMPTY MAP IS A MISAIMED RUN, NOT AN EMPTY LEDGER.
    #
    # This script regenerates each doc's managed link block FROM the map. With an
    # empty map every block regenerates to nothing, so a run that cannot see the
    # ledger deletes every link definition in the tree it was pointed at.
    #
    # ✅ Reproduced 2026-08-25 on a throwaway copy of fran-dash/docs:
    #   `link-doc-refs.py docs/history` -> "ids resolvable: 0 ... across 0 ledger
    #   file(s)", then "linked: 12 narrative doc(s)" -- the word LINKED over an
    #   operation that removed every link. DECISIONS-stamp-history.md went
    #   173 -> 72 lines and 98 link definitions -> 0. Twelve of thirteen files
    #   damaged, exit 0, no warning.
    #
    # ★ The tool already KNEW: it printed `0` and carried on. Treating "I found
    #   nothing" as "there is nothing" is the same defect that made
    #   stamp-doc.py --restore print "nothing to restore" while deleting 367
    #   lines. Finding nothing is a reason to STOP, not a licence to write.
    if not id_map:
        sys.exit(
            f"error: no DEC/G/M/D headings found under {docs_dir!r} — refusing to run.\n"
            f"       Every managed link block would regenerate to EMPTY, deleting\n"
            f"       every link definition in that tree.\n"
            f"       Most likely you aimed this at a SUBDIRECTORY: the ledgers\n"
            f"       (DECISIONS.md and friends) live in the docs root, and cannot be\n"
            f"       seen from inside docs/history/. Point it at the docs root."
        )

    changed_docs = []
    for doc in all_md:
        if doc in frozen_paths:
            continue
        # Self-link mode for the ledger: suppresses links from an entry to itself.
        self_rel = relpaths[doc] if relpaths[doc] == LEDGER else None
        changed, n = rewrite_doc(doc, docs_dir, id_map, dry, self_relpath=self_rel)
        if changed:
            changed_docs.append((os.path.relpath(doc, docs_dir), n))

    # Unresolved reporting stays scoped to docs that merely *reference* IDs. The
    # ledger and the frozen records are excluded: their prose is full of historical
    # G-item mentions that were closed as bullets and never given headings, so
    # including them buries the real signal under ~70 permanent false positives.
    missing = unresolved_in(docs_dir, set(target_files), id_map)

    tag = "[dry-run] would link" if dry else "linked"
    print(f"ids resolvable: {len(id_map)} (DEC/G/M/D across "
          f"{len(target_files)} ledger/record file(s))")
    print(f"{tag}: {len(changed_docs)} narrative doc(s)")
    if not quiet:
        for rel, n in changed_docs:
            print(f"  - {rel}: {n} reference(s)")
    if missing:
        items = ", ".join(f"{k}×{v}" for k, v in sorted(missing.items()))
        print(f"unresolved in THIS ledger — left untouched, NOT rewritten "
              f"(they may live in another repo's ledger; check before calling one a typo): {items}")


if __name__ == "__main__":
    main()
