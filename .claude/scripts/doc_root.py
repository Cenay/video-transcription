#!/usr/bin/env python3
"""Resolve a repo's documentation root. THE single source of that rule.

WHY THIS EXISTS AS ONE FILE.

The routing rule ("docs/ under /mnt/k/Code, .cloaked/docs/ under /mnt/k/_Sites")
was already written down twice before this — in scripts/ensure-doc-folders.sh
and in prose in commands/init-project.md — and the session-desk skill had a
THIRD answer: the hardcoded literal `docs/sessions/`, which is simply wrong in
a _Sites repo. Ruled 2026-08-19 (Session Desk Q1): one resolver, read by
everything, never re-implemented. Two copies of a routing rule is how one
evening produced three scripts giving three different answers about the same
ledger.

THE RULE, in two tiers. EVIDENCE FIRST, then the path.

  1. If a candidate root already HOLDS the doc set, that is the answer.
  2. Otherwise, and only then:
       /mnt/k/_Sites/<repo>   ->  .cloaked/docs   (or cloaked/docs, see below)
       anything else          ->  docs

_Sites/ was the original WordPress codebase; `.cloaked/` is a naming convention
adopted to keep project-management files OUT of the tree WordPress reads, and
gitignored. That history makes the path a reasonable DEFAULT for a repo with no
doc set yet — it is not evidence about a repo that already has one.

WHY TIER 1 EXISTS — a defect this file caused, 2026-08-19 19:04 MST.

The rule was originally path-only, and explicitly rejected probing. That
rejection was right about the probe it considered — "docs/ if it exists, else
.cloaked/docs/" — because EVERY project has a `.cloaked/`, so mere presence
decides nothing. But "which folder exists" and "where the doc set actually
lives" are different questions, and only the first was rejected.

Path-only sent `trfaapi.com`'s Session Desk to `.cloaked/docs/`, which did not
exist, so the migration CREATED it — a second, empty docs root beside a real
one holding 22 tracked files (CURRENT_STATUS, DECISIONS, NEXT_STEPS, TODOS,
LESSONS_LEARNED, guides/, history/, intake/). The desk had been correctly
beside its own doc set and was moved away from it. That is precisely the
"second, empty docs root" failure already filed against ensure-doc-folders.sh;
this file reproduced it by another route.

MEASURED before changing anything, across every git repo on the machine:
NOT ONE has a doc set under `.cloaked/docs/` — zero instances. Exactly one
_Sites repo has a doc set at all (`trfaapi.com`) and it is at `docs/`. So the
path rule had no supporting instance and one contradicting instance; it was
written from the documented convention rather than from the repos.

Tier 1 is deliberately CONSERVATIVE: it only fires where a doc set is already
present, so a fresh WordPress site with no docs still routes to `.cloaked/docs`
exactly as the convention says. It changes the answer only where the old answer
was demonstrably wrong.

Cenay, on finding the desk in the wrong tree: "this is Laravel, not WP and it
HAS a /docs/ folder off the root. That IS where I expected this to move to."
This amends the 2026-08-19 Q1 ruling ("path-based, not folder-probing") on her
correction; the ruling's INTENT — one resolver, read by everything, never a
naive existence probe — is unchanged.

THE TWO SPELLINGS. Verified 2026-08-19 by `ls -d /mnt/k/_Sites/*/cloaked`:
50 repos use `.cloaked/` and 6 use a bare `cloaked/` (cenaynailor.com,
geek2video.com, santbaniashram.org, softwarethatrocks.com, truckingagents.net,
wpgeek.dev). ensure-doc-folders.sh assumed the dotted spelling for all of them,
so in those 6 it would silently mkdir a SECOND, empty docs root beside the real
one and report success. Hence: prefer `.cloaked/` when both exist, otherwise
use whichever is actually there, otherwise default to `.cloaked/`.

The 6 are expected to be renamed to one spelling eventually — blocked on moving
the FTP setup to the PPK, which is path-driven. See docs/TODOS.md. When that
lands, `cloaked` can be dropped from CLOAKED_NAMES and this becomes exact.

USAGE

    # as a library
    from doc_root import doc_root
    doc_root(Path("/mnt/k/Code/foo"))            -> "docs"

    # as a CLI, for shell callers such as ensure-doc-folders.sh
    DOCS=$(python3 scripts/doc_root.py /mnt/k/_Sites/bar)
"""

import sys
from pathlib import Path

# Client sites live here and route their docs into a cloaked folder.
SITES_ROOT = "/mnt/k/_Sites"

# Preference order matters: `.cloaked` wins when a repo somehow has both.
CLOAKED_NAMES = (".cloaked", "cloaked")


def is_site_repo(repo, sites_root=SITES_ROOT):
    """True when `repo` sits directly under the client-sites root.

    Compared as resolved paths so `..` segments and a trailing slash cannot
    change the answer. A repo IS the sites root itself does not count.
    """
    repo = Path(repo).resolve()
    root = Path(sites_root).resolve()
    return root in repo.parents


def cloaked_name(repo, sites_root=SITES_ROOT):
    """Which cloaked spelling this repo uses. Returns None for non-site repos."""
    if not is_site_repo(repo, sites_root):
        return None
    repo = Path(repo)
    for name in CLOAKED_NAMES:
        if (repo / name).is_dir():
            return name
    # Neither present: default to the dotted spelling, which 50 of 56 use.
    return CLOAKED_NAMES[0]


# The standard doc set. A directory holding any of these IS a docs root -- this
# is evidence about the repo, not a guess about which folder happens to exist.
# Kept in step with the doc-set contract in the doc-reconcile skill.
DOC_SET_MARKERS = ("CURRENT_STATUS.md", "DECISIONS.md", "NEXT_STEPS.md",
                   "TODOS.md", "LESSONS_LEARNED.md")


def _marker_count(path):
    try:
        return sum(1 for m in DOC_SET_MARKERS if (path / m).is_file())
    except OSError:
        return 0


def existing_doc_root(repo, sites_root=SITES_ROOT):
    """The docs root this repo ALREADY uses, or None if it has no doc set.

    Ranks candidates by how much of the doc set each holds, so a repo that has
    started a second root somewhere still resolves to the real one. Ties break
    toward the path rule's answer, which keeps the convention authoritative
    whenever the evidence does not actually distinguish the candidates.
    """
    repo = Path(repo)
    cloaked = cloaked_name(repo, sites_root)
    candidates = ["docs"] + ([f"{c}/docs" for c in CLOAKED_NAMES]
                             if cloaked is not None else [])
    scored = [(rel, _marker_count(repo / rel)) for rel in candidates]
    best = max((n for _, n in scored), default=0)
    if best == 0:
        return None
    winners = [rel for rel, n in scored if n == best]
    if len(winners) > 1:
        fallback = "docs" if cloaked is None else f"{cloaked}/docs"
        return fallback if fallback in winners else winners[0]
    return winners[0]


def doc_root(repo, sites_root=SITES_ROOT):
    """The repo's docs root, RELATIVE to the repo. Never an absolute path.

    Relative because every consumer joins it onto a repo path it already holds,
    and because the value ends up in .gitignore lines and glob patterns where
    an absolute path would be wrong.

    Evidence first (see the module docstring), then the path convention.
    """
    found = existing_doc_root(repo, sites_root)
    if found is not None:
        return found
    name = cloaked_name(repo, sites_root)
    return "docs" if name is None else f"{name}/docs"


def desk_path(repo, sites_root=SITES_ROOT):
    """The live desk, relative to the repo. Exactly one per repo."""
    return f"{doc_root(repo, sites_root)}/SESSION-DESK.md"


def archive_dir(repo, sites_root=SITES_ROOT):
    """Where parked desks live, relative to the repo.

    Flat — no `history/`, no `backups/`. Ruled 2026-08-19 (Q3): the live desk
    sits one level ABOVE this folder, so being in here IS what makes a desk an
    archive. That separation is structural, so no naming rule or glob exclusion
    is needed to tell a live desk from a parked one.
    """
    return f"{doc_root(repo, sites_root)}/sessions"


def main(argv):
    if len(argv) != 2:
        print("usage: doc_root.py <repo-path>", file=sys.stderr)
        return 2
    print(doc_root(argv[1]))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
