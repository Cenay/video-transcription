"""Domain term normalization — repair known transcription mishearings.

The speech engine reliably mangles domain vocabulary and names ("Khurram" is
never once transcribed correctly across 83 cached meetings). This module applies
a curated term list to transcripts and analysis output, and reports every
substitution it made.

Design rules, all of which exist for a measured reason — see
plans/term-normalization.md:

  * The AUTHOR only supplies strings. Whether a variant is safe to replace is
    decided here, mechanically, so adding a term costs nothing but the string.
  * A variant whose every token is ordinary English is REFUSED by default,
    because replacing it corrupts prose: "the book he mentioned" must never
    become "the Bookeo mentioned". `force:` overrides this per variant.
  * apply_corrections() returns the substitutions it made. A silent corrector
    is one nobody trusts, and the log is what makes a bad term entry
    discoverable instead of archaeological.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

DEFAULT_TERMS_PATH = Path(__file__).resolve().parent.parent / "config" / "terms.yml"
WORDLIST_PATH = Path("/usr/share/dict/words")

# 1-2 character tokens are not in most word lists in a useful way ("io" IS in
# /usr/share/dict/words, which wrongly made "book io" look like English). These
# are the short tokens that genuinely count as ordinary English.
SHORT_ENGLISH = {
    "a", "i", "am", "an", "as", "at", "be", "by", "do", "go", "he", "if", "in",
    "is", "it", "me", "my", "no", "of", "on", "or", "so", "to", "up", "us",
    "we", "he", "she", "the", "you",
}


def _load_wordlist() -> set[str]:
    if not WORDLIST_PATH.exists():
        return set()
    with WORDLIST_PATH.open(encoding="utf-8", errors="ignore") as fh:
        return {line.strip().lower() for line in fh if line.strip()}


_WORDS = _load_wordlist()


def is_ordinary_english(token: str) -> bool:
    """Would a reader expect to see this word in normal prose?"""
    t = token.lower().strip("'")
    if not t:
        return False
    if len(t) <= 2:
        return t in SHORT_ENGLISH
    return t in _WORDS


def is_risky(variant: str) -> bool:
    """True when replacing this variant could corrupt ordinary prose.

    Risky = every token is an ordinary English word. Such a variant is refused
    unless the term entry explicitly forces it.
    """
    tokens = variant.split()
    return bool(tokens) and all(is_ordinary_english(t) for t in tokens)


@dataclass
class Term:
    correct: str
    heard: list[str] = field(default_factory=list)
    force: list[str] = field(default_factory=list)
    identifier_prefix: str | None = None

    def applied(self) -> list[str]:
        """Variants that will actually be substituted."""
        forced = {f.lower() for f in self.force}
        return [v for v in self.heard if not is_risky(v) or v.lower() in forced]

    def refused(self) -> list[str]:
        forced = {f.lower() for f in self.force}
        return [v for v in self.heard if is_risky(v) and v.lower() not in forced]


@dataclass
class Substitution:
    term: str
    variant: str
    count: int
    forced: bool


def load_terms(path: Path | str | None = None) -> list[Term]:
    """Read the term list. Fails loudly rather than returning an empty list.

    A silently-empty term list would apply zero corrections while every run
    reported success — "clean" is also what a broken checker prints.
    """
    p = Path(path) if path else DEFAULT_TERMS_PATH
    if not p.exists():
        raise FileNotFoundError(
            f"term list not found at {p} — refusing to continue with no corrections. "
            "If the repo moved, update DEFAULT_TERMS_PATH in scripts/terms.py."
        )
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    entries = data.get("terms") or []
    if not entries:
        raise ValueError(f"term list at {p} parsed but contains no terms")
    return [
        Term(
            correct=e["correct"],
            heard=[str(h) for h in e.get("heard", [])],
            force=[str(f) for f in e.get("force", [])],
            identifier_prefix=e.get("identifier_prefix"),
        )
        for e in entries
    ]


def _variant_pattern(variant: str) -> re.Pattern:
    """Word-boundary match, tolerant of whitespace runs in multi-word variants."""
    parts = [re.escape(p) for p in variant.split()]
    return re.compile(r"\b" + r"\s+".join(parts) + r"\b", re.IGNORECASE)


def apply_corrections(
    text: str, terms: list[Term] | None = None
) -> tuple[str, list[Substitution]]:
    """Return (corrected_text, substitutions_made)."""
    if terms is None:
        terms = load_terms()

    changes: list[Substitution] = []

    for term in terms:
        forced = {f.lower() for f in term.force}

        # Longest variants first, so "book io" wins before a shorter overlap.
        for variant in sorted(term.applied(), key=len, reverse=True):
            pattern = _variant_pattern(variant)
            text, n = pattern.subn(term.correct, text)
            if n:
                changes.append(
                    Substitution(term.correct, variant, n, variant.lower() in forced)
                )

        # Constructed identifiers the model builds but nobody speaks:
        # bookio_product_groups -> bookeo_product_groups
        if term.identifier_prefix:
            stem = term.identifier_prefix.rstrip("_")
            for variant in term.heard:
                if " " in variant:
                    continue
                pattern = re.compile(rf"\b{re.escape(variant)}_", re.IGNORECASE)
                text, n = pattern.subn(f"{stem}_", text)
                if n:
                    changes.append(
                        Substitution(term.correct, f"{variant}_ (identifier)", n, False)
                    )

    return text, changes


def spelling_constraint(terms: list[Term] | None = None) -> str:
    """Render the term list as a spelling instruction for the analysis prompt.

    Correcting the transcript is not enough: the model CONSTRUCTS identifiers
    that nobody speaks aloud ("bookio_product_groups" in a summary heading), so
    a term can enter the output without ever appearing in the input. This block
    constrains generation; correct_structure() below catches what it misses.
    """
    if terms is None:
        terms = load_terms()

    lines = [
        "SPELLING (domain vocabulary — these are proper nouns in this company, "
        "and the transcript's speech-to-text mangles them):",
    ]
    for term in terms:
        wrong = ", ".join(sorted(f"{v!r}" for v in term.heard))
        line = f"- Always write {term.correct!r}"
        if term.identifier_prefix:
            line += f" (identifiers use the prefix {term.identifier_prefix!r})"
        if wrong:
            line += f". Never write: {wrong}"
        lines.append(line)
    lines.append(
        "Apply these spellings everywhere — including inside table names, "
        "column names, code identifiers and headings you compose yourself."
    )
    return "\n".join(lines)


def correct_structure(obj, terms: list[Term] | None = None):
    """Apply corrections to every string in a nested dict/list. Returns
    (corrected_obj, substitutions).

    The analysis result is JSON, so the wrong token can be anywhere — a
    heading, a decision's rationale, an action item's owner. Keys are corrected
    as well as values, because a mangled key would break the Notion renderer's
    lookups; in practice keys are fixed schema names and never change.

    `_usage` is skipped: it is our own bookkeeping, not model prose.
    """
    if terms is None:
        terms = load_terms()

    changes: list[Substitution] = []

    def walk(node):
        if isinstance(node, str):
            fixed, subs = apply_corrections(node, terms)
            changes.extend(subs)
            return fixed
        if isinstance(node, list):
            return [walk(v) for v in node]
        if isinstance(node, dict):
            return {
                (k if k == "_usage" else walk(k)): (v if k == "_usage" else walk(v))
                for k, v in node.items()
            }
        return node

    return walk(obj), _merge(changes)


def _merge(changes: list[Substitution]) -> list[Substitution]:
    """Collapse per-string substitutions into one row per (term, variant)."""
    merged: dict[tuple[str, str], Substitution] = {}
    for c in changes:
        key = (c.term, c.variant)
        if key in merged:
            merged[key].count += c.count
        else:
            merged[key] = Substitution(c.term, c.variant, c.count, c.forced)
    return list(merged.values())


def format_report(changes: list[Substitution]) -> str:
    """Human-readable summary for the run output and the log."""
    if not changes:
        return "  No term corrections applied."
    lines = []
    total = sum(c.count for c in changes)
    lines.append(f"  {total} term correction(s) applied:")
    for c in sorted(changes, key=lambda c: -c.count):
        flag = "  [FORCED]" if c.forced else ""
        lines.append(f"    {c.count:5d}x  {c.variant!r} -> {c.term}{flag}")
    return "\n".join(lines)
