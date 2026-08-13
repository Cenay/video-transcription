#!/usr/bin/env python3
"""Tests for scripts/terms.py — the domain term corrector.

Run:  ./venv/bin/python tests/test_terms.py
      ./venv/bin/python tests/test_terms.py --corpus     (also sweeps all cached transcripts)

No pytest dependency — plain asserts, readable top to bottom.

WHY THESE TESTS LOOK LIKE THIS
------------------------------
A corrector that rewrites meeting records is only trustworthy if it has been
shown to REFUSE the dangerous cases, not merely to fix the easy ones. "Nothing
was corrupted" is also what a corrector that does nothing prints. So every
safety test here comes in a pair:

    * the guarded run must leave the sentence untouched, AND
    * the same input with the guard bypassed must visibly corrupt it

If the second half stops failing, the first half has stopped meaning anything.
"""

import json
import glob
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from terms import (  # noqa: E402
    Term,
    apply_corrections,
    is_ordinary_english,
    is_risky,
    load_terms,
)

PASS, FAIL = [], []


def check(name, condition, detail=""):
    (PASS if condition else FAIL).append(name)
    mark = "  ok  " if condition else "  FAIL"
    print(f"{mark} {name}" + (f"\n         {detail}" if detail and not condition else ""))


# ---------------------------------------------------------------------------
# 1. The classifier — which variants are safe to substitute
# ---------------------------------------------------------------------------
def test_classifier():
    print("\n[1] classifier: is a variant safe to auto-replace?")

    # Safe: contains a token that is not ordinary English.
    for v in ["bookio", "booko", "booku", "bookq", "book io", "book eoe",
              "karam", "senay", "milosh", "brandash"]:
        check(f"applies {v!r}", not is_risky(v))

    # Risky: every token is ordinary English -> would corrupt prose.
    for v in ["booking", "booked", "bookings", "bookie", "book it",
              "book he", "book here", "book you", "book a", "book is"]:
        check(f"refuses {v!r}", is_risky(v))

    # The specific trap that broke the first version of this rule: "io" is in
    # /usr/share/dict/words, so a naive dictionary lookup refused "book io".
    check("'io' is not treated as ordinary English", not is_ordinary_english("io"))
    check("'he' IS treated as ordinary English", is_ordinary_english("he"))
    check("'it' IS treated as ordinary English", is_ordinary_english("it"))

    # And the repair that was tried and rejected: a >=3 char minimum would let
    # "book it" and "book he" through, which is the dangerous direction.
    check("'book it' still refused (min-length repair would have allowed it)",
          is_risky("book it"))


# ---------------------------------------------------------------------------
# 2. The guard is load-bearing — paired safe/unsafe runs
# ---------------------------------------------------------------------------
def test_guard_is_load_bearing():
    print("\n[2] guard: refusing risky variants actually prevents damage")

    sentence = "I will book you for Monday and the book he mentioned is a good booking."

    guarded = Term(correct="Bookeo",
                   heard=["bookio", "booking", "book he", "book you", "booked"])
    out_guarded, changes = apply_corrections(sentence, [guarded])
    check("guarded run leaves ordinary English untouched",
          out_guarded == sentence, f"got: {out_guarded}")
    check("guarded run reports no substitutions", changes == [])

    # Same list, but every risky variant force-approved: the damage must appear.
    forced = Term(correct="Bookeo",
                  heard=["booking", "book he", "book you"],
                  force=["booking", "book he", "book you"])
    out_forced, _ = apply_corrections(sentence, [forced])
    check("bypassing the guard DOES corrupt the sentence (proves the guard matters)",
          out_forced != sentence and "Bookeo for Monday" in out_forced,
          f"got: {out_forced}")

    check("classifier splits a mixed list correctly",
          guarded.applied() == ["bookio"],
          f"applied={guarded.applied()} refused={guarded.refused()}")


# ---------------------------------------------------------------------------
# 3. Substitution behaviour
# ---------------------------------------------------------------------------
def test_substitution():
    print("\n[3] substitution mechanics")

    t = Term(correct="Bookeo", heard=["bookio"], identifier_prefix="bookeo_")

    out, _ = apply_corrections("The bookio widget is fine.", [t])
    check("replaces a bare variant", out == "The Bookeo widget is fine.", out)

    out, _ = apply_corrections("BOOKIO and Bookio and bookio", [t])
    check("is case-insensitive on the variant", out == "Bookeo and Bookeo and Bookeo", out)

    out, _ = apply_corrections("Textbookio should not change.", [t])
    check("respects word boundaries", out == "Textbookio should not change.", out)

    out, _ = apply_corrections("bookio_product_groups and bookio_schedules", [t])
    check("rewrites constructed identifiers",
          out == "bookeo_product_groups and bookeo_schedules", out)

    _, changes = apply_corrections("bookio bookio bookio", [t])
    check("reports an accurate count",
          len(changes) == 1 and changes[0].count == 3,
          f"got: {[(c.variant, c.count) for c in changes]}")


def test_possessive_preserved():
    """REGRESSION — found 2026-08-12 while testing against a real transcript.

    Listing "bookio's" as its own variant SWALLOWS the possessive, because
    longest-first ordering matches it before the bare "bookio" and replaces the
    whole thing with "Bookeo". The bare rule alone handles it correctly, since
    the apostrophe is a word boundary. Possessive forms must NOT be listed in
    config/terms.yml.
    """
    print("\n[4] regression: possessives survive")

    bare = Term(correct="Bookeo", heard=["bookio"])
    out, _ = apply_corrections("That is bookio's widget.", [bare])
    check("bare variant preserves the possessive",
          out == "That is Bookeo's widget.", out)

    listed = Term(correct="Bookeo", heard=["bookio", "bookio's"])
    out_bad, _ = apply_corrections("That is bookio's widget.", [listed])
    check("listing the possessive DOES swallow it (this is why it is banned)",
          out_bad == "That is Bookeo widget.", out_bad)

    # And the shipped config must not reintroduce it.
    for term in load_terms():
        offenders = [v for v in term.heard if v.endswith("'s")]
        check(f"config/terms.yml has no possessive variants for {term.correct}",
              not offenders, f"found: {offenders}")


# ---------------------------------------------------------------------------
# 5. The shipped term list
# ---------------------------------------------------------------------------
def test_shipped_config():
    print("\n[5] the shipped config/terms.yml")

    terms = load_terms()
    check("term list loads and is non-empty", len(terms) > 0)

    by_name = {t.correct: t for t in terms}
    check("Bookeo is present", "Bookeo" in by_name)
    check("Bookeo carries an identifier prefix",
          by_name["Bookeo"].identifier_prefix == "bookeo_")

    # Only two entries may use `force:`, and both were justified by measurement.
    forced = {t.correct: t.force for t in terms if t.force}
    check("exactly two terms use force:", len(forced) == 2, f"got: {forced}")
    check("ActiveCampaign forces 'active campaign'",
          "active campaign" in forced.get("ActiveCampaign", []))
    check("fran-dash forces 'fran dash'",
          "fran dash" in forced.get("fran-dash", []))

    # Nothing risky should be applied except the two forced phrases.
    for t in terms:
        unforced_risky = [v for v in t.applied()
                          if is_risky(v) and v.lower() not in {f.lower() for f in t.force}]
        check(f"{t.correct} applies no unforced risky variant",
              not unforced_risky, f"found: {unforced_risky}")


def test_missing_file_fails_loudly():
    print("\n[6] a missing term list fails loudly, never silently empty")
    try:
        load_terms("/nonexistent/path/terms.yml")
        check("raises on a missing term list", False, "no exception raised")
    except FileNotFoundError as e:
        check("raises FileNotFoundError naming the path",
              "/nonexistent/path/terms.yml" in str(e), str(e))
    except Exception as e:  # noqa: BLE001
        check("raises FileNotFoundError (not something else)", False, repr(e))


# ---------------------------------------------------------------------------
# 7. Optional: sweep every cached transcript
# ---------------------------------------------------------------------------
DANGEROUS = ["booking", "bookings", "booked", "bookie", "book it", "book he",
             "book you", "book here", "book a", "books", "make", "take", "nick",
             "real", "active", "campaign"]


def test_corpus():
    print("\n[7] corpus sweep — every cached transcript")
    terms = load_terms()
    files = sorted(glob.glob(str(ROOT / "temp/transcribe-cache/*.json")))
    total, corrupted, seen = 0, [], 0

    for f in files:
        try:
            data = json.load(open(f))
        except Exception:  # noqa: BLE001
            continue
        text = "\n".join(u.get("text", "") for u in data.get("utterances", []))
        if not text:
            continue
        seen += 1
        out, changes = apply_corrections(text, terms)
        total += sum(c.count for c in changes)

        for word in DANGEROUS:
            pat = rf"\b{re.escape(word)}\b"
            before = len(re.findall(pat, text, re.I))
            after = len(re.findall(pat, out, re.I))
            # "active campaign" is intentionally consumed by the forced rule.
            if word in ("active", "campaign"):
                continue
            if before != after:
                corrupted.append((Path(f).name, word, before, after))

    print(f"         {seen} transcripts, {total} corrections applied")
    check("no ordinary-English word was altered anywhere in the corpus",
          not corrupted, f"corrupted: {corrupted[:5]}")
    check("the corpus produced a meaningful number of corrections (>100)",
          total > 100, f"only {total}")


# ---------------------------------------------------------------------------
def main():
    test_classifier()
    test_guard_is_load_bearing()
    test_substitution()
    test_possessive_preserved()
    test_shipped_config()
    test_missing_file_fails_loudly()
    if "--corpus" in sys.argv:
        test_corpus()
    else:
        print("\n[7] corpus sweep — SKIPPED (pass --corpus to run it)")

    print("\n" + "=" * 60)
    print(f"  {len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        print("\n  FAILED:")
        for name in FAIL:
            print(f"    - {name}")
    print("=" * 60)
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
