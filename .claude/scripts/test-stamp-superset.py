#!/usr/bin/env python3
"""Regression suite for the SUPERSET history model ([DEC-053]).

Run:  python3 scripts/test-stamp-superset.py           (exit 0 = pass)
      python3 scripts/test-stamp-superset.py --prove    (the run that counts)

⛔ WHAT THIS GUARDS. `stamp-doc.py` held two incompatible models of what a
history file is, in two functions, and neither knew about the other:

  * `restore_history()` — SUPERSET. It writes every stamp git has ever seen for
    the doc, including the ones still sitting in the doc's `<details>` fold.
  * the roll-down verification in `main()` — DISJOINT. It demanded each prior
    appear EXACTLY ONCE across {doc, history}, so a prior in both was a bug.

★ The collision is not theoretical and it wedges a doc PERMANENTLY. Run
`--restore` (correct), then `--convert-only` (correct), and the second refuses
on every folded prior — reporting `0 prior(s) would be lost, 3 duplicated`,
a message whose first clause is true and whose shape reads like a non-event.
The doc can then never be reshaped again: in `/mnt/k/Code/System` that left a
stale fold pointer at `ai-directed-changes.md:6` that `render_fold()` would have
emitted correctly, with no way to reach it.

✅ Ruled SUPERSET by Cenay, 2026-08-25. Recovery must stay a pure function of
git — the moment it has to consult the doc's current fold, a damaged fold makes
`--restore` write LESS, which is the exact failure class that cost 367 lines on
2026-08-21.

⚠️ THE TRAP THIS SUITE EXISTS TO HOLD DOWN. Legalizing the cross-surface copy is
one line; doing it without dropping the two failures that DO matter is not.
`missing` and `not_durable` must therefore key presence on the RECORD (its
timestamp) and not on the text, because restore keeps the richest wording and
the fold often has a terser one. Case `a_richer_history_wording_is_accepted`
is that trap; it fails against an exact-text implementation.
"""

import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
STAMP = HERE / "stamp-doc.py"

P1 = ("_Prior: 2026-08-19 10:00 MST by an AI session · transcript: "
      "`bbbbbbbb-0000-0000-0000-000000000001` — the first prior._")
P2 = ("_Prior: 2026-08-18 10:00 MST by an AI session · transcript: "
      "`bbbbbbbb-0000-0000-0000-000000000002` — the second prior._")
P3 = ("_Prior: 2026-08-17 10:00 MST by an AI session · transcript: "
      "`bbbbbbbb-0000-0000-0000-000000000003` — the third prior._")

# The same record as P3, in the TERSER wording a fold may carry while
# `--restore` has stored the fuller one in history.
P3_TERSE = "_Prior: 2026-08-17 10:00 MST · transcript: `bbbbbbbb-0000-0000-0000-000000000003`_"

CURRENT = ("_Last updated 2026-08-20 10:00 MST by an AI session · transcript: "
           "`bbbbbbbb-0000-0000-0000-000000000000` — the current one._")


def doc_text(priors):
    fold = "\n".join(f"- {p}" for p in priors)
    return f"""# Current Status

{CURRENT}

<details>
<summary>📜 <strong>Stamp history</strong> — the {len(priors)} previous updates (older ones: <code>history/CURRENT_STATUS-stamp-history.md</code>)</summary>

{fold}

</details>

Body text.
"""


HIST_HEAD = """# Stamp history — CURRENT_STATUS.md

Older traceability stamps, newest first.

---
"""


def repo(priors, history_rows):
    """A git repo whose doc holds `priors` and whose history file holds `history_rows`."""
    d = Path(tempfile.mkdtemp())
    (d / "docs" / "history").mkdir(parents=True)
    doc = d / "docs" / "CURRENT_STATUS.md"
    hist = d / "docs" / "history" / "CURRENT_STATUS-stamp-history.md"
    doc.write_text(doc_text(priors), encoding="utf-8")
    if history_rows is not None:
        rows = "\n".join(f"- {r}" for r in history_rows)
        hist.write_text(HIST_HEAD + "\n" + rows + "\n", encoding="utf-8")
    run = lambda *a: subprocess.run(["git"] + list(a), cwd=d, capture_output=True)
    run("init", "-q")
    run("config", "user.email", "t@t"); run("config", "user.name", "t")
    run("add", "-A"); run("commit", "-qm", "first")
    return d, doc, hist


def convert(doc, keep=3):
    r = subprocess.run(
        [sys.executable, str(STAMP), str(doc), "--convert-only", "--keep", str(keep)],
        capture_output=True, text=True)
    return r.returncode, r.stdout + r.stderr


CASES = []


def case(fn):
    CASES.append(fn)
    return fn


# ---------------------------------------------------------------- the wedge


@case
def the_post_restore_wedge_is_released():
    """3 priors in the fold, the SAME 3 already in history — what --restore leaves.

    ⛔ This is the exact state that used to refuse, permanently.
    """
    _, doc, _ = repo([P1, P2, P3], [P1, P2, P3])
    rc, out = convert(doc, keep=3)
    assert rc == 0, f"refused the superset state: {out}"


@case
def a_richer_history_wording_is_accepted():
    """History holds the FULLER text of a record the fold carries tersely.

    ⚠️ This is what `--restore`'s richest-wins dedupe actually produces. An
    exact-text presence test refuses here, which is a wedge wearing a new hat.

    ⛔ `--keep 1`, and the 1 is the whole point. At `--keep 3` nothing rolls, so
    the terse text stays in the fold and is found verbatim — the case passes
    against an exact-text implementation and proves nothing. ✅ Caught by
    `--prove` reporting this case GREEN under the `exact-text presence only`
    mutant; it is only when the terse prior ROLLS OUT that its sole surviving
    copy is history's richer wording.
    """
    _, doc, _ = repo([P1, P2, P3_TERSE], [P1, P2, P3])
    rc, out = convert(doc, keep=1)
    assert rc == 0, f"refused a richer history wording: {out}"


@case
def roll_down_is_idempotent():
    """Rolling a prior history already holds must not append it a second time."""
    _, doc, hist = repo([P1, P2, P3], [P1, P2, P3])
    rc, out = convert(doc, keep=1)
    assert rc == 0, f"refused: {out}"
    body = hist.read_text(encoding="utf-8")
    for p in (P2, P3):
        n = body.count(p)
        assert n == 1, f"prior appears {n}× in history, expected 1"


@case
def a_genuine_roll_down_still_reaches_history():
    """The ordinary path: priors past --keep land in the history file."""
    _, doc, hist = repo([P1, P2, P3], None)
    rc, out = convert(doc, keep=1)
    assert rc == 0, f"refused: {out}"
    body = hist.read_text(encoding="utf-8")
    assert P2 in body and P3 in body, "rolled priors did not reach history"
    assert P1 in doc.read_text(encoding="utf-8"), "kept prior left the fold"


# ------------------------------------------------- the failures that remain


@case
def a_dropping_writer_is_refused():
    """A writer that silently loses a prior must be stopped.

    ⛔ This is the 2026-08-21 failure class in miniature — 367 lines went out the
    door behind a clean tick. The case is green on healthy code by construction;
    what proves it is `--prove`'s `writer drops a prior` mutant, which makes the
    roll-down lose one and asserts this refuses. Removing the `missing` term on
    top of that mutant makes the loss sail through — that pair is the negative
    test, not this function on its own.
    """
    _, doc, _ = repo([P1, P2, P3], None)
    rc, out = convert(doc, keep=1)
    assert rc == 0, f"control run refused: {out}"
    hist = doc.parent / "history" / "CURRENT_STATUS-stamp-history.md"
    body = hist.read_text(encoding="utf-8")
    assert P2 in body and P3 in body, "a prior was lost on a healthy run"


@case
def a_healthy_doc_passes():
    """The control. If this ever goes red the suite is measuring nothing."""
    _, doc, _ = repo([P1, P2], None)
    rc, out = convert(doc, keep=3)
    assert rc == 0, f"healthy doc refused: {out}"


@case
def duplication_within_one_file_is_refused():
    """Cross-surface is legal now; the same text twice in ONE file is not."""
    _, doc, _ = repo([P1, P2, P3], [P1, P1, P2, P3])
    rc, out = convert(doc, keep=3)
    assert rc != 0, "accepted a history file holding the same prior twice"
    assert "duplicated within a single file" in out, f"wrong refusal: {out}"


def prove():
    """Revert each half of the fix and assert exactly the right cases go red."""
    src = STAMP.read_text(encoding="utf-8")

    disjoint = src.replace(
        "    dupes = [p for p in old_priors\n"
        "             if new_doc.count(p) > 1 or (new_hist or \"\").count(p) > 1]",
        "    dupes = [p for p in old_priors if haystack.count(p) != 1]")
    assert disjoint != src, "could not locate the dupes term to revert"

    exact = src.replace("        if prior in text:\n            return True\n"
                        "        m = AUDIT_DATE_RE.search(prior)\n"
                        "        return bool(m) and m.group(1).strip() in stamp_dates(text)",
                        "        return prior in text")
    assert exact != src, "could not locate the timestamp fallback to revert"

    nodedupe = src.replace("        rolled = [p for p in rolled\n"
                           "                  if not (AUDIT_DATE_RE.search(p)\n"
                           "                          and AUDIT_DATE_RE.search(p).group(1).strip() in have)]",
                           "        rolled = list(rolled)")
    assert nodedupe != src, "could not locate the merge_history dedupe to revert"

    nodurable = src.replace(
        "    not_durable = [p for p in roll if not recorded(p, new_hist or \"\")]",
        "    not_durable = []")
    assert nodurable != src, "could not locate the durability term to revert"

    # ⛔ A BROKEN WRITER, not a broken check: the roll-down silently drops one
    # prior on the floor. This is what the `missing` term is for, and the only
    # honest way to test it — mutating the check itself proves nothing about
    # whether the check would ever fire.
    DROP = ("    keep, roll = priors[: args.keep], priors[args.keep:]",
            "    keep, roll = priors[: args.keep], priors[args.keep + 1:]")
    dropping = src.replace(*DROP)
    assert dropping != src, "could not locate the roll split to break"
    # ...and the same broken writer with the guard removed, which must sail through.
    unguarded = dropping.replace(
        "    missing = [p for p in old_priors if not recorded(p, haystack)]",
        "    missing = []")
    assert unguarded != dropping, "could not locate the missing term to revert"

    expected = {
        "disjoint model restored": (disjoint, {"the_post_restore_wedge_is_released",
                                               "a_richer_history_wording_is_accepted",
                                               "roll_down_is_idempotent"}),
        "exact-text presence only": (exact, {"a_richer_history_wording_is_accepted"}),
        "merge dedupe removed": (nodedupe, {"roll_down_is_idempotent"}),
        "writer drops a prior": (dropping, {"a_dropping_writer_is_refused",
                                            "a_genuine_roll_down_still_reaches_history"}),
    }
    problems = []
    try:
        for name, (mutant, must_fail) in expected.items():
            STAMP.write_text(mutant, encoding="utf-8")
            reds = set()
            for fn in CASES:
                try:
                    fn()
                except Exception:
                    reds.add(fn.__name__)
            print(f"  {name}: {len(reds)} red — {sorted(reds)}")
            for m in sorted(must_fail - reds):
                problems.append(f"    ⛔ {m} stayed GREEN under '{name}'")
            if "a_healthy_doc_passes" in reds:
                problems.append(f"    ⛔ the control went red under '{name}'")
        # ★ THE NEGATIVE TEST'S OTHER HALF, probed DIRECTLY rather than through
        # the case set. ⚠️ Red/green cannot discriminate here: with the guard
        # removed the run succeeds and the prior is genuinely lost, so the case
        # goes red on its CONTENT assertion — the right answer for the wrong
        # reason, and it read as "the guard still fired". Assert the mechanism.
        STAMP.write_text(dropping, encoding="utf-8")
        _, doc, hist = repo([P1, P2, P3], None)
        rc_guarded, out_guarded = convert(doc, keep=1)
        wrote_guarded = hist.exists()

        STAMP.write_text(unguarded, encoding="utf-8")
        _, doc2, hist2 = repo([P1, P2, P3], None)
        rc_open, _ = convert(doc2, keep=1)
        lost = hist2.exists() and P2 not in hist2.read_text(encoding="utf-8")

        print(f"  writer drops a prior — guarded: rc={rc_guarded}, "
              f"history written={wrote_guarded} · unguarded: rc={rc_open}, "
              f"prior lost={lost}")
        if rc_guarded == 0:
            problems.append("    ⛔ the dropping writer was ACCEPTED with the guard in place")
        elif "would be lost" not in out_guarded:
            problems.append(f"    ⛔ refused, but not by the loss term: {out_guarded[:120]}")
        if wrote_guarded:
            problems.append("    ⛔ refused but still wrote the history file — "
                            "'No files changed' is false")
        if not (rc_open == 0 and lost):
            problems.append("    ⛔ with `missing` removed the loss did NOT sail "
                            "through, so that term is not what is catching it")

        # ⚠️ Reported separately: removing the durability term breaks no case
        # above, because every scenario here also satisfies `missing`. Say so
        # rather than let a silent pass read as coverage.
        STAMP.write_text(nodurable, encoding="utf-8")
        reds = {fn.__name__ for fn in CASES if _red(fn)}
        print(f"  durability term removed: {len(reds)} red — {sorted(reds)}")
        if not reds:
            print("    ⓘ NOT COVERED by a case here — the `not_durable` term is a "
                  "belt-and-braces assertion that no scenario in this suite can "
                  "violate without also tripping `missing`. Stated, not claimed.")
    finally:
        STAMP.write_text(src, encoding="utf-8")

    if problems:
        print("\n".join(problems))
        return 1
    print("✅ each revert turns exactly the right cases red, and the control holds")
    return 0


def _red(fn):
    try:
        fn()
        return False
    except Exception:
        return True


def main():
    if "--prove" in sys.argv:
        return prove()
    failed = 0
    for fn in CASES:
        try:
            fn()
            print(f"  ✓ {fn.__name__}")
        except Exception as exc:
            failed += 1
            print(f"  ✗ {fn.__name__}: {exc}")
    if failed:
        print(f"\n{failed} of {len(CASES)} failed")
        return 1
    print(f"\nall {len(CASES)} passed — ⚠️ now run --prove")
    return 0


if __name__ == "__main__":
    sys.exit(main())
