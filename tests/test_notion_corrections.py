#!/usr/bin/env python3
"""Tests for the term-corrections toggle on the Notion page ([DEC-010]).

Run:  ./venv/bin/python tests/test_notion_corrections.py

No pytest dependency — plain asserts, readable top to bottom. The Notion client
is replaced with a recorder, so nothing here touches the network or creates a
page; what is being checked is the block structure that WOULD be sent.

WHY THESE TESTS LOOK LIKE THIS
------------------------------
The block this covers exists so a downstream reader can tell that the
transcript was rewritten. Two ways that fails silently, and both are tested as
pairs — the passing case AND the case that must NOT pass:

    * the toggle renders but omits a correction  -> a rewrite nobody can see
    * the toggle says "none applied" when there were some, or is skipped
      entirely -> silence that reads as "nothing happened"

`None` (the corrector did not run) and `[]` (it ran and found nothing) must
therefore produce visibly different pages.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import notion_output  # noqa: E402
from notion_output import (  # noqa: E402
    CORRECTIONS_TOGGLE_TITLE,
    build_correction_blocks,
    create_meeting_page,
)

PASS, FAIL = [], []


def check(name, condition, detail=""):
    (PASS if condition else FAIL).append(name)
    mark = "  ok  " if condition else "  FAIL"
    print(f"{mark} {name}" + (f"\n         {detail}" if detail and not condition else ""))


def flatten(blocks) -> str:
    """All visible text in a block list, so assertions read as plain strings."""
    out = []
    for b in blocks:
        body = b.get(b["type"], {})
        for rt in body.get("rich_text", []):
            out.append(rt.get("text", {}).get("content", ""))
    return "\n".join(out)


# ---------------------------------------------------------------------------
# A recorder standing in for the Notion client.
# ---------------------------------------------------------------------------
class _Children:
    def __init__(self, log):
        self.log = log
        self.n = 0

    def append(self, block_id, children):
        self.n += 1
        self.log.append((block_id, children))
        return {"results": [{"id": f"block-{self.n}-{i}"} for i, _ in enumerate(children)]}

    def list(self, block_id):
        return {"results": []}


class _Blocks:
    def __init__(self, log):
        self.children = _Children(log)

    def delete(self, block_id):
        pass

    def update(self, block_id, **kwargs):
        pass


class _Pages:
    def create(self, **kwargs):
        return {"id": "page-under-test"}


class FakeNotion:
    def __init__(self):
        self.log = []
        self.blocks = _Blocks(self.log)
        self.pages = _Pages()


def render_page(corrections):
    """Run create_meeting_page against the recorder; return the append log."""
    fake = FakeNotion()
    real = notion_output.notion
    notion_output.notion = fake
    try:
        create_meeting_page(
            title="Test Meeting",
            date="2026-08-20",
            duration_minutes=30,
            analysis={"summary": "s", "action_items": [], "decisions": []},
            transcript="Cenay: we should drop the Bookeo prefix.",
            costs={"transcription": 0.1, "analysis": 0.05, "total": 0.15},
            source_file="test.mp4",
            corrections=corrections,
        )
    finally:
        notion_output.notion = real
    return fake.log


def toggle_titles(log) -> list[str]:
    """Titles of the toggle headings appended to the page itself, in order."""
    titles = []
    for block_id, children in log:
        if block_id != "page-under-test":
            continue
        for b in children:
            if b["type"] == "heading_3" and b["heading_3"].get("is_toggleable"):
                titles.append(b["heading_3"]["rich_text"][0]["text"]["content"])
    return titles


SAMPLE = [
    {"term": "Bookeo", "variant": "bookio", "count": 7,
     "forced": False, "stage": "transcript"},
    {"term": "ActiveCampaign", "variant": "active campaign", "count": 2,
     "forced": True, "stage": "transcript"},
    {"term": "Bookeo", "variant": "bookio_ (identifier)", "count": 1,
     "forced": False, "stage": "analysis"},
]


# ---------------------------------------------------------------------------
# 1. The block contents — every correction is stated
# ---------------------------------------------------------------------------
def test_block_states_every_correction():
    print("\n[1] the toggle states each substitution")
    text = flatten(build_correction_blocks(SAMPLE))

    for row in SAMPLE:
        check(f"names the {row['variant']!r} -> {row['term']} substitution",
              f'"{row["variant"]}" → {row["term"]}' in text, text)

    check("reports the count for each", "7×" in text and "2×" in text, text)
    check("marks a forced substitution", "[forced]" in text, text)
    check("warns that the transcript is not raw speech-to-text",
          "NOT raw speech-to-text" in text, text)
    check("points at the uncorrected copy", "transcribe-cache" in text, text)

    # ...and the load-bearing half: with no corrections it must NOT claim any.
    empty = flatten(build_correction_blocks([]))
    check("...and an empty run claims none of them",
          "Bookeo" not in empty and "No term corrections were applied" in empty,
          empty)


# ---------------------------------------------------------------------------
# 2. Analysis-stage corrections are distinguishable from transcript ones
# ---------------------------------------------------------------------------
def test_analysis_rows_are_flagged():
    print("\n[2] an analysis-stage correction is not shown as a transcript one")
    text = flatten(build_correction_blocks(SAMPLE))

    check("the analysis section says the model wrote it",
          "the model wrote these" in text, text)

    transcript_only = [r for r in SAMPLE if r["stage"] == "transcript"]
    check("...and that section is absent when only the transcript was corrected",
          "the model wrote these" not in flatten(
              build_correction_blocks(transcript_only)),
          flatten(build_correction_blocks(transcript_only)))


# ---------------------------------------------------------------------------
# 3. Placement — the toggle sits directly below the transcript toggle
# ---------------------------------------------------------------------------
def test_toggle_placement():
    print("\n[3] placement: corrections toggle directly under the transcript")
    titles = toggle_titles(render_page(SAMPLE))
    check("both toggles are on the page, corrections last",
          titles == ["Transcript", CORRECTIONS_TOGGLE_TITLE], titles)


# ---------------------------------------------------------------------------
# 4. None vs [] — "did not look" and "found nothing" must look different
# ---------------------------------------------------------------------------
def test_none_and_empty_differ():
    print("\n[4] no list at all vs an empty list")
    none_titles = toggle_titles(render_page(None))
    empty_titles = toggle_titles(render_page([]))

    check("corrections=None renders no corrections toggle",
          none_titles == ["Transcript"], none_titles)
    check("corrections=[] still renders the toggle",
          empty_titles == ["Transcript", CORRECTIONS_TOGGLE_TITLE], empty_titles)
    check("...so the two are distinguishable on the page",
          none_titles != empty_titles, f"{none_titles} == {empty_titles}")


# ---------------------------------------------------------------------------
# 5. Notion's structural limits
# ---------------------------------------------------------------------------
def test_notion_limits():
    print("\n[5] Notion limits: 2000 chars per block, 100 blocks per request")
    long_variant = [{"term": "Bookeo", "variant": "x" * 4000, "count": 1,
                     "forced": False, "stage": "transcript"}]
    blocks = build_correction_blocks(long_variant)
    longest = max(
        len(rt["text"]["content"])
        for b in blocks
        for rt in b[b["type"]].get("rich_text", [])
    )
    check("no block exceeds Notion's 2000-character limit", longest <= 2000,
          f"longest block is {longest} chars")

    many = [{"term": f"Term{i}", "variant": f"heard{i}", "count": 1,
             "forced": False, "stage": "transcript"} for i in range(150)]
    log = render_page(many)
    oversized = [(bid, len(ch)) for bid, ch in log if len(ch) > 100]
    check("no append request carries more than 100 blocks", not oversized,
          f"oversized requests: {oversized}")


# ---------------------------------------------------------------------------
def main():
    test_block_states_every_correction()
    test_analysis_rows_are_flagged()
    test_toggle_placement()
    test_none_and_empty_differ()
    test_notion_limits()

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
