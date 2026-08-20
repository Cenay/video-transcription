# Next Steps — video-transcription

_Last updated 2026-08-20 12:12 MST by an AI session · transcript: `f0912a53-461b-4861-97e4-931cb2f83ba0` — the bookie-o corpus failure is fixed; only the cosmetic Ninthroot failure remains_

<details>
<summary>📜 <strong>Stamp history</strong> — the 3 previous updates (older ones: <code>history/NEXT_STEPS-stamp-history.md</code>)</summary>

- _Prior: 2026-08-20 11:37 MST by an AI session · transcript: `f0912a53-461b-4861-97e4-931cb2f83ba0` — dropped the settled Notion-page question and flagged the pre-existing Ninthroot test failure _
- _Prior: 2026-08-19 15:27 MST by an AI session · transcript: `4325eefc-3918-4756-9846-cdc2fe7683cd` — **queued the `declined:` list** (raised by Cenay 2026-08-19; full sketch in `docs/TODOS.md` → Active). ⛔ **The check belongs at PROPOSAL time — the reconcile skills' `/add-term` offer — not only inside `add-term.py`**, because what costs Cenay time is being asked again, not the write being blocked. ⚠️ **It should exit 0, not 2**: a recorded decline is a satisfied precondition, not an error._
- _Prior: 2026-08-14 00:28 MST by an AI session · transcript: `4c61a822-47ec-4195-b344-607007d9c624` — the term-normalization build is complete; removed the stale NOT-BUILT block for /add-term and both closed measurements, leaving only two rulings and a cleanup_

</details>

> Start every session here. This file answers "what do I pick up right now?" —
> details live in `CURRENT_STATUS.md`, decisions in `DECISIONS.md`.

## Where We Left Off

**The term corrector is wired in and live.** Build-order **steps 1–3** shipped on 2026-08-13 (`7e7339f`): the transcript pass runs at the convergence point ([DEC-004]), and the analysis stage is protected twice — a generated spelling constraint in `ANALYSIS_PROMPT` plus a post-pass over the returned JSON ([DEC-009]).

⚠️ **Steps 4 and 5 of `plans/term-normalization.md` are NOT done** — Layer 2 `word_boost` and Layer 4 `custom_spelling`. They are deferred, not cancelled. Do not read "the corrector is live" as "the plan is finished".

**`/add-term` also shipped that day** — command, script, skill prompt, and an unplanned test script. It is **tooling under [DEC-008], not a build-order step**; the numbering belongs to the plan's five layers. Both remaining unknowns closed the same day:

- **The prompt constraint holds against a live model.** Two real meetings ran end-to-end (10:08 and 23:00 MST), 24 transcript-stage corrections applied, and **zero `[ANALYSIS — …]` entries** in `logs/term-corrections.log`. The model invented no wrong terms. That was the last unverified half of [DEC-009].
- **`/add-term` is in production use, not just built.** Five `chore(terms):` commits landed the same evening, each auto-committing and pushing `config/terms.yml` alone.
- **The global `~/.claude/CLAUDE.md` present-tense wording is now accurate** — re-checked against the script: absolute default path, explicit pathspec on commit, never `-a`, loud failure with the path it tried.

## Pick Up Here

**The next real build is step 4 — Layer 2 `word_boost` at transcription.** Wire-in is specified in `plans/term-normalization.md`: `scripts/transcriber.py`, the `aai.TranscriptionConfig(...)` call in `transcribe_audio()`, adding `word_boost=TERMS.boost_list()` and `boost_param=aai.WordBoost.high`. The case for it is strong — the engine has a **zero-percent** hit rate on `Khurram` and `Cenay`, and names are exactly the `word_boost` payload. It attacks the error upstream instead of repairing it downstream. ⚠️ Unlike steps 1–3, this one **costs money to verify**: `word_boost` changes what AssemblyAI returns, so it cannot be tested `--from-cache` and needs a real transcription run.

Then, smaller and non-blocking:

1. **Decide `Nik` vs `nick`** — 46 occurrences, all reading as the person, but ordinary English so the classifier refuses it. Vet with `./venv/bin/python scripts/preview_corrections.py --all --grep nick`, then either add it under `force:` or record it as deliberately excluded (it is already in the NOT-INCLUDED block in `terms.yml`).
2. **Dead code in `analyzer.py`** — unused `import os`, unused `model` param on `estimate_analysis_cost()`.

⚠️ **One pre-existing test failure in `tests/test_terms.py`, from term DATA added by `/add-term`, not from code:** `Ninthroot forces only variants the classifier refuses` — *needlessly forced: `['9th route', '9th grid']`*. Both contain `9th`, which is not an ordinary English word, so the classifier already applies them and the `force:` entries are redundant. Cosmetic; drop the two entries from `config/terms.yml` and it goes green.

✅ **The corpus-sweep failure is fixed (2026-08-20).** `bookie o` → `Bookeo` was **correct** — every `bookie` in both flagged transcripts is the company (*"one way from bookie O to zero"*, *"put up with Bookie O in the future"*), never a bookmaker. The test was what was wrong: it treated any change in a dangerous word's count as corruption. It now computes how many occurrences a **strictly longer** variant was entitled to swallow and demands the drop equal that exactly. ⛔ **The old hand-written `if word in ("active", "campaign"): continue` skip is gone** — that `continue` had been switching the check off for those two words entirely. See [`LESSONS_LEARNED.md`](LESSONS_LEARNED.md) → 2026-08-20.

**Running the tests:** `./venv/bin/python tests/test_terms.py --corpus` (**88 passed, 1 failed** as of 2026-08-20 — the Ninthroot item above; without `--corpus` it is 86/1) and `./venv/bin/python tests/test_notion_corrections.py` (**16 passed, 0 failed**), which covers the corrections toggle ([DEC-010]) against a stubbed Notion client. ⚠️ **`pytest` is not installed in `venv`**; the suite is a plain script with its own runner, so `python -m pytest` fails with `No module named pytest`. Pass `--corpus` or the 84-transcript sweep silently skips — and that sweep is the check that proves no ordinary-English word is corrupted.

## Decisions Needed

One, listed under **Pick Up Here** above — `Nik` vs `nick`. It blocks nothing. ~~Whether corrections also appear on the Notion page~~ ✅ ruled 2026-08-20 and built ([DEC-010]): a toggle directly under the transcript, carrying both correction passes.

## Watch Out For

⚠️ **The classifier rule in `scripts/terms.py` is more delicate than it looks, and it was wrong on first specification.** A plain dictionary lookup refuses `book io` (because `io` is in `/usr/share/dict/words`); "fixing" that with a minimum token length then wrongly *accepts* `book it` and `book he`, which is the direction that corrupts prose. The verified rule is dictionary lookup for 3+ character tokens **plus** an explicit stop-word set for short ones. Do not simplify it. See [DEC-005] and `tests/test_terms.py`.

⚠️ **Never list a possessive form in `terms.yml`.** `bookio's` as its own variant swallows the apostrophe-s — *"bookio's widget"* becomes *"Bookeo widget"* — because longer variants match first. The bare `bookio` entry already handles it correctly. Regression-tested.

⚠️ **The raw cache at `transcriber.py:59` must stay uncorrected.** Corrections apply downstream only, so the original is always recoverable when a term entry turns out to be wrong. A consequence: a `--from-cache` re-run produces different text than the cache holds. That is by design.

⚠️ **`/add-term` auto-commits and pushes `terms.yml`** — a deliberate exception to the no-auto-commit rule, recorded in the global `~/.claude/CLAUDE.md` so a session in another repo does not "fix" it. Confirmed working 2026-08-13: five commits, one file each.

⚠️ **Never pin a count of `force:` entries in a test.** `tests/test_terms.py` asserted "exactly two terms use `force:`", which `/add-term` broke the moment it forced `ninth root` from another repo — a data change, failing a test the person making it would never run. Replaced with a behavioural check (every forced variant must be one the classifier would actually refuse) plus the corpus sweep. Negative-tested: a decoratively-forced variant does make it fail.

⚠️ **`ANALYSIS_PROMPT` has more than one caller.** Adding the `{spelling}` placeholder broke `scripts/diagnose_analysis.py` with `KeyError: 'spelling'` — it formats the shared constant itself. Grep for `ANALYSIS_PROMPT` before touching its placeholders.

⚠️ **`correct_structure()` corrects dict keys as well as values.** Harmless today because keys are fixed schema names, but a term colliding with a schema key would break `notion_output.py`'s lookups. Noted in [DEC-009], deliberately not guarded.

**A pattern from the session worth carrying:** four separate defects were found here, and **every one surfaced by running something — none by reading it.** The fuzzy miner silently missed the `haram` variant (37 occurrences) because it fell below a similarity threshold, and a silent miss looks exactly like a clean result.

## Queued (unblocked, not yet scheduled)

- **A `declined:` list, so a ruled-against variant is never re-proposed** — raised by Cenay 2026-08-19 after `sine` → `Cenay` came back for the third or fourth time. ⛔ **The cost is the ASK, not the write**, so the check belongs where the variant is *proposed* (the reconcile skills' `/add-term` offer), not only inside `add-term.py`. ⚠️ **The place already exists and is not the problem** — ✅ `config/terms.yml:159`'s `DELIBERATELY NOT INCLUDED` block holds exactly this reasoning, and `add-term.py` preserves it. ⛔ **Nothing reads it**, so it is prose, not a guard. ★ **The job is therefore to promote that block to real YAML, migrating its five measured entries — not to invent a new mechanism.** Full sketch in [`TODOS.md`](TODOS.md) → Active.
- Layer 4 `custom_spelling` — step 5, "only if 1–4 leave residue". ⚠️ Measured caveat from the plan: `custom_spelling` is applied by AssemblyAI to *its own* output and never sees the LLM stage, so on the 2026-07-30 meeting it would have caught **1 occurrence in the transcript and 0 in the summary**. Low expected value.
- Step 4 `word_boost` is no longer "queued" — it is the recommended next build, promoted to **Pick Up Here** above.
- Items tracked in [`TODOS.md`](TODOS.md) and [`plans/term-normalization.md`](../plans/term-normalization.md).

## Blocked / Waiting On

<!-- What's stuck, and on whom or what -->

<!-- link-doc-refs:start (auto-generated — edit the IDs in prose, not this block) -->
[DEC-004]: DECISIONS.md#dec-004-the-substitution-runs-at-the-pipelinepy180-convergence-point
[DEC-005]: DECISIONS.md#dec-005-one-flat-term-table-the-code-classifies-risk-not-the-author
[DEC-008]: DECISIONS.md#dec-008-add-term-will-auto-commit-its-one-file--an-exception-to-the-no-auto-commit-rule
[DEC-009]: DECISIONS.md#dec-009-the-analysis-stage-is-protected-twice--a-prompt-constraint-and-a-post-pass-and-the-gap-between-them-is-the-measurement
[DEC-010]: DECISIONS.md#dec-010-the-corrections-also-go-on-the-notion-page-in-a-toggle-directly-under-the-transcript
<!-- link-doc-refs:end -->
