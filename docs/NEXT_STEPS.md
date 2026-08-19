# Next Steps — video-transcription

_Last updated 2026-08-19 15:27 MST by an AI session · transcript: `4325eefc-3918-4756-9846-cdc2fe7683cd` — **queued the `declined:` list** (raised by Cenay 2026-08-19; full sketch in `docs/TODOS.md` → Active). ⛔ **The check belongs at PROPOSAL time — the reconcile skills' `/add-term` offer — not only inside `add-term.py`**, because what costs Cenay time is being asked again, not the write being blocked. ⚠️ **It should exit 0, not 2**: a recorded decline is a satisfied precondition, not an error._

<details>
<summary>📜 <strong>Stamp history</strong> — the 3 previous updates (older ones: <code>history/NEXT_STEPS-stamp-history.md</code>)</summary>

- _Prior: 2026-08-14 00:28 MST by an AI session · transcript: `4c61a822-47ec-4195-b344-607007d9c624` — the term-normalization build is complete; removed the stale NOT-BUILT block for /add-term and both closed measurements, leaving only two rulings and a cleanup_
- _Prior: 2026-08-13 12:22 MST by an AI session · transcript: `2fa5b28a-7c93-4f78-8239-fc20e8d6cc8f` — rewrote the handoff around the live-meeting measurement; added the shared-prompt and dict-key gotchas; expanded /add-term into three concrete pieces after confirming it does not exist_
- _Prior: 2026-08-12 19:41 MST by an AI session · transcript: `ce166dfb-eca1-4c5a-b935-71755aed3e98` — rewrote the handoff for step 2 (wiring apply_corrections into pipeline.py:180); added the classifier and possessive gotchas._

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
2. **Decide whether corrections also appear on the Notion page**, in addition to `logs/`. Marginal.
3. **Dead code in `analyzer.py`** — unused `import os`, unused `model` param on `estimate_analysis_cost()`.

**Running the tests:** `./venv/bin/python tests/test_terms.py --corpus` — **82 passed, 0 failed** as of 2026-08-13 23:13 MST. ⚠️ **`pytest` is not installed in `venv`**; the suite is a plain script with its own runner, so `python -m pytest` fails with `No module named pytest`. Pass `--corpus` or the 84-transcript sweep silently skips — and that sweep is the check that proves no ordinary-English word is corrupted.

## Decisions Needed

Both are listed under **Pick Up Here** above — `Nik` vs `nick`, and whether corrections also appear on the Notion page. Neither blocks anything.

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
<!-- link-doc-refs:end -->
