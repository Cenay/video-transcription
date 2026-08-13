# Next Steps — video-transcription

_Last updated 2026-08-12 19:41 MST by an AI session · transcript: `ce166dfb-eca1-4c5a-b935-71755aed3e98` — rewrote the handoff for step 2 (wiring apply_corrections into pipeline.py:180); added the classifier and possessive gotchas._

<details>
<summary>📜 <strong>Stamp history</strong> — the 1 previous update (older ones: <code>history/NEXT_STEPS-stamp-history.md</code>)</summary>

- _Prior: 2026-08-12 19:15 MST_

</details>

> Start every session here. This file answers "what do I pick up right now?" —
> details live in `CURRENT_STATUS.md`, decisions in `DECISIONS.md`.

## Where We Left Off

Built and tested the term corrector — `config/terms.yml`, `scripts/terms.py`, `scripts/preview_corrections.py`, `tests/test_terms.py` — but **it is not wired into the pipeline yet**, so it currently affects nothing. Everything so far is standalone and provably safe: 56 tests pass, and a sweep of all 83 cached transcripts produces 630 corrections with zero ordinary-English words altered.

## Pick Up Here

**First, wire it in.** Add the correction pass to `scripts/pipeline.py` at line 180 — the point where the `--from-cache` branch (`:109`) and the fresh-transcription branch (`:170`) converge, so one call covers both ([DEC-004]):

```python
transcript_text, corrections = apply_corrections(transcript_text)
print(format_report(corrections))
```

Print the report and write it to `logs/` ([DEC-006]). Then run a `--from-cache` pass over an already-cached meeting and confirm the Notion page comes out with `bookeo_` and zero `bookio` — that is the plan's stated success criterion, and it costs nothing because the transcript is cached.

**Second, decide about the analysis stage.** The transcript pass alone does not stop Claude from *constructing* a wrong identifier like `bookio_product_groups` in the summary — that is where the original incident actually did its damage. Two options, and `plans/term-normalization.md` argues for doing both and using the difference as a signal: apply the same `apply_corrections()` to the analysis output after `pipeline.py:207`, and/or inject the term list into `ANALYSIS_PROMPT` as a spelling constraint (`analyzer.py:24` — confirmed to be a single prompt with one insertion point).

**Third, if there is appetite:** build `/add-term` ([DEC-008]) so terms can be added from whichever repo you happen to be reconciling in. Designed in detail, not started.

## Decisions Needed

- **`Nik` vs `nick`** — `nick` appears on 42 lines across 8 transcripts. Every sample read as the person, but it is ordinary English so the classifier refuses it by default. Vet with `./venv/bin/python scripts/preview_corrections.py --all --grep nick` and decide whether to `force:` it. Small; does not block anything.
- **Whether corrections also appear on the Notion page**, in addition to `logs/`. Marginal — you review reconciliations anyway, so the page block would only save a manual correction.

## Watch Out For

⚠️ **The classifier rule in `scripts/terms.py` is more delicate than it looks, and it was wrong on first specification.** A plain dictionary lookup refuses `book io` (because `io` is in `/usr/share/dict/words`); "fixing" that with a minimum token length then wrongly *accepts* `book it` and `book he`, which is the direction that corrupts prose. The verified rule is dictionary lookup for 3+ character tokens **plus** an explicit stop-word set for short ones. Do not simplify it. See [DEC-005] and `tests/test_terms.py`.

⚠️ **Never list a possessive form in `terms.yml`.** `bookio's` as its own variant swallows the apostrophe-s — *"bookio's widget"* becomes *"Bookeo widget"* — because longer variants match first. The bare `bookio` entry already handles it correctly. Regression-tested.

⚠️ **The raw cache at `transcriber.py:59` must stay uncorrected.** Corrections apply downstream only, so the original is always recoverable when a term entry turns out to be wrong. A consequence: a `--from-cache` re-run produces different text than the cache holds. That is by design.

⚠️ **`/add-term` will auto-commit `terms.yml`** when built — a deliberate exception to the no-auto-commit rule, recorded in the global `~/.claude/CLAUDE.md` so a session in another repo does not "fix" it.

**A pattern from the session worth carrying:** four separate defects were found here, and **every one surfaced by running something — none by reading it.** The fuzzy miner silently missed the `haram` variant (37 occurrences) because it fell below a similarity threshold, and a silent miss looks exactly like a clean result.

## Queued (unblocked, not yet scheduled)

- Layer 2 `word_boost` and Layer 4 `custom_spelling` — deferred to steps 4 and 5 of the plan's build order. Names are the obvious `word_boost` payload, given the engine's zero-percent hit rate on `Khurram` and `Cenay`.
- Items tracked in [`TODOS.md`](TODOS.md) and [`plans/term-normalization.md`](../plans/term-normalization.md).

## Blocked / Waiting On

<!-- What's stuck, and on whom or what -->

<!-- link-doc-refs:start (auto-generated — edit the IDs in prose, not this block) -->
[DEC-004]: DECISIONS.md#dec-004-the-substitution-runs-at-the-pipelinepy180-convergence-point
[DEC-005]: DECISIONS.md#dec-005-one-flat-term-table-the-code-classifies-risk-not-the-author
[DEC-006]: DECISIONS.md#dec-006-every-correction-is-logged-to-logs
[DEC-008]: DECISIONS.md#dec-008-add-term-will-auto-commit-its-one-file--an-exception-to-the-no-auto-commit-rule
<!-- link-doc-refs:end -->
