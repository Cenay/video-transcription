# Next Steps — video-transcription

_Last updated 2026-08-13 12:22 MST by an AI session · transcript: `2fa5b28a-7c93-4f78-8239-fc20e8d6cc8f` — rewrote the handoff around the live-meeting measurement; added the shared-prompt and dict-key gotchas; expanded /add-term into three concrete pieces after confirming it does not exist_

<details>
<summary>📜 <strong>Stamp history</strong> — the 2 previous updates (older ones: <code>history/NEXT_STEPS-stamp-history.md</code>)</summary>

- _Prior: 2026-08-12 19:41 MST by an AI session · transcript: `ce166dfb-eca1-4c5a-b935-71755aed3e98` — rewrote the handoff for step 2 (wiring apply_corrections into pipeline.py:180); added the classifier and possessive gotchas._
- _Prior: 2026-08-12 19:15 MST_

</details>

> Start every session here. This file answers "what do I pick up right now?" —
> details live in `CURRENT_STATUS.md`, decisions in `DECISIONS.md`.

## Where We Left Off

**The term corrector is wired in and live.** Steps 1–3 of the build order are shipped: the transcript pass runs at the convergence point ([DEC-004]), and the analysis stage is protected twice — a generated spelling constraint in `ANALYSIS_PROMPT` plus a post-pass over the returned JSON ([DEC-009]). 75 assertions pass, and a real `--from-cache` run publishes `bookeo_` with zero `bookio` while leaving ordinary English untouched.

**Committed and pushed** as `7e7339f` (2026-08-13). The tree is clean.

## Pick Up Here

**First, read the live-meeting result.** ⚠️ **The one thing not verified is whether the model obeys the spelling constraint** — only that the constraint renders and reaches the API payload. The run against tonight's meeting answers it, and the number to look at is the analysis post-pass report:

- **Silence** (no `⚠️ The ANALYSIS contained wrong terms…` block) → the constraint held. That is the expected and good outcome.
- **A non-empty report** → the constraint did *not* hold, the post-pass caught it, and the terms it lists are ones the model **invented** rather than heard. Worth reading `logs/term-corrections.log` for the `[ANALYSIS — …]` entry and considering whether the constraint wording needs strengthening.

Either way the published page is correct — the post-pass is the net. The residual is a measurement, not a failure.

**Second, record what that run showed** — in [`TODOS.md`](TODOS.md) (the Active item exists for it) and, if the residual is non-zero, as an amendment to [DEC-009]. The code is already shipped, so this step is measurement and bookkeeping, not a build.

**Third — build `/add-term` ([DEC-008]). It does not exist yet.**

⚠️ **It is easy to believe this is already built, and it is not.** The global `~/.claude/CLAUDE.md` describes it in the **present tense** ("The `/add-term` command — and only that command — commits and pushes its single target file automatically"), and that file loads into every session in every repo. Verified 2026-08-13 by `find /home/cenay/.claude /mnt/k/Code/claude-personal-toolkit -iname "*add-term*"` → **no results**. Nothing exists: no command file, no script, no skill prompt.

Three pieces, per [DEC-008]:

1. **`~/.claude/commands/add-term.md`** — the global slash command, callable from any repo.
2. **`claude-personal-toolkit/scripts/add-term.py`** — writes to this repo's `config/terms.yml` **by absolute path**, then commits and pushes **that one file only**. Never `git commit -a`, never a sweep of whatever the calling repo left dirty. **Fails loudly with the path it tried** if the file is absent, rather than creating a fresh one — a silently-created empty term list would apply zero corrections while every run reported success.
3. **A prompt in the `meeting-reconcile` skill** — offer `/add-term` when a reconciliation review corrects a misheard term. This is the piece that actually makes it get used, since that review is where terms are discovered.

**Why this is the highest-value item left.** Terms are found while reviewing a reconciliation in *another* repo (`fran-dash`, `trfaapi.com`). Today that means switching to this repo, hand-editing YAML, and remembering to commit — so in practice the term never gets added and the mishearing recurs in every later meeting. Adding a term should cost one line typed from wherever you are.

⚠️ **When it is built, the present-tense wording in the global `~/.claude/CLAUDE.md` becomes correct rather than aspirational** — worth re-reading that section then to confirm it matches what was actually built.

## Decisions Needed

- **`Nik` vs `nick`** — `nick` appears on 42 lines across 8 transcripts. Every sample read as the person, but it is ordinary English so the classifier refuses it by default. Vet with `./venv/bin/python scripts/preview_corrections.py --all --grep nick` and decide whether to `force:` it. Small; does not block anything.
- **Whether corrections also appear on the Notion page**, in addition to `logs/`. Marginal — you review reconciliations anyway, so the page block would only save a manual correction.

## Watch Out For

⚠️ **The classifier rule in `scripts/terms.py` is more delicate than it looks, and it was wrong on first specification.** A plain dictionary lookup refuses `book io` (because `io` is in `/usr/share/dict/words`); "fixing" that with a minimum token length then wrongly *accepts* `book it` and `book he`, which is the direction that corrupts prose. The verified rule is dictionary lookup for 3+ character tokens **plus** an explicit stop-word set for short ones. Do not simplify it. See [DEC-005] and `tests/test_terms.py`.

⚠️ **Never list a possessive form in `terms.yml`.** `bookio's` as its own variant swallows the apostrophe-s — *"bookio's widget"* becomes *"Bookeo widget"* — because longer variants match first. The bare `bookio` entry already handles it correctly. Regression-tested.

⚠️ **The raw cache at `transcriber.py:59` must stay uncorrected.** Corrections apply downstream only, so the original is always recoverable when a term entry turns out to be wrong. A consequence: a `--from-cache` re-run produces different text than the cache holds. That is by design.

⚠️ **`/add-term` will auto-commit `terms.yml`** when built — a deliberate exception to the no-auto-commit rule, recorded in the global `~/.claude/CLAUDE.md` so a session in another repo does not "fix" it.

⚠️ **`ANALYSIS_PROMPT` has more than one caller.** Adding the `{spelling}` placeholder broke `scripts/diagnose_analysis.py` with `KeyError: 'spelling'` — it formats the shared constant itself. Grep for `ANALYSIS_PROMPT` before touching its placeholders.

⚠️ **`correct_structure()` corrects dict keys as well as values.** Harmless today because keys are fixed schema names, but a term colliding with a schema key would break `notion_output.py`'s lookups. Noted in [DEC-009], deliberately not guarded.

**A pattern from the session worth carrying:** four separate defects were found here, and **every one surfaced by running something — none by reading it.** The fuzzy miner silently missed the `haram` variant (37 occurrences) because it fell below a similarity threshold, and a silent miss looks exactly like a clean result.

## Queued (unblocked, not yet scheduled)

- Layer 2 `word_boost` and Layer 4 `custom_spelling` — deferred to steps 4 and 5 of the plan's build order. Names are the obvious `word_boost` payload, given the engine's zero-percent hit rate on `Khurram` and `Cenay`.
- Items tracked in [`TODOS.md`](TODOS.md) and [`plans/term-normalization.md`](../plans/term-normalization.md).

## Blocked / Waiting On

<!-- What's stuck, and on whom or what -->

<!-- link-doc-refs:start (auto-generated — edit the IDs in prose, not this block) -->
[DEC-004]: DECISIONS.md#dec-004-the-substitution-runs-at-the-pipelinepy180-convergence-point
[DEC-005]: DECISIONS.md#dec-005-one-flat-term-table-the-code-classifies-risk-not-the-author
[DEC-008]: DECISIONS.md#dec-008-add-term-will-auto-commit-its-one-file--an-exception-to-the-no-auto-commit-rule
[DEC-009]: DECISIONS.md#dec-009-the-analysis-stage-is-protected-twice--a-prompt-constraint-and-a-post-pass-and-the-gap-between-them-is-the-measurement
<!-- link-doc-refs:end -->
