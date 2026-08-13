# TODOs — video-transcription

_Last updated 2026-08-12 19:41 MST by an AI session · transcript: `ce166dfb-eca1-4c5a-b935-71755aed3e98` — added five Active items: wire the corrector, decide Nik/nick, protect the analysis stage, build /add-term, Notion-block question._

<details>
<summary>📜 <strong>Stamp history</strong> — the 1 previous update (older ones: <code>history/TODOS-stamp-history.md</code>)</summary>

- _Prior: 2026-07-31 02:19 MST_

</details>

Small quality-of-life polish items for the transcription pipeline. Add new ideas to
**Active** as they come up; move them to **Completed** with a date, time and timezone when shipped.

> Larger workflow/architecture items live in [`planning.md`](planning.md).
> Migrated 2026-07-31 from `todos/qol-improvements.md` (folder retired — see
> [`DECISIONS.md`](DECISIONS.md)).

## Active

### Wire `apply_corrections()` into the pipeline
Step 2 of the term-normalization build order. Add the call at `scripts/pipeline.py:180` (the convergence point of the `--from-cache` and fresh-transcription branches, [DEC-004]), print the corrections report, and write it to `logs/` ([DEC-006]). Verify with a `--from-cache` run over an already-cached meeting: the Notion page should come out with `bookeo_` and zero `bookio`.

### Decide `Nik` vs `nick`
`nick` appears on 42 lines across 8 cached transcripts. Every sample reads as the person, but it is ordinary English so the classifier refuses it by default. Vet with `./venv/bin/python scripts/preview_corrections.py --all --grep nick`, then either add it to `force:` in `config/terms.yml` or leave it documented as deliberately excluded.

### Protect the analysis stage as well as the transcript
The transcript pass does not stop Claude from *constructing* a wrong identifier (`bookio_product_groups`) in the summary — the stage where the original incident actually did its damage. Either apply `apply_corrections()` to the analysis output after `pipeline.py:207`, or inject the term list into `ANALYSIS_PROMPT` (`analyzer.py:24`, a single prompt with one insertion point), or both and use the difference as a signal.

### Build `/add-term` ([DEC-008])
Global slash command + `claude-personal-toolkit/scripts/add-term.py` + a prompt in the `meeting-reconcile` skill. Auto-commits `config/terms.yml` and only that file; fails loudly with the path it tried rather than creating a fresh one. Designed in `plans/term-normalization.md`, not built.

### Decide whether corrections appear on the Notion page
In addition to `logs/`. Marginal — reconciliations are reviewed by a human anyway, so the page block would only save a manual correction rather than prevent a bad decision entry.

### Clean up dead code in `analyzer.py`
Pylance flags two unused items (harmless, pre-existing):
- Line 4: `import os` is unused — remove it.
- `estimate_analysis_cost()` takes a `model` parameter it never uses in the body — either use it (e.g. for per-model pricing) or drop the param.

## Backlog

<!-- Future tasks -->

## Completed

### Remove the "Quotes" section — _2026-06-08_
Dropped the Key Quotes section from the Notion output (`notion_output.py`) and the `key_quotes` sample in the `__main__` test block. Also removed `key_quotes` from the prompt/JSON schema and the "Pull quotes" guideline in `analyzer.py` so we no longer pay tokens to generate quotes we throw away.

### Put the transcript inside a collapsible Heading 3 toggle — _2026-06-08_
Transcript now lives in a Heading 3 toggle (`is_toggleable = true`), hidden by default. Implementation in `notion_output.py`: non-transcript blocks append to the page first, then the toggle heading is created and its id captured, then the transcript paragraphs are appended as `children` of that heading in 100-block batches (handles long transcripts that exceed a single request).

<!-- link-doc-refs:start (auto-generated — edit the IDs in prose, not this block) -->
[DEC-004]: DECISIONS.md#dec-004-the-substitution-runs-at-the-pipelinepy180-convergence-point
[DEC-006]: DECISIONS.md#dec-006-every-correction-is-logged-to-logs
[DEC-008]: DECISIONS.md#dec-008-add-term-will-auto-commit-its-one-file--an-exception-to-the-no-auto-commit-rule
<!-- link-doc-refs:end -->
