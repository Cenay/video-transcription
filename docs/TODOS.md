# TODOs — video-transcription

_Last updated 2026-08-20 11:37 MST by an AI session · transcript: `f0912a53-461b-4861-97e4-931cb2f83ba0` — closed the Notion-corrections item — [DEC-010] shipped the corrections toggle _

<details>
<summary>📜 <strong>Stamp history</strong> — the 3 previous updates (older ones: <code>history/TODOS-stamp-history.md</code>)</summary>

- _Prior: 2026-08-19 15:27 MST by an AI session · transcript: `4325eefc-3918-4756-9846-cdc2fe7683cd` — **new Active item: a `declined:` list, raised by Cenay after `sine` → `Cenay` was proposed and declined for the third or fourth time.** ⛔ **The cost being paid is the ASK, not the write**, so the check belongs where a variant is *proposed*, not only where it is written. ★ **Two kinds of "no" are conflated and only the wrong one persists** — the classifier's refusal is mechanical and fires forever; Cenay's decline is a ruling, made once, and evaporates. ★ **It also gives `Nik` vs `nick` somewhere to record "deliberately excluded"**, an option that entry already offers with nowhere to write it. Added from a `fran-dash` meeting reconciliation, which is where the gap is felt._
- _Prior: 2026-08-14 00:28 MST by an AI session · transcript: `4c61a822-47ec-4195-b344-607007d9c624` — moved /add-term and the live-model confirmation to Completed; both shipped 2026-08-13_
- _Prior: 2026-08-13 12:22 MST by an AI session · transcript: `2fa5b28a-7c93-4f78-8239-fc20e8d6cc8f` — moved the wire-in and analysis-stage items to Completed; added the live-model confirmation item; flagged /add-term as verified-not-built_

</details>

Small quality-of-life polish items for the transcription pipeline. Add new ideas to
**Active** as they come up; move them to **Completed** with a date, time and timezone when shipped.

> Larger workflow/architecture items live in [`planning.md`](planning.md).
> Migrated 2026-07-31 from `todos/qol-improvements.md` (folder retired — see
> [`DECISIONS.md`](DECISIONS.md)).

## Active

### ⛔ Fix the wordlist dependency — a missing `/usr/share/dict/words` silently disables the risky-variant guard
**Found 2026-08-21 while scoping the Windows port. This is a live latent bug on Linux too, not a Windows-only concern.** Full write-up: `plans/port-transcription-to-windows-brief.md` → §1.

`scripts/terms.py:30` hardcodes `WORDLIST_PATH = Path("/usr/share/dict/words")`, and `_load_wordlist()` (`terms.py:42–46`) **returns an empty set when the file is missing** instead of raising. Empty `_WORDS` → `is_ordinary_english()` (`:52–59`) is always `False` for tokens over 2 chars → `is_risky()` (`:62–69`) is always `False`. **The guard that refuses substitutions which would corrupt ordinary prose never fires**, so every variant is treated as safe.

✅ **Verified by mutation test 2026-08-21** — `terms.py` imported, `WORDLIST_PATH` repointed at a nonexistent path, `_WORDS` reloaded, `is_risky()` compared across both states:

| Variant | with wordlist (102,485 words) | without (0 words) |
|---|---|---|
| `nick` | `True` refused | **`False` allowed** |
| `art` | `True` refused | **`False` allowed** |
| `make` | `True` refused | **`False` allowed** |

Those are exactly the roster terms the guard protects — `nick` → **Nik**, **Art** ("too short and ordinary to auto-correct"), `make` a **Jake** near-miss. It would rewrite ordinary English into teammate names, then feed that into `meeting-reconcile` → `DECISIONS.md`.

**The fix, three parts:**
1. Ship `config/words.txt` in-repo instead of depending on the OS — removes the platform dependency and makes behavior identical everywhere.
2. Make a missing wordlist **loud** — raise, or refuse all risky substitutions rather than allowing all of them. Same principle as `corrections=[]` vs `corrections=None` in [DEC-010]: silence must not mean both "nothing there" and "nobody looked".
3. Add a negative test — assert `is_risky("nick") is True` with the list present, and assert the missing-list path fails loudly. A guard not shown to fail on broken input has not been tested.

⚠️ **Not checked:** whether this changes published output end-to-end. The unit-level inversion is proven; the path from `terms.yml` through `preview_corrections.py` to a real transcript was not traced. Confirm before calling the fix done.

### Fix three hardcoded `/tmp` cache paths
`transcriber.py:42`, `diagnose_analysis.py:17`, `repair_global_options.py:22` all default to `/tmp` where `pipeline.py:86` correctly uses `tempfile.gettempdir()`. ⚠️ **Consequence if missed:** the cache directory diverges from the one `pipeline.py` writes, so `--from-cache` silently misses and re-transcribes at full cost. One-line fix each.

### Make `pipeline.py` emit the Notion page ID instead of printing it for re-parsing
`pipeline.py:337` already sets `result["notion_page_id"]` and then discards it; `transcribe-this.sh:138` scrapes it back out of stdout with GNU-only `grep -oP 'notion\.so/\K[a-f0-9]+'`. Emitting it properly (a `--print-page-id` flag or a JSON line) removes the GNU dependency **and** the fragile stdout coupling. Prerequisite for any wrapper rewrite.

### Fix the `python -c` quoting hazard in the wrapper
`transcribe-this.sh:174–178` interpolates `$S3_URL` straight into a `python -c` string. `$NOTION_PAGE_ID` is grep-extracted hex so it is safe, but `$S3_URL` derives from the filename — an apostrophe in a filename breaks the quoting. ⚠️ **Believed from reading, not tested.** Disappears for free in any wrapper rewrite.

### Commit the desktop integration into the repo
✅ **Verified 2026-08-21** by `ls ~/.local/share/nautilus/scripts/` — **"Transcribe This"** and **"Transcribe This --no-cleanup"** exist entirely outside version control. A machine rebuild loses them silently. Commit them as installable assets (relevant to the Windows port, but worth doing regardless).

### Build step 4 — Layer 2 `word_boost` at transcription
The next real build in `plans/term-normalization.md`, and the strongest remaining lever: the engine has a **zero-percent** hit rate on `Khurram` and `Cenay` across 83 meetings, and names are exactly what `word_boost` is for. Wire-in: `scripts/transcriber.py`, the `aai.TranscriptionConfig(...)` call in `transcribe_audio()` — add `word_boost=TERMS.boost_list()` and `boost_param=aai.WordBoost.high`. ⚠️ **Cannot be verified `--from-cache`** — it changes what AssemblyAI returns, so it costs a real transcription run to measure.

### Build step 5 — Layer 4 `custom_spelling` (only if 1–4 leave residue)
Deliberately last, and possibly never. Measured on the 2026-07-30 meeting it would have caught **1 occurrence in the transcript and 0 in the summary**, because AssemblyAI applies it to its own output and it never sees the LLM stage.

### Build a `declined:` list — so a variant Cenay has ruled against is never proposed again
**Raised by Cenay 2026-08-19**, after `sine` → `Cenay` came back for the **third or fourth** time: *"We should stop trying it if we've ruled it."*

**The loop as it stands.** A reconciliation session in another repo hits the mishearing, offers `/add-term Cenay sine`, the classifier refuses it (all-ordinary-words — correct), the session relays the refusal, Cenay declines. **Nothing anywhere records that she declined.** The next session repeats all four steps, because from its point of view this is a fresh mishearing it just spotted.

⛔ **The cost being paid is the ASK, not the write** — so a guard that only fires when something is written is the wrong guard. **The check has to happen where the variant is PROPOSED**, which is `/add-term`'s offer step in the reconcile skills, not just inside `add-term.py`.

★ **Two different kinds of "no" are being conflated, and only one of them is recorded.** The classifier's refusal is **mechanical and stateless** — `sine` is ordinary English, and it will refuse it identically forever. Cenay's decline is a **ruling**, made once, and it is the one that should persist. Today the mechanical one fires every time and the human one evaporates.

⚠️ **The place to record it ALREADY EXISTS — and that is the sharpest fact here.** ✅ `config/terms.yml` line 159 carries a **`DELIBERATELY NOT INCLUDED`** block, holding exactly this reasoning for `Nik`/`nick`, `Jake`, `Art`/`Arthur`, `TRFA` and others, and `add-term.py` deliberately round-trips it rather than destroying it. ⛔ **But NOTHING READS IT** — ✅ verified: the only two hits in `add-term.py` are about *preserving comments*, not consulting them. **It is a comment block, so a decline written there is invisible to the next session.**

★ **So this is not "invent a place to record declines" — it is "make the place that already exists machine-readable."** A much smaller job, and it explains why the loop survives despite the block being there: `sine` could have been written into that comment today and the next session would still have proposed it.

**Sketch (not a spec — decide the shape when building):**
- Promote the block to real YAML — a `declined:` key per term, each entry carrying the variant, the date, and one line of reason. ⛔ **Migrate the existing five entries rather than starting fresh**; they are measured findings (`Jake`'s near-misses, `TRFA`'s 169 correct occurrences) and losing them to a rewrite would be the whole point of the block, undone.
- `add-term.py` checks it **first** and exits **0** with *"`sine` was declined for `Cenay` on 2026-08-19 — not re-proposing"*, distinct from the exit-2 refusal. ⚠️ **Exit 0, not 2**: this is a satisfied precondition, not an error, and a caller should not treat it as a failure.
- `--force` still overrides, since a ruling can be revisited — but it should say it is overriding a recorded decline, not a classifier refusal.
- The reconcile skills read it before offering, which is what actually stops the question reaching her.

★ **This also gives the `Nik` vs `nick` item below its ending.** That entry offers *"or leave it documented as deliberately excluded"* — and it **is** so documented, in the comment block. ⛔ **Documenting it there did not stop it being re-raised**, which is precisely the evidence that the block needs to be data rather than prose.

**Seed it with what is already ruled:** `sine` → `Cenay` (2026-08-19). ⚠️ **Candidates, not yet ruled — ask before seeding:** `fever` → `Fiverr`, `book you` / `bookie oh` → `Bookeo` (noted 2026-08-19; all three are all-ordinary-words and would be refused anyway, but that is the refusal firing, not a decline being remembered).

### Decide `Nik` vs `nick`
`nick` appears on 42 lines across 8 cached transcripts. Every sample reads as the person, but it is ordinary English so the classifier refuses it by default. Vet with `./venv/bin/python scripts/preview_corrections.py --all --grep nick`, then either add it to `force:` in `config/terms.yml` or leave it documented as deliberately excluded.

### Clean up dead code in `analyzer.py`
Pylance flags two unused items (harmless, pre-existing):
- Line 4: `import os` is unused — remove it.
- `estimate_analysis_cost()` takes a `model` parameter it never uses in the body — either use it (e.g. for per-model pricing) or drop the param.

## Backlog

<!-- Future tasks -->

## Completed

### Decide whether corrections appear on the Notion page — _2026-08-20 11:37 MST_
✅ **Ruled in and built** ([DEC-010]). Cenay: *"place the corrections in a similar toggle as the transcript (can live right below it) that states the correction."* `notion_output.build_correction_blocks()` renders a collapsed `🔤 Term corrections applied by the pipeline` toggle immediately below the transcript toggle, carrying both the transcript pass and the analysis residual. ⚠️ **An empty run still renders the toggle** saying "none applied" — `None` (the corrector never ran) renders nothing, so silence cannot mean both "nothing there" and "nobody looked". Verified by `./venv/bin/python tests/test_notion_corrections.py` → 16 assertions, 0 failures; suppressing the toggle turns 3 red and collapsing the analysis grouping turns 2 red, so the suite is proven able to fail. ⚠️ Not yet seen rendered on a real Notion page.

### Confirm the prompt constraint holds against a live model — _2026-08-13 23:13 MST_
✅ **It held.** The unverified half of [DEC-009] is now measured against real runs, not reasoning. Two live meetings went through the full pipeline on 2026-08-13 (`trfa-tamp-new-class-bookeo` at 10:08, `trfaapi-deletes-new-bookeo-classes` at 23:00), applying 24 transcript-stage corrections between them — and **`logs/term-corrections.log` contains zero `[ANALYSIS — …]` entries**, i.e. the post-pass found nothing to repair either time. Since the transcript pass runs first, the analysis input was already clean, so a residual could only have come from the model *inventing* a wrong term. It invented none. Verified by `grep -c "ANALYSIS" logs/term-corrections.log` → `0`.

### Build `/add-term` ([DEC-008]) — _2026-08-13, built and in production use_
All three designed pieces exist, plus a fourth nobody planned:
- `~/.claude/commands/add-term.md` — the global slash command (source of truth: `claude-personal-toolkit/commands/add-term.md`).
- `claude-personal-toolkit/scripts/add-term.py` — 450 lines. Writes `config/terms.yml` by absolute path (`DEFAULT_TERMS_YML`, overridable via `TERMS_YML`), fails loudly with the path it tried rather than creating a fresh file, backs the file up before writing and **rolls back** if the edited list fails to reload, then commits with an explicit pathspec — never `-a` — and pushes. `--dry-run`, `--no-commit`, `--no-push` escape hatches.
- A prompt in the `meeting-reconcile` skill (`claude-personal-toolkit/skills/meeting-reconcile/SKILL.md`).
- ➕ `claude-personal-toolkit/scripts/test-add-term.py` — not in the design.

**Proven in production the same day:** five `chore(terms):` commits (`5cf3fbd`…`1f2c716`) landed two new terms (Trainual, Ninthroot) and three additions to existing ones (Bookeo `bukio`/`buckio`, Cenay `Sinead`/`Sanay`, Milos `melos`), each committed and pushed automatically, each touching only `terms.yml`.

⚠️ **This item sat in Active reading "NOT BUILT" after it had shipped** — the 2026-08-13 `find` that "verified" its absence was run before the build and never re-run. See [`LESSONS_LEARNED.md`](LESSONS_LEARNED.md).

### Wire `apply_corrections()` into the pipeline — _2026-08-13 12:22 MST_
Step 2 of the term-normalization build order, at the `--from-cache`/fresh convergence point ([DEC-004]). Prints the report and appends to `logs/term-corrections.log` ([DEC-006]) — one cumulative file, since the question it answers ("which meetings did a bad term entry touch?") is a grep across runs. Verified by running: a real `--from-cache` pass over a cached 126-minute meeting produced **0** `bookio` (was 28), 0 `bookio_`, `Cenay`/`Khurram` restored, `booking` and `book it` untouched, and the raw cache unmodified.

### Protect the analysis stage as well as the transcript — _2026-08-13 12:22 MST_
Built **both** defenses per [DEC-009]: `spelling_constraint()` renders `config/terms.yml` into `ANALYSIS_PROMPT` (`analyzer.py:24`), and `correct_structure()` walks the returned JSON in `pipeline.py` after the error guard. The post-pass should normally find nothing — a non-empty result is a signal that the model *invented* a term, logged tagged `[ANALYSIS — prompt constraint missed these]`. 19 new assertions (75 total, 0 failures), plus a stubbed-Claude integration run proving the seam fires on the real path. ⚠️ The prompt half is still unverified against a live model — that is the remaining Active item.

### Remove the "Quotes" section — _2026-06-08_
Dropped the Key Quotes section from the Notion output (`notion_output.py`) and the `key_quotes` sample in the `__main__` test block. Also removed `key_quotes` from the prompt/JSON schema and the "Pull quotes" guideline in `analyzer.py` so we no longer pay tokens to generate quotes we throw away.

### Put the transcript inside a collapsible Heading 3 toggle — _2026-06-08_
Transcript now lives in a Heading 3 toggle (`is_toggleable = true`), hidden by default. Implementation in `notion_output.py`: non-transcript blocks append to the page first, then the toggle heading is created and its id captured, then the transcript paragraphs are appended as `children` of that heading in 100-block batches (handles long transcripts that exceed a single request).

<!-- link-doc-refs:start (auto-generated — edit the IDs in prose, not this block) -->
[DEC-004]: DECISIONS.md#dec-004-the-substitution-runs-at-the-pipelinepy180-convergence-point
[DEC-006]: DECISIONS.md#dec-006-every-correction-is-logged-to-logs
[DEC-008]: DECISIONS.md#dec-008-add-term-will-auto-commit-its-one-file--an-exception-to-the-no-auto-commit-rule
[DEC-009]: DECISIONS.md#dec-009-the-analysis-stage-is-protected-twice--a-prompt-constraint-and-a-post-pass-and-the-gap-between-them-is-the-measurement
[DEC-010]: DECISIONS.md#dec-010-the-corrections-also-go-on-the-notion-page-in-a-toggle-directly-under-the-transcript
<!-- link-doc-refs:end -->
