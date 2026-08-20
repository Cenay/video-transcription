# Lessons Learned

_Last updated 2026-08-20 12:12 MST by an AI session · transcript: `f0912a53-461b-4861-97e4-931cb2f83ba0` — a hand-carved exception inside the corpus guard had switched the check off for two words; the allowance is now computed_

<details>
<summary>📜 <strong>Stamp history</strong> — the 3 previous updates (older ones: <code>history/LESSONS_LEARNED-stamp-history.md</code>)</summary>

- _Prior: 2026-08-14 00:28 MST by an AI session · transcript: `4c61a822-47ec-4195-b344-607007d9c624` — added the stale-NOT-BUILT-warning lesson and the pinned-count test lesson_
- _Prior: 2026-08-13 12:22 MST by an AI session · transcript: `2fa5b28a-7c93-4f78-8239-fc20e8d6cc8f` — added the 2026-08-13 entry: shared prompt constants have multiple callers, stub the expensive call not the path, verification runs corrupt an audit log_
- _Prior: 2026-08-12 19:41 MST by an AI session · transcript: `ce166dfb-eca1-4c5a-b935-71755aed3e98` — added three lessons: four defects all found by running not reading, the engine's total failure on non-English names, and the one-meeting generalisation that produced a wrong plan finding._

</details>

Running log of non-obvious failures and how we diagnosed/fixed them. Append new entries at the top.

---

## 2026-08-20 — A guard's exception was hand-carved, so it silently switched the guard off

### Gotcha: `if word in (...): continue` inside a checker deletes coverage, it does not narrow it

The corpus sweep in `tests/test_terms.py` asserts that no ordinary-English word is altered across every cached transcript — the check that makes the whole corrector trustworthy ([DEC-005]). It carried one hand-written exception: *"`active campaign` is intentionally consumed by the forced rule"*, implemented as `if word in ("active", "campaign"): continue`.

**That `continue` does not narrow the check for those two words — it removes them from the sweep entirely.** Any corruption of `active` or `campaign`, from any term, by any mechanism, was invisible. ✅ Proven by mutation: forcing a bare `active` → `ActiveCampaign` (which destroys the word in 14 places in one transcript alone) passed the old sweep silently.

**How it surfaced:** a *different* variant, `bookie o` → `Bookeo`, added by `/add-term` on 2026-08-18, started eating one `bookie` in two transcripts and turned the sweep red. The tempting fix was the same shape as the existing one — add `bookie` to the skip list — which would have blinded the guard to a third word.

**The substitution was correct and the test was wrong.** ✅ Verified by reading every `bookie` in both transcripts: *"the sync is one way from bookie O to zero"*, *"if we can get data from Bookie Ota"*, *"we will no longer have to put up with Bookie O in the future"*. All the company, never a bookmaker.

**Fix — compute the allowance instead of carving an exception.** `expected_drop(word, changes)` derives how many occurrences a **strictly longer** variant was entitled to swallow, straight from the substitutions `apply_corrections()` reports, and the sweep demands the drop equal that number exactly. The hand-written skip is gone, so `active` and `campaign` are genuinely checked for the first time.

**Two properties worth keeping:**

- **It is derived from the real function's own output**, not a re-implementation of its matching, so it cannot drift when `apply_corrections()` changes.
- **A variant no longer than the dangerous phrase is never "expected."** A bare forced `bookie` or `booking` still fails — ✅ mutation-tested, `allowed 0` against 33 and 255 real occurrences. That also means forcing a bare ordinary word (the open `Nik` vs `nick` question) will turn this red on purpose: it is a decision that should have to edit the `DANGEROUS` list by hand rather than slip through.

**Practice:** when a guard fires on a case you believe is legitimate, ⛔ do not exempt the *input*. Work out what the correct amount of change is and assert that instead. An exemption is indistinguishable from the guard being deleted, and it reads in the diff as diligence.

---

## 2026-08-13 (later) — A doc's "verified" warning outlived the thing it warned about

### Workflow: a ⚠️ NOT BUILT claim needs a re-check date, not just a verification date

`TODOS.md` and `NEXT_STEPS.md` carried a loud, well-evidenced warning that `/add-term` did not exist — quoting the exact `find` that returned nothing. `/add-term` was then built the same day, and the warning stayed. A `/resume` briefing read those docs and reported the tool as unbuilt to the person who had just built it, listing it as the highest-value next task.

**What made it stick:** the warning was *more* convincing than an ordinary note, because it named its evidence. Nothing about it signalled that the evidence had an expiry. The `find` was true when run and false an hour later.

**The tell that was available and missed:** `/add-term` was sitting in the session's own available-skills list, one line, with an accurate description. A tool's presence in the live tool roster is current state; a doc is a claim about the past. **When those disagree, the roster wins** — and the disagreement is worth checking *before* briefing off the doc, not after being corrected.

**Practice:** a "does not exist" claim in a doc gets phrased against a cheap re-check (`find …` / `ls …` in backticks so the next reader can just run it), never as settled fact. And any briefing that is about to recommend *building* something should first check whether it already exists — the check costs one command.

### Gotcha: a test that pins a COUNT breaks on data changes made from another repo

`tests/test_terms.py` asserted `len(forced) == 2` — exactly two terms may use `force:`. Reasonable when written; both forced entries were justified by corpus measurement. Then `/add-term` forced `ninth root` from a session in a different repo, and the suite went red for a **data** change, failed by someone who will never run this suite and would not know what to do about it.

**The fix is not a bigger number.** The count was a proxy for "don't sprinkle `force:` around." Replaced with the property actually worth holding: every forced variant must be one the classifier would genuinely refuse, so `force:` can never be decorative. The real safety net was already there and passing — the 84-transcript corpus sweep asserting no ordinary-English word is altered anywhere.

**Negative-tested before trusting it**, per the standing rule: adding a needlessly-forced variant makes it fail with `Ninthroot forces only variants the classifier refuses`. A passing checker that has never failed is indistinguishable from a broken one.

⚠️ **`pytest` is not installed in this repo's `venv`** — `python -m pytest tests/test_terms.py` dies with `No module named pytest`, which reads like a broken suite. It is a plain script: `./venv/bin/python tests/test_terms.py --corpus`. **Without `--corpus` the sweep silently skips**, and that sweep is the one check standing between a bad term entry and corrupted prose across every meeting.

---

## 2026-08-13 — Wiring the corrector in: a shared prompt constant has more than one caller

### Gotcha: adding a `.format()` placeholder to a shared prompt breaks every other caller

**Symptom:** adding `{spelling}` to `ANALYSIS_PROMPT` (`analyzer.py:24`) and passing it from `analyze_transcript()` looked complete — the pipeline ran fine. But `scripts/diagnose_analysis.py` imports that same constant and formats it *itself*, with only `transcript=`. It would have died with `KeyError: 'spelling'` the next time anyone diagnosed a failed analysis — which is precisely the moment you least want a second failure.

**Why it was caught:** `grep -rn "ANALYSIS_PROMPT" scripts/ tests/` before declaring the change done. Nothing about the pipeline run would have surfaced it, because the broken caller is a separate diagnostic script that only runs after an incident.

**The general rule:** a prompt constant exported from a module is an interface, and changing its `.format()` keys is a **breaking interface change**. Grep for every caller before touching placeholders. Fixed in the same change.

### Verification: to test a seam that sits after an expensive API call, stub the call — don't skip the test

The analysis correction pass runs on the output of `analyze_transcript()`, so exercising it "properly" means paying for a Claude call. The cheap wrong move is to unit-test the function and *assume* the pipeline calls it correctly — which tests the part that was never in doubt.

**What was done instead:** a throwaway harness monkeypatched `pipeline.analyze_transcript` to return a **deliberately poisoned** analysis (`bookio_product_groups`, `karam`, `senay` — terms the transcript never contained) and `pipeline.create_meeting_page` to capture its argument, then ran the **real** `process_video()` over a cached transcript. That proved what the unit tests could not: the seam fires on the real path, in the right order relative to the error guard, and the corrected object is what reaches Notion. Cost: nothing.

**The general rule:** when a code path is guarded by an expensive call, stub the call and run the real path. Stubbing the *dependency* keeps the test honest; stubbing the *path* does not.

### Workflow: a log seeded by verification runs is a corrupted audit trail

`logs/term-corrections.log` exists to answer "which meetings did a bad term entry touch?". After verification it contained four entries — all synthetic, one of them from the poisoned-analysis harness, i.e. **fabricated data in an audit file**. Archived to `.archived/2026-08-12/term-corrections-VERIFICATION-RUNS.log` rather than deleted, so the real log starts empty. A verification artifact that looks exactly like a production record is worse than no record.

---

## 2026-08-12 — Four defects in one session, every one found by running, none by reading

### Workflow: a claim verified only against examples you invented proves nothing

**Date:** 2026-08-12 19:15 MST
**Context:** Building the term corrector — designing a rule to decide which transcription mishearings are safe to auto-replace.
**Problem:** Four separate defects shipped into design or code, and each survived review:

1. **The classifier rule was wrong as specified.** "Refuse a variant whose every token is ordinary English" refuses `book io` — because `io` really is in `/usr/share/dict/words`. The obvious repair (require tokens of 3+ characters) is *worse*: it then accepts `book it` and `book he`, trading one false refusal for two false acceptances in the direction that corrupts prose.
2. **The seeded term list swallowed possessives.** Listing `bookio's` as its own variant beats the bare `bookio` under longest-first matching, so *"bookio's widget"* became *"Bookeo widget"*. The corrections log would have printed `14x 'bookio's' -> Bookeo` and looked perfectly healthy.
3. **The automated miner silently missed a variant.** `haram` (37 occurrences) scored below the fuzzy-match threshold against "Khurram" and never appeared in the seed. It surfaced only when a preview printed a real line: *"And Haram Karam has had to do that before"* — the engine mangling one name two ways in a single sentence.
4. **A documented workflow the tool could not perform.** `docs/new-term-testing.md` step 3 said to vet a candidate word with `--grep`, but `--grep` filtered to lines the term list *already changed* — so vetting a word not yet in the list returned nothing. A second bug in the same code path reported 3 matching lines where there were 42.

**Solution:** Run every claim against data nobody authored for the purpose. Specifically: mine the 83 real cached transcripts rather than reasoning about them; execute the classifier against `/usr/share/dict/words` before describing its behaviour; run each command in a doc before shipping the doc; and mutation-test the test suite itself (disabling the guard must turn assertions red — it turns 13 red here).
**Why:** Every one of these defects was invisible to review and obvious on execution. The pattern is not carelessness — the rules *sounded* right, which is exactly why reading them again could not help. **A silent miss looks identical to a clean result**, whether it comes from a similarity threshold, a regex that matched nothing, or a checker that never ran.

### Tool: the speech engine fails completely on names outside English morphology

**Date:** 2026-08-12 19:15 MST
**Context:** Frequency analysis across 83 cached AssemblyAI transcripts (27,014 utterances).
**Problem:** `Khurram` appears **zero** times — it arrives as `karam` (69) or `haram` (37). `Cenay` appears **zero** times — it arrives as `senay` (13), `cinay`, `cnay`, `cna`. Meanwhile `Arthur` is correct 97 times, `Milos` 104, `Laravel` 198, `TRFA` 169. Every meeting note naming either person has been wrong, in a repo full of records attributing work to people.
**Solution:** Term corrections for the names (`config/terms.yml`), and names are the obvious payload for AssemblyAI's `word_boost` when Layer 2 is built — boosting only shifts probability and cannot corrupt text.
**Why:** The engine's language model is strong on English morphology and has nothing to anchor a name it has never seen. The failure is total rather than occasional, which is *good* news for a term map: a consistently wrong token is trivially correctable, whereas the "mush" the original plan predicted would not have been.

### Architecture: generalising from one meeting produced a wrong plan finding

**Date:** 2026-08-12 19:15 MST
**Context:** `plans/term-normalization.md` was written 2026-07-31 from the 2026-07-30 meeting.
**Problem:** It concluded *"the speech engine does not produce a consistent wrong token — it produces mush"*, and ranked the deterministic term map third on that basis. True of that one meeting (1 `bookio` in the transcript). **False across the archive:** `bookio` appears **361 times**.
**Solution:** The corpus measurement is now recorded *above* the section it revises, rather than the stale text being quietly edited. Layer 3's justification no longer rests on "testable for free" alone.
**Why:** Scope of test = scope of claim. A finding drawn from one sample described as a property of the engine will be wrong at exactly the moment someone relies on it. The counter-discipline is cheap here: 83 transcripts sat in `temp/transcribe-cache/` the whole time.

---

## 2026-06-09 — Notion page published with empty summary ("No summary available")

### Symptom
A `transcribe-this` run produced a Notion page with **only** a "Summary → No summary available." line. Overview, Notes, Keywords, Action Items, and Key Decisions were all missing. The page was still marked Complete and was uploaded to S3.

### Root cause
It was **not** a code regression (yesterday's collapsible-transcript commit was innocent) and **not** a `max_tokens` truncation. On that one run, Claude returned text that failed `json.loads()`. The old `analyzer.py` caught the error and returned `{"error", "raw_response"}`. Then `notion_output.py` silently dropped every section gated by `if items:` — only `summary` had a fallback default (`"No summary available."`), so that lone line survived. The pipeline never noticed and published the empty page anyway.

The trigger was a **non-deterministic bad JSON response** from the model. Replaying the identical cached transcript through the same code parsed perfectly the second time.

### Fixes applied
- `analyzer.py`: `analyze_transcript()` now **retries** (default 3 attempts) on `JSONDecodeError`, nudging the model to return JSON-only; added `_extract_json()` to strip fences/stray prose; bumped `max_tokens` to 8192 and warns loudly on `stop_reason == "max_tokens"`. Only returns the `error` marker after all attempts fail.
- `pipeline.py`: if analysis still contains `"error"`, **aborts before Notion/S3**, saves the raw response to `…-FAILED-ANALYSIS.txt`, and prints a loud message. Never publishes an empty page.
- `notion_output.py`: refactored body-building into `build_meeting_blocks()` + `_append_body()`, shared by `create_meeting_page()` and the new `repair_meeting_page()` (rebuilds an existing page in place, preserving its S3 Meeting Link).

### How to diagnose this again (cheap — no re-transcription)
The raw transcript is cached at `$TEMP_DIR/transcribe-cache/<stem>-raw-transcript.json`, so you can replay analysis for ~$0.09 without paying the AssemblyAI cost again.

```bash
source venv/bin/activate
# Replay analysis on a cached transcript; prints stop_reason, token counts,
# parse result, and saves the raw model reply to <stem>-RAW-ANALYSIS.txt.
python scripts/diagnose_analysis.py <stem>
# e.g. trfa-global-options-status-ninthroot-issue
```

Read `stop_reason` (truncation = `max_tokens`) and the `PARSE` line. The saved `-RAW-ANALYSIS.txt` shows exactly what the model returned.

### How to repair a published page from a recovered analysis (no new Claude call)
If `diagnose_analysis.py` produced a valid `-RAW-ANALYSIS.txt`, rebuild the page body in place:

```python
from notion_output import repair_meeting_page
# repair_meeting_page(page_id, date, duration_min, analysis, transcript, costs)
# Deletes the existing body, rebuilds all sections, re-applies the S3 link.
```

`scripts/repair_global_options.py` is a worked example (finds the page by title via `data_sources.query`, loads the recovered analysis, calls `repair_meeting_page`). Copy and adjust `STEM`/`TITLE`/`DATE`/`COSTS` for a different page.

### Gotcha: notion-client 3.0.0 query API
`notion.databases.query(...)` no longer exists in notion-client 3.0.0. Query a data source instead:
```python
ds_id = notion.databases.retrieve(db_id)["data_sources"][0]["id"]
notion.data_sources.query(data_source_id=ds_id, filter={...})
```
(`pages.create(parent={"database_id": ...})` still works.)
