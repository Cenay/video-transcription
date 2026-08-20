# Plan — domain term normalization ("when you hear XXX, write YYY")

_Last updated 2026-08-20 11:37 MST by an AI session · transcript: `f0912a53-461b-4861-97e4-931cb2f83ba0` — landed the [DEC-010] ruling — corrections also go on the Notion page; US spelling of 'artifact' _

<details>
<summary>📜 <strong>Stamp history</strong> — the 2 previous updates (older ones: <code>history/term-normalization-stamp-history.md</code>)</summary>

- _Prior: 2026-08-12 17:29 MST by an AI session · transcript: `ce166dfb-eca1-4c5a-b935-71755aed3e98` — landed six session-desk rulings (terms.yml location, the pipeline.py:180 seam, logging, one-prompt analyzer, fidelity non-constraint) and added a corpus measurement over 83 cached transcripts that revises the 'no consistent wrong token' finding_
- _Prior: 2026-07-31 02:03 MST by an AI session · transcript: `d3cd0438-54be-4e0f-bbef-9e2cc664c92d`_

</details>

## Why this exists

The 2026-07-30 TRFA API meeting notes rendered the Bookeo table prefix as **`bookio_`** — ten times, in the Overview, a Notes heading, and a Key Decision. That is a table name in a production schema. It was caught only because a reconciliation session cross-checked it against the repo, where `bookeo_` appears 49 times and `bookio` appears zero. Adopting the notes verbatim would have written a wrong table name into `fran-dash/docs/DECISIONS.md`, corrupting [DEC-154], which had closed the day before.

The correction is now permanent on the reading side (a session-memory rule normalizes `bookio` → `bookeo` on sight). This plan fixes the **writing** side, so the bad token is never generated.

## ★ Corpus measurement 2026-08-12 — this REVISES the section below

⚠️ **The section that follows generalised from one meeting, and the generalisation does not hold.** Its claim — *"the speech engine does not produce a consistent wrong token; it produces mush"* — is true of the 2026-07-30 meeting and **false across the archive**. Measured by regex frequency count over **83 cached raw transcripts** in `temp/transcribe-cache/`:

| form | count | character |
|---|---:|---|
| `bookio` | **361** | non-word — a highly consistent wrong token |
| `booking` | 259 | ⚠️ ordinary English, usually legitimate |
| `bookeo` | 177 | already correct |
| `book you` | 99 | ⚠️ ordinary English |
| `booked` | 73 | ⚠️ ordinary English |
| `bookings` | 65 | ⚠️ ordinary English |
| `book it` | 39 | ⚠️ ordinary English |
| `bookie` | 31 | ⚠️ ordinary English word |
| `book here` | 29 | ⚠️ ordinary English |
| `booko` / `booku` / `bookq` | 9 / 7 / 4 | non-words |

**What this changes:**

- **Layer 3 rises in value.** `bookio` alone is 361 occurrences of an unambiguous non-word — the highest-value, zero-risk substitution available. The deterministic pass is not merely the cheap testable starting point; it does the largest share of the work. **The build order below is unchanged, but it no longer rests on the "testable for free" argument alone.**
- **Layer 1 remains necessary**, for the reason the original section gives: it is the only thing that reaches identifiers the model *constructs* (`bookio_product_groups`).
- **`booking` at 259 legitimate occurrences is the hazard to design against.** A naive replace-every-variant map would corrupt ordinary English at scale — the same failure the original incident caused, from the opposite direction.

**Method note, because this is the second time this document has been bitten by it:** the original section generalised from one meeting; this table was produced by running the claim against **83 transcripts nobody authored for the purpose**. Any future claim here about what the engine "does" gets the same treatment before it is written down.

## What the evidence actually showed — read this before choosing an approach

⚠️ **Scope: the 2026-07-30 meeting only.** Read the corpus measurement above first — it revises the "no consistent wrong token" conclusion.

The obvious fix is a "misheard → correct" spelling map. **On this data it would have fixed almost nothing.** Counts from the 2026-07-30 meeting page:

| Source | `bookio` | `bookeo` | What it really contained |
|---|---:|---:|---|
| Raw transcript (AssemblyAI, 32.7 KB) | 1 | 1 | "book you", "book your", "bookie", "book. You", "booking" |
| LLM summary / distillation (16.8 KB) | 10 | 12 | — |

Representative transcript lines:

> "Just get rid of your **bookie** underscore stuff."
> "Drop **book you** underscore prefix from your tables."
> "So now your table should be **book. You** underscore all stuff."
> "Categories, **booking**, underscore schedules."

**The speech engine does not produce a consistent wrong token — it produces mush.** The LLM summarizer then resolves that mush into a single confident, wrong identifier (`bookio_`) and repeats it. So:

- The error is **minted at the summarization step**, not the transcription step.
- A `custom_spelling` map keyed on `"bookio"` would have caught **1 occurrence in the transcript and 0 in the summary**, because `custom_spelling` is applied by AssemblyAI to *its own* output and never sees the LLM stage at all.
- Therefore the highest-value intervention is the **analyzer prompt**, not the transcription config.

This is the whole reason to build it in this order.

## The four layers, in value order

### Layer 1 — Glossary in the analyzer prompt ★ highest value

Pass the domain term list into the summarizing LLM's prompt as an explicit spelling constraint, e.g.:

> These domain terms appear in this meeting. Spell them exactly as written, including case and underscores, regardless of how they were transcribed: `Bookeo`, `bookeo_*` (table prefix), `site_*`, `TRFA`, `ActiveCampaign`, `fran-dash`, `Khurram`, `Eloquent`, `Blade`, `Laravel`, `Bookeo widget`. If the transcript contains a garbled form ("book you", "bookie", "book io"), it refers to Bookeo.

This attacks the step that actually produced the error, and it also fixes the class of error `custom_spelling` structurally cannot reach: identifiers the LLM *constructs* (`bookio_product_groups`) that were never spoken as a single token.

**Wire-in:** `scripts/analyzer.py` — ✅ confirmed 2026-08-12: a single `ANALYSIS_PROMPT` at `:24`, formatted at `:124`, one `client.messages.create` at `:132`. One insertion point, no per-section prompts to chase.

### Layer 2 — `word_boost` at transcription

Bias recognition toward the domain vocabulary so "Bookeo" wins over "book you" in the first place. Verified present in the installed SDK:

- `venv/lib/python3.12/site-packages/assemblyai/types.py:856` — `word_boost: Optional[List[str]]`
- same file `:858` — `boost_param: Optional[WordBoost]`; `WordBoost` enum at `:237`

**Wire-in:** `scripts/transcriber.py`, the `aai.TranscriptionConfig(...)` call in `transcribe_audio()` (currently `speaker_labels` / `language_code` / `punctuate` / `format_text` only). Add `word_boost=TERMS.boost_list()` and `boost_param=aai.WordBoost.high`.

Cheap, native, no correctness risk — boosting only shifts probability, it never rewrites text.

### Layer 3 — Deterministic post-pass over transcript **and** summary

A term map applied after generation, to both artifacts. This is the guarantee layer: testable offline, reviewable in a diff, and it works on transcripts already cached from past meetings.

Design:

- **`config/terms.yml`** — the single source of truth, one entry per term:
  ```yaml
  - correct: Bookeo
    aliases: [bookio, "book io"]        # safe, unambiguous
    identifier_prefix: bookeo_           # rewrites bookio_foo -> bookeo_foo
  - correct: TRFA
    aliases: [tarifa, "t r f a"]
  - correct: ActiveCampaign
    aliases: ["active campaign", "AC"]
  ```
- **`scripts/terms.py`** — `load_terms()` and `apply_corrections(text) -> (text, [changes])`. Word-boundary regex, case-preserving where it matters. **Returns the list of substitutions made** so the pipeline can print them; a silent corrector is one nobody trusts.
- **Wire-in:** `scripts/pipeline.py` — ✅ pinned 2026-08-12 to **`:180`**, the line where the `--from-cache` branch (`:109`) and the fresh branch (`:170`) converge, so one call serves both; and again after analysis at `:207`.

**Keep the raw cache pristine.** `transcriber.py:59` writes the raw transcript to the cache *before* any of this runs. That ordering is already correct and must stay — corrections apply downstream, so the original is always recoverable when a correction turns out to be wrong.

### Layer 4 — `custom_spelling` ⚠️ use narrowly

AssemblyAI's literal "hear X, write Y". Verified: `custom_spelling` at `types.py:895` / `:1014`, setter `set_custom_spelling` at `types.py:1832`.

**The footgun:** it is safe for unambiguous tokens (`bookio` → `Bookeo`) and dangerous for the forms that actually dominate this transcript. Mapping `"book you"` → `Bookeo` would corrupt ordinary English — *"I'll book you for Monday"* becomes *"I'll Bookeo for Monday"*. Given Layer 1 and Layer 3 already cover the ground, this layer earns its place only for single-word confusions with no English meaning.

Populate it from the same `terms.yml` `aliases`, filtered to single tokens flagged `safe_for_asr: true`.

## Suggested build order

1. `config/terms.yml` + `scripts/terms.py` + unit tests (pure functions, no API calls — testable without spending a cent on transcription).
2. Layer 3 wiring in `pipeline.py`, run against the cached 2026-07-30 transcript as the fixture. **Success criterion: the summary comes out with `bookeo_` and zero `bookio`.**
3. Layer 1 glossary in `analyzer.py`, re-run the same fixture — the corrections list from Layer 3 should come back empty or near-empty, proving Layer 1 did the work upstream.
4. Layer 2 `word_boost` on the next real meeting; compare spoken-form noise against this baseline.
5. Layer 4 only if 1–4 leave residue.

## Decisions taken 2026-08-12 — these close the open questions below

Ruled by Cenay in a session-desk discussion. Each replaces an open question or sharpens a wire-in.

- **`terms.yml` lives in THIS repo.** ✅ Ruled — *"this is the tool creating the transcription and notes."* The reasoning is stronger than the original "only one consumer today" framing and does not depend on consumer count: **the list belongs with the tool that produces the artifact it corrects.** Promotion would be triggered by another tool producing transcripts, not by another project wanting the terms — so treat this as settled rather than provisional.
- **The Layer 3 substitution runs at `scripts/pipeline.py:180`.** ✅ Ruled. That is the exact line where the `--from-cache` branch (`:109`) and the fresh-transcription branch (`:170`) converge, so **one call covers both entry paths** and every downstream consumer: the transcribe-only dump (`:186`), the cost estimate (`:204`), the Claude analysis (`:207`), and the Notion page. Rejected alternatives: inside `transcribe_audio()` (would poison the pristine cache), inside `analyze_transcript()` (misses the published transcript), inside `notion_output.py` (too late for the other consumers).
- **Every correction is logged to `logs/`.** ✅ Ruled — *"Absolutely. That makes it discoverable after the fact."* This closes the third open question below. The durability case: a wrong entry in `terms.yml` could quietly rewrite many meetings before anyone notices, and the log is what makes that recoverable rather than archaeological.
- **The corrections ALSO go on the Notion page, in a toggle under the transcript.** ✅ Ruled 2026-08-20 ([DEC-010]) — *"place the corrections in a similar toggle as the transcript (can live right below it) that states the correction."* `logs/` is for after-the-fact archaeology across runs; the page toggle is for the reader of *this* meeting — including `meeting-reconcile`, which reads the page and never sees this repo. Both passes appear in it (transcript substitutions and the [DEC-009] analysis residual, grouped separately). ⚠️ **A run with no corrections still renders the toggle** saying so, because an absent block would otherwise mean either "nothing was rewritten" or "nobody looked".
- **`analyzer.py` builds ONE prompt.** ✅ Answered by inspection, closing the first open question below: a single `ANALYSIS_PROMPT` at `analyzer.py:24`, formatted at `:124`, one `client.messages.create` at `:132` (the JSON-retry path re-sends the same string). The Layer 1 glossary therefore has exactly one insertion point.
- **Transcript fidelity is NOT a constraint.** ✅ Ruled, after being raised as a possible blocker and dismissed. `meeting-reconcile` quotes transcript text as `Said` lines, which auto-correction makes non-verbatim. Cenay: *"These are multiple people speaking in a meeting, and mostly, we understand one another. It's the transcription that doesn't."* The transcript is a record of what a room communicated, not of exact sounds, and it was already wrong when it arrived — so correcting it repairs transcriber damage rather than editing what was said. **The word "verbatim" in that skill exists to stop a session inventing content from memory; it is not a promise about phonetic fidelity.**
- **There is a human review gate.** Meeting reconciliations are reviewed before they apply, and misspellings are corrected inbound there today. This materially lowers the cost of a false-positive substitution, and it is **where new terms get discovered** — which is why adding a term must be possible from any repo (see below).

- **`terms.yml` is ONE FLAT TABLE; the code classifies risk, not the author.** ✅ Ruled. Each entry is a `correct:` term plus a growable list of `heard:` strings:

  ```yaml
  - correct: Bookeo
    heard: [bookio, "book io", "book eoe", booko, booku, bookq]
    identifier_prefix: bookeo_
  ```

  `scripts/terms.py` classifies each `heard:` string **at load time**: a variant whose every word is ordinary English (`booking`, `booked`, `book it`, `bookie`, `book here`) is **refused**; one containing a non-word (`bookio`, `booko`, `booku`, `bookq`) is **applied**.

  ⚠️ **The classifier rule, verified 2026-08-12 — do not simplify it back.** A plain dictionary lookup against `/usr/share/dict/words` (present, 102,485 entries) is **wrong**: it refuses `book io`, this document's own example alias, because `io` is in the dictionary. The obvious repair — a minimum token length of 3 — is **worse**, because it then accepts `book it` and `book he`, and short function words are exactly the dangerous case; that trades one false refusal for two false acceptances, in the direction that corrupts English. **What works: dictionary lookup for tokens of 3+ characters, plus an explicit stop-word set for 1–2 character tokens** (`it`, `he`, `we`, `you`, `is`, `in`, `of`, `to`, `on`, `at`, `a`, `i`, …). ✅ 19/19 cases correct, covering every form the corpus surfaced. **Any change here gets re-run against those cases first** — this rule was broken as originally specified and only running it revealed that. **The binding constraint is authoring cost** — the stated requirement is *"a table I can add to over time as I discover things"*, so adding a row must cost nothing but the string. A hand-flagged safe/risky tier was rejected for exactly this reason: it puts an invisible judgement call on every row, and that is how lists stop being maintained. Validated against the 83-transcript corpus above. An optional per-row override is the escape hatch **only if the mechanical rule proves annoying in practice** — not designed in up front.

- **Terms can be added from ANY repo, via a global `/add-term` that auto-commits.** ✅ Ruled. The discovery moment is a reconciliation review in `fran-dash` or `trfaapi.com`, not a session in this repo, so the file stays here ([above](#)) while the *command* is global. Three pieces:
  1. `~/.claude/commands/add-term.md` — a global slash command, reachable from every repo (that directory already holds 20+).
  2. `claude-personal-toolkit/scripts/add-term.py` — the mechanism, alongside `stamp-doc.py` and `gen-dec-index.py`, distributed by `sync-shared.sh`. Appends a `heard:` string to the matching `correct:` entry, creates the entry when the term is new, **idempotent** (a duplicate is reported, not appended), and prints the resulting entry back.
  3. A prompt in the `meeting-reconcile` skill to offer `/add-term` when a term is corrected inbound — this is what closes the loop rather than merely making it cheaper, since the correction is being made there anyway.

  ⚠️ **`/add-term` AUTO-COMMITS that one file — a deliberate, named exception to the standing no-auto-commit rule.** The rule exists because changes are tested before they are committed; a term addition has no such window — there is nothing to test, it is data not code, and left uncommitted it sits unnoticed in a repo nobody is working in, so the corrections silently do not apply. **Scope is exactly one file**: commit `terms.yml` alone, never `-a`, never a sweep of whatever the other repo left dirty. That scoping is the safety of the exception and belongs in the script, not in a comment.

  ⚠️ **The script FAILS LOUDLY if the canonical path is absent** — exits non-zero naming the path it tried. It must never "helpfully" create a fresh `terms.yml` somewhere unread: a silently-empty term list applies zero corrections while every run reports success.

## ✅ `/add-term` BUILT 2026-08-13 — all three pieces landed

Built in `claude-personal-toolkit` (transcript `777a93e2-2811-4093-b54c-d94264e721b5`), matching the ruling above.

- **`commands/add-term.md`** → symlinked to `~/.claude/commands/add-term.md`, reachable from every repo.
- **`scripts/add-term.py`** → the mechanism. **Toolkit-local, deliberately NOT in `sync-shared.sh`** — a copy in Khurram's repo would be a script pointing at `/mnt/k/Code/TRFA/video-transcription/config/terms.yml`, a path that does not exist on his machine, which is a worse failure than its absence.
- **The `meeting-reconcile` prompt** → a new sub-step under *Step 2 — Distill and classify*, telling the session to keep a running list of the terms it silently corrected and offer them back as concrete `/add-term` calls. Explicitly **offer, never act unasked**, and explicitly **not** filed into the four intake buckets — a mishearing is transcriber damage, not something the meeting decided.

**Design choices made during the build**, each with its reason:

- **The edit is TEXTUAL, never a YAML round-trip.** `yaml.safe_load` + `safe_dump` would silently delete all ~40 comment lines in this file — the corpus counts, the possessive-swallowing warning, the whole *DELIBERATELY NOT INCLUDED* block. Those comments are most of the file's value. A test asserts every original comment line survives a write.
- **It writes, then PROVES the correction will fire.** After writing, the file is re-parsed with this repo's own `scripts/terms.py` and the new variant must come back in `Term.applied()`. If it does not, the write is **rolled back**. "It parsed" was not a strong enough claim — a variant can parse fine and still be refused by the classifier at load time, which is a term that looks added and never fires.
- **Risk is judged by importing `terms.py`, not by re-implementing `is_risky()`.** Two copies of that rule would give two answers to one question — accepted at the desk, refused at runtime.
- **Refusal is exit code 2 with the `--force` command spelled out.** Per Cenay 2026-08-13: warn and offer, never silently accept and never silently refuse. The command file instructs the session to relay the warning and **ask** rather than deciding to `--force` on its own.
- **Commit scope is enforced in code, not in a comment.** `git -C <root> commit -m … -- config/terms.yml` — one pathspec, so whatever else the other repo left dirty is untouched. It pushes too, by the same argument that justifies the commit: a commit sitting unpushed in a repo nobody works in is the same silent failure.

**Tests:** `scripts/test-add-term.py`, 36 assertions, ✅ 36/36 — every case run against a **copy of the real `config/terms.yml`**, not a hand-rolled fixture. Negative tests cover the refusal path, the missing-`terms.yml` path and the missing-`terms.py` path, each asserting non-zero exit **and** an unchanged file. The suite was **mutation-tested**: disabling the refusal guard, the idempotence check and the missing-path guard each produced failures, so a pass means something.

⚠️ **Finding — `book eeo`, this file's own advertised example, is REFUSED.** Verified by `python3 -c "import terms; terms.is_ordinary_english('eeo')"` → `True`: **`eeo` is in `/usr/share/dict/words`** (the EEO acronym), so every token of `book eeo` reads as English. The variant actually listed here is `book eoe`, which passes. This is the *same failure mode* the classifier rule already documents for `io` — the word list contains acronyms and abbreviations, so 3+ character tokens are not the clean signal they look like. **Not changed here**, because the rule was ruled and tested 19/19 and `--force` handles the case; but the `HOW TO ADD A TERM` comment in `terms.yml` advertises an example that does not work, and is worth correcting.

### Still open after this session

- **Whether corrections also appear on the Notion page** (in addition to `logs/`). Marginal, given the human review gate.
- ~~**Where the auto-commit exception is recorded.**~~ ✅ **Closed** — it is recorded in the global `~/.claude/CLAUDE.md` under *Safety Rails → Git*, which is what a session in another repo actually reads, with a ⛔ "do not fix this when you encounter it from another repo" note pointing back here.
- **Team names.** A distributed team (Miami, Orlando, Tampa, Pakistan, Arizona, Serbia) means names are spoken constantly and mangled reliably. `Khurram` is already in the Layer 1 glossary example; the rest of the roster is wanted.

## Open questions

- ~~Does `analyzer.py` build one prompt or several?~~ ✅ **Answered above — one.**
- ~~Should `terms.yml` live in this repo or be shared across TRFA projects?~~ ✅ **Ruled above — this repo.**
- ~~Worth logging every correction to `logs/`?~~ ✅ **Ruled above — yes.**
