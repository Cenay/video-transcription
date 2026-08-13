# Adding and testing a new term

_Last updated 2026-08-12 19:41 MST by an AI session · transcript: `ce166dfb-eca1-4c5a-b935-71755aed3e98` — documented the add-a-term workflow and the read-only preview commands._

<details>
<summary>📜 <strong>Stamp history</strong> — the 1 previous update (older ones: <code>history/new-term-testing-stamp-history.md</code>)</summary>

- _Prior: 2026-08-12 19:15 MST by an AI session · transcript: `ce166dfb-eca1-4c5a-b935-71755aed3e98`_

</details>

How to teach the pipeline a word the transcription keeps getting wrong, and how to check your change before it touches a real meeting.

**Everything in this workflow is read-only and free.** It runs against the 83 transcripts already cached in `temp/transcribe-cache/`, so you can iterate as much as you like without re-transcribing anything or spending a cent.

---

## The mental model

The speech engine mangles domain vocabulary and names — reliably, and in ways you can enumerate. `Khurram` appears **zero** times across 83 meetings; it arrives as `karam` (69) or `haram` (37). `Cenay` also appears **zero** times; it arrives as `senay` (13). Left alone, those wrong forms flow into the summary, into the Notion page, into a meeting reconciliation, and finally into decision documents read by people in six locations.

`config/terms.yml` is the list of known mishearings. `scripts/terms.py` applies it. **Your only job when adding a term is to supply the string you saw** — the code decides whether it is safe to substitute.

### Why the code decides, and not you

Some mishearings are made of ordinary English words, and replacing those corrupts normal prose:

> "I'll **book you** for Monday" → "I'll **Bookeo** for Monday"

So each variant is classified at load time:

| variant contains | verdict | examples |
|---|---|---|
| a token that is not ordinary English | ✅ **applied** | `bookio`, `booko`, `karam`, `haram`, `senay`, `milosh` |
| only ordinary English words | ⛔ **refused** | `booking`, `booked`, `bookie`, `book it`, `book he`, `book here` |

This is not fussiness. In the real archive `booking` appears **259 times**, nearly all of them legitimate — a list that replaced it would corrupt hundreds of sentences.

⚠️ **A refused variant is not a mistake in your entry.** Leave it in the file: it documents the mishearing, it costs nothing, and if you later decide it is worth forcing, the string is already there.

---

## The loop

### 1. Notice a wrong word

Usually while reviewing a meeting reconciliation — that is the natural moment, and it is the one this workflow is built around.

### 2. See what the current list already does

```bash
cd /mnt/k/Code/TRFA/video-transcription

# One meeting — every changed line, before and after
./venv/bin/python scripts/preview_corrections.py tampa

# Every meeting, ranked by how much each would change
./venv/bin/python scripts/preview_corrections.py --all
```

A name fragment is enough (`tampa`, `bookeo`, `khurram`); full paths work too.

### 3. Check how the new word actually behaves in real meetings

**This is the step that matters, and the one worth not skipping.** Before adding a variant, read every place it occurs:

```bash
./venv/bin/python scripts/preview_corrections.py --all --grep nick
```

⚠️ **Grep mode deliberately ignores the term list** — it shows every line containing the word whether or not anything corrects it today. That is the point: you are vetting a word that is usually **not** in the list yet, so filtering to already-changed lines would show you nothing.

Lines are marked:

- `=` — contains the word, not corrected today (the normal case when vetting)
- `-` / `+` — already corrected by the current list

You are looking for one thing: **does this word ever appear with an innocent meaning?** If `nick` is always the person, it is safe to force. If one hit is "nick the file", forcing it would corrupt that sentence forever.

Worked example — `nick` occurs on **42 lines across 8 transcripts**, so this is a real reading job, not a glance. That is the correct amount of friction for a change that rewrites every future meeting.

### 4. Add the string to `config/terms.yml`

```yaml
  - correct: Khurram
    heard:
      - karam           #  69
      - haram           #  37
      - keram           #   1
```

That is the whole edit. Add a count as a comment if you know it — future readers use those numbers to judge whether an entry is still earning its place.

**A brand-new term takes the same shape:**

```yaml
  - correct: Realtop
    heard:
      - real top
      - realtoss
```

### 5. Preview again, and read the diff

```bash
./venv/bin/python scripts/preview_corrections.py --all
```

The total correction count should rise by roughly what you expect. **If it jumps by hundreds, stop and look** — you have probably added something that collides with ordinary English.

### 6. Run the tests

```bash
./venv/bin/python tests/test_terms.py --corpus
```

This sweeps every cached transcript and asserts that **no ordinary-English word was altered anywhere**. It is the backstop for step 3 — if your new variant quietly corrupts prose, this is what catches it.

---

## When a variant is refused but you want it anyway

Some genuine mishearings are made entirely of real words. Two are in the shipped list:

- `active campaign` (61 occurrences) → `ActiveCampaign`
- `fran dash` (46 occurrences) → `fran-dash`

Neither phrase has an innocent meaning in these meetings, so they are force-approved:

```yaml
  - correct: ActiveCampaign
    heard:
      - active campaign
    force:
      - active campaign
```

⚠️ **`force:` is the one place you can hurt yourself.** It disables the guard for that exact string. Use it only after step 3 has shown you every occurrence, and only when none of them is innocent. Forced substitutions are marked `[FORCED]` in the corrections report so they stay visible.

---

## Rules learned the hard way

**⛔ Never list a possessive form.** Writing `bookio's` as its own variant **swallows the apostrophe-s**, because longer variants match first: *"bookio's widget"* becomes *"Bookeo widget"*. The bare `bookio` entry already handles it correctly — *"Bookeo's widget"* — because an apostrophe is a word boundary. Guarded by `tests/test_terms.py::test_possessive_preserved`.

**⛔ Do not add plural or inflected forms of ordinary words.** `bookings`, `booked`, `books` are refused by design, and forcing them would be a mistake.

**✅ Do add every spelling you actually see, however odd.** `bookq`, `booku`, `booko`, `cnay` all appear in real transcripts. Non-words are free — they cannot collide with anything.

**✅ Names are the highest-value entries.** The engine is strong on English morphology and fails completely outside it, so names are where the wins are.

**⚠️ Identifier prefixes are separate.** `identifier_prefix: bookeo_` is what rewrites `bookio_product_groups` → `bookeo_product_groups` — constructed identifiers nobody ever said aloud, which no variant list would match.

---

## Adding a term from another repo

The moment you notice a bad word is usually a reconciliation review in `fran-dash` or `trfaapi.com`, not a session here. A global `/add-term` command is **designed but not yet built** — see `plans/term-normalization.md`. Until it exists, edit `config/terms.yml` directly.

⚠️ **When it is built it will auto-commit that one file** — a deliberate, documented exception to the standing no-auto-commit rule, recorded in `~/.claude/CLAUDE.md`. The reason: a term added from another repo otherwise sits uncommitted and unnoticed here for weeks, and the correction silently never applies.

---

## Command reference

| what you want | command |
|---|---|
| Preview one meeting | `./venv/bin/python scripts/preview_corrections.py <name-fragment>` |
| Preview all, ranked | `./venv/bin/python scripts/preview_corrections.py --all` |
| Preview all, with lines | `./venv/bin/python scripts/preview_corrections.py --all --lines` |
| Vet one word everywhere | `./venv/bin/python scripts/preview_corrections.py --all --grep <word>` |
| Fast tests | `./venv/bin/python tests/test_terms.py` |
| Tests + corpus sweep | `./venv/bin/python tests/test_terms.py --corpus` |
| List cached transcripts | `ls temp/transcribe-cache/` |

---

## Troubleshooting

**"term list not found"** — `scripts/terms.py` refuses to run with an empty list rather than silently applying no corrections. It prints the path it tried; if the repo moved, update `DEFAULT_TERMS_PATH` in `scripts/terms.py`.

**`ModuleNotFoundError: yaml`** — use `./venv/bin/python`, not bare `python3`. PyYAML is in `requirements.txt` and installed in the venv only.

**A variant does nothing** — it is probably being refused. Confirm:

```bash
./venv/bin/python -c "import sys; sys.path.insert(0,'scripts'); from terms import is_risky; print(is_risky('book he'))"
```

`True` means refused. Either accept it, or add the string to `force:` after doing step 3.

**The preview shows no colour** — output is being piped. Colour is ANSI; read it in the terminal directly, or pipe through `less -R`.

---

## Related

- `plans/term-normalization.md` — why this exists, the four-layer design, and every decision taken, with the corpus measurements behind them
- `config/terms.yml` — the list itself, including a documented section of terms deliberately **not** included
- `tests/test_terms.py` — the safety guarantees, written so each one comes with a demonstration that it can fail
