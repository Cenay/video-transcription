# Porting the transcription pipeline to Windows — scoping brief

_Last updated 2026-08-21 17:21 MST by an AI session · transcript: `1c7f1805-eb94-4168-8f2c-91e73e0b7f4d` — initial scope of the Windows port_

**Status:** 🚧 PROPOSED — no code written, no decision taken.
**Question this answers:** can `video-transcription` run on a colleague's Windows machine, and what does it cost?
**Short answer:** yes. The Python core is already portable; the Linux coupling is concentrated in one 237-line bash wrapper plus **one silent correctness bug** that must be fixed before anyone runs this on Windows.

---

## Verification legend

Every factual claim below is marked. ✅ means a command was run this session and its output is quoted or summarized. ⚠️ means it is believed from reading code but not executed.

**What was NOT checked:** nothing was run on an actual Windows machine. Every Windows-side claim is a prediction from reading POSIX-specific code, except the wordlist bug, which was verified by *simulating* the missing file on Linux. No Windows VM, no `pywin32`, no Explorer integration was tested. Effort estimates are judgment, not measurement.

---

## 1. The headline finding — a silent safety inversion (must fix first)

This is the one item that makes the port a correctness question rather than a packaging question.

`scripts/terms.py:30` hardcodes a Linux path:

```python
WORDLIST_PATH = Path("/usr/share/dict/words")
```

`_load_wordlist()` (`terms.py:42–46`) **returns an empty set when the file is missing** rather than raising. On Windows the file is always missing, so `_WORDS` is empty and every downstream guard silently flips.

The chain: `is_ordinary_english()` (`terms.py:52–59`) returns `t in _WORDS` for tokens longer than 2 characters → always `False` on Windows. `is_risky()` (`terms.py:62–69`) is `all(is_ordinary_english(t) for t in tokens)` → also always `False`. **`is_risky` is the guard that refuses substitutions which would corrupt ordinary prose.** With an empty wordlist it never fires, so every term variant is treated as safe to replace.

✅ **Verified** by a mutation test run this session — `terms.py` imported, `WORDLIST_PATH` repointed at a nonexistent path, `_WORDS` reloaded, `is_risky()` compared across both states:

| Variant | Linux (102,485 words loaded) | Windows (0 words loaded) |
|---|---|---|
| `nick` | `True` (refused) | **`False` (allowed)** |
| `art` | `True` (refused) | **`False` (allowed)** |
| `make` | `True` (refused) | **`False` (allowed)** |
| `karam` | `False` | `False` |
| `senay` | `False` | `False` |
| `book io` | `False` | `False` |

Read the first three rows against the roster in `CLAUDE.md`: **Nik** arrives as `nick`, **Art** is flagged there as "too short and ordinary to auto-correct", and `make` is listed as a near-miss for **Jake**. Those are precisely the terms the guard exists to protect. On Windows the tool would rewrite the ordinary English words "nick", "art" and "make" into teammate names mid-sentence.

⚠️ **Believed, not checked:** that this actually changes published output end-to-end. The unit-level inversion is proven; whether a given `terms.yml` entry reaches `is_risky()` on a real transcript was not traced through `preview_corrections.py` / the pipeline. Worth confirming before the fix is called done.

**Why this matters beyond one machine.** Per `CLAUDE.md`, this tool is step 1 of `transcribe → meeting-reconcile → doc-reconcile → checkpoint`, and its Notion page feeds `meeting-reconcile`, which stages edits into `DECISIONS.md` / `CURRENT_STATUS.md` / `TODOS.md` across several repos. A corrupted word propagates into production decision records. It fails silently and it fails *downstream*, which is the expensive combination.

**Fix (small, and worth doing regardless of the port):**

1. Ship a wordlist in-repo — `config/words.txt` — instead of depending on the OS. Removes the platform dependency entirely and makes behavior identical on every machine.
2. Make a missing wordlist **loud**: raise, or refuse all risky substitutions rather than allowing all of them. Silence must not mean both "nothing there" and "nobody looked" — the same principle `notion_output.py` already applies to `corrections=[]` vs `corrections=None` ([DEC-010]).
3. Add a negative test: assert `is_risky("nick") is True` with the wordlist present, and assert the missing-wordlist path fails loudly rather than returning `False`. Per the standing rule, a guard that has not been shown to fail on broken input has not been tested.

> **Recommendation: do this first, on Linux, and ship it independently of the port.** It is a live latent bug — anyone who copies this repo to a machine without `wamerican`/`words` installed inherits it today, Windows or not.

---

## 2. What is already portable (no work required)

✅ **Verified** by grep across all 12 files in `scripts/` (2,370 lines total):

- **No Linux-specific calls in any Python file** other than the `terms.py` wordlist path above.
- **ffmpeg and ffprobe are invoked as argument lists, not shell strings** — `audio_extractor.py:25–37`, `pipeline.py:150–156`. No `shell=True`, no pipes, no quoting hazards. These run unchanged on Windows provided both binaries are on `PATH`.
- **`pipeline.py` already resolves temp correctly** — `pipeline.py:86` and `:98` use `tempfile.gettempdir()` as the `TEMP_DIR` fallback.
- **All 9 dependencies in `requirements.txt` are pure-Python / cross-platform** — assemblyai, anthropic, pydub, watchdog, rq, redis, notion-client, python-dotenv, pyyaml. ⚠️ *Believed, not checked:* none were installed on Windows to confirm wheels resolve. `pydub` additionally needs ffmpeg present, which is already a requirement.
- **Config loading is path-safe** — `terms.py:29` uses `Path(__file__).resolve().parent.parent`, and `.env` loads via `python-dotenv`.

## 3. What is Linux-only (the actual work)

### 3a. Three files hardcode `/tmp`

✅ **Verified** by grep. `transcriber.py:42`, `diagnose_analysis.py:17`, `repair_global_options.py:22` all read:

```python
cache = Path(os.environ.get("TEMP_DIR", "/tmp")) / "transcribe-cache"
```

`pipeline.py` already does this correctly. One-line fix each: `tempfile.gettempdir()`. Note the consequence if missed — the cache directory diverges from the one `pipeline.py` uses, so `--from-cache` silently misses and re-transcribes at full cost.

### 3b. `transcribe-this.sh` — the bulk of the port

✅ **Verified** by reading all 237 lines.

| Lines | Construct | Windows status |
|---|---|---|
| 11–13 | `PIPELINE_DIR="/mnt/k/Code/TRFA/video-transcription"` + bucket constants | Hardcoded to this machine — must become config |
| 102, 111, 123 | `source venv/bin/activate` | Windows is `venv\Scripts\activate` |
| 32 | `realpath` | Not present on Windows |
| 125 | `/tmp/transcribe-pipeline-output-$$.log` | POSIX temp + `$$` PID idiom |
| 133, 161, 199, 236 | `notify-send` (4 calls) | Linux desktop notification daemon |
| 138 | `grep -oP 'notion\.so/\K[a-f0-9]+'` | GNU-only `-P` and `\K` |
| 203–206 | `$HOME/Videos/Zoom` + `gio trash` | Linux path convention + GIO trash |
| 209–212 | `mv` to `.archived/$(date +%Y-%m-%d)` fallback | Already the portable path — good |
| 70–86 | S3 prefix guard | Pure string logic, portable in any language |
| 39–52 | Flag parsing | Portable |
| 155, 195 | `aws s3 cp` / `aws s3 ls` | AWS CLI is cross-platform ✅ |

### 3c. The Explorer/Nautilus right-click integration is not in the repo

✅ **Verified** by `ls ~/.local/share/nautilus/scripts/` — two entries exist, **"Transcribe This"** and **"Transcribe This --no-cleanup"**, living entirely outside version control.

Two consequences: the Windows equivalent must be built from scratch (registry `shell` key, or a `.cmd` shim in the `SendTo` folder), and **the current Linux integration is untracked and unbacked-up** — a machine rebuild loses it silently. Worth committing into the repo as an installable asset regardless of which option below is chosen.

### 3d. A pre-existing quoting hazard, surfaced while reading

⚠️ **Believed from reading, not tested.** `transcribe-this.sh:174–178` interpolates shell variables directly into a `python -c` string:

```bash
python -c "
from scripts.notion_output import update_meeting_link
update_meeting_link('$NOTION_PAGE_ID', '$S3_URL')
"
```

`$NOTION_PAGE_ID` is grep-extracted hex so it is safe, but `$S3_URL` derives from the filename. A filename containing an apostrophe would break the quoting. Not Windows-specific — it is a live bug on Linux today. Any rewrite eliminates it for free by passing arguments properly.

---

## 4. Options

### Option A — WSL2 (fastest, ~1 hour)

Friend installs Ubuntu under WSL2 and runs the repo verbatim.

- ✅ Near-zero code change; identical behavior to a machine that already works.
- ✅ `/usr/share/dict/words` exists (after `apt install wamerican`), so §1 does not bite — **though it should still be fixed**.
- ❌ No Explorer right-click — the workflow becomes copy-path-and-type.
- ❌ Path translation friction (`/mnt/c/Users/...`) on every invocation.
- ❌ Friend must learn enough Linux to maintain it; forks the support burden rather than reducing it.

### Option B — rewrite the wrapper in Python (recommended, ~1 day)

Replace `transcribe-this.sh` with `scripts/transcribe_this.py` as a real CLI, delete the bash file, and have both machines run the same entry point.

Maps cleanly onto the table in §3b:
- `notify-send` → a `notify()` helper branching on `sys.platform` (Linux `notify-send`, Windows toast or a no-op — notifications are non-essential).
- `gio trash` → `send2trash` (one new dependency, works on both platforms), keeping the existing `.archived/` fallback.
- `grep -oP` → **delete entirely.** Have `pipeline.py` return the page ID rather than re-parsing its own stdout. `pipeline.py:337` already sets `result["notion_page_id"]`, so the value exists — it is being thrown away and then scraped back out of terminal output. Emitting it (a `--print-page-id` flag or a JSON line) removes the GNU-grep dependency, removes the fragile stdout coupling, and is the single highest-value change in the port.
- `source venv/...` → unnecessary; the script runs *inside* the venv.
- `realpath`, `/tmp`, `$HOME/Videos/Zoom` → `pathlib`, `tempfile`, and a configurable watch folder.
- Machine constants (§3b line 11–13) → `.env`, alongside the API keys already there.
- §3d disappears — arguments get passed as arguments.

- ✅ One codebase, one wrapper, both platforms. Reduces long-term maintenance rather than doubling it.
- ✅ Fixes two live Linux bugs (§3d, and the stdout-scraping fragility) as a side effect.
- ✅ Explorer integration becomes a thin `.cmd` shim.
- ❌ A day of work, and it touches the entry point that currently works — needs a real regression run on Linux before the friend ever sees it.

### Option C — parallel PowerShell script (~half day, not recommended)

Translate the bash line-for-line into `transcribe-this.ps1`.

- ✅ Native Windows, no WSL.
- ❌ **Two wrappers to keep in sync forever.** Every future change — a new S3 prefix, a new flag — must land twice, and the second one will be forgotten. The S3 prefix guard (`transcribe-this.sh:70–86`) is exactly the kind of safety logic that must not drift.

---

## 5. Recommended sequence

1. **Fix the wordlist bug (§1)** — ship in-repo wordlist + loud failure + negative test. Independent of the port; do it first because it is a live latent bug.
2. **Fix the three `/tmp` defaults (§3a)** — three one-line changes.
3. **Make `pipeline.py` emit the Notion page ID (§4B)** — unblocks the wrapper rewrite and removes the stdout scraping.
4. **Write `scripts/transcribe_this.py` (Option B)**, delete the bash, regression-run on Linux against a real meeting until output is byte-comparable.
5. **Commit the desktop integration into the repo (§3c)** — Nautilus scripts and the Windows `.cmd`, with an install step.
6. **Only then** hand it to the friend, and expect a round of genuine Windows surprises — nothing here was tested on Windows.

Steps 1–3 are worth doing whether or not the port happens.

---

## Open questions for Cenay

- **Q1.** Does the friend need the S3 upload and Notion publish at all, or only transcription? If only transcription, the port shrinks to `--transcribe-only` and most of §3b evaporates.
- **Q2.** Does the friend get their own AssemblyAI / Anthropic / Notion keys, or share yours? Sharing means their usage lands on your bills and their pages land in your Notion database.
- **Q3.** Same Notion database and S3 buckets, or their own? The prefix guard (`transcribe-this.sh:70–86`) assumes TRFA buckets.
- **Q4.** Is the friend technical enough for WSL2 (Option A), or does this need to feel like a normal Windows app (Option B)?
