# Zara AI — Bug Fixes Applied

All 10 issues from the analysis have been fixed. Zara AI behaviour is
unchanged; only the bugs are resolved.

---

## Fix #1 — Path traversal on resume upload  (app.py)
**Was:** `resume_path = os.path.join(UPLOAD_FOLDER, resume_file.filename)`
A crafted filename like `../../etc/cron.d/backdoor` could write outside uploads/.

**Now:**
- `werkzeug.utils.secure_filename` strips `..`, `/`, `\` from the filename.
- A UUID prefix is prepended to prevent collisions between candidates.
- Only `.pdf` files are accepted (extension whitelist).

---

## Fix #2 — Arbitrary file reads on download  (app.py)
**Was:** `send_file(filename, ...)` accepted any filename from the URL, allowing
`/download_report/../../app.py` to expose source files.

**Now:**
- Reports are written to a dedicated `reports/` folder (not the working directory).
- The download route validates that `filename` matches `session["report_file"]`
  (set by `/generate_report`). A different session cannot access another session's report.
- Path is built from `REPORTS_FOLDER + secure_filename(filename)` only.

---

## Fix #3 — Flask session cookie overflow  (app.py)
**Was:** `session["interview_data"]` grew with every answer; Flask's default
cookie-backed session has a 4 KB limit that silently truncates large interviews.

**Now:**
- An in-process dict `_interview_store` (keyed by UUID session ID) holds answers.
- The session cookie only carries small scalars (name, question list, index).
- Drop-in replacement with Redis/DB for production: just swap `_interview_store`.

---

## Fix #4a — tempfile.mktemp race condition  (voice.py)
**Was:** `tempfile.mktemp(suffix=".mp3")` returns a path that doesn't exist yet;
another process could create it between the call and `tts.save(temp_file)`.

**Now:** `tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)` creates the
file atomically. The file is closed immediately so gTTS / soundfile can write to it,
then deleted in a `finally` block.

---

## Fix #4b — Shell injection on mpg123 call  (voice.py)
**Was:** `os.system("mpg123 -q " + temp_file)` — a path containing spaces breaks
the command; a crafted path could inject shell commands.

**Now:** `os.system("mpg123 -q " + shlex.quote(temp_path))` — shlex.quote wraps
the path in single quotes and escapes any special characters.

---

## Fix #5 — API_TIMEOUT too short for HF free tier  (config.py)
**Was:** `API_TIMEOUT = 5` — Mistral-7B cold-start on HF free tier takes 20-40 s,
so almost every LLM call timed out and fell back to templates.

**Now:** `API_TIMEOUT = 30` (overridable via env var `API_TIMEOUT=5` for demos).

---

## Fix #6 — HF_FALLBACK_MODEL never used  (config.py / questions.py / scorer.py)
**Was:** `HF_FALLBACK_MODEL` was defined in config.py but questions.py and scorer.py
never imported or used it; they always hit the primary URL with no model fallback.

**Now:**
- `HF_FALLBACK_API_URL` is exported from config.py.
- Both `questions.py` and `scorer.py` catch HTTP 503 (model loading) on the primary
  and retry once on the fallback model before dropping to templates/heuristics.

---

## Fix #7 — LLM preamble lines admitted as questions  (questions.py)
**Was:** `_parse_questions_from_text` accepted any line >25 chars, letting
intro sentences like "Here are 4 interview questions for..." pollute the list.

**Now:**
- Lines matching known preamble patterns (`here are`, `certainly`, `as requested`, …)
  are rejected before any other check.
- Lines without `?` are only accepted if they carry a numbered/bullet marker AND
  contain a question-verb (`describe`, `how`, `explain`, `walk`, …) in the first 60 chars.

---

## Fix #8 — Heuristic scorer bands too generous  (scorer.py)
**Was:** base score of 5 + up to 3 (length) + 2 (keywords) = 8 for a 25-word answer
with 6 keyword hits — clearly too generous for a very short response.

**Now:** length is the primary gate (sets the score floor: 1/3/5/6/7);
keywords add at most +2 bonus; vague phrases subtract up to 2. A short answer
(<40 words) is capped at 5 regardless of keyword count.

---

## Fix #9 — Score colour thresholds mismatched  (report.py / scorer.py)
**Was:** report.py used `avg >= 8` for green, `avg >= 6` for yellow.
scorer.py used `avg >= 8.5` for "Strong Hire", `avg >= 7` for "Hire".
A score of 7.5 showed a "Hire" verdict but a yellow box — contradictory.

**Now:** both files use the same scale:
  - `≥ 7.0` → green  (Hire / Strong Hire)
  - `≥ 5.5` → yellow (Consider / Maybe)
  - `< 5.5` → red    (No Hire)

---

## Fix #10 — Mid-word truncation in strengths/weaknesses text  (scorer.py)
**Was:** `best_answer["question"][:70]` sliced at exactly 70 bytes, often
cutting mid-word (e.g. "…your end-to-end ML pip").

**Now:** `textwrap.shorten(text, width=70, placeholder="...")` always breaks
on a word boundary and appends a clean `…`.

---

## Fix #11 — Bare bool flag for thread-safe recording stop  (voice.py)
**Was:** `recording_active = False` was a module-level bool toggled from the
Flask request thread and read in the sounddevice callback thread.
Simple bools are effectively atomic in CPython but this is undefined behaviour.

**Now:** replaced with `threading.Event` (`_stop_event`). `stop_recording_stream`
calls `_stop_event.set()`; `record_audio` calls `_stop_event.clear()` at the
start of each recording and polls `_stop_event.is_set()` in the loop.

---

## Fix #12 — Whisper empty transcription not handled  (voice.py)
**Was:** `result["text"].strip()` would raise KeyError if Whisper returned
an unexpected dict shape, and returned `""` silently on silence.

**Now:** `result.get("text", "")` is used; an empty result prints a warning
and returns `""`. The caller in `app.py` already converts `""` to
`"No answer provided."` so behaviour is unchanged.

---

## Fix #13 — Non-Latin Unicode silently corrupted in PDF  (report.py)
**Was:** `text.encode('latin-1', errors='replace').decode('latin-1')` silently
replaced every non-Latin character (common in Whisper transcripts for accented
names, or mixed-language answers) with `?`.

**Now:**
- If `DejaVuSans.ttf` and `DejaVuSans-Bold.ttf` are present next to `report.py`,
  they are registered as a Unicode font and no sanitisation is needed.
- If the font files are absent, the old Latin-1 fallback is used with a startup
  warning directing the user to download the font.
- `textwrap.shorten` is used for truncation in all paths (no mid-word cuts).
