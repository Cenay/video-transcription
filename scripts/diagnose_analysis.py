"""One-off diagnostic: replay a cached transcript through the analyzer to see
exactly what Claude returns and why JSON parsing fails. Does NOT touch Notion.
Saves the raw response to TEMP_DIR/transcribe-cache/<stem>-RAW-ANALYSIS.txt.
"""
import os, sys, json
from pathlib import Path
from dotenv import load_dotenv
import anthropic

sys.path.insert(0, str(Path(__file__).parent))
from analyzer import ANALYSIS_PROMPT, MODEL
from terms import spelling_constraint

load_dotenv(Path(__file__).parent.parent / ".env")

stem = sys.argv[1] if len(sys.argv) > 1 else "trfa-global-options-status-ninthroot-issue"
cache = Path(os.environ.get("TEMP_DIR", "/tmp")) / "transcribe-cache"
data = json.loads((cache / f"{stem}-raw-transcript.json").read_text())
transcript = data["raw_text"]

client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
resp = client.messages.create(
    model=MODEL,
    max_tokens=4096,
    messages=[{"role": "user", "content": ANALYSIS_PROMPT.format(
        transcript=transcript, spelling=spelling_constraint()
    )}],
)
text = resp.content[0].text
out = cache / f"{stem}-RAW-ANALYSIS.txt"
out.write_text(text)

print("stop_reason   :", resp.stop_reason)
print("input_tokens  :", resp.usage.input_tokens)
print("output_tokens :", resp.usage.output_tokens, "(cap was 4096)")
print("response chars:", len(text))
print("first 120     :", repr(text[:120]))
print("last 120      :", repr(text[-120:]))

# Replicate analyzer's parsing logic exactly
t = text
if "```json" in t:
    t = t.split("```json")[1].split("```")[0]
elif "```" in t:
    t = t.split("```")[1].split("```")[0]
try:
    parsed = json.loads(t.strip())
    print("PARSE         : OK  keys=", list(parsed.keys()))
except json.JSONDecodeError as e:
    print("PARSE         : FAILED ->", e)
print("raw saved to  :", out)
