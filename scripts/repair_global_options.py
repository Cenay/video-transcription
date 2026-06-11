"""One-off: rebuild the body of the 'TRFA Global Options Status Ninthroot
Issue' Notion page from the recovered analysis (no new Claude call). Reuses the
RAW-ANALYSIS.txt produced by diagnose_analysis.py.
"""
import os, sys, json
from pathlib import Path
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent))
load_dotenv(Path(__file__).parent.parent / ".env")

from notion_client import Client
from notion_output import repair_meeting_page
from analyzer import _extract_json

STEM = "trfa-global-options-status-ninthroot-issue"
TITLE = "TRFA Global Options Status Ninthroot Issue"
DATE = "2026-06-09"
DURATION_MIN = 120
COSTS = {"transcription": 0.7425, "analysis": 0.0929, "total": 0.8355}

cache = Path(os.environ.get("TEMP_DIR", "/tmp")) / "transcribe-cache"
analysis = json.loads(_extract_json((cache / f"{STEM}-RAW-ANALYSIS.txt").read_text()))
transcript = json.loads((cache / f"{STEM}-raw-transcript.json").read_text())["raw_text"]

notion = Client(auth=os.environ["NOTION_API_KEY"])
db = os.environ["NOTION_DATABASE_ID"]
ds_id = notion.databases.retrieve(db)["data_sources"][0]["id"]
res = notion.data_sources.query(
    data_source_id=ds_id,
    filter={"property": "Name", "title": {"equals": TITLE}},
)["results"]
if not res:
    sys.exit(f"Page not found: {TITLE}")
page_id = res[0]["id"]
print(f"Found page: {page_id}")
print(f"Analysis keys: {list(k for k in analysis if not k.startswith('_'))}")

out = repair_meeting_page(page_id, DATE, DURATION_MIN, analysis, transcript, COSTS)
print(f"Repaired: {out['url']}")
print(f"Restored meeting link: {out['restored_link']}")
