#!/usr/bin/env python3
"""Output analysis results to Notion."""

import os
from datetime import datetime
from notion_client import Client
from dotenv import load_dotenv

load_dotenv()

notion = Client(auth=os.environ.get("NOTION_API_KEY"))
DATABASE_ID = os.environ.get("NOTION_DATABASE_ID")


def create_meeting_page(
    title: str,
    date: str,
    duration_minutes: float,
    analysis: dict,
    transcript: str,
    costs: dict,
    source_file: str = ""
) -> str:
    """
    Create a Notion page with meeting analysis.
    
    Returns the URL of the created page.
    """
    # Create page with database properties
    page = notion.pages.create(
        parent={"database_id": DATABASE_ID},
        properties={
            "Name": {
                "title": [{"text": {"content": title}}]
            },
            "Date": {
                "date": {"start": date}
            },
            "Duration": {
                "rich_text": [{"text": {"content": f"{duration_minutes:.0f} minutes"}}]
            },
            "Status": {
                "select": {"name": "Complete"}
            },
            "Cost": {
                "number": round(costs.get("total", 0), 4)
            },
            "Source File": {
                "rich_text": [{"text": {"content": source_file}}]
            }
        }
    )
    
    page_id = page["id"]
    
    # Build content blocks
    blocks = []
    
    # Summary section
    blocks.append({
        "type": "heading_2",
        "heading_2": {"rich_text": [{"text": {"content": "Summary"}}]}
    })
    
    summary = analysis.get("summary", "No summary available.")
    # Split summary into chunks (Notion has 2000 char limit per block)
    for chunk in chunk_text(summary, 1900):
        blocks.append({
            "type": "paragraph",
            "paragraph": {"rich_text": [{"text": {"content": chunk}}]}
        })
    
    # Action Items section
    action_items = analysis.get("action_items", [])
    if action_items:
        blocks.append({
            "type": "heading_2",
            "heading_2": {"rich_text": [{"text": {"content": "Action Items"}}]}
        })
        
        for item in action_items:
            task = item.get("task", "")
            owner = item.get("owner", "Unassigned")
            deadline = item.get("deadline", "TBD")
            
            blocks.append({
                "type": "to_do",
                "to_do": {
                    "rich_text": [{"text": {"content": f"{task} (@{owner}, due: {deadline})"}}],
                    "checked": False
                }
            })
    
    # Decisions section
    decisions = analysis.get("decisions", [])
    if decisions:
        blocks.append({
            "type": "heading_2",
            "heading_2": {"rich_text": [{"text": {"content": "Key Decisions"}}]}
        })
        
        for i, decision in enumerate(decisions, 1):
            decision_text = decision.get("decision", "")
            rationale = decision.get("rationale", "")
            
            blocks.append({
                "type": "numbered_list_item",
                "numbered_list_item": {
                    "rich_text": [
                        {"type": "text", "text": {"content": decision_text}, "annotations": {"bold": True}},
                        {"type": "text", "text": {"content": f" — {rationale}" if rationale else ""}}
                    ]
                }
            })
    
    # Key Quotes section
    quotes = analysis.get("key_quotes", [])
    if quotes:
        blocks.append({
            "type": "heading_2",
            "heading_2": {"rich_text": [{"text": {"content": "Key Quotes"}}]}
        })
        
        for quote in quotes[:5]:  # Limit to 5 quotes
            quote_text = quote.get("quote", "")
            speaker = quote.get("speaker", "Unknown")
            
        blocks.append({
            "type": "quote",
            "quote": {
                "rich_text": [
                    {"type": "text", "text": {"content": f'"{quote_text}"'}},
                    {"type": "text", "text": {"content": f" — {speaker}"}, "annotations": {"italic": True}}
                ]
            }
        })
    
    # Processing Costs section
    blocks.append({
        "type": "heading_2",
        "heading_2": {"rich_text": [{"text": {"content": "Processing Details"}}]}
    })
    
    cost_text = f"Transcription: ${costs.get('transcription', 0):.4f} | Analysis: ${costs.get('analysis', 0):.4f} | Total: ${costs.get('total', 0):.4f}"
    blocks.append({
        "type": "paragraph",
        "paragraph": {"rich_text": [{"text": {"content": cost_text}}]}
    })
    
    # Full Transcript in a toggle (collapsed by default)
    blocks.append({
        "type": "heading_2",
        "heading_2": {"rich_text": [{"text": {"content": "Full Transcript"}}]}
    })
    
    # Create toggle with transcript chunks
    transcript_chunks = list(chunk_text(transcript, 1900))
    
    toggle_children = []
    for chunk in transcript_chunks[:50]:  # Notion has limits on children
        toggle_children.append({
            "type": "paragraph",
            "paragraph": {"rich_text": [{"text": {"content": chunk}}]}
        })
    
    if len(transcript_chunks) > 50:
        toggle_children.append({
            "type": "paragraph",
            "paragraph": {"rich_text": [{"text": {"content": "[Transcript truncated due to length...]"}}]}
        })
    
    blocks.append({
        "type": "toggle",
        "toggle": {
            "rich_text": [{"text": {"content": "Click to expand full transcript"}}],
            "children": toggle_children
        }
    })
    
    # Append all blocks to the page
    # Notion limits to 100 blocks per request, so batch if needed
    for i in range(0, len(blocks), 100):
        batch = blocks[i:i+100]
        notion.blocks.children.append(page_id, children=batch)
    
    # Return the page URL
    return f"https://notion.so/{page_id.replace('-', '')}"


def chunk_text(text: str, max_length: int = 1900) -> list[str]:
    """Split text into chunks respecting word boundaries."""
    chunks = []
    current = ""
    
    for word in text.split():
        if len(current) + len(word) + 1 > max_length:
            if current:
                chunks.append(current.strip())
            current = word
        else:
            current = f"{current} {word}" if current else word
    
    if current:
        chunks.append(current.strip())
    
    return chunks


if __name__ == "__main__":
    # Test creating a page
    test_analysis = {
        "summary": "This is a test meeting summary.",
        "action_items": [
            {"task": "Review the document", "owner": "John", "deadline": "Friday"}
        ],
        "decisions": [
            {"decision": "Proceed with plan A", "rationale": "Lower risk"}
        ],
        "key_quotes": [
            {"quote": "Let's make it happen", "speaker": "Sarah"}
        ]
    }
    
    url = create_meeting_page(
        title="Test Meeting",
        date=datetime.now().strftime("%Y-%m-%d"),
        duration_minutes=60,
        analysis=test_analysis,
        transcript="This is a test transcript...",
        costs={"transcription": 0.18, "analysis": 0.05, "total": 0.23},
        source_file="test.mp4"
    )
    
    print(f"Created page: {url}")
