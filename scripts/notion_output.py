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

    # Metadata header
    blocks.append({
        "type": "paragraph",
        "paragraph": {"rich_text": [
            {"type": "text", "text": {"content": "🕒 Date: "}, "annotations": {"bold": True}},
            {"type": "text", "text": {"content": date}}
        ]}
    })
    blocks.append({
        "type": "paragraph",
        "paragraph": {"rich_text": [
            {"type": "text", "text": {"content": "🔗 Meeting Link: "}, "annotations": {"bold": True}},
            {"type": "text", "text": {"content": "[Add link after upload]"}}
        ]}
    })
    blocks.append({
        "type": "paragraph",
        "paragraph": {"rich_text": [
            {"type": "text", "text": {"content": "⌛ Duration: "}, "annotations": {"bold": True}},
            {"type": "text", "text": {"content": f"{duration_minutes:.0f} minutes"}}
        ]}
    })
    blocks.append({
        "type": "divider",
        "divider": {}
    })

    # Overview section (concise bullets)
    overview_items = analysis.get("overview", [])
    if overview_items:
        blocks.append({
            "type": "heading_3",
            "heading_3": {"rich_text": [{"text": {"content": "Overview"}}]}
        })
        for item in overview_items:
            blocks.append({
                "type": "bulleted_list_item",
                "bulleted_list_item": {"rich_text": [{"text": {"content": item}}]}
            })

    # Summary section (paragraph narrative)
    blocks.append({
        "type": "heading_2",
        "heading_2": {"rich_text": [{"text": {"content": "Summary"}}]}
    })

    summary = analysis.get("summary", "No summary available.")
    for chunk in chunk_text(summary, 1900):
        blocks.append({
            "type": "paragraph",
            "paragraph": {"rich_text": [{"text": {"content": chunk}}]}
        })

    # Notes section (topical segments with emoji headers)
    notes = analysis.get("notes", [])
    if notes:
        blocks.append({
            "type": "heading_3",
            "heading_3": {"rich_text": [{"text": {"content": "Notes"}}]}
        })
        for note in notes:
            emoji = note.get("emoji", "📌")
            title = note.get("title", "")
            blocks.append({
                "type": "paragraph",
                "paragraph": {"rich_text": [
                    {"type": "text", "text": {"content": f"{emoji} {title}"}, "annotations": {"bold": True}}
                ]}
            })
            for bullet in note.get("bullets", []):
                blocks.append({
                    "type": "bulleted_list_item",
                    "bulleted_list_item": {"rich_text": [{"text": {"content": bullet}}]}
                })

    # Keywords section
    keywords = analysis.get("keywords", [])
    if keywords:
        blocks.append({
            "type": "heading_3",
            "heading_3": {"rich_text": [{"text": {"content": "Keywords"}}]}
        })
        blocks.append({
            "type": "paragraph",
            "paragraph": {"rich_text": [{"text": {"content": ", ".join(keywords)}}]}
        })

    # Action Items section (grouped by person)
    action_items = analysis.get("action_items", [])
    if action_items:
        blocks.append({
            "type": "heading_3",
            "heading_3": {"rich_text": [{"text": {"content": "Action Items"}}]}
        })

        # Group by owner
        owners = {}
        for item in action_items:
            owner = item.get("owner", "Unassigned")
            owners.setdefault(owner, []).append(item)

        for owner, items in owners.items():
            blocks.append({
                "type": "paragraph",
                "paragraph": {"rich_text": [
                    {"type": "text", "text": {"content": f"👤 {owner}"}, "annotations": {"bold": True}}
                ]}
            })
            for item in items:
                task = item.get("task", "")
                deadline = item.get("deadline", "TBD")
                task_text = f"{task} (due: {deadline})" if deadline and deadline != "TBD" else task
                blocks.append({
                    "type": "to_do",
                    "to_do": {
                        "rich_text": [{"text": {"content": task_text}}],
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

        for quote in quotes[:5]:
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

    # Full Transcript section
    blocks.append({
        "type": "heading_3",
        "heading_3": {"rich_text": [{"text": {"content": "Transcript"}}]}
    })

    speaker_turns = [turn.strip() for turn in transcript.split("\n\n") if turn.strip()]

    for turn in speaker_turns:
        if len(turn) > 1900:
            for chunk in chunk_text(turn, 1900):
                blocks.append({
                    "type": "paragraph",
                    "paragraph": {"rich_text": [{"type": "text", "text": {"content": chunk}}]}
                })
        else:
            blocks.append({
                "type": "paragraph",
                "paragraph": {"rich_text": [{"type": "text", "text": {"content": turn}}]}
            })
    
    # Append all blocks to the page
    # Notion limits to 100 blocks per request, so batch if needed
    for i in range(0, len(blocks), 100):
        batch = blocks[i:i+100]
        notion.blocks.children.append(page_id, children=batch)
    
    # Return the page URL and page ID
    page_url = f"https://notion.so/{page_id.replace('-', '')}"
    return {"url": page_url, "page_id": page_id}


def update_meeting_link(page_id: str, meeting_url: str) -> None:
    """Update the Meeting Link block on a Notion page with the S3 URL."""
    # Find and update the Meeting Link placeholder in the page blocks
    blocks = notion.blocks.children.list(page_id)
    for block in blocks["results"]:
        if block["type"] == "paragraph":
            rich_text = block["paragraph"].get("rich_text", [])
            # Find the block that contains "Meeting Link"
            full_text = "".join(rt.get("plain_text", "") for rt in rich_text)
            if "Meeting Link" in full_text:
                notion.blocks.update(
                    block["id"],
                    paragraph={
                        "rich_text": [
                            {"type": "text", "text": {"content": "🔗 Meeting Link: "}, "annotations": {"bold": True}},
                            {"type": "text", "text": {"content": "View meeting", "link": {"url": meeting_url}}}
                        ]
                    }
                )
                return
    print("  Warning: Meeting Link block not found on page")


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
