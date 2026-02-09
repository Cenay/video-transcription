# Quick Start: Creating the Linked Database View

## TL;DR

Your `.env` file has been updated with the correct database ID. Now you just need to create the linked view manually in Notion.

## 5-Step Process

### 1. Open Client Meetings Page
https://www.notion.so/cenay/1009a9bcda8f80a1b620cf84f9872904

### 2. Find the Right Spot
Scroll to find these two sections:
- "Random Links I Need To Save" ← Above
- "Fireflies Summary - Auto Page Insertion" ← Below

Position your cursor between them.

### 3. Create Linked Database
1. Type: `/linked`
2. Press: `Enter`
3. Search for: `Transcriptions`
4. Select it and choose: "Create a new view of database"

### 4. Optional: Add Heading
Above the database:
1. Type: `/heading2`
2. Enter: `Video Transcriptions`

### 5. Done!
New transcriptions will now appear in both:
- Original database: https://www.notion.so/cenay/Transcriptions-1-2e39a9bcda8f80d4a29ec248064b1bad
- Client Meetings page (your new linked view)

## Configuration Summary

✅ `.env` updated with: `NOTION_DATABASE_ID=2e39a9bc-da8f-80d4-a29e-c248064b1bad`
✅ Notion integration has access to Transcriptions database
✅ Ready to process videos

## Test the Setup

```bash
# Verify configuration
python scripts/verify_notion_setup.py

# Test with a video (dry run)
python scripts/pipeline.py /path/to/video.mp4 --dry-run

# Process a video
python scripts/pipeline.py /path/to/video.mp4
```

## Why Can't This Be Automated?

The Notion API doesn't support creating linked database views programmatically (as of February 2026). This is a known limitation. The manual `/linked` command is currently the only way to create these views.

## Need More Details?

See `NOTION_SETUP_GUIDE.md` for comprehensive documentation.
