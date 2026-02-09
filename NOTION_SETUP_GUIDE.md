# Notion Database Setup Guide

## Overview

This guide explains how to set up the Notion integration for the video transcription pipeline, including creating a linked database view on the Client Meetings page.

## Current Configuration

### Database Information

- **Database Name**: Transcriptions
- **Database ID**: `2e39a9bc-da8f-80d4-a29e-c248064b1bad`
- **Original Location**: https://www.notion.so/cenay/Transcriptions-1-2e39a9bcda8f80d4a29ec248064b1bad

### Target Page

- **Page Name**: Client Meetings
- **Page ID**: `1009a9bcda8f80a1b620cf84f9872904`
- **URL**: https://www.notion.so/cenay/Client-Meetings-1009a9bcda8f80a1b620cf84f9872904

## Environment Variables

Your `.env` file has been updated with the correct database ID:

```bash
NOTION_DATABASE_ID=2e39a9bc-da8f-80d4-a29e-c248064b1bad
```

This points to the **Transcriptions** database. All new transcription entries will be created in this database.

## Creating a Linked Database View

### Why Use a Linked Database?

A linked database view allows you to display the same database in multiple locations without duplicating data. Changes made in one view automatically appear in all other views.

### Important: API Limitation

**The Notion API does not support creating linked database views programmatically.** You must create the linked view manually in the Notion interface.

### Step-by-Step Instructions

1. **Open the Client Meetings page** in Notion:
   - URL: https://www.notion.so/cenay/1009a9bcda8f80a1b620cf84f9872904

2. **Navigate to the correct position**:
   - Scroll down to find the "Random Links I Need To Save" section
   - Position your cursor **below** this section
   - The linked database should go **above** the "Fireflies Summary - Auto Page Insertion" section

3. **Create the linked database**:
   - Type `/linked` and press Enter
   - In the search box, type "Transcriptions"
   - Select the "Transcriptions" database from the dropdown
   - Choose "Create a new view of database"

4. **Optional: Add a heading**:
   - Above the linked database, type `/heading2`
   - Enter "Video Transcriptions" as the heading text

5. **Verify the setup**:
   - The linked database should now appear on the Client Meetings page
   - Any existing transcription entries should be visible
   - New entries created by the pipeline will appear in both:
     - The original Transcriptions page
     - The linked view on Client Meetings page

## Database Schema

The Transcriptions database includes the following properties:

| Property | Type | Description |
|----------|------|-------------|
| Name | Title | Meeting/video title |
| Date | Date | Date of the meeting/recording |
| Duration | Text | Length in minutes |
| Status | Select | Processing status (Complete, etc.) |
| Cost | Number | Total API costs for processing |
| Source File | Text | Original video file path |

## How the Pipeline Works

When you run the transcription pipeline:

```bash
python scripts/pipeline.py /path/to/video.mp4
```

The pipeline will:

1. Extract audio from the video
2. Transcribe using AssemblyAI
3. Analyze the transcript with Claude
4. Create a new page in the **Transcriptions** database
5. The new page automatically appears in:
   - The original Transcriptions database view
   - The linked database view on Client Meetings page

## Troubleshooting

### "Could not find database" Error

If you get an error about the database not being found:

1. Verify the Notion integration has access to the Transcriptions database
2. In Notion, open the Transcriptions database
3. Click the "..." menu (top right)
4. Select "Connections" or "Add connections"
5. Add your integration if it's not listed

### Database ID Format

Database IDs in Notion are 32 characters with dashes in UUID format:
```
2e39a9bc-da8f-80d4-a29e-c248064b1bad
```

The `.env` file has been updated with this correct format.

### Linked View Not Updating

If the linked view doesn't show new entries:

1. Refresh the Notion page
2. Verify the filter settings on the linked view (ensure no filters are hiding entries)
3. Check that the original database contains the new entry

## Testing the Setup

After creating the linked database view, test the integration:

```bash
# Test connections
python scripts/test_connections.py

# Estimate costs for a video (dry run)
python scripts/pipeline.py /path/to/test-video.mp4 --dry-run

# Process a short test video
python scripts/pipeline.py /path/to/test-video.mp4
```

Check both locations to verify the new entry appears:
- Original: https://www.notion.so/cenay/Transcriptions-1-2e39a9bcda8f80d4a29ec248064b1bad
- Linked view: On the Client Meetings page

## Page Structure Reference

The Client Meetings page structure (as of the last analysis):

```
[Header content]
  ...
  Random Links I Need To Save (Heading 3)
  [Random links section content]

  ← INSERT LINKED DATABASE HERE

  Fireflies Summary - Auto Page Insertion (Heading 3)
  [Fireflies content]
  ...
```

## Additional Notes

- The linked database view maintains its own view settings (filters, sorts, displayed properties)
- You can customize the linked view without affecting the original database or other views
- All data is stored in the original Transcriptions database
- Deleting a page in any view deletes it from the database entirely
