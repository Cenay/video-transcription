#!/bin/bash
# transcribe-this.sh - Full pipeline: transcribe, upload to S3, update Notion, cleanup
# All steps run in the terminal so you can see progress.
#
# Usage: transcribe-this /path/to/file.mp4
#        transcribe-this /path/to/file.mp4 --dry-run
#        transcribe-this /path/to/file.mp4 --from-cache

PIPELINE_DIR="/mnt/k/Code/TRFA/video-transcription"
S3_BUCKET="cn-client-meetings"
S3_BASE_URL="https://cn-client-meetings.s3.amazonaws.com"

# Check if file argument provided
if [ -z "$1" ]; then
    echo "Usage: transcribe-this /path/to/file.mp4"
    echo "       transcribe-this /path/to/file.mp4 --dry-run"
    echo "       transcribe-this /path/to/file.mp4 --from-cache"
    exit 1
fi

# Check if file exists (skip check if --from-cache since we only need the name)
if [ ! -f "$1" ] && [[ "$*" != *"--from-cache"* ]]; then
    echo "Error: File not found: $1"
    exit 1
fi

# Get the file path and shift to capture any additional args
FILE_PATH="$(realpath "$1")"
FILE_NAME="$(basename "$FILE_PATH")"
FILE_DIR="$(dirname "$FILE_PATH")"
shift
EXTRA_ARGS="$@"

# Determine S3 destination based on filename prefix
if [[ "$FILE_NAME" == trfaapi-* ]]; then
    S3_BUCKET="cn-team-videos"
    S3_BASE_URL="https://cn-team-videos.s3.amazonaws.com"
    S3_PATH="TRFA API/$FILE_NAME"
elif [[ "$FILE_NAME" == trfa-* ]]; then
    S3_PATH="TRFA/$FILE_NAME"
else
    S3_PATH="$FILE_NAME"
fi

S3_URL="$S3_BASE_URL/$S3_PATH"
S3_DEST="s3://$S3_BUCKET/$S3_PATH"

# Guard: if S3 upload would create a new prefix, warn and abort.
# Known prefixes per bucket — add new ones here if needed.
S3_PREFIX="${S3_PATH%%/*}"
case "$S3_BUCKET" in
    cn-client-meetings) KNOWN_PREFIXES="TRFA" ;;
    cn-team-videos)     KNOWN_PREFIXES="TRFA API" ;;
    *)                  KNOWN_PREFIXES="" ;;
esac
# Files in bucket root have no prefix (PREFIX == filename), which is allowed
if [[ "$S3_PREFIX" != "$FILE_NAME" && ! " $KNOWN_PREFIXES " =~ " $S3_PREFIX " ]]; then
    echo ""
    echo "ERROR: Upload would create a NEW S3 folder '$S3_PREFIX' in bucket '$S3_BUCKET'."
    echo "  Full destination: $S3_DEST"
    echo "  Known prefixes for this bucket: $KNOWN_PREFIXES"
    echo ""
    echo "If this is intentional, add '$S3_PREFIX' to KNOWN_PREFIXES in transcribe-this.sh"
    echo "Aborting. No files were uploaded."
    exit 1
fi

echo "============================================================"
echo "Transcribe This - Full Pipeline"
echo "============================================================"
echo "  File: $FILE_NAME"
echo "  S3 destination: $S3_DEST"
echo "  S3 URL: $S3_URL"
echo "============================================================"

# Check for dry-run
if [[ "$EXTRA_ARGS" == *"--dry-run"* ]]; then
    echo ""
    echo "[DRY RUN] Would upload to: $S3_DEST"
    echo "[DRY RUN] Meeting link would be: $S3_URL"
    echo ""
    cd "$PIPELINE_DIR" && source venv/bin/activate && python scripts/pipeline.py "$FILE_PATH" $EXTRA_ARGS
    exit $?
fi

# ============================================================
# STEP 1: Transcription pipeline (interactive for speaker ID)
# ============================================================
echo ""
echo "[Step 1/4] Running transcription pipeline..."
echo "(You may be asked to identify speakers)"
echo ""

cd "$PIPELINE_DIR" && source venv/bin/activate

PIPELINE_LOG="/tmp/transcribe-pipeline-output-$$.log"
python scripts/pipeline.py "$FILE_PATH" $EXTRA_ARGS 2>&1 | tee "$PIPELINE_LOG"
PIPELINE_EXIT=${PIPESTATUS[0]}

if [ $PIPELINE_EXIT -ne 0 ]; then
    echo ""
    echo "Pipeline failed. Aborting."
    rm -f "$PIPELINE_LOG"
    notify-send -u critical "Transcribe This" "Pipeline failed for $FILE_NAME" 2>/dev/null
    exit 1
fi

# Extract Notion page ID from captured output
NOTION_PAGE_ID=$(grep -oP 'notion\.so/\K[a-f0-9]+' "$PIPELINE_LOG" | head -1)
rm -f "$PIPELINE_LOG"

echo ""
echo "============================================================"
echo "[Step 1 Complete] Transcription and Notion page created!"
echo "============================================================"

# ============================================================
# STEP 2: Upload to S3
# ============================================================
echo ""
echo "[Step 2/4] Uploading to S3..."
echo "  Destination: $S3_DEST"
echo "  This may take a while for large files..."
echo ""

aws s3 cp "$FILE_PATH" "$S3_DEST"
S3_EXIT=$?

if [ $S3_EXIT -ne 0 ]; then
    echo ""
    echo "S3 upload failed. Local files preserved."
    notify-send -u critical "Transcribe This" "S3 upload failed for $FILE_NAME" 2>/dev/null
    exit 1
fi

echo ""
echo "  Upload complete!"

# ============================================================
# STEP 3: Update Notion page with meeting link
# ============================================================
echo ""
if [ -n "$NOTION_PAGE_ID" ]; then
    echo "[Step 3/4] Updating Notion page with meeting link..."
    python -c "
from scripts.notion_output import update_meeting_link
update_meeting_link('$NOTION_PAGE_ID', '$S3_URL')
print('  Meeting link updated!')
" 2>&1
else
    echo "[Step 3/4] Skipping Notion update (no page ID captured)"
fi

# ============================================================
# STEP 4: Clean up local Zoom folder
# ============================================================
echo ""
echo "[Step 4/4] Verifying S3 upload before cleanup..."

# Verify the file actually exists in S3 before touching local files
S3_CHECK=$(aws s3 ls "$S3_DEST" 2>&1)
if [ -z "$S3_CHECK" ]; then
    echo "  ERROR: File NOT found in S3! Local files preserved."
    echo "  Expected: $S3_DEST"
    notify-send -u critical "Transcribe This" "S3 verification failed! Local files kept." 2>/dev/null
else
    echo "  Verified in S3: $S3_CHECK"

    ZOOM_DIR="$HOME/Videos/Zoom"
    if [[ "$FILE_DIR" == "$ZOOM_DIR"/* && "$FILE_DIR" != "$ZOOM_DIR" ]]; then
        echo "  Moving Zoom folder to trash: $FILE_DIR"
        gio trash "$FILE_DIR" 2>/dev/null
        if [ $? -ne 0 ]; then
            # Fallback: move to archived folder if gio trash fails
            ARCHIVE_DIR="$HOME/Videos/Zoom/.archived/$(date +%Y-%m-%d)"
            mkdir -p "$ARCHIVE_DIR"
            mv "$FILE_DIR" "$ARCHIVE_DIR/"
            echo "  Moved to: $ARCHIVE_DIR/$(basename "$FILE_DIR")"
        else
            echo "  Moved to trash. Recoverable from trash can."
        fi
    else
        echo "  File is not in a Zoom recording folder. Skipping cleanup."
        echo "  File location: $FILE_DIR"
    fi
fi

# ============================================================
# DONE
# ============================================================
echo ""
echo "============================================================"
echo "All done!"
echo "============================================================"
echo "  S3 URL: $S3_URL"
if [ -n "$NOTION_PAGE_ID" ]; then
    echo "  Notion: https://notion.so/$NOTION_PAGE_ID"
fi
echo "============================================================"

notify-send -i video-x-generic "Transcribe This" "Complete! $FILE_NAME uploaded and processed." 2>/dev/null
