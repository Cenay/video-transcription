#!/bin/bash
# transcribe-this.sh - Run video/audio transcription pipeline
# Usage: transcribe-this /path/to/file.mp4

PIPELINE_DIR="/mnt/k/Code/TRFA-Project-Files/video-transcription"

# Check if file argument provided
if [ -z "$1" ]; then
    echo "Usage: transcribe-this /path/to/file.mp4"
    echo "       transcribe-this /path/to/file.m4a --dry-run"
    exit 1
fi

# Check if file exists
if [ ! -f "$1" ]; then
    echo "Error: File not found: $1"
    exit 1
fi

# Get the file path and shift to capture any additional args (like --dry-run)
FILE_PATH="$1"
shift
EXTRA_ARGS="$@"

# Run the pipeline
cd "$PIPELINE_DIR" && source venv/bin/activate && python scripts/pipeline.py "$FILE_PATH" $EXTRA_ARGS
