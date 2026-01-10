#!/usr/bin/env python3
"""
Main pipeline orchestrator for video transcription and analysis.

Usage:
    python pipeline.py /path/to/video.mp4
    python pipeline.py /path/to/video.mp4 --dry-run
"""

import os
import sys
import json
import argparse
import tempfile
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Import our modules
from audio_extractor import (
    extract_audio, 
    get_audio_duration, 
    needs_chunking,
    chunk_audio_at_silence,
    get_file_size_mb
)
from transcriber import (
    transcribe_audio,
    transcribe_chunked_audio,
    estimate_cost as estimate_transcription_cost
)
from analyzer import (
    analyze_transcript,
    estimate_analysis_cost
)
from notion_output import create_meeting_page

load_dotenv()


def process_video(
    video_path: str,
    dry_run: bool = False,
    keep_temp: bool = False
) -> dict:
    """
    Process a video file through the complete pipeline.
    
    Args:
        video_path: Path to MP4 video
        dry_run: If True, estimate costs without processing
        keep_temp: If True, don't delete temporary files
        
    Returns:
        Result dict with transcript, analysis, and metadata
    """
    video_path = Path(video_path)
    
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    
    print(f"\n{'='*60}")
    print(f"Processing: {video_path.name}")
    print(f"{'='*60}")
    
    # Create temp directory for intermediate files
    temp_dir = Path(os.environ.get("TEMP_DIR", tempfile.gettempdir()))
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    result = {
        "source_file": str(video_path),
        "processed_at": datetime.now().isoformat(),
        "costs": {}
    }
    
    try:
        # Step 1: Extract audio
        print("\n[1/4] Extracting audio...")
        audio_path = temp_dir / f"{video_path.stem}.mp3"
        audio_path = extract_audio(str(video_path), str(audio_path))
        
        duration_sec = get_audio_duration(audio_path)
        duration_min = duration_sec / 60
        file_size_mb = get_file_size_mb(audio_path)
        
        print(f"  Duration: {duration_min:.1f} minutes")
        print(f"  Audio size: {file_size_mb:.1f} MB")
        
        result["duration_minutes"] = duration_min
        
        # Estimate transcription cost
        transcription_cost = estimate_transcription_cost(duration_sec)
        result["costs"]["transcription"] = transcription_cost
        print(f"  Estimated transcription cost: ${transcription_cost:.4f}")
        
        if dry_run:
            # Estimate analysis cost based on typical transcript length
            # ~150 words per minute of speech, ~4 chars per token
            estimated_words = int(duration_min * 150)
            estimated_chars = estimated_words * 5
            analysis_estimate = {"estimated_cost_usd": estimated_chars / 4 * 3 / 1_000_000}
            result["costs"]["analysis"] = analysis_estimate["estimated_cost_usd"]
            result["costs"]["total"] = transcription_cost + analysis_estimate["estimated_cost_usd"]
            
            print(f"\n[DRY RUN] Estimated total cost: ${result['costs']['total']:.4f}")
            return result
        
        # Step 2: Transcribe
        print("\n[2/4] Transcribing audio...")
        
        if needs_chunking(audio_path):
            print(f"  File exceeds 25MB limit, chunking at silence points...")
            chunks = chunk_audio_at_silence(audio_path)
            print(f"  Created {len(chunks)} chunks")
            transcription = transcribe_chunked_audio(chunks)
            
            # Clean up chunk files
            if not keep_temp:
                for chunk_path, _ in chunks:
                    Path(chunk_path).unlink(missing_ok=True)
        else:
            transcription = transcribe_audio(audio_path)
        
        transcript_text = transcription.get("text", "")
        print(f"  Transcription complete: {len(transcript_text)} characters")
        
        result["transcript"] = transcript_text
        result["transcription_metadata"] = {
            "duration": transcription.get("duration"),
            "language": transcription.get("language"),
            "segment_count": len(transcription.get("segments", []))
        }
        
        # Step 3: Analyze with Claude
        print("\n[3/4] Analyzing transcript with Claude...")
        
        analysis_estimate = estimate_analysis_cost(transcript_text)
        print(f"  Estimated analysis cost: ${analysis_estimate['estimated_cost_usd']:.4f}")
        
        analysis = analyze_transcript(transcript_text)
        
        if "_usage" in analysis:
            actual_cost = (
                (analysis["_usage"]["input_tokens"] / 1_000_000) * 3.00 +
                (analysis["_usage"]["output_tokens"] / 1_000_000) * 15.00
            )
            result["costs"]["analysis"] = actual_cost
            print(f"  Actual analysis cost: ${actual_cost:.4f}")
        
        result["analysis"] = analysis
        
        # Calculate total cost
        result["costs"]["total"] = (
            result["costs"].get("transcription", 0) +
            result["costs"].get("analysis", 0)
        )
        
        # Step 4: Output to Notion
        print("\n[4/4] Creating Notion page...")
        
        # Generate title from filename or use summary
        title = video_path.stem.replace("-", " ").replace("_", " ").title()
        date = datetime.now().strftime("%Y-%m-%d")
        
        notion_url = create_meeting_page(
            title=title,
            date=date,
            duration_minutes=duration_min,
            analysis=analysis,
            transcript=transcript_text,
            costs=result["costs"],
            source_file=video_path.name
        )
        
        result["notion_url"] = notion_url
        print(f"  Created: {notion_url}")
        
        # Cleanup
        if not keep_temp:
            Path(audio_path).unlink(missing_ok=True)
        
        # Summary
        print(f"\n{'='*60}")
        print("Processing Complete!")
        print(f"{'='*60}")
        print(f"  Duration: {duration_min:.1f} minutes")
        print(f"  Total cost: ${result['costs']['total']:.4f}")
        print(f"  Notion: {notion_url}")
        print(f"  Action items found: {len(analysis.get('action_items', []))}")
        print(f"  Decisions found: {len(analysis.get('decisions', []))}")
        
        return result
        
    except Exception as e:
        result["error"] = str(e)
        print(f"\n❌ Error: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Process video files for transcription and analysis"
    )
    parser.add_argument("video", help="Path to video file (MP4)")
    parser.add_argument("--dry-run", action="store_true", 
                       help="Estimate costs without processing")
    parser.add_argument("--keep-temp", action="store_true",
                       help="Keep temporary audio files")
    parser.add_argument("--output-json", type=str,
                       help="Save full result to JSON file")
    
    args = parser.parse_args()
    
    result = process_video(
        args.video,
        dry_run=args.dry_run,
        keep_temp=args.keep_temp
    )
    
    if args.output_json:
        output_path = Path(args.output_json)
        # Don't include full transcript in JSON (too large)
        result_slim = {k: v for k, v in result.items() if k != "transcript"}
        result_slim["transcript_length"] = len(result.get("transcript", ""))
        
        with open(output_path, "w") as f:
            json.dump(result_slim, f, indent=2)
        print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
