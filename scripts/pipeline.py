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
import subprocess
import tempfile
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Import our modules
from audio_extractor import extract_audio
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
    keep_temp: bool = False,
    from_cache: bool = False,
    transcribe_only: bool = False
) -> dict:
    """
    Process a video file through the complete pipeline.

    Args:
        video_path: Path to MP4 video
        dry_run: If True, estimate costs without processing
        keep_temp: If True, don't delete temporary files
        from_cache: If True, skip transcription and use cached transcript
        transcribe_only: If True, stop after transcription (no analysis or Notion)

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
        if from_cache:
            # Load cached transcript — skip audio extraction and transcription
            cache_dir = Path(os.environ.get("TEMP_DIR", tempfile.gettempdir())) / "transcribe-cache"
            cache_file = cache_dir / f"{video_path.stem}-raw-transcript.json"

            if not cache_file.exists():
                raise FileNotFoundError(f"No cached transcript found: {cache_file}")

            print("\n[1/4] Skipping audio extraction (using cache)")
            print(f"\n[2/4] Loading cached transcript: {cache_file}")

            cache_data = json.loads(cache_file.read_text())
            duration_sec = cache_data.get("duration", 0)
            duration_min = duration_sec / 60

            print(f"  Duration: {duration_min:.1f} minutes")

            result["duration_minutes"] = duration_min
            result["costs"]["transcription"] = 0  # Already paid for

            # Re-run speaker identification on cached utterances
            from transcriber import identify_speakers, format_with_speakers

            class FakeUtterance:
                def __init__(self, d):
                    self.speaker = d["speaker"]
                    self.text = d["text"]
                    self.start = d.get("start", 0)
                    self.end = d.get("end", 0)

            utterances = [FakeUtterance(u) for u in cache_data.get("utterances", [])]

            if utterances:
                transcript_text = identify_speakers(utterances)
            else:
                transcript_text = cache_data.get("raw_text", "")

            print(f"  Transcript loaded: {len(transcript_text)} characters")

            result["transcript"] = transcript_text
            result["transcription_metadata"] = {
                "duration": duration_sec,
                "language": "en",
                "utterance_count": len(utterances),
                "confidence": None
            }
        else:
            # Step 1: Extract audio with error tolerance
            print("\n[1/4] Extracting audio...")
            audio_path = temp_dir / f"{video_path.stem}.wav"
            audio_path = extract_audio(str(video_path), str(audio_path))

            # Get duration from extracted audio
            probe_cmd = [
                'ffprobe',
                '-v', 'error',
                '-show_entries', 'format=duration',
                '-of', 'default=noprint_wrappers=1:nokey=1',
                str(audio_path)
            ]
            probe_result = subprocess.run(probe_cmd, capture_output=True, text=True)
            if probe_result.returncode != 0:
                raise RuntimeError(f"ffprobe failed: {probe_result.stderr}")

            duration_sec = float(probe_result.stdout.strip())
            duration_min = duration_sec / 60
            file_size_mb = os.path.getsize(audio_path) / (1024 * 1024)

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

            # Step 2: Transcribe audio
            print("\n[2/4] Transcribing audio...")
            transcription = transcribe_audio(str(audio_path))

            transcript_text = transcription.get("text", "")
            print(f"  Transcription complete: {len(transcript_text)} characters")

            result["transcript"] = transcript_text
            result["transcription_metadata"] = {
                "duration": transcription.get("duration"),
                "language": transcription.get("language"),
                "utterance_count": len(transcription.get("utterances", [])),
                "confidence": transcription.get("confidence")
            }
        
        # Transcribe-only mode: save transcript and stop
        if transcribe_only:
            output_dir = Path(__file__).parent.parent / "output"
            output_dir.mkdir(parents=True, exist_ok=True)
            output_file = output_dir / f"{video_path.stem}-transcript.txt"
            output_file.write_text(transcript_text)

            result["costs"]["total"] = result["costs"].get("transcription", 0)
            result["output_file"] = str(output_file)

            print(f"\n{'='*60}")
            print("Transcription Complete (transcribe-only mode)")
            print(f"{'='*60}")
            print(f"  Duration: {duration_min:.1f} minutes")
            print(f"  Total cost: ${result['costs']['total']:.4f}")
            print(f"  Saved to: {output_file}")
            print(f"  Characters: {len(transcript_text)}")

            return result

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

        # Guard: if analysis failed to parse after retries, abort BEFORE creating
        # a Notion page or uploading to S3. Publishing an empty "No summary
        # available" page (and paying to host it) is worse than failing loudly.
        if "error" in analysis:
            cache_dir = temp_dir / "transcribe-cache"
            cache_dir.mkdir(parents=True, exist_ok=True)
            raw_path = cache_dir / f"{video_path.stem}-FAILED-ANALYSIS.txt"
            raw_path.write_text(analysis.get("raw_response", ""))
            print(f"\n❌ Analysis failed: {analysis['error']}")
            print(f"   Raw response saved to: {raw_path}")
            print("   Aborting — no Notion page created, no S3 upload.")
            print("   Re-run with --from-cache once resolved (no re-transcription cost).")
            result["error"] = analysis["error"]
            return result

        # Calculate total cost
        result["costs"]["total"] = (
            result["costs"].get("transcription", 0) +
            result["costs"].get("analysis", 0)
        )
        
        # Step 4: Output to Notion
        print("\n[4/4] Creating Notion page...")
        
        # Generate title from filename or use summary
        # Known acronyms that should stay uppercase
        UPPERCASE_WORDS = {"trfa", "trfaapi", "fdd", "crm", "api", "ac", "lsp", "psi"}
        raw_title = video_path.stem.replace("-", " ").replace("_", " ").title()
        title = " ".join(
            word.upper() if word.lower() in UPPERCASE_WORDS else word
            for word in raw_title.split()
        )
        date = datetime.now().strftime("%Y-%m-%d")
        
        notion_result = create_meeting_page(
            title=title,
            date=date,
            duration_minutes=duration_min,
            analysis=analysis,
            transcript=transcript_text,
            costs=result["costs"],
            source_file=video_path.name
        )

        notion_url = notion_result["url"]
        result["notion_url"] = notion_url
        result["notion_page_id"] = notion_result["page_id"]
        print(f"  Created: {notion_url}")

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
    parser.add_argument("--from-cache", action="store_true",
                       help="Skip transcription, use cached transcript from previous run")
    parser.add_argument("--transcribe-only", action="store_true",
                       help="Stop after transcription (no analysis or Notion page)")
    parser.add_argument("--output-json", type=str,
                       help="Save full result to JSON file")

    args = parser.parse_args()

    result = process_video(
        args.video,
        dry_run=args.dry_run,
        keep_temp=args.keep_temp,
        from_cache=args.from_cache,
        transcribe_only=args.transcribe_only
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
