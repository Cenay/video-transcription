#!/usr/bin/env python3
"""Transcribe audio using OpenAI Whisper API."""

import os
import json
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# Initialize client
client = OpenAI()

# Cost per minute (GPT-4o Mini Transcribe)
COST_PER_MINUTE = 0.003


def transcribe_audio(
    audio_path: str,
    model: str = "whisper-1",
    response_format: str = "verbose_json",
    language: str = "en"
) -> dict:
    """
    Transcribe a single audio file using Whisper API.
    
    Args:
        audio_path: Path to audio file (mp3, m4a, wav, etc.)
        model: Whisper model to use
        response_format: Output format (json, text, srt, verbose_json, vtt)
        language: ISO language code
        
    Returns:
        Transcription result dict with text and metadata
    """
    with open(audio_path, "rb") as audio_file:
        response = client.audio.transcriptions.create(
            model=model,
            file=audio_file,
            response_format=response_format,
            language=language,
            timestamp_granularities=["word", "segment"] if response_format == "verbose_json" else None
        )
    
    if response_format == "verbose_json":
        return response.model_dump()
    else:
        return {"text": response}


def transcribe_chunked_audio(
    chunks: list[tuple[str, int]],
    model: str = "whisper-1"
) -> dict:
    """
    Transcribe multiple audio chunks and merge results.
    
    Args:
        chunks: List of (chunk_path, start_offset_ms) tuples
        model: Whisper model to use
        
    Returns:
        Merged transcription with adjusted timestamps
    """
    all_segments = []
    all_words = []
    full_text_parts = []
    total_duration = 0
    
    for chunk_path, offset_ms in chunks:
        print(f"  Transcribing chunk: {Path(chunk_path).name}")
        
        result = transcribe_audio(chunk_path, model=model)
        offset_sec = offset_ms / 1000.0
        
        # Adjust segment timestamps
        if "segments" in result:
            for segment in result["segments"]:
                adjusted_segment = segment.copy()
                adjusted_segment["start"] += offset_sec
                adjusted_segment["end"] += offset_sec
                all_segments.append(adjusted_segment)
        
        # Adjust word timestamps
        if "words" in result:
            for word in result["words"]:
                adjusted_word = word.copy()
                adjusted_word["start"] += offset_sec
                adjusted_word["end"] += offset_sec
                all_words.append(adjusted_word)
        
        full_text_parts.append(result.get("text", ""))
        
        if "duration" in result:
            total_duration = max(total_duration, offset_sec + result["duration"])
    
    # Deduplicate overlapping segments (from the 30-second overlap)
    all_segments = deduplicate_segments(all_segments)
    
    return {
        "text": " ".join(full_text_parts),
        "segments": all_segments,
        "words": all_words,
        "duration": total_duration,
        "language": "en"
    }


def deduplicate_segments(segments: list[dict], tolerance: float = 1.0) -> list[dict]:
    """Remove duplicate segments from overlapping chunks."""
    if not segments:
        return []
    
    # Sort by start time
    segments = sorted(segments, key=lambda x: x["start"])
    
    deduped = [segments[0]]
    for segment in segments[1:]:
        last = deduped[-1]
        # If this segment starts before the last one ends (overlap), skip it
        if segment["start"] < last["end"] - tolerance:
            continue
        deduped.append(segment)
    
    return deduped


def estimate_cost(duration_seconds: float) -> float:
    """Estimate transcription cost based on duration."""
    minutes = duration_seconds / 60
    return minutes * COST_PER_MINUTE


def format_timestamp(seconds: float) -> str:
    """Format seconds as HH:MM:SS."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        audio = sys.argv[1]
        print(f"Transcribing: {audio}")
        result = transcribe_audio(audio)
        print(f"Text length: {len(result.get('text', ''))} characters")
        print(f"First 500 chars: {result.get('text', '')[:500]}...")
