#!/usr/bin/env python3
"""Extract and chunk audio from video files for transcription processing."""

import os
import subprocess
import tempfile
from pathlib import Path
from pydub import AudioSegment
from pydub.silence import detect_silence

def extract_audio(video_path: str, output_path: str = None) -> str:
    """
    Extract audio from video, optimized for transcription (16kHz mono MP3).

    Args:
        video_path: Path to input video file
        output_path: Optional output path. If None, creates temp file.

    Returns:
        Path to extracted audio file
    """
    video_path = Path(video_path)

    if output_path is None:
        output_path = video_path.with_suffix('.mp3')

    cmd = [
        'ffmpeg',
        '-i', str(video_path),
        '-ar', '16000',      # 16kHz sample rate
        '-ac', '1',          # Mono
        '-b:a', '64k',       # 64kbps bitrate (~28MB per hour)
        '-y',                # Overwrite output
        str(output_path)
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg failed: {result.stderr}")
    
    return str(output_path)


def get_audio_duration(audio_path: str) -> float:
    """Get duration of audio file in seconds."""
    audio = AudioSegment.from_file(audio_path)
    return len(audio) / 1000.0


def chunk_audio_at_silence(
    audio_path: str,
    max_chunk_duration_ms: int = 600000,  # 10 minutes
    min_silence_len: int = 700,            # 700ms silence
    silence_thresh: int = -40,             # dB threshold
    overlap_ms: int = 30000                # 30 second overlap
) -> list[tuple[str, int]]:
    """
    Split audio at silence boundaries for better transcription.
    
    Returns list of (chunk_path, start_offset_ms) tuples.
    """
    audio = AudioSegment.from_file(audio_path)
    audio_path = Path(audio_path)
    
    # Find silence points
    silences = detect_silence(
        audio,
        min_silence_len=min_silence_len,
        silence_thresh=silence_thresh
    )
    
    chunks = []
    start = 0
    chunk_num = 0
    
    for silence_start, silence_end in silences:
        # If we've accumulated enough audio, split here
        if silence_start - start >= max_chunk_duration_ms:
            split_point = (silence_start + silence_end) // 2
            
            chunk = audio[start:split_point]
            chunk_path = audio_path.parent / f"{audio_path.stem}_chunk{chunk_num:03d}.mp3"
            chunk.export(str(chunk_path), format="mp3")
            
            chunks.append((str(chunk_path), start))
            chunk_num += 1
            
            # Start next chunk with overlap for context continuity
            start = split_point - overlap_ms
    
    # Don't forget the last chunk
    if start < len(audio):
        chunk = audio[start:]
        chunk_path = audio_path.parent / f"{audio_path.stem}_chunk{chunk_num:03d}.mp3"
        chunk.export(str(chunk_path), format="mp3")
        chunks.append((str(chunk_path), start))
    
    return chunks


def get_file_size_mb(file_path: str) -> float:
    """Get file size in megabytes."""
    return os.path.getsize(file_path) / (1024 * 1024)


def needs_chunking(audio_path: str, max_size_mb: float = 24.0) -> bool:
    """Check if audio file exceeds size threshold (AssemblyAI handles large files natively)."""
    return get_file_size_mb(audio_path) > max_size_mb


if __name__ == "__main__":
    # Quick test
    import sys
    if len(sys.argv) > 1:
        video = sys.argv[1]
        print(f"Extracting audio from: {video}")
        audio = extract_audio(video)
        print(f"Audio saved to: {audio}")
        print(f"Size: {get_file_size_mb(audio):.2f} MB")
        print(f"Duration: {get_audio_duration(audio)/60:.1f} minutes")
        print(f"Needs chunking: {needs_chunking(audio)}")
