#!/usr/bin/env python3
"""Extract audio from video files by simply copying the audio stream."""

import os
import subprocess
from pathlib import Path

def extract_audio(video_path: str, output_path: str = None) -> str:
    """
    Extract audio from video with error tolerance for corrupted streams.
    Outputs WAV format which is most compatible with transcription services.

    Args:
        video_path: Path to input video file
        output_path: Optional output path. If None, creates temp file with .wav extension.

    Returns:
        Path to extracted audio file
    """
    video_path = Path(video_path)

    if output_path is None:
        output_path = video_path.with_suffix('.wav')

    cmd = [
        'ffmpeg',
        '-err_detect', 'ignore_err',     # Ignore decoding errors
        '-fflags', '+genpts+igndts',     # Generate timestamps, ignore DTS errors
        '-i', str(video_path),
        '-vn',                           # No video
        '-acodec', 'pcm_s16le',          # Convert to uncompressed WAV
        '-ar', '16000',                  # 16kHz for transcription
        '-ac', '1',                      # Mono
        '-y',                            # Overwrite output
        str(output_path)
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    # Check if output file was created (FFmpeg may fail but produce partial audio)
    if not Path(output_path).exists():
        raise RuntimeError(f"FFmpeg failed to create output: {result.stderr}")

    # Verify the output file has content
    file_size = Path(output_path).stat().st_size
    if file_size == 0:
        raise RuntimeError(f"FFmpeg produced empty output file")

    # Warn if FFmpeg reported errors but still produced output
    if result.returncode != 0:
        print(f"  ⚠ Warning: FFmpeg reported errors but created {file_size / (1024*1024):.1f}MB audio file")
        print(f"  This may be a partial conversion due to stream corruption")

    return str(output_path)


def get_file_size_mb(file_path: str) -> float:
    """Get file size in megabytes."""
    return os.path.getsize(file_path) / (1024 * 1024)


def get_audio_duration(audio_path: str) -> float:
    """Get duration of audio file in seconds using ffprobe."""
    cmd = [
        'ffprobe',
        '-v', 'error',
        '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        str(audio_path)
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {result.stderr}")

    return float(result.stdout.strip())


def needs_chunking(audio_path: str, max_size_mb: float = 100.0) -> bool:
    """Check if audio file exceeds size threshold (AssemblyAI handles large files natively)."""
    return get_file_size_mb(audio_path) > max_size_mb


def chunk_audio_at_silence(
    audio_path: str,
    max_chunk_duration_ms: int = 600000,
    min_silence_len: int = 700,
    silence_thresh: int = -40,
    overlap_ms: int = 30000
) -> list[tuple[str, int]]:
    """
    Chunking not needed - AssemblyAI handles large files natively.
    Kept for backward compatibility.
    """
    raise NotImplementedError("Chunking not needed - AssemblyAI handles large files")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        video = sys.argv[1]
        print(f"Extracting audio from: {video}")
        audio = extract_audio(video)
        print(f"Audio saved to: {audio}")
        print(f"Size: {get_file_size_mb(audio):.2f} MB")
        print(f"Duration: {get_audio_duration(audio)/60:.1f} minutes")
        print(f"Needs chunking: {needs_chunking(audio)}")
