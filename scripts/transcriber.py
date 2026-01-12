#!/usr/bin/env python3
"""Transcribe audio using AssemblyAI API with speaker diarization."""

import os
import assemblyai as aai
from dotenv import load_dotenv

load_dotenv()

# Configure AssemblyAI
aai.settings.api_key = os.environ.get("ASSEMBLYAI_API_KEY")

# Cost per minute (AssemblyAI with speaker diarization)
COST_PER_MINUTE = 0.0062


def transcribe_audio(
    audio_path: str,
    speaker_labels: bool = True,
    language_code: str = "en"
) -> dict:
    """
    Transcribe an audio file using AssemblyAI.
    """
    config = aai.TranscriptionConfig(
        speaker_labels=speaker_labels,
        language_code=language_code,
        punctuate=True,
        format_text=True
    )
    
    transcriber = aai.Transcriber()
    transcript = transcriber.transcribe(audio_path, config=config)
    
    if transcript.status == aai.TranscriptStatus.error:
        raise RuntimeError(f"Transcription failed: {transcript.error}")
    
    # Build formatted text with speaker labels
    formatted_text = format_with_speakers(transcript.utterances) if speaker_labels else transcript.text
    
    result = {
        "text": formatted_text,
        "raw_text": transcript.text,
        "duration": transcript.audio_duration,
        "language": language_code,
        "confidence": transcript.confidence,
        "utterances": []
    }
    
    if transcript.utterances:
        for utterance in transcript.utterances:
            result["utterances"].append({
                "speaker": utterance.speaker,
                "text": utterance.text,
                "start": utterance.start / 1000,
                "end": utterance.end / 1000,
                "confidence": utterance.confidence
            })
    
    return result


def format_with_speakers(utterances) -> str:
    """Format utterances with speaker labels and paragraph breaks."""
    if not utterances:
        return ""
    
    lines = []
    current_speaker = None
    
    for utterance in utterances:
        speaker = utterance.speaker
        text = utterance.text.strip()
        
        if speaker != current_speaker:
            if current_speaker is not None:
                lines.append("")
            lines.append(f"Speaker {speaker}: {text}")
            current_speaker = speaker
        else:
            lines[-1] += f" {text}"
    
    return "\n".join(lines)


def estimate_cost(duration_seconds: float) -> float:
    """Estimate transcription cost based on duration."""
    minutes = duration_seconds / 60
    return minutes * COST_PER_MINUTE


def transcribe_chunked_audio(chunks: list, model: str = None) -> dict:
    """AssemblyAI handles large files natively. Kept for compatibility."""
    if chunks:
        audio_path = chunks[0][0]
        return transcribe_audio(audio_path)
    return {"text": "", "utterances": []}


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        audio = sys.argv[1]
        print(f"Transcribing: {audio}")
        result = transcribe_audio(audio)
        print(f"\nDuration: {result.get('duration', 0) / 60:.1f} minutes")
        print(f"Confidence: {result.get('confidence', 0):.1%}")
        print(f"\n--- Transcript Preview ---\n")
        print(result.get("text", "")[:1500])