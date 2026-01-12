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
    
    # Build formatted text with speaker labels (interactive prompt)
    if speaker_labels and transcript.utterances:
        formatted_text = identify_user_speaker(transcript.utterances)
    else:
        formatted_text = transcript.text
    
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


def identify_speakers(utterances, user_name: str = "Cenay") -> str:
    """
    Interactive prompt to identify all speakers in the transcript.

    Args:
        utterances: List of utterance objects from AssemblyAI
        user_name: Default name for the user

    Returns:
        Formatted transcript with speaker names replacing labels
    """
    if not utterances:
        return ""

    # Find first utterance from each speaker
    speaker_previews = {}
    for utterance in utterances:
        speaker = utterance.speaker
        if speaker not in speaker_previews:
            preview = utterance.text.strip()[:80]
            if len(utterance.text.strip()) > 80:
                preview += "..."
            speaker_previews[speaker] = preview

    speakers = sorted(speaker_previews.keys())

    # Show previews
    print("\n  Speaker identification:")
    print("  " + "-" * 50)
    for speaker in speakers:
        print(f"  Speaker {speaker} first said: \"{speaker_previews[speaker]}\"")
    print("  " + "-" * 50)

    # Ask which speaker is the user
    prompt = f"  Which speaker are you? [{'/'.join(speakers)}/skip]: "
    speaker_names = {}

    while True:
        choice = input(prompt).strip().upper()
        if choice.lower() == "skip":
            print("  → Keeping original speaker labels")
            return format_with_speakers(utterances)
        elif choice in speakers:
            speaker_names[choice] = user_name
            print(f"  → Speaker {choice} = {user_name}")
            break
        else:
            print(f"  Invalid choice. Enter {', '.join(speakers)}, or skip.")

    # Ask for names of other speakers
    other_speakers = [s for s in speakers if s not in speaker_names]
    for speaker in other_speakers:
        preview = speaker_previews[speaker][:50] + "..." if len(speaker_previews[speaker]) > 50 else speaker_previews[speaker]
        name = input(f"  Name for Speaker {speaker} (\"{preview}\") [enter to skip]: ").strip()
        if name:
            speaker_names[speaker] = name
            print(f"  → Speaker {speaker} = {name}")
        else:
            print(f"  → Keeping Speaker {speaker}")

    return format_with_speaker_names(utterances, speaker_names)


def format_with_speaker_names(utterances, speaker_names: dict) -> str:
    """
    Format utterances with custom names for speakers.

    Args:
        utterances: List of utterance objects
        speaker_names: Dict mapping speaker labels to names (e.g., {"A": "Cenay", "B": "John"})
    """
    if not utterances:
        return ""

    lines = []
    current_speaker = None

    for utterance in utterances:
        speaker = utterance.speaker
        text = utterance.text.strip()

        # Use custom name if provided, otherwise keep Speaker X format
        label = speaker_names.get(speaker, f"Speaker {speaker}")

        if speaker != current_speaker:
            if current_speaker is not None:
                lines.append("")
            lines.append(f"{label}: {text}")
            current_speaker = speaker
        else:
            lines[-1] += f" {text}"

    return "\n".join(lines)


# Keep old function name as alias for compatibility
def identify_user_speaker(utterances, user_name: str = "Cenay") -> str:
    """Alias for identify_speakers for backward compatibility."""
    return identify_speakers(utterances, user_name)


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
        