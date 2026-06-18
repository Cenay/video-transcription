#!/usr/bin/env python3
"""Analyze transcripts using Claude API."""

import os
import json
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

client = Anthropic()

# Claude model used for analysis. Bump this one line to change models everywhere
# (test_connections.py and diagnose_analysis.py import it from here).
# Use a no-date alias so a new dated snapshot doesn't 404 the pipeline; a major
# version bump (e.g. to Opus, or Sonnet 5) is a deliberate edit here.
MODEL = "claude-sonnet-4-6"

# Pricing per million tokens (Claude Sonnet 4.6)
INPUT_COST_PER_M = 3.00
OUTPUT_COST_PER_M = 15.00


ANALYSIS_PROMPT = """Analyze this meeting transcript and extract structured information.

Return your analysis as JSON with this exact structure:
{{
  "overview": [
    "Concise bullet point summarizing a key topic or outcome",
    "Another bullet point covering a different aspect of the meeting",
    "Continue with additional high-level points as needed (typically 4-8 bullets)"
  ],
  "summary": "2-3 paragraph executive summary of what was discussed and decided",
  "notes": [
    {{
      "emoji": "📧",
      "title": "Short descriptive title for this topic section",
      "bullets": [
        "Key point or detail discussed in this topic",
        "Another relevant detail or outcome",
        "Additional context as needed"
      ]
    }}
  ],
  "keywords": ["Keyword1", "Keyword2", "Keyword3"],
  "action_items": [
    {{
      "task": "Description of the task",
      "owner": "Person responsible (or 'Unassigned' if unclear)",
      "deadline": "Due date if mentioned (or 'TBD')"
    }}
  ],
  "decisions": [
    {{
      "decision": "What was decided",
      "rationale": "Why this was decided",
      "participants": "Who was involved in the decision"
    }}
  ],
  "meeting_metadata": {{
    "apparent_purpose": "What the meeting was about",
    "tone": "General tone (collaborative, tense, brainstorming, etc.)",
    "participation_notes": "Any notes about participation patterns"
  }}
}}

Guidelines:
- Overview bullets should be concise, high-level takeaways (one line each)
- Notes sections should break the meeting into logical topic segments (typically 5-12 sections)
- Pick a contextually appropriate emoji for each notes section title
- Keywords should be useful for searching/finding this meeting later (typically 4-8 keywords)
- Group action items by owner — include all action items for each person
- Look for commitments: "I'll do X", "will review Y", "let's plan to Z"
- Capture explicit decisions: "we agreed", "approved", "decided to"
- If speaker names aren't clear, use "Speaker" or describe by role if apparent
- Mark uncertain items with (unclear) or (approximate)

TRANSCRIPT:
{transcript}

Return ONLY valid JSON, no additional text or markdown formatting."""


def _extract_json(text: str) -> str:
    """Pull the JSON object out of a model reply that may wrap it in markdown
    fences or surround it with stray prose."""
    t = text.strip()
    if "```json" in t:
        t = t.split("```json", 1)[1].split("```", 1)[0]
    elif "```" in t:
        t = t.split("```", 1)[1].split("```", 1)[0]
    t = t.strip()
    # Trim anything before the first { and after the last } (defensive against
    # leading/trailing prose the model sometimes adds despite instructions).
    start, end = t.find("{"), t.rfind("}")
    if start != -1 and end != -1 and end > start:
        t = t[start:end + 1]
    return t


def analyze_transcript(
    transcript: str,
    model: str = MODEL,
    max_tokens: int = 8192,
    max_attempts: int = 3,
) -> dict:
    """
    Analyze a transcript using Claude.

    Retries on JSON parse failure (the model occasionally returns malformed
    JSON or stray prose) before falling back to an explicit error marker, so a
    one-off bad response can't silently produce an empty page.

    Args:
        transcript: Full transcript text
        model: Claude model to use
        max_tokens: Maximum response tokens
        max_attempts: How many times to retry on a JSON parse failure

    Returns:
        Parsed analysis dict, or {"error", "raw_response", "_usage"} if every
        attempt failed to parse.
    """
    prompt = ANALYSIS_PROMPT.format(transcript=transcript)
    messages = [{"role": "user", "content": prompt}]

    total_input = total_output = 0
    last_error = None
    last_response = ""

    for attempt in range(1, max_attempts + 1):
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            messages=messages,
        )
        total_input += response.usage.input_tokens
        total_output += response.usage.output_tokens
        last_response = response.content[0].text

        # A truncated response is the classic cause of invalid JSON — surface it.
        if response.stop_reason == "max_tokens":
            print(f"  ⚠️  Analysis hit max_tokens ({max_tokens}) on attempt "
                  f"{attempt} — response truncated, JSON will be incomplete")

        try:
            analysis = json.loads(_extract_json(last_response))
            analysis["_usage"] = {
                "input_tokens": total_input,
                "output_tokens": total_output,
                "model": model,
                "attempts": attempt,
            }
            return analysis
        except json.JSONDecodeError as e:
            last_error = e
            print(f"  ⚠️  Analysis JSON parse failed on attempt "
                  f"{attempt}/{max_attempts}: {e}")
            # Nudge the model toward strictly valid JSON on the retry.
            messages = [{
                "role": "user",
                "content": prompt + "\n\nIMPORTANT: Your previous reply was not "
                "valid JSON. Reply with ONLY a single valid JSON object — no "
                "prose, no markdown fences, no trailing commentary.",
            }]

    # Every attempt failed — return an explicit error so the pipeline aborts
    # instead of publishing an empty page.
    return {
        "error": f"Failed to parse JSON after {max_attempts} attempts: {last_error}",
        "raw_response": last_response,
        "_usage": {
            "input_tokens": total_input,
            "output_tokens": total_output,
            "model": model,
            "attempts": max_attempts,
        },
    }


def estimate_analysis_cost(transcript: str, model: str = MODEL) -> dict:
    """
    Estimate the cost to analyze a transcript.
    
    Rough estimate: 1 token ≈ 4 characters for English text
    """
    # Estimate input tokens (transcript + prompt template)
    prompt_overhead = 800  # Approximate tokens in prompt template
    transcript_tokens = len(transcript) / 4
    input_tokens = transcript_tokens + prompt_overhead
    
    # Estimate output tokens (structured JSON response)
    output_tokens = 2000  # Typical analysis response size
    
    input_cost = (input_tokens / 1_000_000) * INPUT_COST_PER_M
    output_cost = (output_tokens / 1_000_000) * OUTPUT_COST_PER_M
    
    return {
        "estimated_input_tokens": int(input_tokens),
        "estimated_output_tokens": int(output_tokens),
        "estimated_cost_usd": round(input_cost + output_cost, 4)
    }


if __name__ == "__main__":
    # Quick test with sample text
    sample = """
    John: Okay, let's discuss the Q4 roadmap. I think we need to prioritize the API redesign.
    Sarah: Agreed. I'll create the technical spec by Friday.
    John: Perfect. And Mike, can you review the current performance metrics?
    Mike: Sure, I'll have that ready by end of week.
    Sarah: We should also decide on the database migration timeline.
    John: Let's target January for that. All agreed?
    Mike: Yes, January works.
    Sarah: Agreed. That gives us time to finish testing.
    """
    
    print("Estimating cost...")
    estimate = estimate_analysis_cost(sample)
    print(f"Estimated cost: ${estimate['estimated_cost_usd']}")
    
    print("\nAnalyzing sample transcript...")
    result = analyze_transcript(sample)
    print(json.dumps(result, indent=2))
