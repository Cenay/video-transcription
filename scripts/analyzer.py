#!/usr/bin/env python3
"""Analyze transcripts using Claude API."""

import os
import json
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

client = Anthropic()

# Pricing per million tokens (Claude Sonnet 4.5)
INPUT_COST_PER_M = 3.00
OUTPUT_COST_PER_M = 15.00


ANALYSIS_PROMPT = """Analyze this meeting transcript and extract structured information.

Return your analysis as JSON with this exact structure:
{{
  "summary": "2-3 paragraph executive summary of what was discussed and decided",
  "action_items": [
    {{
      "task": "Description of the task",
      "owner": "Person responsible (or 'Unassigned' if unclear)",
      "deadline": "Due date if mentioned (or 'TBD')",
      "context": "Brief context for why this task exists"
    }}
  ],
  "decisions": [
    {{
      "decision": "What was decided",
      "rationale": "Why this was decided",
      "participants": "Who was involved in the decision"
    }}
  ],
  "key_quotes": [
    {{
      "quote": "The exact quote",
      "speaker": "Who said it (if identifiable)",
      "timestamp": "Approximate time in transcript",
      "significance": "Why this quote matters"
    }}
  ],
  "topics_discussed": ["Topic 1", "Topic 2"],
  "follow_up_items": ["Items needing follow-up discussion"],
  "meeting_metadata": {{
    "apparent_purpose": "What the meeting was about",
    "tone": "General tone (collaborative, tense, brainstorming, etc.)",
    "participation_notes": "Any notes about participation patterns"
  }}
}}

Guidelines:
- Look for commitments: "I'll do X", "will review Y", "let's plan to Z"
- Capture explicit decisions: "we agreed", "approved", "decided to"
- Pull quotes that represent key insights or turning points
- If speaker names aren't clear, use "Speaker" or describe by role if apparent
- Mark uncertain items with (unclear) or (approximate)

TRANSCRIPT:
{transcript}

Return ONLY valid JSON, no additional text or markdown formatting."""


def analyze_transcript(
    transcript: str,
    model: str = "claude-sonnet-4-20250514",
    max_tokens: int = 4096
) -> dict:
    """
    Analyze a transcript using Claude.
    
    Args:
        transcript: Full transcript text
        model: Claude model to use
        max_tokens: Maximum response tokens
        
    Returns:
        Parsed analysis dict
    """
    prompt = ANALYSIS_PROMPT.format(transcript=transcript)
    
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    
    # Extract text from response
    response_text = response.content[0].text
    
    # Parse JSON from response
    try:
        # Handle potential markdown code blocks
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0]
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0]
        
        analysis = json.loads(response_text.strip())
    except json.JSONDecodeError as e:
        # Return raw text if JSON parsing fails
        analysis = {
            "error": f"Failed to parse JSON: {e}",
            "raw_response": response_text
        }
    
    # Add usage stats
    analysis["_usage"] = {
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
        "model": model
    }
    
    return analysis


def estimate_analysis_cost(transcript: str, model: str = "claude-sonnet-4-20250514") -> dict:
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
