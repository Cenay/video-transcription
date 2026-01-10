---
name: meeting-transcript-analyzer
description: Analyze meeting transcripts to extract summaries, action items with owners and deadlines, key decisions, and important quotes. Use when processing transcripts from video recordings, Zoom meetings, audio recordings, or any meeting notes. Triggers on requests to "analyze this transcript", "summarize this meeting", "extract action items", or when user uploads a transcript file.
---

# Meeting Transcript Analyzer

## Core Workflow

1. Parse the transcript identifying any speaker patterns or timestamps
2. Extract ALL action items—search for commitment language:
   - "I'll", "I will", "will do", "going to"
   - "need to", "should", "must", "have to"
   - "by [date]", "before [event]", "next week"
   - "action item:", "TODO:", "follow up"
3. Identify decisions with rationale:
   - "we decided", "agreed to", "approved"
   - "let's go with", "the plan is", "we'll do"
4. Pull significant quotes worth preserving
5. Generate summary focused on outcomes, not just topics

## Output Format

Return analysis as structured JSON:
```json
{
  "summary": "2-3 paragraph executive summary",
  "action_items": [
    {
      "task": "Clear description of the task",
      "owner": "Person responsible (or 'Unassigned')",
      "deadline": "Due date (or 'TBD')",
      "context": "Why this task exists"
    }
  ],
  "decisions": [
    {
      "decision": "What was decided",
      "rationale": "Why",
      "participants": "Who was involved"
    }
  ],
  "key_quotes": [
    {
      "quote": "Exact words",
      "speaker": "Who said it",
      "significance": "Why it matters"
    }
  ],
  "topics_discussed": ["Topic 1", "Topic 2"],
  "follow_up_items": ["Items needing future discussion"]
}
```

## Quality Guidelines

- Don't miss commitments buried in casual discussion
- Mark unclear deadlines as "TBD - needs confirmation"  
- Preserve technical terms and project names exactly as stated
- Infer assignees from context when not explicitly stated
- If multiple people discuss ownership, note the ambiguity
- Prioritize substance over volume—fewer, higher-quality extractions

## Handling Edge Cases

- **No clear action items**: Note "No explicit action items identified" but look for implicit commitments
- **Unclear speakers**: Use "Speaker A/B" or describe by apparent role
- **Partial transcripts**: Note that analysis is based on available content
- **Non-English content**: Identify the language and proceed if capable
