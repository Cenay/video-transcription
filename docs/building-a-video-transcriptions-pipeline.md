# Building a video transcription and AI analysis pipeline

**Transcribing 60-240 minute videos with AssemblyAI and processing them with Claude is both practical and cost-effective—expect roughly $1-2 per 2-hour video for the complete pipeline.** The optimal approach combines cloud APIs (for simplicity and reliability) with speaker diarization and a Claude Skill for consistent analysis. For regular use on Ubuntu 24.04, a Python-based automation pipeline with Redis queues and Notion output creates a maintainable, repeatable workflow.

## AssemblyAI transcription: Speaker diarization included

AssemblyAI charges **$0.0062/minute** with speaker diarization included ($0.37/hour)—making a 2-hour video cost about $0.74. Unlike Whisper, **AssemblyAI handles large files natively** with no file size limits, eliminating the need for audio chunking.

Key advantages over Whisper:
- **Speaker diarization built-in** - identifies who said what
- **No file size limits** - upload hours of audio directly
- **Higher accuracy** for conversational audio and multiple speakers

| Option | Cost (2-hour video) | Best for |
|--------|---------------------|----------|
| AssemblyAI (with speakers) | $0.74 | Meetings, interviews, multi-speaker |
| AssemblyAI (no speakers) | $0.44 | Single speaker, lectures |

The API supports `mp3`, `m4a`, `wav`, `flac`, and `mp4` directly. Speaker diarization returns utterances with speaker labels and timestamps.

## Audio extraction

Extract audio from MP4 using FFmpeg optimized for transcription:

```bash
# Optimal extraction: 16kHz mono, 64kbps MP3 (~28MB per hour)
ffmpeg -i input.mp4 -ar 16000 -ac 1 -b:a 64k output.mp3
```

**Note:** AssemblyAI handles large files natively—no chunking required. The chunking code below is retained for reference if using other transcription services with file size limits:

```python
from pydub import AudioSegment
from pydub.silence import detect_silence

def chunk_at_silence(audio_path, max_duration_ms=600000):  # 10 minutes
    audio = AudioSegment.from_file(audio_path)
    silences = detect_silence(audio, min_silence_len=700, silence_thresh=-40)

    chunks, start = [], 0
    for silence_start, silence_end in silences:
        if silence_start - start >= max_duration_ms:
            split_point = (silence_start + silence_end) // 2
            chunks.append((audio[start:split_point], start))
            start = split_point - 30000  # 30-second overlap
    chunks.append((audio[start:], start))
    return chunks
```

## LLM analysis fits comfortably in context

A 2-hour meeting generates roughly **25,000-35,000 words** with speaker labels, translating to **35,000-47,000 tokens**. This fits easily within Claude's 200K context window—no chunking required for most meetings.

Cost for Claude Sonnet 4.5 to analyze a 2-hour transcript: **$0.13-0.17** (input: ~$0.12, output: ~$0.03 for 2K tokens). For higher volume, Claude Haiku 4.5 drops this to **$0.04-0.06** per meeting with minimal quality loss for structured extraction tasks.

The most effective prompt structure requests structured JSON output:

```python
analysis_prompt = """Analyze this meeting transcript and return JSON:
{
  "summary": "2-3 paragraph executive summary",
  "action_items": [{"task": "", "assignee": "", "deadline": "", "context": ""}],
  "decisions": [{"decision": "", "rationale": "", "decided_by": ""}],
  "key_quotes": [{"speaker": "", "quote": "", "timestamp": "", "significance": ""}],
  "follow_ups": [""]
}

Look for commitments ("I'll do X", "will review Y"), explicit decisions ("agreed", 
"approved"), and significant statements worth preserving.

TRANSCRIPT:
{transcript}"""
```

For cost optimization, use a **tiered approach**: run Haiku for initial extraction of action items and decisions (structured, lower complexity), then optionally use Sonnet for nuanced summarization requiring synthesis.

## Claude Skills create reusable analysis workflows

Claude Skills work through **progressive disclosure**: Claude loads only the skill's name and description (~100 words) at startup, then fetches the full SKILL.md content when relevant. This enables complex workflows without consuming context unnecessarily.

A skill for meeting analysis needs this YAML frontmatter:

```yaml
---
name: meeting-transcript-analyzer  
description: Process meeting transcripts to extract summaries, action items with 
owners and deadlines, key decisions, and important quotes. Use when analyzing 
recordings, meeting notes, or transcripts.
---
```

The SKILL.md body defines the analysis workflow:

```markdown
# Meeting Transcript Analyzer

## Instructions
1. Parse the transcript identifying speakers and timestamps
2. Extract ALL action items—look for "I'll", "will", "needs to", "by [date]"
3. Identify decisions with their rationale and who approved
4. Pull significant quotes with attribution and timestamps
5. Generate summary focusing on outcomes, not just topics discussed

## Output Format
### Summary
[1-2 paragraphs capturing what was decided and why it matters]

### Action Items
| Owner | Task | Deadline | Context |
|-------|------|----------|---------|

### Key Decisions  
1. **[Decision]**: [Description] — Decided by [Name]

### Notable Quotes
> "[Quote]" — [Speaker] at [timestamp]

## Guidelines
- Don't miss commitments buried in discussion
- Mark unclear deadlines as "TBD - needs confirmation"
- Preserve technical terms and project names exactly
- Infer assignees from context when not explicit
```

Skills require code execution to be enabled (available on Pro, Max, Team, and Enterprise plans). The skill automatically activates when users upload transcripts or mention meeting analysis.

## Automation architecture for Ubuntu 24.04

The recommended stack uses **watchdog** for file monitoring, **RQ (Redis Queue)** for reliable job processing, and **systemd timers** for scheduled maintenance:

```
Watch Folder (/input) → watchdog → Redis Queue → Worker → Notion
                                       ↓
                              [Extract Audio]
                              [Transcribe (AssemblyAI)]
                              [Analyze (Claude)]
                              [Create Page (Notion)]
                              [Notify (Slack)]
```

Key implementation patterns:

**State persistence for resumability**—save progress after each stage so failures don't require restarting from scratch:

```python
class VideoPipeline:
    def __init__(self, video_path):
        self.state_file = Path(f"/tmp/pipeline/{Path(video_path).stem}/state.json")
    
    def run(self):
        state = self.load_state()
        
        if state["stage"] == "start":
            state["audio_path"] = self.extract_audio()
            state["stage"] = "audio_complete"
            self.save_state(state)
            
        if state["stage"] == "audio_complete":
            state["transcript"] = self.transcribe()
            state["stage"] = "transcribed"
            self.save_state(state)
        # Continue through analysis and output...
```

**RQ job configuration** with automatic retry:

```python
from rq import Queue, Retry
from redis import Redis

q = Queue('transcription', connection=Redis())
job = q.enqueue(
    process_video,
    '/path/to/video.mp4',
    retry=Retry(max=3, interval=[60, 300, 900]),
    job_timeout='2h'
)
```

## Notion integration for structured output

The `notion-client` library (official SDK) creates database entries with rich content:

```python
from notion_client import Client

notion = Client(auth=os.environ["NOTION_TOKEN"])

def create_meeting_page(db_id, title, date, summary, action_items, transcript):
    # Create page with properties
    page = notion.pages.create(
        parent={"database_id": db_id},
        properties={
            "Name": {"title": [{"text": {"content": title}}]},
            "Date": {"date": {"start": date}},
            "Status": {"select": {"name": "Processed"}}
        }
    )
    
    # Add content blocks
    blocks = [
        {"type": "heading_2", "heading_2": {"rich_text": [{"text": {"content": "Summary"}}]}},
        {"type": "paragraph", "paragraph": {"rich_text": [{"text": {"content": summary}}]}},
        {"type": "heading_2", "heading_2": {"rich_text": [{"text": {"content": "Action Items"}}]}},
    ]
    
    # Add action items as to-do blocks
    for item in action_items:
        blocks.append({
            "type": "to_do",
            "to_do": {
                "rich_text": [{"text": {"content": f"{item['task']} (@{item['assignee']})"}}],
                "checked": False
            }
        })
    
    # Add transcript in a toggle (collapsed)
    blocks.append({
        "type": "toggle",
        "toggle": {
            "rich_text": [{"text": {"content": "Full Transcript"}}],
            "children": [{"type": "paragraph", "paragraph": {"rich_text": [{"text": {"content": transcript[:2000]}}]}}]
        }
    })
    
    notion.blocks.children.append(page["id"], children=blocks)
    return page
```

Notion's API rate limit is **3 requests per second**—implement a simple rate limiter for batch processing.

## Complete cost breakdown for regular use

For processing **20 two-hour meetings per month**:

| Component | Per meeting | Monthly (20 meetings) |
|-----------|-------------|----------------------|
| AssemblyAI (with speaker diarization) | $0.74 | $14.80 |
| Claude Sonnet 4 analysis | $0.15 | $3.00 |
| **Total** | **$0.89** | **$17.80** |

Using Claude Haiku instead of Sonnet reduces LLM costs to ~$1/month. AssemblyAI without speaker diarization costs $0.0037/minute (~$0.44 per 2-hour video).

## Conclusion

The most practical implementation combines **AssemblyAI** (speaker diarization included, no file size limits), **Claude Sonnet 4** through a reusable Skill (for consistent high-quality analysis), and a **Python pipeline with RQ** for reliability. Extract audio at 16kHz/64kbps, upload directly to AssemblyAI (no chunking needed), then process the full transcript in Claude's 200K context window—no summarization chunking needed for meetings under 6-8 hours.

For the Claude Skill, focus the SKILL.md on clear extraction rules for action items and decisions (the highest-value outputs) and use structured JSON output for automation compatibility. The entire pipeline runs reliably on Ubuntu 24.04 with Python 3.10+, Redis, and FFmpeg as the only system dependencies.