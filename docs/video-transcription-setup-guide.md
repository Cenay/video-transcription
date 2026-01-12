# Video Transcription Pipeline: Setup Guide for Ubuntu 24.04

## Pre-Flight Checklist

Based on your environment, here's what you likely have vs. what needs setup:

| Component | Status | Action Required |
|-----------|--------|-----------------|
| Python 3.x | ✅ Likely installed | Verify version |
| FFmpeg | ⚠️ May need install | Check/install via APT |
| Redis | ⚠️ Likely not installed | Install via APT |
| AssemblyAI API Key | ⚠️ May need to create | Get from assemblyai.com |
| Anthropic API Key | ✅ Have | Verify in environment |
| Notion API Token | ⚠️ May need to create | Create integration |
| Python packages | ❌ Need install | pip install |

---

## Step 1: Verify Existing Components

Run these checks first to see what you already have:

```bash
# Check Python version (need 3.10+)
python3 --version

# Check if FFmpeg is installed
ffmpeg -version

# Check if Redis is installed
redis-server --version

# Check for existing API keys in environment
echo "AssemblyAI: ${ASSEMBLYAI_API_KEY:0:10}..."
echo "Anthropic: ${ANTHROPIC_API_KEY:0:10}..."
```

---

## Step 2: Install System Dependencies (APT)

These use APT as you prefer for system-level packages:

```bash
# Update package list
sudo apt update

# Install FFmpeg (audio extraction from video)
sudo apt install ffmpeg -y

# Install Redis (job queue backend)
sudo apt install redis-server -y

# Start Redis and enable on boot
sudo systemctl start redis-server
sudo systemctl enable redis-server

# Verify Redis is running
redis-cli ping
# Should return: PONG
```

---

## Step 3: Create Project Directory Structure

```bash
# Create project directory
mkdir -p ~/video-transcription-pipeline
cd ~/video-transcription-pipeline

# Create subdirectories
mkdir -p input output logs temp scripts

# Create the directory structure
# input/     - Drop MP4 files here for processing
# output/    - Processed transcripts and analysis
# logs/      - Pipeline logs
# temp/      - Temporary audio chunks (auto-cleaned)
# scripts/   - Python scripts for the pipeline
```

---

## Step 4: Set Up Python Virtual Environment

Using a venv keeps your system Python clean:

```bash
cd ~/video-transcription-pipeline

# Create virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip
```

---

## Step 5: Install Python Dependencies

```bash
# With venv activated:
pip install assemblyai       # AssemblyAI transcription API
pip install anthropic        # Claude API
pip install pydub            # Audio processing
pip install watchdog         # File monitoring
pip install rq               # Redis Queue
pip install redis            # Redis client
pip install notion-client    # Notion API
pip install python-dotenv    # Environment variables
```

Create a requirements.txt for reproducibility:

```bash
cat > requirements.txt << 'EOF'
assemblyai>=0.30.0
anthropic>=0.18.0
pydub>=0.25.1
watchdog>=3.0.0
rq>=1.15.0
redis>=4.5.0
notion-client>=2.0.0
python-dotenv>=1.0.0
EOF
```

---

## Step 6: Configure Environment Variables

Create a `.env` file for your API keys:

```bash
cat > .env << 'EOF'
# AssemblyAI (for transcription with speaker diarization)
ASSEMBLYAI_API_KEY=your-assemblyai-key-here

# Anthropic (for Claude analysis)
ANTHROPIC_API_KEY=sk-ant-your-anthropic-key-here

# Notion (for output)
NOTION_API_KEY=secret_your-notion-key-here
NOTION_DATABASE_ID=your-database-id-here

# Pipeline settings
INPUT_DIR=/home/YOUR_USERNAME/video-transcription-pipeline/input
OUTPUT_DIR=/home/YOUR_USERNAME/video-transcription-pipeline/output
TEMP_DIR=/home/YOUR_USERNAME/video-transcription-pipeline/temp
LOG_DIR=/home/YOUR_USERNAME/video-transcription-pipeline/logs
EOF
```

**Important:** Replace `YOUR_USERNAME` with your actual username, and fill in your API keys.

---

## Step 7: Create Notion Integration

If you don't already have a Notion integration:

1. Go to https://www.notion.so/my-integrations
2. Click "+ New integration"
3. Name it "Video Transcription Pipeline"
4. Select your workspace
5. Copy the "Internal Integration Secret" → This is your `NOTION_API_KEY`

### Create the Target Database

1. In Notion, create a new page
2. Add a Database (Full page or Inline)
3. Add these properties:
   - **Name** (Title) - default, already there
   - **Date** (Date) - for meeting date
   - **Duration** (Text) - video length
   - **Status** (Select) - options: "Processing", "Complete", "Error"
   - **Cost** (Number) - processing cost in dollars
   - **Source File** (Text) - original filename

4. Get the database ID:
   - Open the database as a full page
   - Copy the URL: `https://www.notion.so/yourworkspace/abc123def456...`
   - The database ID is the 32-character string before the `?v=`
   - Format it with dashes: `abc123de-f456-7890-abcd-ef1234567890`

5. Share with integration:
   - Click "..." menu on the database page
   - Click "Connections" → "Add connections"
   - Select "Video Transcription Pipeline"

---

## Step 8: Verify API Connections

Create a test script to verify all connections work:

```bash
cat > scripts/test_connections.py << 'EOF'
#!/usr/bin/env python3
"""Test all API connections before running the pipeline."""

import os
import sys
from dotenv import load_dotenv

load_dotenv()

def test_assemblyai():
    """Test AssemblyAI connection."""
    print("Testing AssemblyAI connection...", end=" ")
    try:
        import assemblyai as aai
        api_key = os.environ.get("ASSEMBLYAI_API_KEY")
        if not api_key:
            print("❌ ASSEMBLYAI_API_KEY not set")
            return False
        aai.settings.api_key = api_key
        print("✅ API key configured")
        return True
    except Exception as e:
        print(f"❌ Failed: {e}")
        return False

def test_anthropic():
    """Test Anthropic/Claude connection."""
    print("Testing Anthropic connection...", end=" ")
    try:
        import anthropic
        client = anthropic.Anthropic()
        # Minimal API call to verify
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=10,
            messages=[{"role": "user", "content": "Say 'connected' and nothing else."}]
        )
        print(f"✅ {response.content[0].text}")
        return True
    except Exception as e:
        print(f"❌ Failed: {e}")
        return False

def test_notion():
    """Test Notion connection."""
    print("Testing Notion connection...", end=" ")
    try:
        from notion_client import Client
        notion = Client(auth=os.environ.get("NOTION_API_KEY"))
        db_id = os.environ.get("NOTION_DATABASE_ID")
        
        if not db_id:
            print("❌ NOTION_DATABASE_ID not set")
            return False
            
        # Try to retrieve the database
        db = notion.databases.retrieve(database_id=db_id)
        print(f"✅ Connected to: {db['title'][0]['plain_text']}")
        return True
    except Exception as e:
        print(f"❌ Failed: {e}")
        return False

def test_redis():
    """Test Redis connection."""
    print("Testing Redis connection...", end=" ")
    try:
        import redis
        r = redis.Redis(host='localhost', port=6379, db=0)
        r.ping()
        print("✅ PONG")
        return True
    except Exception as e:
        print(f"❌ Failed: {e}")
        return False

def test_ffmpeg():
    """Test FFmpeg installation."""
    print("Testing FFmpeg...", end=" ")
    try:
        import subprocess
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True)
        if result.returncode == 0:
            version = result.stdout.split('\n')[0]
            print(f"✅ {version[:50]}...")
            return True
        else:
            print("❌ FFmpeg not working")
            return False
    except FileNotFoundError:
        print("❌ FFmpeg not installed")
        return False

if __name__ == "__main__":
    print("\n" + "="*50)
    print("Video Transcription Pipeline - Connection Tests")
    print("="*50 + "\n")
    
    results = {
        "FFmpeg": test_ffmpeg(),
        "Redis": test_redis(),
        "AssemblyAI": test_assemblyai(),
        "Anthropic": test_anthropic(),
        "Notion": test_notion(),
    }
    
    print("\n" + "="*50)
    print("Summary")
    print("="*50)
    
    all_passed = True
    for name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False
    
    print()
    if all_passed:
        print("All systems ready! You can run the pipeline.")
        sys.exit(0)
    else:
        print("Fix the failing connections before running the pipeline.")
        sys.exit(1)
EOF

chmod +x scripts/test_connections.py
```

Run the test:

```bash
source venv/bin/activate
python scripts/test_connections.py
```

---

## Step 9: Create the Core Pipeline Scripts

### 9a. Audio Extraction Module

```bash
cat > scripts/audio_extractor.py << 'EOF'
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
EOF
```

### 9b. Transcription Module

```bash
cat > scripts/transcriber.py << 'EOF'
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

    Args:
        audio_path: Path to audio file
        speaker_labels: Enable speaker diarization
        language_code: ISO language code

    Returns:
        Transcription result dict with text and metadata
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
    if speaker_labels and transcript.utterances:
        formatted_text = format_with_speakers(transcript.utterances)
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
EOF
```

### 9c. Claude Analysis Module

```bash
cat > scripts/analyzer.py << 'EOF'
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
{
  "summary": "2-3 paragraph executive summary of what was discussed and decided",
  "action_items": [
    {
      "task": "Description of the task",
      "owner": "Person responsible (or 'Unassigned' if unclear)",
      "deadline": "Due date if mentioned (or 'TBD')",
      "context": "Brief context for why this task exists"
    }
  ],
  "decisions": [
    {
      "decision": "What was decided",
      "rationale": "Why this was decided",
      "participants": "Who was involved in the decision"
    }
  ],
  "key_quotes": [
    {
      "quote": "The exact quote",
      "speaker": "Who said it (if identifiable)",
      "timestamp": "Approximate time in transcript",
      "significance": "Why this quote matters"
    }
  ],
  "topics_discussed": ["Topic 1", "Topic 2"],
  "follow_up_items": ["Items needing follow-up discussion"],
  "meeting_metadata": {
    "apparent_purpose": "What the meeting was about",
    "tone": "General tone (collaborative, tense, brainstorming, etc.)",
    "participation_notes": "Any notes about participation patterns"
  }
}

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
EOF
```

### 9d. Notion Output Module

```bash
cat > scripts/notion_output.py << 'EOF'
#!/usr/bin/env python3
"""Output analysis results to Notion."""

import os
from datetime import datetime
from notion_client import Client
from dotenv import load_dotenv

load_dotenv()

notion = Client(auth=os.environ.get("NOTION_API_KEY"))
DATABASE_ID = os.environ.get("NOTION_DATABASE_ID")


def create_meeting_page(
    title: str,
    date: str,
    duration_minutes: float,
    analysis: dict,
    transcript: str,
    costs: dict,
    source_file: str = ""
) -> str:
    """
    Create a Notion page with meeting analysis.
    
    Returns the URL of the created page.
    """
    # Create page with database properties
    page = notion.pages.create(
        parent={"database_id": DATABASE_ID},
        properties={
            "Name": {
                "title": [{"text": {"content": title}}]
            },
            "Date": {
                "date": {"start": date}
            },
            "Duration": {
                "rich_text": [{"text": {"content": f"{duration_minutes:.0f} minutes"}}]
            },
            "Status": {
                "select": {"name": "Complete"}
            },
            "Cost": {
                "number": round(costs.get("total", 0), 4)
            },
            "Source File": {
                "rich_text": [{"text": {"content": source_file}}]
            }
        }
    )
    
    page_id = page["id"]
    
    # Build content blocks
    blocks = []
    
    # Summary section
    blocks.append({
        "type": "heading_2",
        "heading_2": {"rich_text": [{"text": {"content": "Summary"}}]}
    })
    
    summary = analysis.get("summary", "No summary available.")
    # Split summary into chunks (Notion has 2000 char limit per block)
    for chunk in chunk_text(summary, 1900):
        blocks.append({
            "type": "paragraph",
            "paragraph": {"rich_text": [{"text": {"content": chunk}}]}
        })
    
    # Action Items section
    action_items = analysis.get("action_items", [])
    if action_items:
        blocks.append({
            "type": "heading_2",
            "heading_2": {"rich_text": [{"text": {"content": "Action Items"}}]}
        })
        
        for item in action_items:
            task = item.get("task", "")
            owner = item.get("owner", "Unassigned")
            deadline = item.get("deadline", "TBD")
            
            blocks.append({
                "type": "to_do",
                "to_do": {
                    "rich_text": [{"text": {"content": f"{task} (@{owner}, due: {deadline})"}}],
                    "checked": False
                }
            })
    
    # Decisions section
    decisions = analysis.get("decisions", [])
    if decisions:
        blocks.append({
            "type": "heading_2",
            "heading_2": {"rich_text": [{"text": {"content": "Key Decisions"}}]}
        })
        
        for i, decision in enumerate(decisions, 1):
            decision_text = decision.get("decision", "")
            rationale = decision.get("rationale", "")
            
            blocks.append({
                "type": "numbered_list_item",
                "numbered_list_item": {
                    "rich_text": [
                        {"text": {"content": decision_text, "annotations": {"bold": True}}},
                        {"text": {"content": f" — {rationale}" if rationale else ""}}
                    ]
                }
            })
    
    # Key Quotes section
    quotes = analysis.get("key_quotes", [])
    if quotes:
        blocks.append({
            "type": "heading_2",
            "heading_2": {"rich_text": [{"text": {"content": "Key Quotes"}}]}
        })
        
        for quote in quotes[:5]:  # Limit to 5 quotes
            quote_text = quote.get("quote", "")
            speaker = quote.get("speaker", "Unknown")
            
            blocks.append({
                "type": "quote",
                "quote": {
                    "rich_text": [
                        {"text": {"content": f'"{quote_text}"'}},
                        {"text": {"content": f" — {speaker}", "annotations": {"italic": True}}}
                    ]
                }
            })
    
    # Processing Costs section
    blocks.append({
        "type": "heading_2",
        "heading_2": {"rich_text": [{"text": {"content": "Processing Details"}}]}
    })
    
    cost_text = f"Transcription: ${costs.get('transcription', 0):.4f} | Analysis: ${costs.get('analysis', 0):.4f} | Total: ${costs.get('total', 0):.4f}"
    blocks.append({
        "type": "paragraph",
        "paragraph": {"rich_text": [{"text": {"content": cost_text}}]}
    })
    
    # Full Transcript in a toggle (collapsed by default)
    blocks.append({
        "type": "heading_2",
        "heading_2": {"rich_text": [{"text": {"content": "Full Transcript"}}]}
    })
    
    # Create toggle with transcript chunks
    transcript_chunks = list(chunk_text(transcript, 1900))
    
    toggle_children = []
    for chunk in transcript_chunks[:50]:  # Notion has limits on children
        toggle_children.append({
            "type": "paragraph",
            "paragraph": {"rich_text": [{"text": {"content": chunk}}]}
        })
    
    if len(transcript_chunks) > 50:
        toggle_children.append({
            "type": "paragraph",
            "paragraph": {"rich_text": [{"text": {"content": "[Transcript truncated due to length...]"}}]}
        })
    
    blocks.append({
        "type": "toggle",
        "toggle": {
            "rich_text": [{"text": {"content": "Click to expand full transcript"}}],
            "children": toggle_children
        }
    })
    
    # Append all blocks to the page
    # Notion limits to 100 blocks per request, so batch if needed
    for i in range(0, len(blocks), 100):
        batch = blocks[i:i+100]
        notion.blocks.children.append(page_id, children=batch)
    
    # Return the page URL
    return f"https://notion.so/{page_id.replace('-', '')}"


def chunk_text(text: str, max_length: int = 1900) -> list[str]:
    """Split text into chunks respecting word boundaries."""
    chunks = []
    current = ""
    
    for word in text.split():
        if len(current) + len(word) + 1 > max_length:
            if current:
                chunks.append(current.strip())
            current = word
        else:
            current = f"{current} {word}" if current else word
    
    if current:
        chunks.append(current.strip())
    
    return chunks


if __name__ == "__main__":
    # Test creating a page
    test_analysis = {
        "summary": "This is a test meeting summary.",
        "action_items": [
            {"task": "Review the document", "owner": "John", "deadline": "Friday"}
        ],
        "decisions": [
            {"decision": "Proceed with plan A", "rationale": "Lower risk"}
        ],
        "key_quotes": [
            {"quote": "Let's make it happen", "speaker": "Sarah"}
        ]
    }
    
    url = create_meeting_page(
        title="Test Meeting",
        date=datetime.now().strftime("%Y-%m-%d"),
        duration_minutes=60,
        analysis=test_analysis,
        transcript="This is a test transcript...",
        costs={"transcription": 0.18, "analysis": 0.05, "total": 0.23},
        source_file="test.mp4"
    )
    
    print(f"Created page: {url}")
EOF
```

### 9e. Main Pipeline Orchestrator

```bash
cat > scripts/pipeline.py << 'EOF'
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
import tempfile
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Import our modules
from audio_extractor import (
    extract_audio, 
    get_audio_duration, 
    needs_chunking,
    chunk_audio_at_silence,
    get_file_size_mb
)
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
    keep_temp: bool = False
) -> dict:
    """
    Process a video file through the complete pipeline.
    
    Args:
        video_path: Path to MP4 video
        dry_run: If True, estimate costs without processing
        keep_temp: If True, don't delete temporary files
        
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
        # Step 1: Extract audio
        print("\n[1/4] Extracting audio...")
        audio_path = temp_dir / f"{video_path.stem}.mp3"
        audio_path = extract_audio(str(video_path), str(audio_path))
        
        duration_sec = get_audio_duration(audio_path)
        duration_min = duration_sec / 60
        file_size_mb = get_file_size_mb(audio_path)
        
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
        
        # Step 2: Transcribe
        print("\n[2/4] Transcribing audio...")

        # AssemblyAI handles large files natively - no chunking needed
        transcription = transcribe_audio(audio_path)
        
        transcript_text = transcription.get("text", "")
        print(f"  Transcription complete: {len(transcript_text)} characters")
        
        result["transcript"] = transcript_text
        result["transcription_metadata"] = {
            "duration": transcription.get("duration"),
            "language": transcription.get("language"),
            "utterance_count": len(transcription.get("utterances", [])),
            "confidence": transcription.get("confidence")
        }
        
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
        
        # Calculate total cost
        result["costs"]["total"] = (
            result["costs"].get("transcription", 0) +
            result["costs"].get("analysis", 0)
        )
        
        # Step 4: Output to Notion
        print("\n[4/4] Creating Notion page...")
        
        # Generate title from filename or use summary
        title = video_path.stem.replace("-", " ").replace("_", " ").title()
        date = datetime.now().strftime("%Y-%m-%d")
        
        notion_url = create_meeting_page(
            title=title,
            date=date,
            duration_minutes=duration_min,
            analysis=analysis,
            transcript=transcript_text,
            costs=result["costs"],
            source_file=video_path.name
        )
        
        result["notion_url"] = notion_url
        print(f"  Created: {notion_url}")
        
        # Cleanup
        if not keep_temp:
            Path(audio_path).unlink(missing_ok=True)
        
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
    parser.add_argument("--output-json", type=str,
                       help="Save full result to JSON file")
    
    args = parser.parse_args()
    
    result = process_video(
        args.video,
        dry_run=args.dry_run,
        keep_temp=args.keep_temp
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
EOF

chmod +x scripts/pipeline.py
```

---

## Step 10: Create the Claude Skill

This creates a reusable skill you can upload to Claude.ai:

```bash
mkdir -p ~/video-transcription-pipeline/skill/meeting-transcript-analyzer

cat > ~/video-transcription-pipeline/skill/meeting-transcript-analyzer/SKILL.md << 'EOF'
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
EOF
```

To upload this skill to Claude.ai:
1. Go to claude.ai → Settings → Skills
2. Click "Add Skill" 
3. Upload the SKILL.md file or paste its contents

---

## Step 11: Test the Pipeline

Run a complete test:

```bash
cd ~/video-transcription-pipeline
source venv/bin/activate

# First, verify all connections
python scripts/test_connections.py

# Then do a dry run to see estimated costs
python scripts/pipeline.py /path/to/your/video.mp4 --dry-run

# If costs look good, run the full pipeline
python scripts/pipeline.py /path/to/your/video.mp4

# Save results to JSON as well
python scripts/pipeline.py /path/to/your/video.mp4 --output-json output/results.json
```

---

## Quick Reference: Common Commands

```bash
# Activate environment
cd ~/video-transcription-pipeline && source venv/bin/activate

# Process a video
python scripts/pipeline.py /path/to/video.mp4

# Check costs first
python scripts/pipeline.py /path/to/video.mp4 --dry-run

# Process and keep temp files for debugging
python scripts/pipeline.py /path/to/video.mp4 --keep-temp

# Test all connections
python scripts/test_connections.py

# Check Redis status
sudo systemctl status redis-server

# View Redis queue (if using worker mode)
rq info
```

---

## Troubleshooting

**FFmpeg not found:**
```bash
sudo apt install ffmpeg
```

**Redis connection refused:**
```bash
sudo systemctl start redis-server
sudo systemctl enable redis-server
```

**Notion "Could not find database":**
- Verify the database ID format (32 chars with dashes)
- Ensure you shared the database with your integration
- Check the integration has correct permissions

**AssemblyAI transcription error:**
- Verify ASSEMBLYAI_API_KEY is set correctly
- AssemblyAI handles large files natively (no chunking needed)
- Check your account has sufficient credits at assemblyai.com

**Claude JSON parsing error:**
- Usually means response was cut off
- Increase `max_tokens` in `analyzer.py`

---

## Cost Summary (per 2-hour video)

| Component | Cost |
|-----------|------|
| AssemblyAI (with speaker diarization) | $0.74 |
| Claude Sonnet 4 | ~$0.15 |
| **Total** | **~$0.89** |
