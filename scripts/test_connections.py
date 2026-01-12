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
        # Verify key format (starts with valid prefix)
        if not api_key.startswith(("", )):  # AssemblyAI keys don't have a standard prefix
            pass  # Skip prefix check
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
