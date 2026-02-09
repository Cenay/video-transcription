#!/usr/bin/env python3
"""Verify Notion setup is correct."""

import os
from notion_client import Client
from dotenv import load_dotenv

load_dotenv()

def verify_setup():
    """Verify Notion configuration."""

    print("=" * 70)
    print("NOTION SETUP VERIFICATION")
    print("=" * 70)

    # Check environment variables
    print("\n1. Checking environment variables...")
    api_key = os.getenv("NOTION_API_KEY")
    db_id = os.getenv("NOTION_DATABASE_ID")

    if not api_key:
        print("   ❌ NOTION_API_KEY not found in .env")
        return False
    else:
        print(f"   ✓ NOTION_API_KEY found (starts with: {api_key[:10]}...)")

    if not db_id:
        print("   ❌ NOTION_DATABASE_ID not found in .env")
        return False
    else:
        print(f"   ✓ NOTION_DATABASE_ID: {db_id}")

    # Verify database access
    print("\n2. Verifying database access...")
    notion = Client(auth=api_key)

    try:
        db = notion.databases.retrieve(database_id=db_id)
        title = db.get('title', [{}])[0].get('plain_text', 'Untitled')
        print(f"   ✓ Successfully connected to database: '{title}'")
        print(f"   ✓ Database URL: {db.get('url')}")

        # Check if it's the Transcriptions database
        if title == "Transcriptions":
            print("   ✓ Database name matches expected 'Transcriptions'")
        else:
            print(f"   ⚠ Warning: Database name is '{title}', expected 'Transcriptions'")

        return True

    except Exception as e:
        print(f"   ❌ Failed to access database: {e}")
        print("\n   Troubleshooting:")
        print("   - Ensure the Notion integration has access to the database")
        print("   - In Notion, open the database and add the integration via the '...' menu")
        return False

if __name__ == "__main__":
    success = verify_setup()

    print("\n" + "=" * 70)
    if success:
        print("✅ SETUP VERIFIED")
        print("=" * 70)
        print("\nYour Notion integration is correctly configured.")
        print("\nNext steps:")
        print("1. Create the linked database view manually (see NOTION_SETUP_GUIDE.md)")
        print("2. Test with: python scripts/pipeline.py /path/to/video.mp4 --dry-run")
    else:
        print("❌ SETUP INCOMPLETE")
        print("=" * 70)
        print("\nPlease fix the issues above before running the pipeline.")
        print("Refer to NOTION_SETUP_GUIDE.md for detailed instructions.")
    print("=" * 70)
