# Checkpoint - Usage Guide

Here's what it does and how to use it.

## What It Does

Checkpoint preserves session context and captures lessons by:
- Updating project status, TODOs, and lessons learned files
- Ensuring the next session can resume without context loss
- Flagging and recording gotchas discovered during work
- Categorizing lessons for easy future reference

## Installation

1. The skill is symlinked to `~/.claude/skills/checkpoint`
2. Available immediately in all Claude Code sessions
3. Source of truth: `/mnt/k/Code/claude-personal-toolkit/skills/checkpoint/`

## When to Use

Use checkpoint when:
- Ending a work session
- Before DISTILL or context compact
- A lesson or gotcha is discovered
- Switching between projects

**Never use for:**
- Mid-task progress tracking (use task list for that)
- Project priming (use project-primer)

## Example Conversations

### End of Session

```
You: "/checkpoint"

Claude: Updating project docs:
- docs/CURRENT_STATUS.md — Updated: completed API endpoints, next is frontend integration
- docs/TODOS.md — Added: "Handle rate limiting on external API"
- docs/LESSONS_LEARNED.md — No new lessons this session

Session state saved. Safe to close.
```

### Lesson Captured Mid-Work

```
Claude: "Lesson spotted: ActiveCampaign API returns first contact instead of empty
array when no email match found. Add to docs/LESSONS_LEARNED.md?"

You: "Yes"

Claude: Added under Integration: ActiveCampaign GetMany False Positive
```

## Tips for Best Results

1. **Docs before commit** - Always update docs/ files BEFORE committing and pushing. Stage everything (code + docs) in one atomic commit. Never commit code first then update docs — that leaves uncommitted files behind.
2. **Be specific in status** - "Completed X, next step is Y" not "Made progress"
3. **Capture lessons immediately** - Don't wait until session end
4. **Use categories** - Makes lessons searchable across projects
