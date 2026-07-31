# Project Skills — video-transcription

This directory holds project-specific Claude Code skills. Unlike global skills (in ~/.claude/skills/), these are scoped to this project and auto-load when Claude works in this directory.

## When to Create a Project Skill

Create a skill when Claude repeatedly needs domain-specific knowledge that isn't obvious from the code:
- **Business rules** (billing logic, validation rules, approval workflows)
- **Data schemas** (field mappings, required formats, API contracts)
- **Domain conventions** (naming standards, architectural patterns specific to this project)

## How to Create a Skill

```bash
mkdir -p .claude/skills/{skill-name}/guides
```

Create `.claude/skills/{skill-name}/SKILL.md`:

```yaml
---
name: skill-name
description: One-line description of what this skill encodes
triggers:
  - "trigger phrase that activates this skill"
---

Skill prompt content here. Encode the domain knowledge Claude needs.
```

Optionally add reference docs in the `guides/` subdirectory.

## Examples of Good Project Skills

- `billing-logic` — Encodes daily-to-monthly conversion rules so Claude never gets billing wrong
- `api-schema` — Documents the exact request/response format for all endpoints
- `parser-conventions` — Enforces field naming standards across all parser nodes
