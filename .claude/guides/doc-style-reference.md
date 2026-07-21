# Documentation Style Reference

_Last updated 2026-07-17 12:09 MST by an AI session · transcript: `4088ac65-e4b4-4ce7-a94b-8bae61562311`_

This guide governs how documentation is written across projects, so that docs read consistently no matter who — or which AI session — wrote them. Its companion, `bug-report-style.md`, governs bug-ledger entries specifically; where it is silent, this guide applies.

## General Principles

### 1. Clarity Over Brevity
- Explain concepts thoroughly.
- Include context for *why* something exists, not just what it does.
- Document gotchas and edge cases.
- Assume the reader is a developer who does not know the project history.

### 2. Practical Examples
- Always include real-world usage examples.
- Show both simple and complex use cases.
- Include expected output where relevant.

### 3. Honesty About Limitations
- Be upfront about what you can and cannot do.
- Don't pretend to complete actions that failed.
- Admit when you don't have the information (see "Don't Know" Situations below).

## Markdown Formatting Standards

### No hard-wrapped prose

Write each paragraph and list item as **one continuous physical line** and let editors soft-wrap. Do not hand-wrap prose at ~80/90 columns — hard breaks mid-sentence render as broken text and make diffs noisy. Hard breaks are structural only: lists, tables, code blocks, blockquotes. (Where the project ships `reflow-md.py`, it enforces this mechanically.)

### Blockquotes for Reference Content

Use blockquotes (`>`) for content that is meant to be referenced but not necessarily copied.

**✅ DO use blockquotes for:**
- API endpoints and routes
- Command-line examples
- File paths (when referring to them, not editing them)
- Usage examples
- Configuration values
- Return values or responses

**Example:**

> **Usage:** `GET /api/resource/{id}/{option?}`
>
> **Returns:** a response object or URL string
>
> **Configuration:** set `SOME_KEY` in the project's config

**❌ DON'T use blockquotes for:**
- Code blocks (use fenced code blocks with language tags)
- Headings or section titles
- Table content
- Regular explanatory text

### Code Blocks

Always use fenced code blocks with a language identifier (` ```php `, ` ```js `, ` ```bash `, ` ```json `, ` ```sql `, ` ```md `, …). The language tag drives syntax highlighting and signals intent.

```
$result = doSomething($input);
return $result;
```

### Headings Hierarchy

```md
# Document Title (H1) — only one per document
## Main Sections (H2)
### Subsections (H3)
#### Sub-subsections (H4) — use sparingly
```

### Lists

**Unordered lists:**
- Use for items without a specific order.
- Use `-` (dash) for consistency.
- Indent nested items with 2 spaces.

**Ordered lists:**
1. Use for sequential steps.
2. Use for ranked items.
3. Let Markdown auto-number (every item may be `1.`).

### Tables

Use tables for structured comparisons. Always include a header row and a separator row.

```md
| Column 1 | Column 2 | Column 3 |
|----------|----------|----------|
| Value A  | Value B  | Value C  |
```

### Emphasis

- **Bold** (`**text**`) — important terms, labels, emphasis.
- *Italic* (`*text*`) — parameter names, light emphasis.
- `Code` (`` `text` ``) — inline code, filenames, field names, values.

## Documentation Structure

### Component / Class / Module Documentation

Every component doc should include, as applicable:

1. **Overview** — what it does in 2-3 sentences.
2. **Purpose** — why it exists, what problem it solves.
3. **Dependencies** — what it requires.
4. **Entry Points** — public functions/methods called from routes or external code.
5. **Helper Methods** — internal helpers, clearly explained.
6. **Gotchas** — known issues, edge cases, warnings.
7. **Configuration** — required config / environment variables.
8. **Usage Examples** — real-world examples.
9. **Testing Considerations** — how to test it.
10. **Related Files** — other files a developer should know about.

### Function / Method Documentation Format

Use this copyable template:

```md
### `functionName(param1, param2, optional = null)`

One-line description.

**Purpose:** Why it exists and what problem it solves.

**Parameters:**
- `param1` — what it represents and accepted formats/constraints
- `param2` — what it represents and accepted formats/constraints
- `optional` — optional parameter description (default: `null`)

**Returns:** Return type and meaning. Include error conditions if applicable.

**Usage:**
​```
result = functionName(param1, param2);
​```

**Special Logic:**
- Important behavior
- Edge cases
- Side effects
```

## Content Guidelines

### Explain Acronyms
- First use: write out fully, then the abbreviation in parentheses — "Content Management System (CMS)…". Thereafter, the abbreviation is fine.

### Date Formats
- Use ISO format: `YYYY-MM-DD`.
- Include the timezone when relevant: `2026-07-17 14:30 MST`. A bare time with no zone is ambiguous — get the stamp from `date "+%Y-%m-%d %H:%M %Z"`, never guess it.

### Version References
- Specify which version of the code a doc describes, and note version differences where they exist.

### Magic Numbers and Hardcoded IDs
- Always document what a magic number or hardcoded ID means.
- Provide a reference table where several exist.
- Explain *why* they're hardcoded, if applicable.

## Warning and Note Callouts

Use clear markers for important information. Prefer bold labels and keep the text concise.

**⚠️ Warning:** This action writes files that can fill disk space.

**📝 Note:** The flag must be the string `'false'`, not a boolean.

**🔧 Deprecated:** Uses an old API — refactor needed.

**✅ Best Practice:** Always validate input before acting on it.

## "Don't Know" Situations

When you don't have the information, say so and point at where it lives — never guess.

**✅ DO say:**

> **Note:** The exact route definition isn't documented here. Check the project's route files for the actual registration.

**❌ DON'T say:**

> The route is probably at `/api/...`.

## Consistency Rules

### Terminology
Use consistent terms throughout a project. Pick one term for each concept and stick to it (e.g. "method" vs "function" when referring to class members; "query parameter" vs "URL parameter"). In prose, prefer the human phrasing ("contact ID") over the code identifier (`contactId`), reserving the code form for code contexts.

### File Paths
- Always use forward slashes: `app/Http/Controllers/SomeController.php`.
- Always relative to the project root unless specified.
- Always in code formatting.

### Code References
- Function/method names with parentheses: `doSomething()`.
- Class names without extension: `SomeController`.
- Config references in the project's own idiom.

## Code Examples

### Good example format

Use this structure for end-to-end examples — scenario, input, output, then why:

```md
### Example: Fetch a resource URL

**Scenario:** Building a pre-filled form URL for an existing record

**Request:**
`GET /api/resource/123/false`

**Response:**
`https://example.com/form?set-form=1&first=John&last=Doe`

**Explanation:** The `false` option prevents an automatic redirect and returns the URL as a string instead.
```

### Poor example format

❌ Don't do this — it gives an input and an outcome with no scenario, no real output, and no reason:

```
Example: /api/resource/123/false Returns a URL
```

## File Delivery

When delivering documentation content:

1. **Single block** — provide complete content in one copyable Markdown block when asked for a full file.
2. **No split sections** — don't split content across messages unless the file is extremely large (>2000 lines).
3. **Complete and ready** — content should paste directly into a file.
4. **No placeholders** — don't use "..." or "[continued]"; provide the full content.

---

**Usage:** Reference this guide when creating documentation, explainers, or code comments. AI sessions should follow these conventions to keep documentation consistent across a project. Project-specific conventions (stack, framework, domain vocabulary) layer on top of — and never contradict — this baseline.

## Project-Specific Notes

Everything above is stack-neutral and identical in every repo — **edit it in the toolkit, never in a repo copy.** This section is the one part a repo fills in for itself. When adopting the guide, fill it in and delete the guidance italics; leave the body untouched.

- **Stack:** _(language(s), framework(s), and versions — e.g. "Laravel 12, PHP 8.1, Vue 3")_
- **Code root:** _(where source lives, e.g. `app/`, `src/`, `site/app/`)_
- **Code-doc folder:** _(where per-class/module docs live, if any — else "none")_
- **External systems:** _(CRMs, APIs, or services the docs will reference)_
- **Glossary:** _(project-specific terms and the one canonical spelling for each)_
- **Common patterns:** _(recurring idioms a doc writer should know — e.g. how config IDs are stored)_
