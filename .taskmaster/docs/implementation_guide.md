# Implementation Guide: Voice Transformation System

## Overview

This guide provides exact code changes for each task based on analysis of the existing codebase.

---

## TASK-001: Add source_type field to signal schema

### Current State
Posts are stored as dicts in `fetch_reddit.py` (line 73-84) and commits in `fetch_github.py` (line 139-153).

### Changes Required

**No schema change needed** - SQLite is flexible. Just add the field to the dicts being created.

If you want to formally track it, add to database schema:
```sql
ALTER TABLE posts ADD COLUMN source_type TEXT DEFAULT 'external';
ALTER TABLE github_commits ADD COLUMN source_type TEXT DEFAULT 'internal';
```

---

## TASK-002: Update fetch_reddit.py

### File: `execution/fetch_reddit.py`
### Location: Lines 73-84

### Current Code:
```python
post = {
    'subreddit': subreddit_name,
    'title': entry.get('title', ''),
    'url': entry.get('link', ''),
    'author': entry.get('author', 'unknown'),
    'content': content,
    'timestamp': timestamp,
    'upvotes': 0,
    'num_comments': 0,
    'retrieved_at': int(time.time())
}
```

### New Code:
```python
post = {
    'subreddit': subreddit_name,
    'title': entry.get('title', ''),
    'url': entry.get('link', ''),
    'author': entry.get('author', 'unknown'),
    'content': content,
    'timestamp': timestamp,
    'upvotes': 0,
    'num_comments': 0,
    'retrieved_at': int(time.time()),
    'source_type': 'external'  # NEW: Voice transformation field
}
```

---

## TASK-003: Update fetch_github.py

### File: `execution/fetch_github.py`
### Location: Lines 139-153

### Current Code:
```python
commit_info = {
    "repo_owner": owner,
    "repo_name": repo,
    "commit_sha": sha,
    # ... other fields ...
    "retrieved_at": int(time.time())
}
```

### New Code:
```python
commit_info = {
    "repo_owner": owner,
    "repo_name": repo,
    "commit_sha": sha,
    # ... other fields ...
    "retrieved_at": int(time.time()),
    "source_type": "internal"  # NEW: Voice transformation field
}
```

---

## TASK-004: Create voice_rules.md directive

### File: `directives/voice_rules.md` (NEW)

### Content:
```markdown
# Voice Transformation Rules

## Purpose
Automatically adjust narrative voice based on content source to maintain authenticity.

## Source Types

| Source | source_type | Voice |
|--------|-------------|-------|
| Reddit, HN, Twitter | external | Journalist Observer |
| GitHub, Internal docs | internal | Practitioner Owner |
| Mixed/Unknown | external | Journalist Observer (safe default) |

## External Voice (Journalist Observer)

### Forbidden Words
- we, We, WE
- our, Our, OUR
- my, My, MY
- I built, I created, I developed, I designed, I implemented
- we built, we created, we developed, we designed, we implemented
- we discovered, we found, we learned, we realized
- our team, our project, our system, our approach
- my team, my project, my system, my approach

### Allowed Alternatives
| Instead of | Use |
|------------|-----|
| we found | teams have found |
| we discovered | engineers discovered |
| our approach | one approach / this approach |
| my method | this method |
| we reported | developers reported |
| our team | the team / a team |
| our project | the project |
| I built | was built / the implementation |

### Tone Guidelines
- Authoritative technology reporter sharing insights
- Observer documenting learnings from the community
- Confident but not claiming ownership
- "Here's what I've learned from observing..." framing

## Internal Voice (Practitioner Owner)

### Encouraged Words
- we, our, my, I
- I built, I created, we developed
- our team, my project
- Personal anecdotes and war stories

### Tone Guidelines
- Battle-scarred engineer sharing experience
- Authentic first-person storytelling
- "Here's what we built and learned..."

## Quality Invariants (Both Voices)

Regardless of voice, preserve:
1. Emotional hooks and dramatic openings
2. Specific metrics and technical details
3. Actionable takeaways
4. Confident, authoritative tone
```

---

## TASK-005: External Voice Prompt Template

### Create: `execution/prompts/external_voice.py` (or add to existing prompt config)

```python
EXTERNAL_VOICE_PROMPT = """
You are writing as an authoritative technology reporter sharing insights from the engineering community.

CRITICAL: You are an OBSERVER documenting learnings. You did NOT do this work yourself.

## FORBIDDEN (Never use these):
- "we", "our", "my" (ownership pronouns)
- "I built", "I created", "I developed", "I designed", "I implemented"
- "we built", "we created", "we discovered", "we found", "we learned"
- "our team", "our project", "our system", "our approach"
- "my team", "my project", "my experience building"

## USE INSTEAD:
- "teams have found" (not "we found")
- "engineers discovered" (not "we discovered")
- "one approach" or "this approach" (not "our approach")
- "the implementation" (not "my implementation")
- "developers reported" (not "we reported")
- "a team recently shared" (not "our team")
- "one engineer's experience shows" (not "my experience")

## PRESERVE:
- Emotional hooks and dramatic openings (stories still need tension)
- Specific metrics and numbers (these add credibility)
- Actionable takeaways and lessons learned
- Confident, authoritative tone (you're an expert observer)
- Technical depth and accuracy

## EXAMPLE TRANSFORMATIONS:
❌ "I still remember the day our RAG pipeline started hallucinating"
✅ "The story of a RAG pipeline hallucinating is all too common"

❌ "We spent three weeks debugging this issue"
✅ "The team spent three weeks debugging this issue"

❌ "Our approach reduced latency by 50%"
✅ "This approach reduced latency by 50%"

Write with authority about what you've OBSERVED and LEARNED from the community, not what you personally built.
"""
```

---

## TASK-006: Internal Voice Prompt Template

```python
INTERNAL_VOICE_PROMPT = """
You are writing as an experienced engineer sharing your own war stories and hard-won lessons.

## ENCOURAGED:
- Use "we", "our", "my", "I" authentically
- Share personal anecdotes and emotional moments
- "I built", "I created", "we discovered", "our team"
- Specific personal experiences and realizations

## TONE:
- Battle-scarred engineer sharing experience
- Confident and opinionated
- "Here's what we built and what we learned the hard way"
- Personal, conversational, authentic

## PRESERVE:
- Emotional hooks and dramatic openings
- Specific metrics and numbers
- Actionable takeaways
- Technical depth

This is YOUR story. Own it.
"""
```

---

## TASK-007: Voice Selector in generate_drafts.py

### File: `execution/generate_drafts.py`
### Add after line 25 (after OUTPUT_DIR definition):

```python
# Voice prompt templates
EXTERNAL_VOICE_CONTEXT = """
VOICE RULE: You are an OBSERVER, not the person who did this work.
- FORBIDDEN: "we", "our", "my", "I built", "we discovered"
- USE: "teams found", "engineers discovered", "this approach", "the implementation"
"""

INTERNAL_VOICE_CONTEXT = """
VOICE RULE: This is YOUR experience. Use ownership voice authentically.
- ENCOURAGED: "we", "our", "my", "I built", "we discovered"
"""

def get_voice_context(source_type: str) -> str:
    """Return voice context based on source type."""
    if source_type == "internal":
        return INTERNAL_VOICE_CONTEXT
    return EXTERNAL_VOICE_CONTEXT  # Default to external (safe)
```

### Modify generate_medium_draft function (line 92):

```python
def generate_medium_draft(post_id, subreddit, title, content, url, reasoning, market_strategy, source_type="external"):
    """Generate a Medium article draft with source-aware voice."""

    voice_context = get_voice_context(source_type)

    # Include voice_context in LLM prompt when implemented
    # For now, add to template header:

    draft = f"""# {title}

{voice_context}

*Insights from the AI Engineering Community*
...
"""
```

---

## TASK-008: Voice Selector in generate_medium_full.py

### File: `execution/generate_medium_full.py`
### This is more complex - voice needs to be injected into the SpecialistAgents

### Option A: Pass source_type as argument
Add `--source-type` argument and modify agent prompts.

### Option B: Detect from topic/signal
If the signal/topic includes source metadata, extract it.

### Recommended Changes:

**Line 29-30 (argparse):**
```python
parser.add_argument("--topic", help="Topic or Signal to write about", required=True)
parser.add_argument("--source-type", choices=["external", "internal"], default="external",
                    help="Source type for voice selection (default: external)")
```

**Line 83-96 (story_agent):**
```python
# Modify constraint_instruction based on source_type
if args.source_type == "external":
    story_instruction = """Your job is to make this feel like it's written by a REAL technology journalist.

1. You are an OBSERVER sharing what you've learned from the community.
2. NEVER use "we", "our", "my", "I built" - you didn't do this work.
3. USE: "teams have found", "engineers discovered", "one approach"
4. Add brief observations: "I've seen this pattern..." or "What's striking about this..."
5. The personality should feel like a thoughtful analyst, not a participant.

Do NOT claim ownership of any work described."""
else:
    story_instruction = """Your job is to make this feel like it's written by a REAL engineer.

1. Add 1-2 brief personal moments: a frustration, a realization, a late-night debugging session.
2. Use "I" and "we" authentically - this is YOUR story.
3. The personality should feel like a smart colleague explaining something over coffee."""

story_agent = SpecialistAgent(
    constraint_name="Storytelling Architect",
    constraint_instruction=story_instruction
)
```

---

## TASK-010: Create validate_voice.py

### File: `execution/validate_voice.py` (NEW)

```python
#!/usr/bin/env python3
"""
Validate article voice based on source type.
Scans for forbidden pronouns in external-sourced content.
"""

import re
import json
from pathlib import Path
from datetime import datetime

# Forbidden patterns for external voice
FORBIDDEN_PATTERNS = [
    (r'\b[Ww]e\b', "we"),
    (r'\b[Oo]ur\b', "our"),
    (r'\b[Mm]y\b', "my"),
    (r'\bI built\b', "I built"),
    (r'\bI created\b', "I created"),
    (r'\bI developed\b', "I developed"),
    (r'\bI designed\b', "I designed"),
    (r'\bI implemented\b', "I implemented"),
    (r'\b[Ww]e built\b', "we built"),
    (r'\b[Ww]e created\b', "we created"),
    (r'\b[Ww]e discovered\b', "we discovered"),
    (r'\b[Ww]e found\b', "we found"),
    (r'\b[Ww]e learned\b', "we learned"),
    (r'\b[Oo]ur team\b', "our team"),
    (r'\b[Oo]ur project\b', "our project"),
    (r'\b[Mm]y team\b', "my team"),
    (r'\b[Mm]y project\b', "my project"),
]

# Suggested replacements
REPLACEMENTS = {
    "we": "teams / engineers / developers",
    "our": "the / this",
    "my": "the / this",
    "I built": "was built / the implementation",
    "we found": "teams have found",
    "we discovered": "engineers discovered",
    "our team": "the team / a team",
    "our project": "the project",
}


def validate_external_voice(content: str) -> dict:
    """
    Validate that external-sourced content doesn't use ownership pronouns.

    Returns:
        dict with 'valid', 'violations', and 'suggestions'
    """
    violations = []
    lines = content.split('\n')

    for line_num, line in enumerate(lines, 1):
        for pattern, label in FORBIDDEN_PATTERNS:
            matches = re.finditer(pattern, line)
            for match in matches:
                violations.append({
                    'line': line_num,
                    'column': match.start(),
                    'word': match.group(),
                    'label': label,
                    'context': line.strip()[:100],
                    'suggestion': REPLACEMENTS.get(label, f"Rephrase without '{label}'")
                })

    return {
        'valid': len(violations) == 0,
        'violation_count': len(violations),
        'violations': violations
    }


def validate_content(content: str, source_type: str) -> dict:
    """
    Validate content based on source type.

    Args:
        content: Article content
        source_type: 'external' or 'internal'

    Returns:
        Validation result dict
    """
    if source_type == "internal":
        # Internal sources can use any voice
        return {'valid': True, 'violation_count': 0, 'violations': [], 'note': 'Internal source - no restrictions'}

    return validate_external_voice(content)


def print_validation_report(result: dict, filename: str = None):
    """Print a formatted validation report."""
    if filename:
        print(f"\n📄 Validating: {filename}")

    if result['valid']:
        print("✅ PASSED - No forbidden ownership pronouns found")
        return

    print(f"❌ FAILED - {result['violation_count']} violations found:\n")

    for v in result['violations'][:20]:  # Limit output
        print(f"  Line {v['line']}: '{v['word']}'")
        print(f"    Context: ...{v['context']}...")
        print(f"    Suggestion: {v['suggestion']}\n")

    if result['violation_count'] > 20:
        print(f"  ... and {result['violation_count'] - 20} more violations")


def log_validation(result: dict, filename: str, log_path: str = ".tmp/validation_log.json"):
    """Append validation result to log file."""
    log_file = Path(log_path)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    # Load existing log
    if log_file.exists():
        with open(log_file) as f:
            log = json.load(f)
    else:
        log = []

    # Append new entry
    log.append({
        'timestamp': datetime.now().isoformat(),
        'filename': filename,
        'valid': result['valid'],
        'violation_count': result['violation_count']
    })

    # Save
    with open(log_file, 'w') as f:
        json.dump(log, f, indent=2)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Validate article voice')
    parser.add_argument('file', help='Markdown file to validate')
    parser.add_argument('--source-type', choices=['external', 'internal'], default='external')
    parser.add_argument('--log', action='store_true', help='Log result to validation_log.json')

    args = parser.parse_args()

    with open(args.file, 'r', encoding='utf-8') as f:
        content = f.read()

    result = validate_content(content, args.source_type)
    print_validation_report(result, args.file)

    if args.log:
        log_validation(result, args.file)
        print(f"\n📝 Logged to .tmp/validation_log.json")

    # Exit with error code if invalid
    exit(0 if result['valid'] else 1)
```

---

## Testing Commands

After implementation, test with:

```bash
# Test source detection
python execution/fetch_reddit.py --subreddits LocalLLaMA --max-posts 5
# Verify posts have source_type='external'

python execution/fetch_github.py --repos microsoft/semantic-kernel --max-commits 5
# Verify commits have source_type='internal'

# Test voice validation
python execution/validate_voice.py drafts/medium_polished_7_lies_llm_evals.md --source-type external

# Test full pipeline
python execution/generate_medium_full.py --topic "Test signal" --source-type external
python execution/validate_voice.py drafts/medium_full_*.md --source-type external
```

---

## Summary: Files to Create/Modify

| File | Action | Task |
|------|--------|------|
| `execution/fetch_reddit.py` | Modify | TASK-002 |
| `execution/fetch_github.py` | Modify | TASK-003 |
| `directives/voice_rules.md` | Create | TASK-004 |
| `execution/prompts/external_voice.py` | Create | TASK-005 |
| `execution/prompts/internal_voice.py` | Create | TASK-006 |
| `execution/generate_drafts.py` | Modify | TASK-007 |
| `execution/generate_medium_full.py` | Modify | TASK-008 |
| `execution/validate_voice.py` | Create | TASK-010 |
