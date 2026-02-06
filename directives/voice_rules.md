> **DEPRECATED**: This file's content has been merged into `directives/writing_rules.md`. This file is kept for reference only.

# Voice Transformation Rules

> SOP for source-aware narrative voice in content generation

## Purpose

Automatically adjust narrative voice based on content source to maintain authenticity. We never claim ownership of work we didn't do, but we also don't break the narrative by citing sources explicitly.

## Source Type Classification

| Source | source_type | Voice Style |
|--------|-------------|-------------|
| Reddit, HN, Twitter | `external` | Journalist Observer |
| GitHub commits, Internal docs | `internal` | Practitioner Owner |
| Mixed/Unknown | `external` | Journalist Observer (safe default) |

---

## External Voice (Journalist Observer)

Use when `source_type == "external"` — content is from Reddit, Hacker News, Twitter, or other external sources.

### Forbidden Words & Phrases

**Never use these in external-sourced content:**

```
# Ownership pronouns
we, We, WE
our, Our, OUR
my, My, MY

# Ownership claims
I built, I created, I developed, I designed, I implemented
we built, we created, we developed, we designed, we implemented
we discovered, we found, we learned, we realized
our team, our project, our system, our approach
my team, my project, my system, my approach
```

### Allowed Alternatives

| Instead of... | Use... |
|---------------|--------|
| we found | teams have found |
| we discovered | engineers discovered |
| our approach | one approach / this approach |
| my method | this method |
| we reported | developers reported |
| our team | the team / a team |
| our project | the project |
| I built | was built / the implementation |
| we learned | the lesson learned |
| my experience | one engineer's experience |

### Tone Guidelines

- **Role**: Authoritative technology reporter sharing insights from the community
- **Framing**: "Here's what I've learned from observing..." not "Here's what I built..."
- **Authority**: Confident and knowledgeable, but as an informed observer
- **Examples**:
  - "A team recently shared how they tackled this problem..."
  - "One engineer's experience shows that..."
  - "The approach that emerged from this discussion..."
  - "What's striking about this solution is..."

### What to Preserve

Even in observer voice, maintain:
- Emotional hooks and dramatic openings
- Specific metrics and numbers
- Actionable takeaways
- Technical depth and accuracy
- Confident, engaging tone

---

## Internal Voice (Practitioner Owner)

Use when `source_type == "internal"` — content is from your own GitHub commits, internal documentation, or original work.

### Encouraged Words & Phrases

```
# Ownership pronouns (use freely)
we, our, my, I

# Ownership claims (encouraged)
I built, I created, we developed
we discovered, we learned
our team, our project
my experience building...
```

### Tone Guidelines

- **Role**: Battle-scarred engineer sharing war stories
- **Framing**: "Here's what we built and what we learned the hard way..."
- **Authority**: Confident practitioner with firsthand experience
- **Examples**:
  - "I still remember the day our pipeline crashed..."
  - "We spent three weeks debugging this issue..."
  - "Our approach was unconventional, but it worked..."
  - "What I learned from building this system..."

---

## Quality Invariants (Both Voices)

Regardless of voice type, every article must preserve:

1. **Emotional Hooks**: Stories need tension, drama, relatable moments
2. **Specific Metrics**: Numbers add credibility ("reduced latency by 50%")
3. **Actionable Takeaways**: Reader should learn something useful
4. **Technical Accuracy**: Facts must be correct
5. **Engaging Tone**: Never dry or academic

---

## Implementation

### Detection
- `source_type` field is set at signal ingestion (fetch_reddit.py, fetch_github.py)
- Missing/unknown `source_type` defaults to `"external"` (safe choice)

### Application
- Voice rules are applied in prompt templates during generation
- generate_drafts.py and generate_medium_full.py select prompts based on `source_type`

### Validation
- validate_voice.py scans output for forbidden pronouns
- Violations are flagged with line numbers and suggestions
- Run after generation, before publishing

---

## Examples

### Before (Ownership Voice on External Source)
> "I still remember the day **our** RAG pipeline started hallucinating. **We** had built what **we** thought was a robust system, but **our** confidence was misplaced. **My** team and **I** spent weeks debugging..."

### After (Observer Voice on External Source)
> "The story of a RAG pipeline hallucinating is all too familiar. **A team** had built what **they** thought was a robust system, but **their** confidence was misplaced. **The engineers** spent weeks debugging..."

---

*Last Updated: 2026-01-07*
*Version: 1.0*
