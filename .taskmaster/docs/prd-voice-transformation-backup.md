# PRD: Source-Aware Voice Transformation System

**Version**: 1.0
**Created**: 2026-01-07
**Status**: Approved
**Owner**: GhostWriter Content Pipeline

---

## 1. Executive Summary

GhostWriter generates Medium articles from external sources (Reddit, HN) and internal sources (GitHub). Currently, all content uses first-person ownership voice ("we built", "our project") regardless of source — creating intellectual dishonesty when the author didn't actually do the work. This PRD defines a Voice Transformation System that automatically adjusts narrative voice based on content source: "Journalist Observer" for external sources and "Practitioner Owner" for internal sources, while maintaining engagement quality.

---

## 2. Problem Statement

### Current Situation
- Content is sourced from Reddit, Hacker News, Twitter (external) and GitHub commits (internal)
- All generated articles use ownership voice: "we", "our", "my project", "I built"
- This creates implicit plagiarism — claiming ownership of others' work

### User Impact
- **Readers**: May feel deceived if they discover the author didn't do the work described
- **Author**: Credibility risk if fact-checked; ethical concerns
- **Platform**: Trust erosion with Medium audience

### Business Impact
- Reputation damage if exposed as claiming others' work
- Potential platform violations (Medium's authenticity policies)
- Limits ability to scale content without ethical concerns

### Why Solve Now
- Content pipeline is operational but ethically flawed
- Fixing voice at generation time is cheaper than post-hoc editing
- Building credibility early prevents future reputation damage

---

## 3. Goals & Success Metrics

| Goal | Metric | Baseline | Target | Timeframe |
|------|--------|----------|--------|-----------|
| Eliminate false ownership claims | Ownership pronouns in external-sourced articles | ~15 per article | 0 | Immediate |
| Maintain content quality | Reader engagement score (subjective) | Current baseline | No degradation | Ongoing |
| Preserve emotional hooks | Hook strength rating | Current baseline | No degradation | Ongoing |
| Invisible transformation | Reader detection rate | N/A | 0% notice difference | Ongoing |
| Internal source unchanged | Ownership voice preserved | Current | 100% maintained | Immediate |

---

## 4. User Stories

### US-001: External Source Voice Transformation
**As a** content creator using GhostWriter
**I want** articles from Reddit/HN to use observer voice automatically
**So that** I don't falsely claim ownership of others' work

**Acceptance Criteria:**
- [ ] Articles from Reddit sources contain zero instances of: "we", "our", "my", "I built", "I created", "we discovered"
- [ ] Observer voice maintains authoritative tone
- [ ] Emotional hooks are preserved (stories still engaging)
- [ ] Technical details and metrics are preserved

### US-002: Internal Source Voice Preservation
**As a** content creator documenting my own GitHub work
**I want** articles from my GitHub commits to use ownership voice
**So that** I can authentically share my experiences

**Acceptance Criteria:**
- [ ] Articles from GitHub sources maintain full ownership voice
- [ ] No voice transformation applied to internal sources
- [ ] "We", "our", "I built" language preserved

### US-003: Mixed Source Handling
**As a** content creator with multi-source signals
**I want** mixed sources to default to observer voice
**So that** I never accidentally claim ownership of external content

**Acceptance Criteria:**
- [ ] Mixed source signals detected and flagged
- [ ] Default to "external" (observer voice) for safety
- [ ] Clear indication in metadata that mixed sources were detected

### US-004: Voice Validation
**As a** content creator
**I want** automatic validation that voice rules are followed
**So that** I can catch violations before publishing

**Acceptance Criteria:**
- [ ] Post-generation scan identifies forbidden pronouns
- [ ] Violations flagged with line numbers and suggestions
- [ ] Option to auto-correct or manual review

---

## 5. Functional Requirements

### Source Detection

| ID | Requirement | Priority | Implementation Hint |
|----|-------------|----------|---------------------|
| REQ-001 | Add `source_type` field to signal schema with values: "external", "internal" | Must | Modify signal data structure in fetch scripts |
| REQ-002 | Classify Reddit, HN, Twitter sources as "external" | Must | Update fetch_reddit.py, add source classification |
| REQ-003 | Classify GitHub commits, internal docs as "internal" | Must | Update fetch_github.py, add source classification |
| REQ-004 | Default mixed/unknown sources to "external" | Must | Add fallback logic in signal processing |
| REQ-005 | Persist source_type through entire pipeline | Must | Pass source_type to all downstream functions |

### Voice Rules

| ID | Requirement | Priority | Implementation Hint |
|----|-------------|----------|---------------------|
| REQ-006 | Create external voice prompt template (Journalist Observer) | Must | New prompt template file or directive |
| REQ-007 | Create internal voice prompt template (Practitioner Owner) | Must | Existing prompt or minor modifications |
| REQ-008 | Forbidden words for external: "we", "our", "my", "I built", "I created", "we discovered", "our team", "my project" | Must | Define in voice rules config |
| REQ-009 | Allowed phrases for external: "teams have found", "engineers discovered", "one approach", "this method", "developers reported" | Must | Include in prompt template |
| REQ-010 | Select voice template based on source_type at generation time | Must | Conditional logic in generate_drafts.py |

### Validation

| ID | Requirement | Priority | Implementation Hint |
|----|-------------|----------|---------------------|
| REQ-011 | Create validation function to scan for forbidden pronouns | Must | New Python function or script |
| REQ-012 | Validation runs automatically after draft generation | Must | Hook into generate_drafts.py or generate_medium_full.py |
| REQ-013 | Flag violations with: line number, forbidden word, suggested replacement | Should | Return structured validation report |
| REQ-014 | Option to auto-correct common violations | Could | Regex replacement with confirmation |
| REQ-015 | Log validation results for quality tracking | Should | Append to .tmp/validation_log.json |

### Quality Invariants

| ID | Requirement | Priority | Implementation Hint |
|----|-------------|----------|---------------------|
| REQ-016 | Emotional hooks preserved regardless of voice | Must | Prompt engineering, explicit instruction |
| REQ-017 | Specific metrics/numbers preserved | Must | Prompt engineering |
| REQ-018 | Actionable takeaways preserved | Must | Prompt engineering |
| REQ-019 | Article structure unchanged (intro, sections, conclusion) | Must | Voice change only affects pronouns, not structure |

---

## 6. Non-Functional Requirements

### Performance
- Voice transformation adds < 100ms to generation time
- Validation scan completes in < 500ms per article

### Reliability
- Voice selection never fails silently (always explicit)
- Missing source_type defaults safely to "external"

### Maintainability
- Voice rules configurable without code changes (directive file)
- Easy to add new source types in future

### Testability
- Each voice template testable in isolation
- Validation function has unit tests with known inputs

---

## 7. Technical Considerations

### Architecture

```
┌─────────────────┐
│ Signal Sources  │
│ (Reddit/GitHub) │
└────────┬────────┘
         │ + source_type field
         ▼
┌─────────────────┐
│ Signal Storage  │
│ (with metadata) │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Voice Selector  │◄── source_type → prompt template
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Draft Generator │◄── voice-aware prompt
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Voice Validator │◄── scan for violations
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Final Draft     │
└─────────────────┘
```

### Files to Modify

| File | Changes |
|------|---------|
| `execution/fetch_reddit.py` | Add source_type="external" to signal |
| `execution/fetch_github.py` | Add source_type="internal" to signal |
| `execution/fetch_all.py` | Pass through source_type |
| `execution/generate_drafts.py` | Voice selector logic |
| `execution/generate_medium_full.py` | Voice selector logic |
| `directives/produce_content.md` | Update with voice rules |

### New Files to Create

| File | Purpose |
|------|---------|
| `directives/voice_rules.md` | Voice transformation SOP |
| `execution/validate_voice.py` | Post-generation validation script |
| `.tmp/validation_log.json` | Validation results log |

### Voice Prompt Templates

**External (Journalist Observer):**
```
You are writing as an authoritative technology reporter sharing insights
from the engineering community. You are an observer documenting learnings,
NOT someone who did this work yourself.

FORBIDDEN (never use):
- "we", "our", "my", "I built", "I created", "we discovered"
- "our team", "my project", "we implemented"

USE INSTEAD:
- "teams have found", "engineers discovered", "one approach"
- "this method", "developers reported", "the implementation"
- "a team recently shared", "one engineer's experience shows"

PRESERVE:
- Emotional hooks and dramatic openings
- Specific metrics and technical details
- Actionable takeaways and lessons learned
- Authoritative, confident tone
```

**Internal (Practitioner Owner):**
```
You are writing as an experienced engineer sharing your own war stories
and hard-won lessons. Use first-person ownership voice authentically.

ENCOURAGED:
- "we", "our", "my", "I built", "I created", "we discovered"
- "our team", "my project", "we implemented"
- Personal anecdotes and emotional moments

PRESERVE:
- Emotional hooks and dramatic openings
- Specific metrics and technical details
- Actionable takeaways and lessons learned
- Battle-scarred, authentic tone
```

### Database/Schema Changes

Signal schema addition:
```python
{
    "signal_id": "...",
    "title": "...",
    "source": "reddit",           # existing
    "source_type": "external",    # NEW FIELD
    "content": "...",
    # ... rest of schema
}
```

---

## 8. Implementation Roadmap

### Phase 1: Source Detection (Tasks 1-3)
- Add source_type field to signal schema
- Update fetch scripts to classify sources
- Verify source_type persists through pipeline

### Phase 2: Voice Templates (Tasks 4-6)
- Create voice rules directive
- Create external voice prompt template
- Create internal voice prompt template

### Phase 3: Voice Selection (Tasks 7-9)
- Implement voice selector in generate_drafts.py
- Implement voice selector in generate_medium_full.py
- Test with sample signals from each source

### Phase 4: Validation (Tasks 10-12)
- Create validate_voice.py script
- Integrate validation into generation pipeline
- Add logging and reporting

### Phase 5: Testing & Polish (Tasks 13-15)
- End-to-end testing with real signals
- Quality comparison (before/after)
- Documentation updates

---

## 9. Out of Scope

- **Anonymization**: NOT removing company names, usernames, or identifying details (per user decision)
- **Source citation**: NOT adding explicit "Source: Reddit" disclaimers
- **Retroactive fixing**: NOT transforming existing drafts (only new generations)
- **User-selectable voice**: NOT allowing per-article voice override (source determines voice)
- **Sentiment analysis**: NOT adjusting voice based on content sentiment

---

## 10. Open Questions & Risks

### Open Questions

| Question | Owner | Status |
|----------|-------|--------|
| Should validation auto-correct or just flag? | User | Decide during implementation |
| How to handle edge case pronouns ("one's", "oneself")? | Dev | Research during Task 11 |

### Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Voice transformation degrades quality | Medium | High | Extensive prompt engineering, A/B testing |
| False positives in validation (valid "we" flagged) | Medium | Low | Context-aware validation, manual review option |
| LLM ignores voice instructions | Low | High | Explicit examples, validation as safety net |
| Performance impact on generation | Low | Low | Voice selection is simple conditional |

---

## 11. Validation Checkpoints

| Checkpoint | After Task | Validation Criteria |
|------------|------------|---------------------|
| Source Detection Works | Task 3 | source_type correctly set for Reddit AND GitHub signals |
| Voice Templates Ready | Task 6 | Both templates generate distinct, quality content |
| Voice Selection Works | Task 9 | Correct template selected based on source_type |
| Validation Catches Violations | Task 12 | 100% of planted violations detected |
| End-to-End Quality | Task 15 | External articles have 0 ownership pronouns, quality maintained |

---

## Appendix A: Forbidden/Allowed Word Lists

### External Voice - Forbidden
```
we, We, WE
our, Our, OUR
my, My, MY
I built, I created, I developed, I designed, I implemented
we built, we created, we developed, we designed, we implemented
we discovered, we found, we learned, we realized
our team, our project, our system, our approach
my team, my project, my system, my approach
```

### External Voice - Allowed Alternatives
```
teams have found → replaces "we found"
engineers discovered → replaces "we discovered"
one approach → replaces "our approach"
this method → replaces "my method"
developers reported → replaces "we reported"
the team → replaces "our team"
the project → replaces "our project"
a recent implementation → replaces "our implementation"
```

---

## Appendix B: Example Transformations

### Before (Ownership Voice)
> "I still remember the day **our** RAG pipeline started hallucinating. **We** had built what **we** thought was a robust system, but **our** confidence was misplaced. **My** team and **I** spent weeks debugging..."

### After (Observer Voice)
> "The story of a RAG pipeline hallucinating is all too common. **A team** had built what **they** thought was a robust system, but **their** confidence was misplaced. **The engineers** spent weeks debugging..."

---

*PRD Generated: 2026-01-07*
*Quality Score: Pending Validation*
