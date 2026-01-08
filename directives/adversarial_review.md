# Directive: Adversarial Expert Panel Review

> Quality gate that ensures no substandard content reaches publication

## Purpose

Implement a multi-expert adversarial review loop that critiques content through the lens of world-class copywriting agencies until it meets elite standards (score >= 7/10).

## The Expert Panel

The adversarial panel simulates critique from:

### Agency Experts
| Expert | Focus | Pet Peeves |
|--------|-------|------------|
| Digital Commerce Partners | Conversion | Weak CTAs, no value prop |
| AWISEE | Data-driven storytelling | Claims without evidence, placeholders |
| Feldman Creative | Extreme clarity | Jargon, passive voice, melodrama |

### Brand Giants
| Expert | Focus | Pet Peeves |
|--------|-------|------------|
| Google | SEO & UX | No keyword strategy, poor structure |
| Apple | Simplicity | Tonal inconsistency, visual chaos |
| Zomato/Swiggy | Engagement | Zero personality, nothing quotable |

### SEO Specialists
| Expert | Focus | Pet Peeves |
|--------|-------|------------|
| Victorious | Technical SEO | Missing keywords, broken hierarchy |
| Compose.ly | Production quality | HTML artifacts, incomplete sections |

### Creative Houses
| Expert | Focus | Pet Peeves |
|--------|-------|------------|
| Serviceplan | Creative excellence | AI-generated feel, no narrative |
| Emirates Graphic | Global quality | Scroll-past openings, no voice |

---

## Kill Phrases (Instant Failures)

These phrases trigger automatic score caps and must be eliminated:

```
PLACEHOLDER_TEXT:     "..."
WEAK_CTA:             "What's been your experience?"
TEMPLATE_PHRASE:      "This aligns with what I'm seeing"
BORING_OPENER:        "In this article"
HTML_ARTIFACT:        "<!-- SC_OFF", "<div class="
METADATA_LEAK:        "Contains some technical content"
CLICHE_CLOSER:        "As AI engineering matures"
HASHTAG_SPAM:         "#AIEngineering #MachineLearning #LLMOps #ProductionAI"
TEMPLATE_OPENER:      "🚀 Interesting insight from r/"
```

**If any kill phrase is found, score is capped at 4/10 regardless of other qualities.**

---

## Quality Scoring (1-10)

| Score | Meaning | Action |
|-------|---------|--------|
| 1-3 | Unpublishable garbage | Full rewrite |
| 4-5 | Major rewrites needed | Significant revision |
| 6 | Mediocre, needs work | Targeted fixes |
| **7** | **Minimum acceptable** | **Pass threshold** |
| 8 | Good quality | Minor polish |
| 9-10 | Exceptional (rare) | Ready to publish |

---

## Review Loop Flow

```
┌─────────────────────────────────────────────────────────┐
│                     QUALITY GATE                        │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Draft v1 ──► Adversarial Panel Review                  │
│                      │                                  │
│                      ▼                                  │
│              Score >= 7.0? ──► YES ──► APPROVED         │
│                      │                                  │
│                      NO                                 │
│                      │                                  │
│                      ▼                                  │
│              Generate Fix Instructions                  │
│                      │                                  │
│                      ▼                                  │
│              Writer Revises Draft                       │
│                      │                                  │
│                      ▼                                  │
│              Loop (max 3 iterations)                    │
│                      │                                  │
│                      ▼                                  │
│              Still < 7.0? ──► ESCALATE                  │
│                              (Human review)             │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## Tools / Scripts

### 1. Quality Gate (`execution/quality_gate.py`)

**Purpose**: Orchestrate the REVIEW <-> FIX loop.

**Usage**:
```bash
# Review a single draft
python execution/quality_gate.py --input drafts/draft.md --platform medium

# With custom iteration limit
python execution/quality_gate.py --input draft.md --max-iterations 5

# JSON output for integration
python execution/quality_gate.py --input draft.md --json
```

**Output**: Approved or escalated draft with quality metadata.

---

### 2. Quality-Gated Draft Generation (`execution/generate_drafts_v2.py`)

**Purpose**: Generate drafts with built-in quality gate.

**Usage**:
```bash
# Generate LinkedIn drafts with quality gate
python execution/generate_drafts_v2.py --platform linkedin --limit 5

# Generate Medium articles
python execution/generate_drafts_v2.py --platform medium --limit 3

# Filter by source
python execution/generate_drafts_v2.py --unified --source reddit --platform both
```

**Output**:
- Approved drafts in `drafts/` with quality metadata
- Escalated drafts marked for human review

---

### 3. Adversarial Panel Agent (`execution/agents/adversarial_panel.py`)

**Purpose**: Multi-expert critique engine.

**Usage** (in code):
```python
from execution.agents.adversarial_panel import AdversarialPanelAgent

panel = AdversarialPanelAgent()
verdict = panel.review_content(
    content="Your draft here...",
    platform="medium",
    iteration=1
)

print(f"Score: {verdict.average_score}/10")
print(f"Passed: {verdict.passed}")
print(f"Critical Failures: {verdict.critical_failures}")
```

---

## Quality Checklist

Every piece of content must pass:

- [ ] **Hook Test**: First 10 words create curiosity or tension
- [ ] **Specificity Test**: At least 3 specific numbers/metrics
- [ ] **Memorable Test**: At least 1 quotable/screenshot-worthy line
- [ ] **Completeness Test**: No placeholders, no "..."
- [ ] **Cleanliness Test**: No HTML artifacts, no metadata leaks
- [ ] **Voice Test**: Consistent tone throughout
- [ ] **CTA Test**: Specific, urgent, value-offering call-to-action
- [ ] **Kill Phrase Test**: None of the forbidden phrases present

---

## Edge Cases

### 1. Content Cannot Pass After Max Iterations

**Scenario**: Draft fails 3 iterations, still below 7.0.

**Action**:
- Mark as "escalated" in output
- Save to `drafts/{platform}_{timestamp}_escalated.md`
- Human must review and manually improve
- Consider if the source material is too weak

### 2. All Experts Give Conflicting Feedback

**Scenario**: One expert says "too technical", another says "not technical enough".

**Action**:
- Weight feedback by platform fit (LinkedIn = accessible, Medium = depth)
- Prioritize fix instructions by frequency (mentioned by 2+ experts)
- If truly conflicting, escalate for human decision

### 3. Kill Phrase in Source Material

**Scenario**: Original Reddit post contains "..."

**Action**:
- Writer must transform, not copy
- Ensure content sanitization happens BEFORE review
- If kill phrase persists after revision, investigate writer prompt

---

## Success Metrics

| Metric | Target | Current |
|--------|--------|---------|
| First-pass approval rate | > 30% | TBD |
| Avg iterations to approval | < 2.5 | TBD |
| Escalation rate | < 20% | TBD |
| Kill phrase violations | 0% | TBD |

---

## Integration with Existing Pipeline

### Before (Template-based)
```
Signal → Template → Draft → Publish
(No quality gate, template phrases everywhere)
```

### After (Agent-based with Quality Gate)
```
Signal → Writer Agent → Draft v1 → Quality Gate → Approved Draft → Publish
                                      ↓
                              (Loop if < 7.0)
```

### Migration
1. Use `generate_drafts_v2.py` instead of `generate_drafts.py`
2. Run existing drafts through `quality_gate.py` before publishing
3. Monitor approval rates and adjust thresholds as needed

---

## Maintenance

- **Weekly**: Review escalated drafts, identify patterns in failures
- **Monthly**: Analyze which expert critiques are most predictive of engagement
- **Quarterly**: Update kill phrase list based on new bad patterns discovered

---

*Last Updated: 2026-01-07*
*Version: 1.0*
