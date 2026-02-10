# GhostWriter 3.0: WSJ-Tier AI Writing Agency
## Product Requirements Document

**Version:** 3.0.0
**Created:** 2026-01-08
**Status:** Ready for TaskMaster Parsing
**Complexity:** Complex (4 Phases, ~60 Tasks)
**Previous PRD:** Voice Transformation System (backed up to prd-voice-transformation-backup.md)

---

## 1. Executive Summary

GhostWriter is being transformed from a functional multi-agent content pipeline into a **WSJ-tier AI writing agency** that meets the editorial standards of the world's leading publications (WSJ, BBC, CNN, CBC Radio). This transformation addresses critical infrastructure gaps, implements mandatory fact verification, modernizes the architecture with LangGraph state machines, and adds enterprise-grade quality controls including multi-model adversarial review panels, C2PA content provenance for EU AI Act compliance, and human-in-the-loop review workflows.

The transformation will enable GhostWriter to produce content that passes the **WSJ Four Showstoppers** (Attribution, Source Quality, Tone, No Surprises), meets BBC accuracy standards ("well-sourced, based on sound evidence, thoroughly tested and presented in clear, precise language"), and fully complies with **EU AI Act Article 50** disclosure requirements effective August 2026.

**Note:** The Voice Transformation System from PRD v1.0 is incorporated into this comprehensive transformation as part of the Style/Quality requirements.

---

## 2. Problem Statement

### 2.1 Current Situation

The existing GhostWriter codebase (45/100 production readiness score) has significant gaps:

- **Hardcoded Windows paths** breaking cross-platform execution (`OUTPUT_DIR = Path("n:/RedditNews/drafts")`)
- **No mandatory fact verification** - articles can publish with unverified claims
- **Scattered configuration** across 34 Python files with no centralization
- **Single-model quality review** - no adversarial critique from different model families
- **No content provenance** - fails EU AI Act compliance requirements
- **No human review workflow** - fully autonomous with no approval gates
- **Untested in production** - missing test suite and monitoring
- **Voice inconsistency** - ownership pronouns used for external content (per v1.0 PRD)

### 2.2 User Impact

- **Readers**: Risk receiving AI-generated content with unverified facts
- **Publishers**: Legal exposure from EU AI Act non-compliance (up to EUR 35M penalties)
- **Content creators**: No tools to review/approve AI drafts before publication
- **Business**: Reputation risk from publishing low-quality AI content

### 2.3 Business Impact

- **Compliance Risk**: EU AI Act Article 50 requires machine-readable AI disclosure by August 2026
- **Quality Risk**: Without adversarial review, hallucinations and bias slip through
- **Scalability Risk**: Hardcoded paths prevent deployment to cloud environments
- **Trust Risk**: No provenance trail undermines content credibility

### 2.4 Why Solve Now

- EU AI Act deadlines approaching (August 2026 for full Article 50)
- Current codebase has foundational issues blocking production deployment
- Multi-model adversarial review is now table-stakes for AI content quality
- Competitive landscape demands WSJ-tier output quality

---

## 3. Goals & Success Metrics

### 3.1 Primary Goals

| Goal | Metric | Baseline | Target | Timeframe |
|------|--------|----------|--------|-----------|
| **G1: Production Ready** | Production readiness score | 45/100 | 90/100 | Phase 1-2 |
| **G2: Fact Verification** | % claims with verified sources | ~30% | 95%+ | Phase 1 |
| **G3: Quality Score** | Articles passing quality gate | Unknown | 85%+ | Phase 2 |
| **G4: EU AI Act Compliance** | C2PA manifests embedded | 0% | 100% | Phase 3 |
| **G5: Human Review Integration** | HITL approval rate | N/A | 100% for high-risk | Phase 3 |

### 3.2 Secondary Goals

| Goal | Metric | Baseline | Target |
|------|--------|----------|--------|
| **G6: Kill Phrase Detection** | Journalese/bias terms detected | 0 | 100% |
| **G7: Multi-Model Review** | Articles reviewed by 3+ model families | 0% | 100% |
| **G8: Cost Optimization** | Cost per article | Unknown | Track + 20% reduction |
| **G9: Test Coverage** | Code coverage | 0% | 80%+ |
| **G10: Cross-Platform** | Windows/Linux/Mac support | Windows only | All platforms |
| **G11: Voice Accuracy** | Ownership pronouns in external content | ~15/article | 0 |

---

## 4. User Stories

### US-001: Content Quality Assurance
**As a** content editor
**I want to** review AI-generated articles with fact-check annotations
**So that** I can quickly verify claims and approve publication

**Acceptance Criteria:**
- [ ] Each claim in the article is highlighted with verification status
- [ ] Sources are displayed alongside claims with one-click verification links
- [ ] Editor can approve, request revision, or reject with notes
- [ ] All decisions are logged to an immutable audit trail

### US-002: Adversarial Quality Review
**As a** quality manager
**I want to** have AI content reviewed by multiple model families
**So that** bias and hallucinations from any single model are caught

**Acceptance Criteria:**
- [ ] At least 3 different model families review each article
- [ ] Each reviewer provides structured feedback on accuracy, tone, and completeness
- [ ] Conflicting assessments trigger escalation for human review
- [ ] Quality scores are aggregated with weighted voting

### US-003: EU AI Act Compliance
**As a** compliance officer
**I want to** ensure all AI-generated content has proper provenance metadata
**So that** we meet EU AI Act Article 50 disclosure requirements

**Acceptance Criteria:**
- [ ] C2PA manifests are embedded in all published content
- [ ] Visible AI disclosure statement is included in content body
- [ ] Schema.org JSON-LD metadata marks content as trainedAlgorithmicMedia
- [ ] Multi-agent provenance chain is recorded and exportable

### US-004: Cross-Platform Deployment
**As a** DevOps engineer
**I want to** deploy GhostWriter to any environment
**So that** we can run in cloud, containers, or local development

**Acceptance Criteria:**
- [ ] No hardcoded absolute paths in codebase
- [ ] All paths use pathlib.Path with cross-platform resolution
- [ ] Configuration loaded from environment variables with sensible defaults
- [ ] Works on Windows, Linux, and macOS without code changes

### US-005: Fact Verification Pipeline
**As a** journalist
**I want to** ensure every factual claim is verified against authoritative sources
**So that** published content meets BBC accuracy standards

**Acceptance Criteria:**
- [ ] Claims are automatically extracted from generated content
- [ ] Each claim is verified against at least 2 independent sources
- [ ] Unverified claims are flagged and require human approval
- [ ] Verification sources are cited in final output

### US-006: Human-in-the-Loop Review
**As an** editor-in-chief
**I want to** require human approval for high-risk content
**So that** sensitive topics receive appropriate editorial oversight

**Acceptance Criteria:**
- [ ] Review dashboard shows pending items with priority indicators
- [ ] Side-by-side comparison view shows AI draft vs editor revisions
- [ ] Escalation workflow routes to senior editors when needed
- [ ] Review decisions are stored with timestamps and rationale

### US-007: Voice-Aware Content Generation
**As a** content creator
**I want** articles from external sources to use observer voice automatically
**So that** I don't falsely claim ownership of others' work

**Acceptance Criteria:**
- [ ] External source articles contain zero ownership pronouns
- [ ] Internal source articles maintain authentic ownership voice
- [ ] Voice validation catches violations before publication
- [ ] Quality and engagement preserved regardless of voice

---

## 5. Functional Requirements

### Phase 1: Critical Infrastructure (REQ-1xx)

| ID | Requirement | Priority | Task Hint |
|----|-------------|----------|-----------|
| **REQ-101** | Centralize all configuration in execution/config.py using dataclasses | Must | Migrate hardcoded values |
| **REQ-102** | Replace all hardcoded paths with config.paths properties | Must | Search/replace all files |
| **REQ-103** | Implement environment variable loading with .env support | Must | Use python-dotenv |
| **REQ-104** | Add path validation on startup with clear error messages | Must | Validate paths exist |
| **REQ-105** | Create mandatory fact verification step before quality review | Must | FactVerificationAgent |
| **REQ-106** | Require minimum 2 verified sources per factual claim | Must | Claim extraction + verification |
| **REQ-107** | Block publication of articles with >1 unverified claim | Must | Quality gate check |
| **REQ-108** | Implement claim extraction using NLP | Should | Use spaCy or similar |
| **REQ-109** | Create fact verification API wrapper (Perplexity, Gemini) | Must | Unified verification interface |
| **REQ-110** | Add verification status to article metadata | Must | Extend article schema |

### Phase 2: Architecture Modernization (REQ-2xx)

| ID | Requirement | Priority | Task Hint |
|----|-------------|----------|-----------|
| **REQ-201** | Migrate orchestration to LangGraph state machine | Must | StateGraph with typed state |
| **REQ-202** | Define typed state schema for article workflow | Must | TypedDict with all fields |
| **REQ-203** | Implement node-level caching with SQLite checkpointer | Should | LangGraph MemorySaver |
| **REQ-204** | Add interrupt_before hooks for human approval gates | Must | LangGraph interrupt pattern |
| **REQ-205** | Implement multi-model adversarial review panel | Must | Claude + GPT + Gemini |
| **REQ-206** | Create weighted voting system for quality scores | Must | Aggregate with weights |
| **REQ-207** | Add WSJ Four Showstoppers checklist as review criteria | Must | Attribution/Source/Tone/No Surprises |
| **REQ-208** | Implement kill phrase detection | Must | Regex + NLP for journalese |
| **REQ-209** | Add bias indicator scoring | Should | Sentiment + loaded language |
| **REQ-210** | Create escalation logic for conflicting reviews | Must | Human review trigger |
| **REQ-211** | Implement source-aware voice selection | Must | External vs Internal voice |
| **REQ-212** | Create voice validation post-processing | Must | Forbidden pronoun detection |

### Phase 3: WSJ-Tier Quality Systems (REQ-3xx)

| ID | Requirement | Priority | Task Hint |
|----|-------------|----------|-----------|
| **REQ-301** | Implement C2PA manifest generation | Must | Use c2pa-python library |
| **REQ-302** | Embed provenance metadata in output files | Must | JSON-LD + inline disclosure |
| **REQ-303** | Record multi-agent workflow chain | Must | Track agent sequence |
| **REQ-304** | Add EU AI Act compliant disclosure statements | Must | Human + machine readable |
| **REQ-305** | Create Streamlit HITL review dashboard | Must | Queue + Review + Audit |
| **REQ-306** | Implement side-by-side diff view for editor revisions | Should | streamlit-diff-viewer |
| **REQ-307** | Add fact-check highlighting panel | Must | Claim status visualization |
| **REQ-308** | Create audit trail with timestamp, user, action | Must | Immutable log |
| **REQ-309** | Implement escalation workflow (Editor->Senior->Legal) | Should | Hierarchy-based routing |
| **REQ-310** | Integrate Agent-Lightning for prompt optimization | Could | RL-based improvement |

### Phase 4: Production Hardening (REQ-4xx)

| ID | Requirement | Priority | Task Hint |
|----|-------------|----------|-----------|
| **REQ-401** | Create comprehensive test suite | Must | pytest with 80%+ coverage |
| **REQ-402** | Add unit tests for all agents | Must | Mock LLM responses |
| **REQ-403** | Add integration tests for pipeline | Must | End-to-end workflow |
| **REQ-404** | Implement cost tracking per article | Should | Token counting + pricing |
| **REQ-405** | Add observability with structured logging | Must | JSON logs with trace IDs |
| **REQ-406** | Create performance metrics dashboard | Should | Grafana/Prometheus |
| **REQ-407** | Implement rate limiting and error recovery | Must | Retry with backoff |
| **REQ-408** | Add health checks for all external services | Should | API monitoring |
| **REQ-409** | Create deployment documentation | Must | Docker + env setup |
| **REQ-410** | Implement graceful shutdown and state persistence | Should | Checkpoint on interrupt |

---

## 6. Non-Functional Requirements

### 6.1 Performance

| Metric | Target | Rationale |
|--------|--------|-----------|
| Article generation time | < 5 minutes | User expectation |
| Fact verification time | < 2 minutes per claim | Parallel verification |
| Quality review time | < 3 minutes | Multi-model parallel |
| HITL dashboard load time | < 2 seconds | Responsive UX |
| API response p95 | < 500ms | Acceptable latency |

### 6.2 Reliability

| Metric | Target |
|--------|--------|
| Uptime | 99.5% |
| Error rate | < 1% |
| Recovery time | < 5 minutes |
| Data loss | Zero (checkpoint persistence) |

### 6.3 Security

| Requirement | Implementation |
|-------------|----------------|
| API key protection | Environment variables only |
| Audit trail integrity | Append-only log, hash chain |
| User authentication | Streamlit authenticator |
| Data encryption | TLS for API calls |

### 6.4 Compliance

| Requirement | Standard |
|-------------|----------|
| AI disclosure | EU AI Act Article 50 |
| Content provenance | C2PA 2.2 specification |
| Data protection | GDPR for EU users |

### 6.5 Scalability

| Dimension | Target |
|-----------|--------|
| Articles per day | 100+ |
| Concurrent reviews | 10+ |
| Historical retention | 1 year |

---

## 7. Technical Considerations

### 7.1 Architecture Overview

```
INPUT SOURCES (Reddit, HN, GitHub)
         |
         v
+------------------+
| LANGGRAPH STATE  |
| MACHINE          |
+------------------+
         |
    +----+----+
    |         |
    v         v
+-------+  +--------+
|RESEARCH|  |GENERATE|
| PHASE  |  | PHASE  |
+-------+  +--------+
    |         |
    v         v
+------------------+
| FACT VERIFICATION|
| (Mandatory)      |
+------------------+
         |
         v
+------------------+
| ADVERSARIAL      |
| REVIEW PANEL     |
| (Claude+GPT+     |
|  Gemini)         |
+------------------+
         |
         v
+------------------+
| WSJ QUALITY GATE |
| Four Showstoppers|
+------------------+
         |
    +----+----+
    |         |
    v         v
Score>=80  Score<80
    |         |
    v         v
+-------+  +--------+
|PUBLISH|  | HITL   |
|       |  |DASHBOARD|
+-------+  +--------+
    |         |
    v         v
+------------------+
| C2PA PROVENANCE  |
| METADATA         |
+------------------+
```

### 7.2 LangGraph State Schema

```python
from typing import TypedDict, Literal, List, Optional
from datetime import datetime

class ArticleState(TypedDict):
    # Input
    article_id: str
    topic: str
    source_posts: List[dict]
    source_type: Literal["external", "internal"]  # Voice selection

    # Research
    research_data: dict
    verified_sources: List[dict]

    # Generation
    outline: str
    draft: str
    final_content: str

    # Fact Verification
    claims: List[dict]
    verification_results: List[dict]
    unverified_claim_count: int

    # Quality Review
    adversarial_reviews: List[dict]
    quality_scores: dict
    aggregate_score: float
    wsj_showstoppers: dict
    voice_violations: List[dict]

    # Human Review
    human_review_required: bool
    human_decision: Optional[str]
    editor_notes: str
    editor_revisions: str

    # Provenance
    workflow_id: str
    agent_chain: List[dict]
    c2pa_manifest: dict

    # Metadata
    created_at: datetime
    status: Literal["draft", "in_review", "approved", "rejected", "published"]
```

### 7.3 Multi-Model Adversarial Panel Configuration

```python
ADVERSARIAL_PANEL = {
    "ethics_reviewer": {
        "model": "claude-sonnet-4-20250514",
        "provider": "anthropic",
        "focus": ["bias", "harm", "fairness"],
        "weight": 0.30
    },
    "accuracy_reviewer": {
        "model": "gemini-2.0-flash",
        "provider": "google",
        "focus": ["factual_accuracy", "source_quality"],
        "weight": 0.35
    },
    "structure_reviewer": {
        "model": "gpt-4o",
        "provider": "openai",
        "focus": ["coherence", "flow", "completeness"],
        "weight": 0.20
    },
    "style_reviewer": {
        "model": "llama-3.3-70b-versatile",
        "provider": "groq",
        "focus": ["tone", "journalese", "readability"],
        "weight": 0.15
    }
}
```

### 7.4 Kill Phrase Detection Rules

```python
KILL_PHRASES = {
    "journalese": [
        "amid", "blaze", "probe", "bid", "axe", "boost", "slam", "blast"
    ],
    "vague_attribution": [
        "sources say", "experts say", "some argue", "many believe"
    ],
    "loaded_language": [
        "unprecedented", "shocking", "explosive", "bombshell",
        "groundbreaking", "revolutionary", "game-changing"
    ],
    "pr_terms": [
        "disruptor", "cutting-edge", "best-in-class", "innovative"
    ],
    "ownership_external": [  # From Voice Transformation PRD
        "we built", "our team", "my project", "I created", "we discovered"
    ]
}
```

### 7.5 Dependencies

```
# Core
langgraph>=0.2.0
langchain>=0.3.0
pydantic>=2.0

# LLM Providers
anthropic>=0.39.0
openai>=1.50.0
google-generativeai>=0.8.0
groq>=0.11.0

# Fact Verification
spacy>=3.7.0

# Content Provenance
c2pa-python>=0.5.0

# HITL Dashboard
streamlit>=1.40.0
streamlit-diff-viewer

# Testing
pytest>=8.0
pytest-asyncio
pytest-cov

# Monitoring
structlog
```

---

## 8. Implementation Roadmap

### Phase 1: Critical Infrastructure (Tasks 1-15)
**Focus**: Fix blocking issues, add mandatory fact verification

1. Create centralized config module (REQ-101)
2. Migrate all hardcoded paths to config (REQ-102)
3. Add environment variable support (REQ-103)
4. Validate paths on startup (REQ-104)
5. Create FactVerificationAgent (REQ-105)
6. Implement claim extraction (REQ-108)
7. Build verification API wrapper (REQ-109)
8. Add verification status to schema (REQ-110)
9. Create quality gate blocking rule (REQ-107)
10. Update pipeline to require verification (REQ-106)
11. Test cross-platform path handling
12. Document configuration options
13. Add config validation tests
14. Migration script for existing code
15. USER-TEST-1: Verify fact verification works

### Phase 2: Architecture Modernization (Tasks 16-32)
**Focus**: LangGraph migration, adversarial review panel

16. Design LangGraph state schema (REQ-202)
17. Create StateGraph builder (REQ-201)
18. Migrate research phase to LangGraph node
19. Migrate generation phase to LangGraph node
20. Migrate verification phase to LangGraph node
21. Add SQLite checkpointer (REQ-203)
22. Implement multi-model adversarial panel (REQ-205)
23. Create ethics reviewer (Claude) (REQ-205)
24. Create accuracy reviewer (Gemini) (REQ-205)
25. Create structure reviewer (GPT-4o) (REQ-205)
26. Implement weighted voting aggregation (REQ-206)
27. Add WSJ Four Showstoppers checklist (REQ-207)
28. Implement kill phrase detection (REQ-208)
29. Add source-aware voice selection (REQ-211)
30. Create voice validation post-processing (REQ-212)
31. Add escalation logic (REQ-210)
32. USER-TEST-2: Verify adversarial review works

### Phase 3: WSJ-Tier Quality (Tasks 33-48)
**Focus**: C2PA, HITL dashboard, Agent-Lightning

33. Implement C2PA manifest generation (REQ-301)
34. Add Schema.org JSON-LD output (REQ-302)
35. Create inline disclosure generator (REQ-304)
36. Build provenance tracking module (REQ-303)
37. Create Streamlit dashboard shell (REQ-305)
38. Implement review queue view (REQ-305)
39. Add side-by-side comparison view (REQ-306)
40. Create fact-check highlighting panel (REQ-307)
41. Implement approval workflow (REQ-305)
42. Add audit trail display (REQ-308)
43. Implement escalation UI (REQ-309)
44. Add LangGraph interrupt integration (REQ-204)
45. Research Agent-Lightning integration (REQ-310)
46. Create optimization feedback loop (REQ-310)
47. End-to-end integration testing
48. USER-TEST-3: Verify HITL dashboard works

### Phase 4: Production Hardening (Tasks 49-64)
**Focus**: Testing, monitoring, deployment

49. Create test infrastructure (REQ-401)
50. Add unit tests for config module (REQ-401)
51. Add unit tests for agents (REQ-402)
52. Add unit tests for quality gate (REQ-402)
53. Create integration test suite (REQ-403)
54. Add end-to-end pipeline test (REQ-403)
55. Implement cost tracking (REQ-404)
56. Add structured logging (REQ-405)
57. Create health check endpoints (REQ-408)
58. Implement rate limiting (REQ-407)
59. Add error recovery logic (REQ-407)
60. Create Docker configuration (REQ-409)
61. Write deployment documentation (REQ-409)
62. Performance optimization
63. Security audit
64. USER-TEST-4: Full system verification

---

## 9. Out of Scope

The following are **NOT** included in this PRD:

- **Video/Audio content generation** - Focus is on text articles only
- **Real-time streaming updates** - Batch processing model
- **Multi-language support** - English only for v3.0
- **Custom LLM fine-tuning** - Using off-the-shelf models
- **Mobile application** - Web dashboard only
- **Social media auto-posting** - Manual publication workflow
- **User registration/accounts** - Single-tenant for now
- **Revenue/billing system** - No monetization in v3.0
- **SynthID text watermarking** - C2PA metadata sufficient for compliance

---

## 10. Open Questions & Risks

### 10.1 Open Questions

| Question | Owner | Decision Deadline |
|----------|-------|-------------------|
| Should we require human approval for ALL articles or only high-risk? | Product | Phase 3 start |
| What is the SLA for human review turnaround? | Operations | Phase 3 start |
| Should we integrate with existing CMS or build standalone? | Engineering | Phase 3 start |
| What is the budget for multi-model API calls? | Finance | Phase 2 start |

### 10.2 Risks

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| API costs exceed budget | High | Medium | Cost tracking early, cascaded routing |
| Human reviewers become bottleneck | High | Medium | Batch approval, priority queue |
| LangGraph migration breaks existing pipeline | High | Low | Parallel implementation |
| EU AI Act requirements change | Medium | Low | Flexible disclosure layer |
| False positives in kill phrase detection | Medium | Medium | Override with reviewer note |

---

## 11. Validation Checkpoints

### USER-TEST-1: Fact Verification (After Task 15)
- [ ] Generate article with 5+ factual claims
- [ ] Verify each claim has verification status
- [ ] Confirm unverified claims trigger warning
- [ ] Test publication blocking with >1 unverified claim

### USER-TEST-2: Adversarial Review (After Task 32)
- [ ] Generate article and observe multi-model review
- [ ] Verify 3+ different models provide feedback
- [ ] Confirm weighted scoring produces aggregate
- [ ] Test escalation when reviewers disagree
- [ ] Test voice validation for external content

### USER-TEST-3: HITL Dashboard (After Task 48)
- [ ] Load review queue with test articles
- [ ] Verify side-by-side comparison works
- [ ] Test approval/reject workflow
- [ ] Confirm audit trail records all actions
- [ ] Test C2PA manifest generation

### USER-TEST-4: Production Readiness (After Task 64)
- [ ] Run full pipeline end-to-end
- [ ] Verify test coverage meets 80% target
- [ ] Confirm cross-platform operation
- [ ] Test error recovery scenarios
- [ ] Review deployment documentation

---

## 12. Appendix: Research Sources

This PRD was informed by comprehensive adversarial research from 11 specialized agents:

1. **WSJ Editorial Standards** - Four Showstoppers, accuracy, attribution
2. **BBC Journalism Standards** - Impartiality, accuracy, AI stance
3. **CNN Editorial Philosophy** - Facts First, human oversight
4. **CBC Radio Journalism** - Intimate voice, documentary approach
5. **FirstPost Tech Journalism** - Indian perspective, opinion-forward
6. **AI Content Agency Ethics** - EU AI Act, FTC, disclosure requirements
7. **Multi-Agent Architectures** - LangGraph, CrewAI, adversarial review
8. **Editorial Quality Frameworks** - Kill phrases, fact-checking, scoring
9. **C2PA Content Provenance** - EU AI Act Article 50, metadata
10. **HITL UI Patterns** - Streamlit, LangGraph interrupt, workflows
11. **Codebase Audit** - Technical debt, pipeline flow, configuration

---

## Approval

| Role | Name | Date | Signature |
|------|------|------|-----------|
| Product Owner | | | |
| Engineering Lead | | | |
| Compliance | | | |

---

*This PRD is optimized for TaskMaster parsing and will generate approximately 64 atomic tasks across 4 phases.*
