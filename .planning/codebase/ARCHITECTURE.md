# Architecture

**Analysis Date:** 2026-02-09

## Pattern Overview

**Overall:** Multi-Agent AI Content Generation Pipeline with Quality Gates and Provenance Tracking

**Key Characteristics:**
- **LangGraph-based state machine** with checkpoint persistence (6-phase workflow: RESEARCH → GENERATE → VERIFY → REVIEW → REVISE → APPROVE)
- **Multi-LLM provider routing** (Gemini/Groq/Perplexity) with fallback and provider-specific exception handling
- **Adversarial review loop** - content cannot publish without passing expert panel (score ≥ 7.0)
- **Post-generation fact verification** - all claims extracted and verified before approval
- **Tone-adaptive writing** - six built-in voice presets plus custom inference from writing samples
- **Content provenance tracking** - C2PA manifest, Schema.org JSON-LD, AI disclosure per SEO standards

## Layers

**Layer 1: Directive (What to do)**
- Purpose: Human-readable SOPs defining behavior, goals, and constraints
- Location: `N:\RedditNews\directives\`
- Contains: Markdown instruction files for voice, style, writing rules, tone presets
- Key files:
  - `writing_rules.md` - Expert Pragmatist voice and quality standards
  - `tone_system.md` - 6 preset voices and tone inference methodology
  - `style_enforcement.md` - 5-dimension voice fingerprinting (burstiness, lexical diversity, AI-tell detection, authenticity, framework compliance)
  - `framework_rules.md` - 5-Pillar Architected Writing Framework (Contrast Hook, Human Variable, Takeaway Density, Tradeoff Perspective, Visual Anchor)
  - `adversarial_review.md` - Panel expert definitions and scoring criteria
  - `pulse_monitoring.md` - Source configuration and trend aggregation
- Depends on: Nothing (static configuration)
- Used by: Layer 2 (orchestration) to parameterize agents

**Layer 2: Orchestration (Decision making)**
- Purpose: Intelligent routing, state management, error handling
- Location: `N:\RedditNews\execution\` (Python scripts for deterministic logic)
- Contains: Config management, pipeline orchestration, quality gates
- Key files:
  - `config.py` - Pydantic-based centralized config (paths, API keys, quality thresholds, model selection)
  - `article_state.py` - ArticleState schema (Pydantic) for LangGraph state with dict-style compatibility
  - `pipeline.py` - LangGraph StateGraph orchestrator (6 pipeline phases, node timeouts, checkpoint persistence)
  - `quality_gate.py` - Adversarial review loop orchestrator
  - `provenance.py` - C2PA/Schema.org generation for content transparency
- Depends on: Layer 1 (directives via file paths)
- Used by: Layer 3 (execution agents) for policy decisions; Layer 1 (UI) for state updates

**Layer 3: Execution (Doing the work)**
- Purpose: Deterministic, testable, fast implementations
- Location: `N:\RedditNews\execution\agents\` (agent classes) + `execution/sources/` (content ingestion)
- Contains: 16 specialized agents + 5 content source handlers
- Key files:
  - `agents/base_agent.py` - BaseAgent abstract class with multi-provider LLM routing (Groq/OpenAI/Gemini), typed exception handling, transient error detection, retry loops
  - `agents/writer.py` - WriterAgent: Senior ghostwriter with hook/CTA patterns, forbidden phrases, tone profile integration
  - `agents/adversarial_panel.py` - AdversarialPanelAgent: 4-expert review panel with structured JSON output
  - `agents/editor.py` - EditorAgent: Final polish and consistency checks
  - `agents/critic.py` - CriticAgent: Content quality evaluation
  - `agents/style_enforcer.py` - StyleEnforcerAgent: Quantitative voice scoring (5 dimensions, 0-100 composite)
  - `agents/fact_verification_agent.py` - FactVerificationAgent: Claim extraction + multi-source verification
  - `agents/gemini_researcher.py` - GeminiResearchAgent: Grounded research with web search
  - `agents/perplexity_researcher.py` - PerplexityResearchAgent: Sonar search API fallback
  - `agents/tone_inference.py` - ToneInferenceEngine: Custom tone profile inference from writing samples
  - `sources/base_source.py` - ContentSource abstract class (Reddit, Gmail, GitHub, RSS, HackerNews)
  - `sources/reddit_source.py`, `gmail_source.py`, `hackernews_source.py`, `rss_source.py` - Concrete source implementations
  - `sources/database.py` - SQLite persistence for content items
- Depends on: Layer 2 (config, state, quality thresholds)
- Used by: Layer 2 (pipeline orchestration calls agents)

## Data Flow

**Full Pipeline: RESEARCH → GENERATE → VERIFY → REVIEW → REVISE → APPROVE**

1. **RESEARCH Phase** (3-minute timeout)
   - Input: Topic + source content (from `article_state.topic`, `article_state.source_content`)
   - Process:
     - `research_node()` in `pipeline.py` calls GeminiResearchAgent or PerplexityResearchAgent
     - Agents perform web search, extract facts with confidence scores and source URLs
     - Results cached in `article_state.research_facts` (List[Dict])
   - Output: Verified facts ready for writing

2. **GENERATE Phase** (5-minute timeout)
   - Input: Topic + research facts
   - Process:
     - `generate_node()` calls WriterAgent.generate_content()
     - WriterAgent injects:
       - Tone profile instructions (if provided)
       - Research facts as context
       - Forbidden phrases + hook/CTA patterns
     - LLM generates draft in target voice
   - Output: Draft stored in `article_state.draft` (str)

3. **VERIFY Phase** (Fact verification)
   - Input: Draft content
   - Process:
     - QualityGate.verify_facts() calls FactVerificationAgent.verify_article()
     - Agent extracts claims via LLM
     - Verifies each claim against research facts + web search (Gemini/Perplexity)
     - Returns structured verification_report with:
       - `claims`: List of extracted claims
       - `results`: Verification details (verified, unverified, false)
       - `passes_quality_gate`: Boolean (0 false claims + ≤1 unverified + ≥3 verified)
   - Output: Verification status in `article_state.verification_results`
   - Gate: If fails → requires_human_review=True, escalation triggered

4. **REVIEW Phase** (Multi-model adversarial panel)
   - Input: Draft content
   - Process:
     - `review_node()` calls AdversarialPanelAgent.review()
     - Panel of 4 experts scores content:
       - Conversion Strategist (agency focus, CTA strength)
       - Brand Strategist (brand voice alignment)
       - SEO Specialist (keyword placement, structure)
       - Creative Director (uniqueness, engagement)
     - Each expert scores 1-10 with specific failures + fixes
     - Critical failures (mentioned by 2+ experts) aggregated
   - Output: PanelVerdict with:
     - `average_score`: Composite expert score
     - `expert_scores`: Dict of individual scores
     - `critical_failures`: Array of common issues
     - `priority_fixes`: Top 5 fixes to implement
   - Gate: If average_score < 7.0 → revision needed

5. **REVISE Phase** (Fix loop)
   - Input: Draft + PanelVerdict with fix instructions
   - Process:
     - WriterAgent.revise() takes fix instructions and regenerates content
     - EditorAgent.edit() applies final polish
     - Loop back to REVIEW until:
       - Score ≥ 7.0 (PASS), OR
       - iteration_count ≥ MAX_ITERATIONS (3) → ESCALATE
   - Output: Revised draft or escalation

6. **APPROVE Phase** (Human-in-the-loop)
   - Input: Approved draft from quality gate
   - Process:
     - `approve_node()` sets requires_human_review=True
     - UI displays draft + provenance metadata
     - Human reviewer accepts/rejects
   - Output: Final content or request for changes

## State Management

**ArticleState (Pydantic BaseModel)**
- Location: `execution/article_state.py`
- Flow: Initialized → passed through each phase → updated with phase outputs
- Key fields:
  - Core: `topic`, `source_content`, `draft`, `final_content`
  - Research: `research_facts`, `research_sources`
  - Verification: `claims`, `verification_results`, `unverified_claim_count`, `false_claim_count`, `verification_passed`
  - Quality: `quality_score`, `quality_passed`, `review_iterations`
  - Panel: `panel_scores`, `panel_verdict`, `critical_failures`, `priority_fixes`
  - Voice: `voice_type`, `voice_violations`, `voice_validated`
  - Tone: `tone_profile_name`
  - Provenance: `c2pa_manifest`, `ai_disclosure`
  - Workflow: `current_phase`, `next_action`, `requires_human_review`
- Checkpoint: LangGraph SqliteSaver persists state after each node

**State Transitions:**
```
init → research → generate → verify → review → revise (loop) → approve → done
                                        ↓ (fail)
                                    escalate
```

## Key Abstractions

**BaseAgent**
- Purpose: Shared LLM interaction layer with multi-provider routing
- Location: `execution/agents/base_agent.py`
- Pattern: Abstract base class with concrete implementations for Writer, Editor, Critic, etc.
- Responsibilities:
  - Provider routing: Tries provider in order (Groq → OpenAI → Gemini)
  - Typed exception handling: Catches provider-specific errors (groq.RateLimitError, google.api_core.exceptions.ResourceExhausted)
  - Transient detection: Identifies retryable errors vs. fatal
  - Retry logic: tenacity-based exponential backoff (3 attempts)
  - Token accounting: Tracks usage for cost analysis
- Used by: WriterAgent, EditorAgent, CriticAgent, AdversarialPanelAgent, FactVerificationAgent

**ContentSource & ContentItem**
- Purpose: Polymorphic abstraction for ingesting content from multiple sources
- Location: `execution/sources/base_source.py`
- Pattern: Strategy + Factory
- ContentItem: Normalized output regardless of source
  - Fields: `source_type`, `source_id`, `title`, `content`, `url`, `author`, `timestamp`, `trust_tier`, `metadata`
  - Properties: `unique_key` (for deduplication), `should_evaluate`, `is_auto_signal`, `is_blocked`
- TrustTier: A (Curated/auto-signal), B (Semi-trusted), C (Untrusted/full eval), X (Blocked)
- Implementations:
  - `RedditSource` - Fetches trending posts from subreddits
  - `GmailSource` - Ingests newsletter emails
  - `GitHubSource` - Monitors trending repositories
  - `RSSSource` - Configurable Atom/RSS feeds
  - `HackerNewsSource` - HN API integration

**ToneProfile**
- Purpose: Adaptively parameterize voice and style
- Location: `execution/tone_profiles.py`
- Pattern: Data class + preset manager
- Contains:
  - `name`: Preset name (Expert Pragmatist, Thought Leader, Technical Deep Dive, etc.)
  - `instructions`: LLM instructions prepended to prompts
  - `forbidden_phrases`: Voice-specific kill phrases
  - `war_story_keywords`: Authentic moment triggers
  - `hook_style`, `cta_style`: Specific pattern preferences
  - `style_baselines`: Quantitative targets for StyleEnforcerAgent
- Flow: User selects preset OR provides writing sample → ToneInferenceEngine infers custom profile → passed to WriterAgent.generate()

**VerificationState**
- Purpose: Structured verification progress tracking
- Location: `execution/article_state.py`
- Contains: claims_extracted, claims_verified, claims_unverified, claims_false, detailed_results
- Property: `passes_gate` - boolean check against config thresholds (MIN_VERIFIED_FACTS=3, MAX_UNVERIFIED_CLAIMS=1)

**PanelVerdict**
- Purpose: Aggregated adversarial review output
- Location: `execution/agents/adversarial_panel.py`
- Contains: expert_critiques (List[ExpertCritique]), critical_failures, priority_fixes, average_score, passed
- Converts to JSON for quality gate loop instructions

**ContentProvenance**
- Purpose: C2PA/Schema.org compliance for content transparency
- Location: `execution/provenance.py`
- Tracks: creation metadata, source info, quality metrics, models used, action history, human involvement
- Generates: C2PA manifest, Schema.org JSON-LD, inline AI disclosure text

## Entry Points

**UI Entry Point: Streamlit App**
- Location: `N:\RedditNews\app.py`
- Triggers: `streamlit run app.py`
- Responsibilities:
  - Session management (selected tone, topic, source)
  - Phase visualization (renders phase indicators, progress bars)
  - LLM provider selection (Groq/OpenAI/Gemini selector)
  - Draft preview (Markdown rendering with syntax highlighting)
  - Provenance display (C2PA manifest, fact verification table)
  - Human approval interface (accept/reject with notes)
  - Configuration panel (model selection, quality thresholds)
- Uses: config singleton, tone_profiles, pipeline runner

**Script Entry Points:**

1. `pulse_aggregator.py` - Trend discovery
   - Fetches content_items from SQLite
   - Clusters by keyword frequency (TF-IDF if sklearn available)
   - Scores topics by cross-source presence
   - Outputs daily pulse summary

2. `fetch_reddit.py` - Reddit content ingestion
   - Calls RedditSource to fetch trending posts
   - Deduplicates via ContentItem.unique_key
   - Stores in SQLite via sources/database.py

3. `quality_gate.py` - Standalone quality review
   - Command-line interface: `python quality_gate.py --input draft.md`
   - Runs adversarial panel + revision loop

4. `validate_voice.py` - Voice validation
   - Standalone checker for forbidden phrases
   - Reports violations

5. `generate_drafts_v2.py` - Batch generation
   - Takes topic list, generates drafts in parallel

## Error Handling

**Strategy:** Typed exceptions + transient detection + provider fallback

**Patterns:**

1. **Provider-Specific Typing**
   - BaseAgent._is_transient() checks exception type:
     - `groq.RateLimitError` → transient (retry)
     - `google.api_core.exceptions.ResourceExhausted` → transient (retry)
     - `openai.APIError` → transient (retry)
     - `TypeError`, `ValueError` → fatal (fail immediately)
   - Preserves per-provider granularity (not normalized like LiteLLM)

2. **Cascading Provider Fallback**
   - BaseAgent._call_llm() tries in order:
     1. Groq (free tier, generous limits)
     2. OpenAI (fallback)
     3. Gemini (fallback)
   - Stops on first success; raises AllProvidersFailedError if all exhausted

3. **Retry with Exponential Backoff**
   - tenacity.retry decorator: `@retry(stop=stop_after_attempt(3), wait=wait_exponential())`
   - Applied to all LLM calls in BaseAgent

4. **Pipeline-Level Timeouts**
   - Per-node timeouts via with_timeout() decorator (threading-based)
   - RESEARCH: 3 min, GENERATE: 5 min, etc.
   - Raises NodeTimeoutError if exceeded

5. **Quality Gate Escalation**
   - Fact verification fails → requires_human_review=True
   - Panel score < escalation_threshold (6.0) → flags for escalation
   - Max iterations exceeded → escalates with reason code
   - False claims detected → auto-escalate (config.quality.FALSE_CLAIM_AUTO_ESCALATE)

6. **Provenance Tracking**
   - All errors logged to article_state.error_messages
   - Provenance.actions records failed agents + timestamps
   - Human reviewer sees full error history

## Cross-Cutting Concerns

**Logging:**
- Path: `execution/utils/logging.py`
- Approach: Structured logging to `logs/` directory with timestamps
- Each agent logs to phase-specific handler

**Validation:**
- Input: Pydantic models (ArticleState, QualityGateInput) for type safety
- Output: StyleEnforcerAgent scores against 5 dimensions; PanelVerdict aggregates expert verdicts
- Config: GhostWriterConfig.validate() checks API keys, paths, fact verification availability

**Authentication:**
- API keys: Loaded from .env file via pydantic_settings
- Paths: Centralized in config.PathConfig (cross-platform via pathlib)
- Secrets: .env in .gitignore; no secrets in code

**Quality Gates:**
- FACT_VERIFICATION_REQUIRED: If true, all content must pass claim verification
- PASS_THRESHOLD: 7.0 (panel score must meet this to auto-approve)
- ESCALATION_THRESHOLD: 6.0 (below this triggers human review)
- KILL_PHRASE_MAX_SCORE: 4.0 (kill phrases alone can fail content)
- MAX_ITERATIONS: 3 (revision loop limit before escalation)

**Voice & Tone Adaptation:**
- WriterAgent merges tone_profile.forbidden_phrases + tone_profile.instructions into prompts
- StyleEnforcerAgent adjusts scoring baselines per tone_profile.style_baselines
- Tone inference happens once (via ToneInferenceEngine) when user provides sample
- Profiles stored in tone_presets.json (built-in) or inferred dynamically

**Provenance & Transparency:**
- C2PA manifest: action_type (created/modified/verified/reviewed), agent, timestamp, model
- Schema.org JSON-LD: Article schema for search engines
- Inline AI disclosure: "This article was written by GhostWriter AI, reviewed by [models], fact-checked by [method]..."
- All tracked in article_state.c2pa_manifest and article_state.ai_disclosure

---

*Architecture analysis: 2026-02-09*
