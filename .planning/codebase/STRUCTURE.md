# Codebase Structure

**Analysis Date:** 2026-02-09

## Directory Layout

```
N:\RedditNews\
├── .claude/                    # Claude Code skills
│   └── skills/                 # Custom skill implementations
├── .planning/                  # GSD planning outputs
│   └── codebase/              # Architecture analysis (this file)
├── .tmp/                       # Intermediate files (regenerated, not committed)
│   ├── drafts/                # Generated draft articles
│   └── audit/                 # Audit intermediate outputs
├── .taskmaster/               # Task orchestration (GSD helper)
│   ├── docs/                  # Task documentation
│   └── tasks/                 # Task definitions
├── directives/                # Markdown SOPs (Layer 1 instruction set)
│   ├── sources/               # Source-specific configuration
│   ├── writing_rules.md       # Expert Pragmatist voice standards
│   ├── tone_system.md         # 6 preset voices + inference methodology
│   ├── style_enforcement.md   # 5-dimension voice fingerprinting spec
│   ├── framework_rules.md     # 5-Pillar Architected Writing Framework
│   ├── adversarial_review.md  # Expert panel definitions
│   ├── pulse_monitoring.md    # Trend aggregation configuration
│   ├── technical_rules.md     # Research methodology + accuracy standards
│   ├── voice_rules.md         # Voice attribution and authenticity
│   └── produce_content.md     # Content pipeline SOP
├── execution/                 # Deterministic Python tools (Layer 2 + Layer 3)
│   ├── agents/                # 16 specialized LLM agents
│   │   ├── base_agent.py      # Abstract base with multi-provider routing
│   │   ├── writer.py          # Senior ghostwriter (hooks, CTAs, tone)
│   │   ├── editor.py          # Polish and consistency
│   │   ├── critic.py          # Quality evaluation
│   │   ├── adversarial_panel.py  # 4-expert review loop
│   │   ├── style_enforcer.py  # 5-dimension voice scoring
│   │   ├── fact_verification_agent.py  # Claim extraction + verification
│   │   ├── fact_researcher.py # Pre-generation research support
│   │   ├── gemini_researcher.py       # Web search via Gemini + grounding
│   │   ├── perplexity_researcher.py   # Sonar search fallback
│   │   ├── topic_researcher.py        # Topic analysis
│   │   ├── technical_supervisor.py    # Technical accuracy oversight
│   │   ├── specialist.py      # Domain-specific expertise
│   │   ├── original_thought_agent.py  # Unique insight generation
│   │   ├── copywriter_agent.py        # Copywriting specialization
│   │   ├── commit_analyzer.py # GitHub commit analysis
│   │   └── visuals.py         # Image prompt generation (ByteByteGo style)
│   ├── sources/               # Content ingestion (5 source types)
│   │   ├── base_source.py     # Abstract ContentSource interface
│   │   ├── reddit_source.py   # Reddit trending posts
│   │   ├── gmail_source.py    # Newsletter email ingestion
│   │   ├── github_source.py   # GitHub trending repos
│   │   ├── hackernews_source.py  # HackerNews API
│   │   ├── rss_source.py      # Configurable RSS/Atom feeds
│   │   ├── database.py        # SQLite persistence (content_items table)
│   │   ├── circuit_breaker.py # Rate limit + outage handling
│   │   └── __init__.py        # Source factory
│   ├── utils/                 # Shared utilities
│   │   ├── logging.py         # Structured logging setup
│   │   ├── json_parser.py     # LLM JSON extraction with fallbacks
│   │   ├── file_ops.py        # File operations (paths, write, read)
│   │   ├── datetime_utils.py  # UTC timestamps (utc_now, utc_iso)
│   │   ├── health.py          # Health check endpoints
│   │   ├── research_templates.py  # Research prompt templates
│   │   └── __init__.py
│   ├── dashboard/             # Web UI components
│   │   └── app.py            # Streamlit dashboard implementation
│   ├── prompts/              # Prompt templates
│   │   └── voice_templates.py # Voice-specific prompt scaffolding
│   ├── config.py             # Centralized config (paths, API keys, thresholds)
│   ├── article_state.py      # ArticleState Pydantic model + helpers
│   ├── pipeline.py           # LangGraph state machine orchestrator
│   ├── quality_gate.py       # Adversarial review loop CLI
│   ├── provenance.py         # C2PA/Schema.org generation
│   ├── tone_profiles.py      # ToneProfile data model + 6 presets
│   ├── tone_inference.py     # Custom tone inference from samples
│   ├── user_preferences.py   # User tone preferences + adaptive learning
│   ├── voice_utils.py        # Voice validation and rule enforcement
│   ├── pulse_aggregator.py   # Daily trend clustering + scoring
│   ├── evaluate_posts.py     # Signal/noise evaluation
│   ├── evaluate_content.py   # Content evaluation framework
│   ├── validate_voice.py     # Standalone voice validation
│   ├── puter_bridge.py       # External service integration (unused currently)
│   ├── optimization.py       # Performance tuning and profiling
│   ├── init_db.py            # SQLite initialization
│   ├── fetch_reddit.py       # Reddit content fetching script
│   ├── fetch_github.py       # GitHub content fetching script
│   ├── fetch_all.py          # Multi-source fetching orchestration
│   ├── generate_drafts.py    # Batch draft generation (v1)
│   ├── generate_drafts_v2.py # Batch draft generation (v2)
│   ├── generate_medium_full.py  # Medium-specific full article generation
│   ├── exceptions.py         # Custom exception types
│   ├── test_auxiliary_agents.py  # Agent testing harness
│   └── __pycache__/          # Python bytecode (not committed)
├── content_archive/          # Historical content for voice calibration
│   └── articles/             # Archive of published articles
├── drafts/                    # Output directory for generated articles
│   └── images/               # Generated infographic drafts
├── scripts/                   # Utility scripts
├── templates/                 # HTML/Jinja templates
├── tests/                     # Unit and integration tests
│   ├── test_config.py        # Config validation tests
│   ├── test_provenance.py    # Provenance generation tests
│   ├── test_tone_inference.py    # Tone inference tests
│   ├── test_tone_integration.py  # Tone system integration tests
│   ├── test_tone_profiles.py # Preset loading tests
│   ├── test_user_preferences.py  # User preference storage tests
│   ├── test_voice_utils.py   # Voice validation tests
│   ├── __pycache__/          # Python bytecode (not committed)
│   └── __init__.py
├── logs/                      # Runtime logs (created on startup)
├── assets/                    # Static assets (images, icons)
├── docs/                      # Documentation
│   └── hardening/            # Security hardening guides
├── app.py                     # Main Streamlit UI entry point
├── CLAUDE.md                  # Agent instructions (architecture SOPs)
├── .env                       # Environment variables (secrets, API keys)
├── .gitignore                 # Git ignore rules
├── requirements.txt           # Python dependencies
├── reddit_content.db          # SQLite database (content_items, pulse_daily tables)
├── pyproject.toml             # Python project metadata
└── README.md                  # Project overview
```

## Directory Purposes

**`directives/`** - Layer 1: Human-Readable Instruction SOPs
- Purpose: Define goals, constraints, voice, evaluation criteria as Markdown
- Contains: 13 SOP files covering writing rules, tone system, style enforcement, research methodology
- Key files:
  - `writing_rules.md`: Expert Pragmatist voice baseline (specificity, memorable moments, real data)
  - `tone_system.md`: 6 preset voices + ToneProfile structure + inference approach
  - `style_enforcement.md`: 5-dimension scoring spec (burstiness, lexical diversity, AI-tell detection, authenticity, framework compliance)
  - `adversarial_review.md`: Expert panel definitions (agency, brand, SEO, creative specialists)
- Not directly used by code; referenced by agents via persona prompts hardcoded in agent classes

**`execution/`** - Layer 2 & 3: Deterministic Python Implementation
- Purpose: Reliable, testable, fast tools for LLM calls, data processing, API integrations
- Contains: 50+ Python scripts and modules organized by function
- Lifecycle: Most scripts are re-runnable utilities (fetch_reddit.py, generate_drafts_v2.py), not one-time setup
- Key subsystems:
  - `agents/`: 16 specialized LLM wrappers inheriting from BaseAgent
  - `sources/`: 5 content source implementations (Reddit, Gmail, GitHub, RSS, HackerNews)
  - `utils/`: Logging, JSON parsing, file ops, datetime handling
  - Pipeline: config.py + article_state.py + pipeline.py orchestrate state machine
  - Quality: quality_gate.py + adversarial_panel.py implement review loop
  - Provenance: C2PA/Schema.org generation for transparency

**`.tmp/`** - Intermediate Files (Regenerated)
- Purpose: Scratch space for drafts, audit outputs, temporary data
- Contents: Do NOT commit; always regenerated by scripts
- Contains:
  - `.tmp/drafts/`: Draft articles during pipeline execution
  - `.tmp/audit/`: Audit intermediate files (dossiers, fact-check results)

**`content_archive/articles/`** - Historical Content
- Purpose: Voice calibration reference
- Contains: Published articles for ToneInferenceEngine to learn from
- Used by: tone_inference.py when inferring custom profiles

**`drafts/`** - Output Directory
- Purpose: Final article drafts awaiting human approval
- Contains: Markdown files, images
- Directory: Configurable via GHOSTWRITER_OUTPUT_DIR env var (defaults to `drafts/`)

**`tests/`** - Unit and Integration Tests
- Purpose: Verify config, tone system, provenance generation, voice rules
- 7 test modules, ~300 lines total
- Pattern: pytest-based, uses fixtures for tone profiles and state
- Coverage: Mostly system-level integration tests (tone inference, provenance generation, voice validation)

**`logs/`** - Runtime Logs
- Purpose: Audit trail for pipeline execution
- Created: On startup via config.py (path.LOGS_DIR.mkdir())
- Format: Structured logs with phase, agent, timestamp, error messages

## Key File Locations

**Entry Points:**

1. `N:\RedditNews\app.py` - **Streamlit UI**
   - Purpose: Interactive draft generation, tone selection, approval interface
   - Triggers: `streamlit run app.py`
   - Uses: config, tone_profiles, pipeline runner, quality_gate
   - Outputs: Renders phase progression, provenance metadata, human approval buttons

2. `N:\RedditNews\execution\pipeline.py` - **LangGraph Pipeline Orchestrator**
   - Purpose: 6-phase state machine (RESEARCH → GENERATE → VERIFY → REVIEW → REVISE → APPROVE)
   - Entry function: `create_pipeline()` → `run_pipeline(pipeline, state)`
   - Uses: All agent classes, quality gates, provenance tracking
   - Outputs: ArticleState with final_content and approval status

3. `N:\RedditNews\execution\quality_gate.py` - **Standalone Quality Review CLI**
   - Purpose: Run adversarial panel + revision loop on any draft
   - Entry function: `QualityGate(max_iterations=3).run(content, topic)`
   - Command-line: `python quality_gate.py --input draft.md`
   - Outputs: QualityGateResult with final score and escalation reason

**Configuration:**

1. `N:\RedditNews\execution\config.py` - **Centralized Config Singleton**
   - Purpose: Single source of truth for paths, API keys, model selection, quality thresholds
   - Pattern: Pydantic BaseSettings with env_file=".env"
   - Sections:
     - PathConfig: PROJECT_ROOT, OUTPUT_DIR, TEMP_DIR, LOGS_DIR, DIRECTIVES_DIR
     - APIConfig: GROQ_API_KEY, GOOGLE_API_KEY, PERPLEXITY_API_KEY, ANTHROPIC_API_KEY
     - QualityConfig: PASS_THRESHOLD (7.0), ESCALATION_THRESHOLD (6.0), MAX_ITERATIONS (3)
     - ModelConfig: DEFAULT_WRITER_MODEL, RESEARCH_MODEL_PRIMARY, etc.
     - VoiceConfig: VOICE_EXTERNAL, VOICE_INTERNAL, publication STYLE_* constants
   - Usage: `from execution.config import config; path = config.paths.OUTPUT_DIR`

2. `N:\RedditNews\.env` - **Environment Variables**
   - Purpose: API keys, secrets, environment flags
   - Contents: GROQ_API_KEY, ANTHROPIC_API_KEY, GOOGLE_API_KEY, PERPLEXITY_API_KEY, GHOSTWRITER_ENV
   - Security: .gitignore'd, never committed

**Core Logic:**

1. `N:\RedditNews\execution\article_state.py` - **State Schema**
   - Purpose: Pydantic model for pipeline state (ArticleState + VerificationState + QualityGateInput)
   - Pattern: Dict-style compatibility (supports `state["topic"]` access for backward compatibility)
   - Fields: 40+ fields covering content, metadata, verification, quality, panel scores, voice, provenance
   - Functions: `create_initial_state()`, `update_verification_state()`

2. `N:\RedditNews\execution\agents/base_agent.py` - **LLM Abstraction**
   - Purpose: Multi-provider routing (Groq → OpenAI → Gemini) with typed exception handling
   - Abstract methods: `_build_prompt()` (per-agent customization)
   - Concrete methods: `_call_llm()`, `generate_content()`, retry loops
   - Exception hierarchy: LLMError → ProviderError → AllProvidersFailedError
   - Transient detection: Provider-specific exception types (groq.RateLimitError, google.api_core.exceptions.*)

3. `N:\RedditNews\execution\agents/writer.py` - **Senior Ghostwriter**
   - Purpose: Generate content with hook patterns, forbidden phrases, tone adaptation
   - Inherits from: BaseAgent
   - Methods: `generate_content(topic, research_facts, tone_profile=None)` → draft
   - Forbidden phrases: Kill words (game-changer, paradigm shift, etc.)
   - Hook patterns: 5 proven openers (burned $X, everyone talking, specific number engineers, etc.)
   - CTA patterns: 4 call-to-action formulas
   - Tone integration: Merges tone_profile.forbidden_phrases + .instructions into prompts

4. `N:\RedditNews\execution\agents/adversarial_panel.py` - **Expert Review Panel**
   - Purpose: 4-expert critique with structured JSON output
   - Inherits from: BaseAgent
   - Experts: Conversion Strategist, Brand Strategist, SEO Specialist, Creative Director
   - Output: PanelVerdict with average_score, expert_scores, critical_failures, priority_fixes
   - Scoring: 1-10 per expert; average ≥7.0 passes

5. `N:\RedditNews\execution\agents/style_enforcer.py` - **Voice Fingerprinting**
   - Purpose: Quantitative scoring across 5 dimensions (0-100 composite)
   - Dimensions:
     - Burstiness (20% weight): Sentence length variation
     - Lexical diversity (15%): Unique word ratio (VOCD, TTR)
     - AI-tell detection (25%): Forbidden phrases + patterns
     - Authenticity (25%): War story keywords, specific metrics
     - Framework compliance (15%): Contrast hook, tradeoff presence
   - Thresholds: ≥80 passes, 60-79 needs revision, <60 rejected
   - Not inherited from BaseAgent (no LLM calls)

6. `N:\RedditNews\execution\agents/fact_verification_agent.py` - **Claim Verification Gatekeeper**
   - Purpose: Extract claims + verify against research + web search
   - Inherits from: BaseAgent (for Gemini/Perplexity fallback)
   - Methods: `verify_article(content, topic, research_facts)` → verification_report
   - Output: List of claims with status (verified, unverified, false) + pass/fail on quality gate
   - Quality gate: 0 false claims + ≤1 unverified + ≥3 verified

7. `N:\RedditNews\execution\sources/base_source.py` - **Content Source Interface**
   - Purpose: Polymorphic abstraction for 5 content sources
   - Abstract methods: `fetch()`, `parse_items()` (per-source implementation)
   - ContentItem: Normalized output (source_type, source_id, title, content, url, author, trust_tier)
   - TrustTier: A (auto-signal), B (semi-trusted), C (full eval), X (blocked)
   - Factory: sources/__init__.py creates appropriate source based on config

**Quality Gate:**

1. `N:\RedditNews\execution\quality_gate.py` - **Review Loop Orchestrator**
   - Purpose: Coordinate adversarial panel + writer revisions
   - Class: QualityGate
   - Methods:
     - `__init__(max_iterations=3, require_verification=True, tone_profile=None)`
     - `run(content, topic, platform="medium")` → QualityGateResult
     - `verify_facts(content, topic)` → dict with verification_summary
   - Loop: REVIEW → (if score<7.0) FIX → REVIEW (until pass or max_iterations)
   - Escalation: Auto-escalate if max_iterations hit, false claims detected, or kill phrase present

**Tone System:**

1. `N:\RedditNews\execution\tone_profiles.py` - **Tone Preset Manager**
   - Purpose: Define 6 built-in voice presets + ToneProfile data model
   - ToneProfile fields:
     - `name`, `instructions` (prepended to prompts)
     - `forbidden_phrases`, `war_story_keywords`
     - `hook_style`, `cta_style`
     - `style_baselines` (quantitative targets for StyleEnforcerAgent)
   - Presets: Expert Pragmatist, Thought Leader, Technical Deep Dive, Conversational Engineer, News Reporter, Contrarian Challenger
   - Functions: `list_presets()`, `get_preset(name)`, `create_custom_profile()`

2. `N:\RedditNews\execution\tone_inference.py` - **Custom Profile Inference**
   - Purpose: Infer custom tone from writing samples
   - Method: ToneInferenceEngine.infer(writing_samples: List[str]) → ToneProfile
   - Uses: LLM to analyze samples and extract tone characteristics

3. `N:\RedditNews\execution\user_preferences.py` - **Adaptive Learning**
   - Purpose: Track user feedback + adjust tone profiles over time
   - Methods: `save_feedback()`, `get_adaptive_profile()` (applies user edits to baseline)

**Provenance & Transparency:**

1. `N:\RedditNews\execution\provenance.py` - **C2PA/Schema.org Generation**
   - Purpose: Content provenance tracking per industry standards
   - Classes:
     - ProvenanceAction: Single action in chain (created/modified/verified/reviewed)
     - ContentProvenance: Full provenance record with metadata, models used, action history
   - Functions:
     - `generate_c2pa_manifest()` → JSON manifest
     - `generate_schema_org_jsonld()` → Article schema for search engines
     - `generate_inline_disclosure()` → Human-readable AI disclosure text
   - Tracked in: article_state.c2pa_manifest, article_state.ai_disclosure

**Testing:**

1. `N:\RedditNews\tests/test_config.py` - Config validation
2. `N:\RedditNews\tests/test_provenance.py` - Provenance generation
3. `N:\RedditNews\tests/test_tone_*.py` - Tone system integration
4. `N:\RedditNews\tests/test_voice_utils.py` - Voice validation

## Naming Conventions

**Files:**

- Snake_case for Python modules: `writer.py`, `article_state.py`, `fact_verification_agent.py`
- Descriptive agent names: `{role}_agent.py` (e.g., `technical_supervisor.py`, `fact_researcher.py`)
- Test files: `test_{feature}.py` (e.g., `test_tone_inference.py`)
- Directive files: descriptive kebab-case: `writing_rules.md`, `framework_rules.md`
- Config files: single word or underscore: `config.py`, `pyproject.toml`, `.env`

**Directories:**

- Plural for collections: `agents/`, `sources/`, `utils/`, `tests/`, `directives/`, `logs/`
- Descriptive and functional: `execution/`, `content_archive/`, `.tmp/`, `drafts/`

**Classes:**

- PascalCase: `WriterAgent`, `BaseAgent`, `AdversarialPanelAgent`, `StyleEnforcerAgent`
- Suffixes: `*Agent` for LLM agents, `*Source` for content sources
- Data classes: `ArticleState`, `ContentItem`, `PanelVerdict`, `VerificationState`

**Functions:**

- Snake_case: `create_initial_state()`, `update_verification_state()`, `create_pipeline()`
- Private with leading underscore: `_is_transient()`, `_call_llm()`, `_build_prompt()`

**Variables:**

- Snake_case: `topic`, `research_facts`, `verification_passed`, `article_id`
- Constants: UPPER_SNAKE_CASE: `PASS_THRESHOLD = 7.0`, `MAX_ITERATIONS = 3`
- State fields (dict-like): lowercase: `state["draft"]`, `state["panel_scores"]`

## Where to Add New Code

**New LLM Agent:**
1. Create `N:\RedditNews\execution\agents\{role}_agent.py`
2. Inherit from `BaseAgent`
3. Implement `_build_prompt(self, **kwargs) → str`
4. Call `self._call_llm(prompt)` for inference
5. Parse output (JSON or text)
6. Test with `tests/test_{role}_agent.py` (if critical)
7. Wire into pipeline: Add node to `execution/pipeline.py` or call from quality_gate.py

**New Content Source:**
1. Create `N:\RedditNews\execution\sources/{source_name}_source.py`
2. Inherit from `ContentSource` (abstract base in `execution/sources/base_source.py`)
3. Implement `fetch()` and `parse_items()` methods
4. Return List[ContentItem] with normalized fields
5. Register in `execution/sources/__init__.py` factory
6. Add config entry to `directives/pulse_monitoring.md`
7. Update database schema in `execution/sources/database.py` if needed

**New Utility Module:**
1. Create `N:\RedditNews\execution/utils/{feature}.py`
2. Keep functions pure (no side effects where possible)
3. Import logging via `from execution.utils.logging import get_logger`
4. Test with `pytest N:\RedditNews\tests/`

**New Directive (Instruction SOP):**
1. Create `N:\RedditNews\directives/{topic}.md`
2. Follow format: Goals, Inputs, Tools/Scripts, Outputs, Edge Cases, Examples
3. Reference in CLAUDE.md agent instructions if it affects behavior
4. Link from relevant agent class docstrings

**New Test:**
1. Create `N:\RedditNews\tests/test_{feature}.py`
2. Use pytest + fixtures for tone profiles, state objects
3. Place in tests/ (never commit .tmp/ artifacts)
4. Run: `pytest N:\RedditNews\tests/`

**Batch Processing Script:**
1. Create `N:\RedditNews\execution/{task}.py`
2. Follow pattern: argparse for CLI args, main() function, script entry point
3. Import config singleton: `from execution.config import config`
4. Log to structured logger: `from execution.utils.logging import get_logger`
5. Return JSON or write to `.tmp/` for intermediate outputs
6. Deliverables (Google Sheets, final articles) go to cloud/drafts/, not .tmp/

## Special Directories

**`.tmp/`** - Regenerated Intermediates
- Purpose: Scratch space for drafts, audit outputs, temp data
- Generated: By fetch_*.py, generate_*.py, quality_gate.py during execution
- Committed: No; always in .gitignore
- Cleanup: Safe to delete; will be regenerated on next run

**`content_archive/articles/`** - Voice Calibration Reference
- Purpose: Published articles used by ToneInferenceEngine to learn custom profiles
- Committed: Yes; essential for tone learning
- Format: Markdown files, one per article
- Indexed by: tone_inference.py to infer custom profiles from user samples

**`directives/`** - Living Instruction Documents
- Purpose: Human-readable SOPs defining behavior, constraints, voice standards
- Committed: Yes; core to system
- Format: Markdown with clear structure (Goals, Inputs, Tools, Outputs, Edge Cases)
- Versioning: Update when system behavior changes or new learnings discovered
- Reference: Agent classes hardcode directives as persona prompts; file paths are NOT dynamically loaded

**`logs/`** - Runtime Audit Trail
- Purpose: Structured logs for debugging, auditing, compliance
- Committed: No; generated at runtime
- Format: Plain text logs per phase (research.log, generate.log, quality_gate.log)
- Retention: Kept for audit; cleared periodically

**`drafts/`** - Final Output
- Purpose: Generated articles awaiting human approval
- Committed: Configurable (via GHOSTWRITER_OUTPUT_DIR); defaults to `drafts/`
- Format: Markdown files + images/
- Delivery: User downloads from drafts/ or via Google Slides export

---

*Structure analysis: 2026-02-09*
