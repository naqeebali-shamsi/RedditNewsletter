# Technology Stack

**Analysis Date:** 2026-02-09

## Languages

**Primary:**
- Python 3.x - Core application language for all agents, pipelines, and utilities

**Secondary:**
- Markdown - Directives and documentation
- YAML/JSON - Configuration files and data formats

## Runtime

**Environment:**
- Python 3.6+ (no specific version pinned, compatible with latest)

**Package Manager:**
- pip
- Lockfile: `requirements.txt` (45 dependencies)

## Frameworks

**Core:**
- Streamlit 1.30.0+ - Web UI for dashboard and main application (`app.py`, `execution/dashboard/app.py`)
- Pydantic 2.0.0+ - Data validation and settings management (`execution/config.py`)
- SQLAlchemy 2.0.0+ - ORM and database abstraction (`execution/sources/database.py`)

**Testing:**
- pytest (implied via test files in `tests/` and `execution/`)
- unittest (standard library)

**Build/Dev:**
- python-dotenv 1.0.0 - Environment variable management
- structlog 24.1.0+ - Structured logging (`execution/utils/logging.py`)

## Key Dependencies

**Critical:**
- google-generativeai 0.3.0+ - Google Gemini API and search grounding (research agent, fact verification)
- openai 1.0.0+ - OpenAI API for GPT models (writer, critic, editor agents)
- anthropic 0.18.0+ - Claude API for ethics review (`execution/agents/adversarial_panel.py`)
- groq 7.7.0+ - Groq Llama inference (fast/lightweight model tasks)
- perplexity-py (via openai-compatible API) - Sonar Pro research (`execution/agents/perplexity_researcher.py`)

**Data Processing:**
- feedparser 6.0.11 - RSS/Atom feed parsing (`execution/sources/rss_source.py`)
- requests 2.31.0+ - HTTP client for API calls (GitHub, HN, Reddit)
- praw 7.7.0+ - Reddit API client (optional, RSS fallback used by default)
- pandas - Data manipulation (used in dashboard)
- Pillow - Image processing

**NLP & Analysis:**
- vaderSentiment 3.3.2+ - Sentiment analysis for content evaluation
- scikit-learn 1.3.0+ - Machine learning utilities for style scoring
- nltk 3.8+ - Natural language toolkit for text analysis
- lexicalrichness 0.5+ - Lexical diversity metrics for style enforcement

**Resilience & Retry:**
- tenacity 8.2.0+ - Retry logic with exponential backoff (transient error handling)

**Authentication:**
- google-auth 2.x - OAuth 2.0 for Gmail integration
- google-auth-oauthlib - OAuth authentication flow
- googleapiclient - Google Workspace API client

**Database:**
- SQLite (file-based, `reddit_content.db` default)

## Configuration

**Environment Variables (.env):**
```
# LLM API Keys (choose one or more)
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
GOOGLE_API_KEY=...          # For Gemini
GEMINI_API_KEY=...          # Alternative name for same
GROQ_API_KEY=...
PERPLEXITY_API_KEY=...

# Reddit API (optional)
REDDIT_CLIENT_ID=...
REDDIT_CLIENT_SECRET=...
REDDIT_USER_AGENT=RedditNewsBot/1.0

# GitHub Integration
GITHUB_TOKEN=...            # For higher rate limits (5000/hr vs 60/hr)
GITHUB_REPOS=...            # Comma-separated list

# Database
DB_PATH=reddit_content.db

# Application
GHOSTWRITER_ENV=development|production
GHOSTWRITER_OUTPUT_DIR=./drafts
GHOSTWRITER_TEMP_DIR=./.tmp
DASHBOARD_PASSWORD=...      # Optional, for dashboard auth
```

**Configuration System:**
- Centralized in `execution/config.py`
- Pydantic-based with environment variable override support
- Sections: PathConfig, APIConfig, QualityConfig, ModelConfig, VoiceConfig
- Multi-provider LLM routing (Gemini → OpenAI → Anthropic → Groq fallback)

**Quality Gate Config:**
- Pass threshold: 7.0
- Escalation threshold: 6.0
- Max iterations: 3
- Max revision attempts: 2
- Fact verification required: True
- Multi-model review required: True

**Model Selection:**
- Writer/Critic/Editor: llama-3.3-70b-versatile (Groq)
- Research primary: gemini-2.5-flash-lite
- Research fallback: sonar-pro (Perplexity)
- Ethics reviewer: claude-sonnet-4-20250514
- Structure reviewer: gpt-4o
- Fact reviewer: llama-3.3-70b-versatile

## Platform Requirements

**Development:**
- Windows 10+ (tested on Windows 11)
- Python 3.6+
- pip package manager
- Git (for version control)

**Production:**
- Python 3.6+ runtime
- SQLite (file-based database)
- Network access for LLM APIs (OpenAI, Google, Anthropic, Groq, Perplexity)
- Network access for content sources (Reddit, HN, RSS feeds, GitHub, Gmail)

**Deployment Target:**
- Can run on any platform with Python (local machine, cloud VM, Docker)
- No external database required (SQLite embedded)
- Streamlit handles web serving for UI

---

*Stack analysis: 2026-02-09*
