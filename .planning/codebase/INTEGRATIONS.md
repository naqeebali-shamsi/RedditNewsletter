# External Integrations

**Analysis Date:** 2026-02-09

## APIs & External Services

**LLM Providers:**
- **OpenAI (GPT-4, GPT-3.5-turbo)**
  - What it's used for: Draft generation, editing, criticism
  - SDK/Client: `openai` package
  - Auth: `OPENAI_API_KEY` env var
  - Usage: WriterAgent, EditorAgent, CriticAgent, AdversarialPanelAgent (structure review)
  - Fallback: Multi-provider routing in `execution/agents/base_agent.py`

- **Google Gemini (gemini-2.5-pro, gemini-2.5-flash-lite)**
  - What it's used for: Research with real-time web search, fact verification, content evaluation
  - SDK/Client: `google-generativeai` package
  - Auth: `GOOGLE_API_KEY` or `GEMINI_API_KEY` env var
  - Usage: GeminiResearchAgent (`execution/agents/gemini_researcher.py`), content evaluation
  - Features: Google Search grounding (native web search integration), citation metadata

- **Anthropic (Claude Sonnet, Claude Opus)**
  - What it's used for: Ethics review, adversarial panel assessment
  - SDK/Client: `anthropic` package
  - Auth: `ANTHROPIC_API_KEY` env var
  - Usage: AdversarialPanelAgent for ethics/tone validation
  - Location: `execution/agents/adversarial_panel.py`

- **Groq (Llama 3.3 70B)**
  - What it's used for: Fast inference for writers, editors, critics (all supporting agents)
  - SDK/Client: `groq` package (OpenAI-compatible API)
  - Auth: `GROQ_API_KEY` env var
  - Usage: Default fast model across WriterAgent, EditorAgent, CriticAgent, FactVerificationAgent
  - Features: ~700 tokens/second throughput, generous free tier

- **Perplexity Sonar Pro**
  - What it's used for: Grounded fact verification with real-time web search and citations
  - SDK/Client: OpenAI-compatible API via `openai` package
  - Auth: `PERPLEXITY_API_KEY` env var
  - Usage: PerplexityResearchAgent (`execution/agents/perplexity_researcher.py`) for fact-checking
  - Features: F-score 0.858 on SimpleQA benchmark, 1200 tokens/sec, numbered citations built-in

## Data Sources

**Content Sources:**
- **Reddit (RSS & API)**
  - Integration: `execution/sources/reddit_source.py`
  - Auth: Optional `REDDIT_CLIENT_ID`, `REDDIT_CLIENT_SECRET`, `REDDIT_USER_AGENT`
  - Default: RSS feeds (no auth required)
  - With API: Full data access, more metadata
  - Subreddits monitored: LocalLLaMA, LLMDevs, LanguageTechnology, MachineLearning, deeplearning, mlops, learnmachinelearning

- **Hacker News (Firebase API)**
  - Integration: `execution/sources/hackernews_source.py`
  - Auth: Not required (public Firebase API)
  - Endpoint: `https://hacker-news.firebaseio.com/v0`
  - Fetches: Top stories metadata and details

- **RSS/Atom Feeds**
  - Integration: `execution/sources/rss_source.py`
  - Default feeds: Lobsters, Dev.to, Hacker Noon
  - Configurable via `DEFAULT_FEEDS` list
  - Library: `feedparser`
  - Features: Tracking parameter stripping (UTM/ref), URL normalization

- **Gmail Newsletters**
  - Integration: `execution/sources/gmail_source.py`
  - Auth: OAuth 2.0 via Google Cloud Console
  - Scope: `https://www.googleapis.com/auth/gmail.readonly` (read-only)
  - Credentials: `credentials_gmail.json` (from Google Cloud setup)
  - Token: `token_gmail.json` (generated after OAuth flow)
  - Libraries: `google-auth-oauthlib`, `googleapiclient`
  - Features: Label-based filtering, sender trust tier tracking

- **GitHub Repositories**
  - Integration: `execution/fetch_github.py`
  - Auth: `GITHUB_TOKEN` (personal access token) for higher rate limits
  - Without token: 60 requests/hour
  - With token: 5000 requests/hour
  - Endpoint: GitHub REST API v3
  - Default repos: microsoft/semantic-kernel, langchain-ai/langchain, run-llama/llama_index, vllm-project/vllm, huggingface/transformers
  - Configurable via: `GITHUB_REPOS` env var (comma-separated)

## Data Storage

**Databases:**
- **SQLite (primary)**
  - File: `reddit_content.db` (at project root)
  - Configurable: `DB_PATH` env var
  - ORM: SQLAlchemy 2.0+
  - Tables:
    - `content_items` - Unified content from all sources (Reddit, HN, RSS, Gmail)
    - `posts` - Legacy Reddit posts (being phased out)
    - `newsletter_senders` - Gmail sender trust tiers
    - Review decision history and audit trails
  - Location: `execution/sources/database.py`

**File Storage:**
- **Local filesystem**
  - Output drafts: `./drafts/` (configurable via `GHOSTWRITER_OUTPUT_DIR`)
  - Temp files: `./.tmp/` (intermediate processing, ephemeral)
  - Logs: `./logs/`
  - No external cloud storage integration
  - All paths centralized in `execution/config.py`

**Caching:**
- None configured at system level
- Tenacity provides in-memory retry caching for transient errors
- Per-request state in ArticleState (`execution/article_state.py`)

## Authentication & Identity

**Auth Providers:**
- **OAuth 2.0** (for Gmail)
  - Provider: Google Cloud Console
  - Flow: InstalledAppFlow (desktop/server OAuth)
  - Token refresh: Automatic with `google.auth.transport.requests.Request`
  - Scope: Gmail read-only (`gmail.readonly`)

- **API Key Authentication** (for LLM providers)
  - All providers use API keys (no OAuth)
  - Keys stored in `.env` file
  - Environment variable override in `execution/config.py`

- **Personal Access Token** (for GitHub)
  - Optional enhancement for rate limit increase
  - Token generated in GitHub settings
  - Header-based authentication

- **Dashboard Auth** (optional)
  - Password: `DASHBOARD_PASSWORD` env var (optional)
  - Implemented in `execution/dashboard/app.py` via Streamlit
  - No auth when password unset (development mode)

## Monitoring & Observability

**Error Tracking:**
- Not detected - Custom exception hierarchy in `execution/agents/base_agent.py`
- Exception types: LLMError, ProviderError, AllProvidersFailedError, LLMNotConfiguredError
- Transient error detection: Typed exceptions + string-based fallback patterns

**Logs:**
- Structured logging via `structlog` 24.1.0+
- Location: `execution/utils/logging.py`
- Destination: `./logs/` directory
- No external log aggregation (local files only)

**Health Checks:**
- Module: `execution/utils/health.py`
- Validates API key availability per provider
- Used in dashboard and configuration validation

## CI/CD & Deployment

**Hosting:**
- Not configured (runs locally or on cloud VM)
- No cloud platform tie-ins
- Standalone Python application

**CI Pipeline:**
- Not detected
- Git repository tracked but no GitHub Actions/CI configured

**Deployment:**
- Manual: `python app.py` for Streamlit UI
- Manual: `python execution/[script].py` for pipelines
- Docker: Not configured (but feasible)
- Systemd/scheduler: Not configured (manual or cron setup required)

## Environment Configuration

**Required env vars (at minimum one LLM provider):**
- One of: `OPENAI_API_KEY`, `GOOGLE_API_KEY`, `ANTHROPIC_API_KEY`, `GROQ_API_KEY`, `PERPLEXITY_API_KEY`
- At least one research API: `GOOGLE_API_KEY` or `PERPLEXITY_API_KEY`

**Optional env vars:**
- `REDDIT_CLIENT_ID`, `REDDIT_CLIENT_SECRET`, `REDDIT_USER_AGENT` (for enhanced Reddit access)
- `GITHUB_TOKEN` (for higher GitHub API rate limits)
- `GITHUB_REPOS` (custom repository list)
- `GHOSTWRITER_ENV` (development or production)
- `GHOSTWRITER_OUTPUT_DIR` (custom drafts directory)
- `GHOSTWRITER_TEMP_DIR` (custom temp directory)
- `DB_PATH` (custom database location)
- `DASHBOARD_PASSWORD` (for dashboard authentication)

**Secrets location:**
- `.env` file at project root (git-ignored)
- `.env.example` provided as template
- OAuth tokens: `credentials_gmail.json`, `token_gmail.json` (git-ignored)

**Validation:**
- `execution/config.py` provides `validate_config()` function
- Checks for required API keys, fact verification availability
- Called in `app.py` startup and available via CLI

## Webhooks & Callbacks

**Incoming Webhooks:**
- None detected - Unidirectional data fetching only

**Outgoing Callbacks:**
- None detected - No reverse-calling to external systems
- All integrations are pull-based (fetch data from sources)

**Retry Logic:**
- Tenacity library with exponential backoff + random jitter
- Configured in: `execution/fetch_github.py`, `execution/sources/reddit_source.py`, `execution/sources/hackernews_source.py`
- Transient error detection: Typed exceptions (OpenAI, Groq, Google API) + string matching fallback
- Location: `execution/agents/base_agent.py` (_is_transient function)

---

*Integration audit: 2026-02-09*
