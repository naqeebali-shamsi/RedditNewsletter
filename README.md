# GhostWriter

An autonomous AI ghostwriting pipeline that extracts high-signal posts from AI engineering subreddits and converts them into LinkedIn posts and Medium articles.

## 🎯 Purpose

Build thought leadership in AI Engineering by:
- Monitoring 7+ high-value subreddits (r/LocalLLaMA, r/LLMDevs, etc.)
- Filtering for "Signal" (technical, practical) vs "Noise" (hype, speculation)
- Generating platform-specific content drafts for LinkedIn and Medium

## 🏗️ Architecture

This project follows a **3-layer architecture**:

1. **Layer 1: Directives** (`directives/`) - SOPs defining what to do
2. **Layer 2: Orchestration** (You, the Agent) - Decision making and routing
3. **Layer 3: Execution** (`execution/`) - Deterministic Python scripts

### Database Schema

SQLite database (`reddit_content.db`) with 3 tables:
- `posts` - Raw Reddit posts from RSS feeds
- `evaluations` - LLM classifications (Signal vs Noise)
- `drafts` - Generated LinkedIn/Medium content

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
# Copy template
cp .env.example .env

# Edit .env and add your LLM API key (OpenAI, Anthropic, or Google)
```

### 3. Initialize Database

```bash
python execution/init_db.py
```

### 4. Run the Pipeline

```bash
# Step 1: Fetch posts from Reddit RSS feeds
python execution/fetch_reddit.py --all

# Step 2: Evaluate posts (Signal vs Noise)
python execution/evaluate_posts.py --limit 50

# Step 3: Generate content drafts
python execution/generate_drafts.py --platform both --limit 10
```

### 5. Review Drafts

Drafts are saved to `.tmp/drafts/`:
- `linkedin_*.txt` - LinkedIn posts (short, conversational)
- `medium_*.txt` - Medium articles (long-form, technical)

## 📁 Project Structure

```
RedditNews/
├── directives/              # SOPs and strategy
│   ├── market_strategy.md   # Subreddit tiers, persona, content themes
│   └── produce_content.md   # Workflow SOP
├── execution/               # Python scripts
│   ├── schema.sql           # Database schema
│   ├── init_db.py           # Initialize SQLite database
│   ├── fetch_reddit.py      # Collect posts from RSS
│   ├── evaluate_posts.py    # Signal/Noise classification
│   └── generate_drafts.py   # Draft generation
├── .tmp/                    # Temporary files (gitignored)
│   └── drafts/              # Generated content drafts
├── .env.example             # Environment variables template
├── requirements.txt         # Python dependencies
└── reddit_content.db        # SQLite database (created on init)
```

## 🎨 Content Themes

Based on market research, these themes resonate with AI engineering audiences:

1. **Production Reality vs. Hype** - "RAG doesn't solve your problem: When it fails"
2. **Technical Decision Frameworks** - "vLLM vs TensorRT: When to use each"
3. **Postmortems & Lessons** - "We built LLM evaluation wrong for 4 months"
4. **Ecosystem Deep Dives** - "Vector databases for RAG: Weaviate vs Pinecone"
5. **Scaling & Optimization** - "From 1M to 100M inferences: What changed"

## 🎯 Target Subreddits

### S+ Tier (Highest Priority)
- r/LocalLLaMA (45k) - Local LLM deployment, quantization
- r/LLMDevs (5k) - Hyper-specialized, LLMOps focus
- r/LanguageTechnology (45k) - Professional NLP

### S Tier (High Priority)
- r/MachineLearning (3M) - Research implementations
- r/deeplearning (100k) - Architecture-focused
- r/mlops (50k) - Operations/Infrastructure
- r/learnmachinelearning (355k) - Educational

## 🔧 Advanced Usage

### Fetch Specific Subreddits

```bash
python execution/fetch_reddit.py --subreddits LocalLLaMA LLMDevs
```

### Adjust Time Window

```bash
# Only posts from last 24 hours
python execution/fetch_reddit.py --hours 24
```

### Generate Only LinkedIn Posts

```bash
python execution/generate_drafts.py --platform linkedin --limit 20
```

## 📊 Expected Metrics

- **Signal Rate**: 20-40% of posts (healthy range)
- **LinkedIn Engagement**: 100+ reactions per post
- **Medium Engagement**: 1K+ views per article
- **Time Investment**: 4-6 hours/week
- **ROI**: $50K-$100K salary increase (from thought leadership)

## 🔄 Recommended Schedule

- **Fetch**: 3x daily (morning, afternoon, evening)
- **Evaluate**: 3x daily (after each fetch)
- **Generate Drafts**: 1x weekly (Sunday evening)
- **Publish LinkedIn**: 2-3x per week (Mon, Wed, Fri)
- **Publish Medium**: 1-2x per month

## 🛠️ Current Status

**⚠️ LLM Integration**: The evaluation and draft generation scripts currently use simple keyword heuristics. To enable full LLM-powered processing:

1. Add your API key to `.env`
2. Update `evaluate_posts.py` and `generate_drafts.py` to call your chosen LLM API
3. Test with a small batch first

See `directives/produce_content.md` for detailed implementation guidance.

## 📝 Next Steps

1. **Add LLM Integration** - Replace placeholder evaluation logic with real LLM calls
2. **Scheduler** - Set up cron jobs (Linux/Mac) or Task Scheduler (Windows)
3. **Analytics** - Track which posts generate most engagement
4. **Refinement** - Adjust evaluation criteria based on results

## 📚 Documentation

- **Market Strategy**: `directives/market_strategy.md`
- **Workflow SOP**: `directives/produce_content.md`
- **Agent Instructions**: `AGENTS.md`, `CLAUDE.md`, `GEMINI.md`

---

**Built with the 3-layer Agentic Architecture** - Separating directives, orchestration, and execution for maximum reliability and self-improvement.
