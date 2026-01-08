# Quick Reference Guide

## 🚀 Daily Workflow

### Option 1: Streamlit Web UI (Recommended)
```bash
streamlit run app.py
```
This launches the GhostWriter dashboard for full article generation with visual progress tracking.

### Option 2: Run Individual Steps
```bash
# Fetch latest posts
python execution/fetch_reddit.py

# Evaluate for Signal/Noise
python execution/evaluate_posts.py --limit 50

# Generate drafts
python execution/generate_drafts.py --platform both --limit 10
```

---

## 📋 Common Commands

### Fetching Posts

```bash
# S+ tier only (default)
python execution/fetch_reddit.py

# All tiers (S+ and S)
python execution/fetch_reddit.py --all

# Specific subreddits
python execution/fetch_reddit.py --subreddits LocalLLaMA LLMDevs

# Adjust time window (default 72 hours)
python execution/fetch_reddit.py --hours 24
```

### Evaluating Posts

```bash
# Evaluate 50 posts (default)
python execution/evaluate_posts.py

# Evaluate more/fewer
python execution/evaluate_posts.py --limit 100
```

## 1. Generate Medium Article ("The Agency")

Run the multi-agent pipeline to generate a high-quality article with infographics.

```powershell
python -m execution.generate_medium_full --topic "Your Topic Here"
```

**Output:**
- Draft: `drafts/medium_full_[date].md`
- Visuals Dashboard: `drafts/visuals_dashboard.html` (Open in browser to gen images)

### Generating Drafts

```bash
# Both platforms (default 10 posts)
python execution/generate_drafts.py --platform both --limit 10

# LinkedIn only
python execution/generate_drafts.py --platform linkedin --limit 20

# Medium only
python execution/generate_drafts.py --platform medium --limit 5
```

---

## 🎯 Quick Modes

### Streamlit Dashboard (Full Featured)
```bash
streamlit run app.py
# Interactive UI with 10-phase article generation pipeline
```

### CLI: Generate Medium Article
```bash
python -m execution.generate_medium_full --topic "Your Topic"
# Generates article + visuals dashboard
```

### CLI: Quick Draft Generation
```bash
python execution/generate_drafts.py --platform both --limit 5
# Fast batch generation for LinkedIn/Medium
```

---

## 🔧 Setup (One-time)

1. **Set up Virtual Environment (Recommended)**:
   ```bash
   python -m venv .venv
   .\.venv\Scripts\activate  # Windows
   source .venv/bin/activate  # Linux/Mac
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure environment**:
   ```bash
   cp .env.example .env
   # Edit .env and add your LLM API key
   ```

3. **Initialize database**:
   ```bash
   python execution/init_db.py
   ```

---

## 📁 Where to Find Things

| Item | Location |
|------|----------|
| **Generated drafts** | `.tmp/drafts/` |
| **Database** | `reddit_content.db` |
| **Strategy & persona** | `directives/market_strategy.md` |
| **Workflow SOP** | `directives/produce_content.md` |
| **Scripts** | `execution/*.py` |

---

## 🎨 Content Strategy at a Glance

### S+ Tier Subreddits (Target First)
- **r/LocalLLaMA** - Local deployment, quantization
- **r/LLMDevs** - LLMOps, production challenges
- **r/LanguageTechnology** - Professional NLP

### S Tier Subreddits (Second wave)
- **r/MachineLearning** - Research implementations
- **r/deeplearning** - Neural architecture
- **r/mlops** - Operations/Infrastructure
- **r/learnmachinelearning** - Educational

### Content Themes
1. Production Reality vs. Hype
2. Technical Decision Frameworks
3. Postmortems & Hard-Won Lessons
4. Ecosystem Deep Dives
5. Scaling & Optimization

---

## 📊 Database Queries (SQLite)

### Check post counts
```bash
sqlite3 reddit_content.db "SELECT subreddit, COUNT(*) FROM posts GROUP BY subreddit;"
```

### View signal posts
```bash
sqlite3 reddit_content.db "SELECT p.title, p.subreddit FROM posts p JOIN evaluations e ON p.id = e.post_id WHERE e.is_signal = 1 LIMIT 10;"
```

### Check signal rate
```bash
sqlite3 reddit_content.db "SELECT is_signal, COUNT(*) FROM evaluations GROUP BY is_signal;"
```

---

## ⏰ Recommended Schedule

- **Fetch**: 3x daily (9 AM, 2 PM, 8 PM)
- **Evaluate**: 3x daily (after each fetch)
- **Generate**: 1x weekly (Sunday 8 PM)
- **Publish LinkedIn**: Mon, Wed, Fri mornings
- **Publish Medium**: 1-2x per month

---

## 🆘 Troubleshooting

### "No posts found to evaluate"
→ Run `python execution/fetch_reddit.py` first

### "No Signal posts found"
→ Adjust evaluation criteria or fetch more posts

### "Database not found"
→ Run `python execution/init_db.py`

### "Module not found"
→ Run `pip install -r requirements.txt`

---

## 🔄 Next Steps

1. **First run**: `streamlit run app.py` or `python -m execution.generate_medium_full`
2. **Review drafts**: Check `drafts/` folder
3. **Edit & publish**: Customize and post to LinkedIn/Medium
4. **Iterate**: Adjust evaluation criteria based on results
5. **Automate**: Set up scheduled tasks (cron/Task Scheduler)

---

**For detailed documentation, see `README.md`**
