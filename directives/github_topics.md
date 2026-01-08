# GitHub Topics: Generate Content from Commit Activity

## Goal
Extract insights from GitHub commit activity in leading AI/ML repositories and convert them into thought leadership content that demonstrates deep technical awareness.

## Why GitHub?
- **Reddit**: Shows what people are TALKING about
- **GitHub**: Shows what people are BUILDING

For technical credibility, demonstrating awareness of actual code changes > commentary.

## Target Persona
Mid-level SWE transitioning to AI Engineer, using open source intelligence to differentiate content from generic AI commentary.

## Repository Tier List

### Tier S+ (High Signal for AI Engineers)
- `microsoft/semantic-kernel` - Enterprise AI orchestration
- `langchain-ai/langchain` - LLM application framework
- `run-llama/llama_index` - Data framework for LLMs

### Tier S (High Activity, Good Patterns)
- `vllm-project/vllm` - High-performance inference
- `huggingface/transformers` - Foundation models
- `openai/openai-python` - Official SDK patterns

### Tier A (Specialized)
- `anthropics/anthropic-sdk-python` - Claude patterns
- `lm-sys/FastChat` - LLM serving patterns
- `ray-project/ray` - Distributed ML

## Tools / Scripts

### 1. Data Collection: `execution/fetch_github.py`

**Usage**:
```bash
# Fetch from default repos (first 3)
python execution/fetch_github.py

# Fetch from all default repos
python execution/fetch_github.py --all

# Fetch specific repos
python execution/fetch_github.py --repos microsoft/semantic-kernel langchain-ai/langchain

# Adjust time window
python execution/fetch_github.py --hours 48  # Last 2 days
```

**Output**: Commits stored in `reddit_content.db` (github_commits table)

**Frequency**: Run 1x daily (commits don't change like Reddit posts)

### 2. Theme Extraction: `execution/agents/commit_analyzer.py`

The CommitAnalysisAgent:
- Analyzes batches of commits from the database
- Extracts themes: problems solved, technologies used, patterns
- Returns topic dict compatible with existing pipeline

**Usage**: Automatically invoked when "GitHub (Commits)" is selected in app.py

### 3. Content Generation

Same pipeline as Reddit - only Phase 1 (topic selection) changes:
- Select "GitHub (Commits)" in the app.py sidebar
- Click Generate
- Pipeline continues with EditorAgent -> WriterAgent -> etc.

## Outputs
- **LinkedIn Posts**: 2-3 per week (insights from open source activity)
- **Medium Articles**: 1-2 per month (deep dives on patterns)
- **Files**: Same location as Reddit: `drafts/medium_full_*.md`

## Edge Cases

### 1. GitHub API Rate Limits
**Scenario**: Unauthenticated requests limited to 60/hour

**Action**:
- Add `GITHUB_TOKEN` to `.env` for 5,000 requests/hour
- Script implements automatic delays between requests
- Consider using GitHub App for higher limits if needed

### 2. No Significant Themes Found
**Scenario**: Commits are routine maintenance without content-worthy patterns

**Action**:
- Fallback to "What Recent Framework Updates Reveal" generic topic
- Consider adjusting repository list
- Try longer time window (`--hours 336` = 2 weeks)

### 3. Stale Data
**Scenario**: Database has old commits from weeks ago

**Action**:
- `fetch_github.py` uses `--hours` parameter
- Re-run fetch to get fresh commits

## Content Themes from Commits

1. **Breaking Changes**: What's changing that practitioners need to know?
2. **Performance Improvements**: Optimizations in inference engines, memory usage
3. **API Evolution**: How are interfaces changing? What does this mean?
4. **Bug Patterns**: Common issues being fixed - what can we learn?
5. **Dependency Updates**: What's the ecosystem doing?

## Quick Start

1. **Add GitHub Token** (optional but recommended):
   ```bash
   # Add to .env
   GITHUB_TOKEN=ghp_your_token_here
   ```

2. **Fetch Commits**:
   ```bash
   python execution/fetch_github.py --all
   ```

3. **Generate Content**:
   - Open app: `streamlit run app.py`
   - Select "GitHub (Commits)" in sidebar
   - Click Generate

## Success Metrics
- **Theme Quality**: Themes should be specific, not generic
- **Uniqueness**: Content angle not found in typical AI news
- **Technical Depth**: Insights require reading commit history
- **Engagement**: GitHub-sourced posts should match or exceed Reddit-sourced

## Maintenance
- **Weekly**: Review which repos generate best themes
- **Monthly**: Update repository list based on industry trends
- **Quarterly**: Evaluate if commits are providing unique value vs Reddit
