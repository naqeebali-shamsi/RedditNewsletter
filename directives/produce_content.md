# Directive: Produce AI Engineering Content from Reddit

## Goal
Extract high-signal posts from AI engineering subreddits and convert them into LinkedIn posts and Medium articles that establish authority in the AI engineering space.

## Target Persona
Mid-level SWE transitioning to AI Engineer, building thought leadership on LinkedIn/Medium to secure premium AI engineering roles ($250K+ TC).

## Inputs
- **Subreddit List**: Defined in `market_strategy.md` (S+ and S tier)
- **Time Window**: Last 72 hours
- **Volume**: Up to 100 posts per subreddit per run

## Tools / Scripts

### 1. Data Collection (`execution/fetch_reddit.py`)
**Purpose**: Pull latest posts from target subreddits via RSS.

**Usage**:
```bash
# Fetch from S+ tier (default)
python execution/fetch_reddit.py

# Fetch from all tiers
python execution/fetch_reddit.py --all

# Fetch specific subreddits
python execution/fetch_reddit.py --subreddits LocalLLaMA mlops
```

**Output**: Posts stored in `reddit_content.db` (posts table).

**Frequency**: Run 2-3x per day (morning, afternoon, evening).

---

### 2. Signal/Noise Evaluation (`execution/evaluate_posts.py`)
**Purpose**: Use LLM to classify posts based on market strategy criteria.

**Evaluation Criteria** (from `market_strategy.md`):
- **Signal**: Concrete details, production challenges, technical insights, frameworks, postmortems
- **Noise**: Generic news, speculation, "will AI take over?", surface-level trends

**Usage**:
```bash
# Evaluate up to 50 posts
python execution/evaluate_posts.py --limit 50
```

**Output**: Evaluations stored in `reddit_content.db` (evaluations table).

**Frequency**: Run after each fetch cycle.

---

### 3. Draft Generation (`execution/generate_drafts.py`)
**Purpose**: Convert Signal posts into platform-specific content.

**Content Themes** (from `market_strategy.md`):
1. Production Reality vs. Hype
2. Technical Decision Frameworks
3. Postmortems & Hard-Won Lessons
4. Ecosystem Deep Dives
5. Scaling & Optimization

**Usage**:
```bash
# Generate LinkedIn drafts
python execution/generate_drafts.py --platform linkedin --limit 10

# Generate Medium drafts
python execution/generate_drafts.py --platform medium --limit 5

# Generate both
python execution/generate_drafts.py --platform both --limit 10
```

**Output**: 
- Drafts stored in `reddit_content.db` (drafts table)
- Text files exported to `.tmp/drafts/`

**Frequency**: Run weekly (Sunday evenings) to prepare content for the week.

---

### 4. Writing Framework Gates (`directives/framework_rules.md`)
**Purpose**: Apply the 5-Pillar Architected Writing Framework to ensure quality.

**Pre-Generation Gate** (before draft):
- Define the "Status Quo" being challenged (Contrast Hook)
- Identify the war story to weave in (Human Variable)
- List target takeaways (Takeaway Density)

**Post-Generation Gate** (after draft, before publish):
- Quality checklist from `framework_rules.md`:
  1. Does the hook challenge a status quo? (with evidence)
  2. Is any paragraph over the limit? (LinkedIn: 3 lines, Articles: 4-5 lines)
  3. Are tradeoffs stated where they exist?
  4. Does it sound like a battle-scarred engineer, not an AI bot?
  5. No meta-labels in output?
- Voice validation: `python execution/validate_voice.py`
- Style scoring: `python execution/validate_voice.py --score`

**References**:
- Writing voice & style: `directives/writing_rules.md`
- 5-Pillar framework: `directives/framework_rules.md`
- Technical standards: `directives/technical_rules.md`

---

## Outputs
- **LinkedIn Posts**: 2-3 per week (Mon, Wed, Fri)
- **Medium Articles**: 1-2 per month (deep dives)
- **Files**: `.tmp/drafts/linkedin_*.txt` and `.tmp/drafts/medium_*.txt`

## Edge Cases

### 1. No Signal Posts Found
**Scenario**: All posts classified as Noise.

**Action**: 
- Lower evaluation threshold slightly
- Check if subreddit list needs adjustment
- Verify LLM prompt is not too strict

### 2. API Rate Limits (Reddit RSS)
**Scenario**: Reddit blocks excessive requests.

**Action**:
- Implement 2-second delay between subreddit fetches
- Run fetch cycle max 3x per day
- Use Reddit API with authentication if RSS fails

### 3. Duplicate Content Across Subreddits
**Scenario**: Same post appears in multiple subreddits.

**Action**:
- Database enforces unique URLs (handled automatically)
- LLM evaluation considers "cross-posted" context

### 4. Outdated Content
**Scenario**: Posts older than 72 hours.

**Action**:
- `fetch_reddit.py` filters by timestamp (default 72h)
- Adjust `--hours` parameter if needed

### 5. LLM API Failures
**Scenario**: OpenAI/Anthropic API down or rate limited.

**Action**:
- Implement exponential backoff (3 retries)
- Fall back to simple keyword heuristic (already in code)
- Log failures for manual review

---

## Success Metrics
- **Signal Rate**: 20-40% of posts classified as Signal (healthy range)
- **Draft Quality**: 80%+ of drafts require minimal editing
- **Engagement**: LinkedIn posts average 100+ reactions, Medium articles 1K+ views

## Maintenance
- **Weekly**: Review top Signal posts manually, update evaluation criteria if needed
- **Monthly**: Analyze which subreddits produce most Signal, adjust tier list
- **Quarterly**: Update `market_strategy.md` with new content themes

---

## Quick Start

1. **Initialize Database**:
   ```bash
   python execution/init_db.py
   ```

2. **First Run** (Full Pipeline):
   ```bash
   # Fetch from S+ tier
   python execution/fetch_reddit.py
   
   # Evaluate posts
   python execution/evaluate_posts.py --limit 50
   
   # Generate drafts
   python execution/generate_drafts.py --platform both --limit 10
   ```

3. **Review Drafts**:
   - Open `.tmp/drafts/`
   - Edit as needed
   - Publish to LinkedIn/Medium

4. **Schedule** (via cron or Task Scheduler):
   - Run `fetch_reddit.py` 3x daily
   - Run `evaluate_posts.py` 3x daily (after fetch)
   - Run `generate_drafts.py` 1x weekly (Sunday 8 PM)
