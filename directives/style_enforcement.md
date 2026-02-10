# Directive: Style Enforcement

## Purpose

Quantitative voice fingerprinting that scores content across 5 dimensions before publication. Replaces subjective "does this sound right?" with measurable thresholds.

## Inputs

- Article/post content (Markdown text)
- Content type: `linkedin`, `article`, or `longform`
- Optional: `voice_profile.json` for baseline comparison

## Tools

- `execution/agents/style_enforcer.py` — StyleEnforcerAgent (standalone, no LLM needed)
- `execution/voice_profile.json` — Baseline voice profile
- `execution/validate_voice.py --score` — CLI access to style scoring
- `execution/quality_gate.py` — Integrated as Step 0.5 in the review loop

## The 5 Dimensions

| Dimension | Weight | What It Measures |
|-----------|--------|------------------|
| Burstiness | 20% | Sentence length variation. Humans: 0.3-0.6. AI: 0.15-0.25. |
| Lexical Diversity | 15% | Vocabulary richness via VOCD or TTR. |
| AI-Tell Detection | 25% | Forbidden phrases and hedging language. |
| Authenticity Markers | 25% | War story keywords + specific metrics/numbers. |
| Framework Compliance | 15% | Contrast hook, tradeoffs, paragraph limits. |

## Score Thresholds

| Score Range | Action |
|-------------|--------|
| 80-100 | **Publish** — Content passes style gate. |
| 60-79 | **Revision** — Content needs targeted fixes. Check which dimensions scored low. |
| 0-59 | **Rejection** — Content needs significant rework or is likely AI-generated without editing. |

## Usage

### CLI

```bash
# Score an article
python execution/agents/style_enforcer.py articles/my-post.md

# Score with voice profile baseline
python execution/agents/style_enforcer.py articles/my-post.md --profile execution/voice_profile.json

# JSON output for pipeline integration
python execution/agents/style_enforcer.py articles/my-post.md --json

# Via validate_voice.py
python execution/validate_voice.py --file articles/my-post.md --score
```

### In Pipeline (quality_gate.py)

Style enforcement runs automatically as Step 0.5 in the quality gate, after fact verification and before the adversarial review loop. It logs the score and any AI tells found. It does not block publication on its own — it provides signal to the review loop.

### Programmatic

```python
from execution.agents.style_enforcer import StyleEnforcerAgent

enforcer = StyleEnforcerAgent(profile_path="execution/voice_profile.json")
result = enforcer.score(article_text, content_type="article")

if result.passed:
    print("Ready for publication")
elif result.needs_revision:
    print(f"Fix these dimensions: ...")
    print(enforcer.format_report(result))
else:
    print("Needs significant rework")
```

## How to Recalibrate voice_profile.json

The voice profile starts with null baselines. To calibrate:

1. **Gather 5-10 published articles** that represent the target voice
2. **Run the scorer on each** and collect raw metrics:
   ```bash
   python execution/agents/style_enforcer.py articles/sample1.md --json > .tmp/scores.json
   ```
3. **Average the baseline metrics** (burstiness_ratio, avg_sentence_length, sentence_length_std, ttr)
4. **Update voice_profile.json** with the averaged values:
   ```json
   "baseline_metrics": {
     "avg_sentence_length": 14.2,
     "sentence_length_std": 8.7,
     "burstiness_ratio": 0.45,
     "vocd_score": 62.3,
     "ttr": 0.58
   }
   ```
5. **Set `calibrated_from`** to the list of articles used
6. **Set `updated_at`** to the calibration date

## Integration Points

1. **quality_gate.py** — Step 0.5, runs after fact verification
2. **validate_voice.py** — `--score` flag for standalone scoring
3. **Adversarial panel** — Style score can inform panel review context
4. **CI/pre-commit** — Can be added as a pre-publish check

## ToneProfile Overrides

When a `ToneProfile` is provided (from `execution/tone_profiles.py`), the style enforcer adjusts its scoring parameters via `profile.to_style_overrides()`. This allows different tones to be scored against appropriate baselines rather than a single fixed standard.

### What Gets Overridden

| Parameter | Default | Override Source |
|-----------|---------|----------------|
| Forbidden phrases | Global list | `profile.forbidden_phrases` (each preset has its own) |
| War story keywords | Global list | `profile.war_story_keywords` (e.g. News Reporter has none) |
| Burstiness thresholds | high variance targets | Based on `sentence_style.length_variance` (high/medium/low) |
| Authenticity weight | 25% | Drops to 10% when profile has no war story keywords |
| AI-tell weight | 25% | Increases to 30% for formal profiles (formality >= 0.8) |
| Framework compliance weight | 15% | Absorbs remaining weight after authenticity/AI-tell redistribution |

### Backward Compatibility

No ToneProfile = original behavior. All overrides are additive -- the scoring logic checks for the presence of a ToneProfile and falls back to default thresholds when none is provided. Existing pipelines, CLI usage, and `voice_profile.json` calibration continue to work unchanged.

### Example

```python
from execution.agents.style_enforcer import StyleEnforcerAgent
from execution.tone_profiles import get_preset

profile = get_preset("News Reporter")
overrides = profile.to_style_overrides()

enforcer = StyleEnforcerAgent(profile_path="execution/voice_profile.json")
result = enforcer.score(article_text, content_type="article", tone_overrides=overrides)
```

For full tone system documentation, see `directives/tone_system.md`.

## Self-Annealing

When the writing style intentionally evolves:

1. Re-run calibration on the latest 5-10 published articles
2. Update `voice_profile.json` baseline metrics
3. If new forbidden phrases are discovered, add them to the profile
4. If new war story keywords emerge, add them to `required_markers`
5. Update `updated_at` and `calibrated_from`
6. Document the change in `.planning/SUMMARY.md`

When the scorer produces false positives/negatives:

1. Adjust dimension weights if one dimension dominates unfairly
2. Tune thresholds in the scoring bands (e.g., burstiness >= 0.3 instead of 0.4 for full marks)
3. Add/remove phrases from forbidden or hedging lists
4. Update this directive with the rationale

## Edge Cases

- **Short content (<100 words)**: Burstiness and lexical diversity may be unreliable. Weight AI-tell and authenticity scores higher.
- **Code-heavy articles**: Code blocks inflate word count and skew metrics. Consider stripping code blocks before scoring.
- **Missing dependencies**: `lexicalrichness` and `nltk` are optional. Without them, lexical diversity falls back to simple TTR.
- **LinkedIn vs. articles**: LinkedIn has stricter paragraph limits (3 lines vs. 5). Pass the correct `content_type`.

## Dependencies

Required: None (pure Python with stdlib)

Optional (for better accuracy):
- `lexicalrichness` — VOCD scoring
- `nltk` — Better sentence tokenization
