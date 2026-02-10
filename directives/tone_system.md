# Directive: Tone System

## Purpose

Customizable writing voice system that lets GhostWriter produce content in different tones -- from battle-scarred engineer to neutral news reporter -- while maintaining quality gates. Users can select preset tones, infer tones from writing samples, save custom profiles, and benefit from adaptive learning that refines the voice over time based on their edits.

## Overview

The tone system has four components that work together:

1. **Tone Profiles** (`execution/tone_profiles.py`) -- Pydantic data model defining all voice dimensions (formality, technical depth, personality, sentence style, vocabulary, hooks, CTAs, forbidden phrases, war story keywords). Ships with 6 built-in presets.

2. **Tone Inference** (`execution/tone_inference.py`) -- Analyzes a writing sample (raw text or URL) using deterministic metrics + LLM classification to produce a ToneProfile. Useful when a user wants to match an existing author's voice.

3. **User Preferences** (`execution/user_preferences.py`) -- Persists the active profile, custom profiles, feedback log, and learned micro-adjustments to `user_preferences.json`. Thread-safe.

4. **Adaptive Learning** -- After 5+ feedback events, the system compares original vs. edited content to detect formality shifts, sentence length changes, and burstiness changes. These accumulate as micro-adjustments that are applied on top of the active profile.

**Flow:** `select/infer tone → generate content (WriterAgent) → score (StyleEnforcerAgent) → review (QualityGate) → feedback → learn → refine`

## Inputs

- Preset name (string) -- selects a built-in tone
- Writing sample text (string, min 50 chars) -- for inference
- URL (string) -- for inference from a web page
- Feedback events (accepted/edited/rejected) -- for adaptive learning

## Tools

- `execution/tone_profiles.py` -- ToneProfile model, TonePresetManager, `get_preset()`, `list_presets()`
- `execution/tone_presets.json` -- Built-in preset data (6 presets)
- `execution/tone_inference.py` -- ToneInferenceEngine (async + sync wrappers)
- `execution/user_preferences.py` -- UserPreferences (persistence, learning)
- `execution/agents/style_enforcer.py` -- Accepts ToneProfile for scoring overrides
- `execution/agents/writer.py` -- Accepts ToneProfile for prompt construction
- `execution/quality_gate.py` -- Passes ToneProfile through the review pipeline

## Available Presets

| Preset | Personality | Formality | Tech Depth | Best For |
|--------|-------------|-----------|------------|----------|
| **Expert Pragmatist** | Authoritative | 0.6 (professional) | 0.8 (deep) | Default GhostWriter voice. Battle-tested engineer, data-heavy, zero fluff. |
| **Thought Leader** | Visionary | 0.7 (polished) | 0.5 (moderate) | Big-picture strategy pieces, trend analysis, forward-looking content. |
| **Technical Deep Dive** | Analytical | 0.8 (formal) | 1.0 (maximum) | Architecture decisions, benchmarks, code walkthroughs. Minimal narrative. |
| **Conversational Engineer** | Friendly | 0.3 (casual) | 0.6 (moderate) | Blog-style posts, approachable explainers, "coffee chat" format. |
| **News Reporter** | Objective | 0.9 (formal) | 0.5 (moderate) | Third-person tech journalism. WSJ-level precision, no opinion, lead-driven. |
| **Contrarian Challenger** | Provocative | 0.5 (balanced) | 0.7 (deep) | Debate-oriented pieces that question consensus with data. Strong opinions. |

### Selecting a Preset

```python
from execution.tone_profiles import get_preset, list_presets

# List all available presets
print(list_presets())

# Get a specific preset
profile = get_preset("News Reporter")
instructions = profile.to_writer_instructions()  # For WriterAgent prompts
overrides = profile.to_style_overrides()          # For StyleEnforcerAgent scoring
```

## Inferring Tone from a Sample

When you want to match an existing author's voice:

```python
from execution.tone_inference import ToneInferenceEngine

engine = ToneInferenceEngine()

# From raw text (sync wrapper for UI/scripts)
profile = engine.infer_from_text_sync("Your sample text here...")

# From a URL (fetches and strips HTML)
profile = engine.infer_from_url_sync("https://example.com/article")

# Async versions available: infer_from_text(), infer_from_url()
```

**How inference works:**
1. Deterministic analysis -- sentence burstiness, lexical diversity (TTR/VOCD), war story keyword detection, AI-tell phrase detection
2. LLM classification -- formality, personality, hook style, CTA style, vocabulary preferences (uses a fast model, temperature 0.3)
3. Merge -- deterministic metrics inform sentence_style, LLM results fill subjective fields
4. Confidence scoring -- based on sample length: <100 words = 0.3, 100-500 = 0.6, 500-1500 = 0.8, 1500+ = 0.95

**Saving an inferred profile:**
```python
from execution.user_preferences import UserPreferences

prefs = UserPreferences()
prefs.save_custom_profile(profile)
prefs.set_active_profile(profile.name)
```

## Adaptive Learning System

The system learns from user behavior over time. Here is how it works:

### Feedback Loop

1. **Log feedback** after article generation:
   ```python
   prefs.log_feedback("article-123", "accepted")
   prefs.log_feedback("article-456", "edited", original="...", edited="...")
   prefs.log_feedback("article-789", "rejected")
   ```

2. **Activation threshold** -- Learning activates after 5 feedback events (configurable via `_LEARNING_THRESHOLD`).

3. **What it learns from edits** -- When `action="edited"` with original + edited text, the system compares:
   - **Sentence length shift** -- Did the user shorten or lengthen sentences?
   - **Burstiness shift** -- Did the user increase or decrease sentence length variation?
   - **Formality shift** -- Did the user make the text more or less formal? (heuristic based on formal/informal word ratios)

4. **Micro-adjustments** -- Each edit produces a small delta (max 0.1 per cycle) that accumulates in `LearnedAdjustments`. Adjustments are clamped to prevent runaway drift:
   - `formality_delta`: -0.5 to +0.5
   - `technical_depth_delta`: -0.5 to +0.5
   - `avg_sentence_length_delta`: -10 to +10 words
   - `burstiness_delta`: -0.3 to +0.3

5. **Effective profile** -- `prefs.get_effective_profile()` returns the active profile with learned adjustments merged. This is what the pipeline should use.

### Resetting Learning

```python
prefs.reset_learning()  # Clears adjustments, keeps feedback log
```

## Integration Points

### WriterAgent (`execution/agents/writer.py`)

When a ToneProfile is provided:
- `profile.to_writer_instructions()` generates natural language instructions injected into the writer prompt
- Controls formality, technical depth, sentence rhythm, vocabulary, hook style, CTA style, example phrases, and forbidden phrases
- The writer follows these instructions alongside standard writing rules

### StyleEnforcerAgent (`execution/agents/style_enforcer.py`)

When a ToneProfile is provided:
- `profile.to_style_overrides()` adjusts scoring parameters:
  - **Burstiness thresholds** change based on `length_variance` (high/medium/low)
  - **Dimension weights** shift -- profiles without war stories reduce authenticity weight (25% -> 10%); formal profiles increase AI-tell weight (25% -> 30%)
  - **Forbidden phrases** and **war story keywords** come from the profile instead of global defaults
- Backward compatible -- no ToneProfile = original behavior unchanged

### QualityGate (`execution/quality_gate.py`)

- ToneProfile name is tracked in article state for provenance
- The profile flows through to StyleEnforcerAgent during the quality gate scoring step
- Review panel context includes the active tone for more relevant feedback

### Article State (`execution/article_state.py`)

- `tone_profile_name` field records which tone was used for each article
- Enables traceability and auditing of tone choices

## UI (Streamlit)

The tone system is accessible through the Streamlit dashboard (`app.py`):

- **Preset selector** -- dropdown to choose from 6 built-in presets
- **Inference panel** -- paste text or enter a URL to infer a tone profile
- **Custom profile management** -- save, activate, delete custom profiles
- **Feedback stats** -- view acceptance rate, learned adjustments, learning status

## Edge Cases and Limitations

- **Short samples (<100 words)** -- Inference confidence is low (0.3). Burstiness and lexical diversity metrics are unreliable. The LLM classification carries more weight but has less to work with.
- **Non-English text** -- The inference engine assumes English. Formal/informal word lists and war story keywords are English-only. Results for other languages will be unreliable.
- **URL fetch failures** -- If the URL times out (15s) or yields <50 chars of extractable text, `infer_from_url` raises a `ValueError`.
- **Preset names are case-sensitive** -- "expert pragmatist" will not match "Expert Pragmatist".
- **News Reporter has no war stories** -- Its `war_story_keywords` list is empty, so authenticity weight drops to 10%. This is intentional -- reporters don't share personal experiences.
- **Learning drift** -- Micro-adjustments are clamped to prevent extreme drift, but after many edits the effective profile may diverge noticeably from the base preset. Use `reset_learning()` to start fresh.
- **Thread safety** -- `UserPreferences` file writes are guarded by a lock. Safe for Streamlit concurrent access but not for multi-process deployments.

## Self-Annealing

### Adding a New Preset

1. Add the preset JSON to `execution/tone_presets.json` following the existing schema (see any existing preset for the full field list).
2. Verify it loads: `python -c "from execution.tone_profiles import get_preset; print(get_preset('New Name'))"`
3. Test writer instructions: `profile.to_writer_instructions()` should produce coherent instructions.
4. Test style overrides: `profile.to_style_overrides()` should produce valid scoring parameters.
5. Update the "Available Presets" table in this directive.

### Recalibrating an Inferred Profile

If an inferred profile doesn't match the target voice well:
1. Provide a longer sample (1500+ words for 0.95 confidence).
2. Manually edit the saved custom profile's fields via `UserPreferences.save_custom_profile()`.
3. Use `profile.merge_with_adjustments({...})` to apply specific overrides.

### Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| Style enforcer scores seem wrong for a tone | ToneProfile overrides not being passed | Verify `to_style_overrides()` is called and passed to the enforcer |
| Inferred profile has low confidence | Sample too short | Provide 500+ words for reliable inference |
| Learning isn't activating | Fewer than 5 feedback events | Continue logging feedback; threshold is 5 |
| Writer ignores tone instructions | Profile not passed to WriterAgent | Verify `to_writer_instructions()` output is in the prompt |
| Preset not found error | Case-sensitive name mismatch | Use exact casing from `list_presets()` |
| `user_preferences.json` corrupted | Invalid manual edits | Delete the file; defaults will regenerate on next load |

### When to Update This Directive

- New preset added to `tone_presets.json`
- New dimension added to `ToneProfile` model
- Learning algorithm changes (thresholds, clamping ranges, heuristics)
- New integration point added to the pipeline
- Edge case discovered during production use

## Dependencies

Required:
- `pydantic` -- ToneProfile data model
- LLM API access -- For tone inference (uses `config.models.DEFAULT_FAST_MODEL`)

Optional (for better accuracy):
- `lexicalrichness` -- VOCD scoring in inference
- `nltk` -- Better sentence tokenization in inference and learning
