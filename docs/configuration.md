# GhostWriter Configuration Guide

## Overview

GhostWriter uses a centralized configuration system located in `execution/config.py`. All configuration is managed through dataclasses with environment variable support.

## Configuration Sections

### PathConfig

Controls file system paths. All paths derive from PROJECT_ROOT.

| Property | Default | Env Variable | Description |
|----------|---------|--------------|-------------|
| PROJECT_ROOT | Auto-detected | - | Project root directory |
| OUTPUT_DIR | `{PROJECT_ROOT}/drafts` | `GHOSTWRITER_OUTPUT_DIR` | Article output directory |
| TEMP_DIR | `{PROJECT_ROOT}/.tmp` | `GHOSTWRITER_TEMP_DIR` | Temporary files |
| LOGS_DIR | `{PROJECT_ROOT}/logs` | - | Log files |
| DIRECTIVES_DIR | `{PROJECT_ROOT}/directives` | - | SOP directives |
| EXECUTION_DIR | `{PROJECT_ROOT}/execution` | - | Python scripts |

### APIConfig

API keys loaded from environment variables.

| Property | Env Variable | Required For |
|----------|--------------|--------------|
| GROQ_API_KEY | `GROQ_API_KEY` | Fast inference (Writer, Critic) |
| GOOGLE_API_KEY | `GOOGLE_API_KEY` | Gemini models, web search |
| PERPLEXITY_API_KEY | `PERPLEXITY_API_KEY` | Perplexity research |
| ANTHROPIC_API_KEY | `ANTHROPIC_API_KEY` | Claude models |
| OPENAI_API_KEY | `OPENAI_API_KEY` | GPT models |
| GMAIL_CREDENTIALS_PATH | `GMAIL_CREDENTIALS_PATH` | Gmail integration |

### QualityConfig

Quality gate thresholds.

| Property | Default | Description |
|----------|---------|-------------|
| PASS_THRESHOLD | 7.0 | Minimum score to pass quality gate |
| ESCALATION_THRESHOLD | 6.0 | Score below which escalation is considered |
| KILL_PHRASE_MAX_SCORE | 4.0 | Max score when kill phrases detected |
| MAX_ITERATIONS | 3 | Maximum revision iterations |
| MAX_REVISION_ATTEMPTS | 2 | Max attempts per revision |
| FACT_VERIFICATION_REQUIRED | True | Whether fact verification is mandatory |
| MULTI_MODEL_REVIEW_REQUIRED | True | Whether multi-model review is required |
| HUMAN_REVIEW_REQUIRED_FOR_PUBLISH | True | Whether HITL is required |
| MIN_VERIFIED_FACTS | 3 | Minimum verified facts required |
| MAX_UNVERIFIED_CLAIMS | 1 | Maximum allowed unverified claims |

### ModelConfig

Model selection for different tasks.

| Property | Default | Description |
|----------|---------|-------------|
| DEFAULT_WRITER_MODEL | llama-3.3-70b-versatile | Primary writing model |
| DEFAULT_CRITIC_MODEL | llama-3.3-70b-versatile | Critique model |
| DEFAULT_EDITOR_MODEL | llama-3.3-70b-versatile | Editing model |
| RESEARCH_MODEL_PRIMARY | gemini-2.0-flash | Primary research model |
| RESEARCH_MODEL_FALLBACK | sonar-pro | Fallback research model |
| ETHICS_REVIEWER_MODEL | claude-sonnet-4 | Ethics review (adversarial panel) |
| STRUCTURE_REVIEWER_MODEL | gpt-4o | Structure review (adversarial panel) |
| FACT_REVIEWER_MODEL | gemini-2.0-flash | Fact verification model |

### VoiceConfig

Voice and style configuration.

| Property | Value | Description |
|----------|-------|-------------|
| VOICE_EXTERNAL | "Journalist Observer" | For external source content |
| VOICE_INTERNAL | "Practitioner Owner" | For internal/own content |
| STYLE_WSJ | "wsj" | WSJ Four Showstoppers style |
| STYLE_BBC | "bbc" | BBC impartial style |
| STYLE_CBC | "cbc" | CBC conversational style |
| STYLE_CNN | "cnn" | CNN Facts First style |
| STYLE_MEDIUM | "medium" | Medium hook-driven style |

## Usage

### Importing Configuration

```python
from execution.config import config, OUTPUT_DIR, PROJECT_ROOT

# Access paths
output_path = config.paths.OUTPUT_DIR / "article.md"

# Check API availability
if config.api.has_key("gemini"):
    # Use Gemini
    pass

# Access quality settings
if config.quality.FACT_VERIFICATION_REQUIRED:
    # Run fact verification
    pass
```

### Validating Configuration

```python
from execution.config import validate_config

# Returns True if valid, prints issues/warnings
is_valid = validate_config()

# Or get detailed result
result = config.validate()
print(result["issues"])
print(result["warnings"])
```

### Environment Variables

Create a `.env` file in the project root:

```env
# Required
GROQ_API_KEY=your_groq_key
GOOGLE_API_KEY=your_google_key

# Optional
GHOSTWRITER_OUTPUT_DIR=/custom/output/path
GHOSTWRITER_TEMP_DIR=/custom/temp/path
GHOSTWRITER_ENV=production

# For multi-model review
ANTHROPIC_API_KEY=your_anthropic_key
OPENAI_API_KEY=your_openai_key
PERPLEXITY_API_KEY=your_perplexity_key
```

## Architecture

```
GhostWriterConfig (Singleton)
├── paths: PathConfig
│   ├── PROJECT_ROOT
│   ├── OUTPUT_DIR
│   ├── TEMP_DIR
│   └── ...
├── api: APIConfig
│   ├── GROQ_API_KEY
│   ├── GOOGLE_API_KEY
│   └── ...
├── quality: QualityConfig
│   ├── PASS_THRESHOLD
│   ├── MAX_ITERATIONS
│   └── ...
├── models: ModelConfig
│   ├── DEFAULT_WRITER_MODEL
│   └── ...
└── voice: VoiceConfig
    ├── VOICE_EXTERNAL
    └── ...
```

The configuration is a singleton instance, ensuring consistent settings across the application.
