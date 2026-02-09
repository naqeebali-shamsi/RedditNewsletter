# Coding Conventions

**Analysis Date:** 2026-02-09

## Naming Patterns

**Files:**
- Snake case with descriptive names: `base_agent.py`, `fact_verification_agent.py`, `quality_gate.py`
- Agents suffixed with `_agent.py`: `writer.py`, `critic.py`, `editor.py`
- Test files prefixed with `test_`: `test_config.py`, `test_provenance.py`, `test_hardening.py`
- Source module files: `reddit_source.py`, `rss_source.py`, `hackernews_source.py`
- Utility modules grouped in `utils/`: `json_parser.py`, `datetime_utils.py`, `health.py`

**Classes:**
- PascalCase: `BaseAgent`, `WriterAgent`, `AdversarialPanelAgent`, `StyleEnforcerAgent`
- Dataclasses for lightweight data models: `ExpertCritique`, `PanelVerdict`, `StyleScore`, `ArticleState`
- Agent classes inherit from `BaseAgent`: All specialized agents extend the base class
- Exception classes inherit from base exception: `LLMError`, `ProviderError`, `QualityGateError`

**Functions:**
- Snake case: `_is_transient()`, `_build_transient_types()`, `call_llm()`, `verify_article_facts()`
- Private functions prefixed with single underscore: `_get_api_key()`, `_setup_client()`, `_validate_response()`
- Async functions prefixed with `async def`: `call_llm_async()`, `generate_async()`, `infer_from_text()`
- Static methods used for utilities: `@staticmethod` decorators on `_get_key_for_provider()`, `_log_retry()`

**Variables:**
- Snake case for all variables: `api_key`, `model_name`, `provider`, `tone_profile`
- Constants in ALL_CAPS: `PASS_THRESHOLD`, `MAX_ITERATIONS`, `KILL_PHRASES`, `FORBIDDEN_PHRASES`
- Private module variables with leading underscore: `_TRANSIENT_TYPES`, `_TRANSIENT_SUBSTRINGS`, `_KILL_PHRASE_META`

**Enums and Types:**
- PascalCase: `VerificationStatusEnum`, `VerificationStatus`
- Enum values: `PENDING`, `IN_PROGRESS`, `PASSED`, `FAILED`, `ESCALATED`
- Type hints throughout (Python 3.10+): `List[Dict]`, `Optional[str]`, `Dict[str, Any]`

## Code Style

**Formatting:**
- No explicit linter/formatter configured in repo (no `.eslintrc`, `.prettierrc`, `ruff.toml`, etc.)
- Convention: 4-space indentation (Python standard)
- Line length: ~80-100 characters (observed in code, not enforced)
- Imports organized by category (stdlib, third-party, local)

**Type Hints:**
- Required on function signatures: `def call_llm(self, prompt: str, system_instruction: str = None) -> str:`
- Used on class attributes via Pydantic models: `from typing import Optional, List, Dict, Any`
- Pydantic BaseModel for configuration and state: `from pydantic import BaseModel, Field`
- Optional types preferred over `None`: `Optional[str]`, `Optional[Dict]`

**Docstrings:**
- Module-level docstrings required: Each `.py` file starts with triple-quoted description
- Function docstrings for public methods: Shows args, return type, and raises section
- Class docstrings: Describe purpose and responsibility
- Format: Google/NumPy style (Args, Returns, Raises sections)
- Examples from `base_agent.py`:
```python
def call_llm(self, prompt, system_instruction=None, temperature=0.7,
             system_prompt=None):
    """Standardized call to the underlying LLM.

    Args:
        prompt: The user/dynamic prompt content.
        system_instruction: Additional instructions appended to the
            default role/persona system prompt.
        temperature: Sampling temperature.
        system_prompt: Optional fully-formed static system prompt.
            When provided, this replaces the auto-generated
            role/persona preamble entirely.

    Raises:
        LLMNotConfiguredError: No provider configured.
        ProviderError: The configured provider failed after retries.
    """
```

## Import Organization

**Order:**
1. Standard library imports (`os`, `sys`, `time`, `json`, `argparse`)
2. Third-party imports (`pydantic`, `tenacity`, `google.genai`, `openai`)
3. Local/relative imports (`from execution.config import config`, `from .base_agent import BaseAgent`)

**Path Aliases:**
- Project-root relative imports preferred: `from execution.config import config`
- Relative imports within modules: `from .base_agent import BaseAgent`
- No complex path aliasing; direct module references

**Examples from codebase:**
```python
# execution/agents/writer.py
from .base_agent import BaseAgent
from execution.config import config

# execution/quality_gate.py
from execution.agents.adversarial_panel import AdversarialPanelAgent, PanelVerdict
from execution.agents.writer import WriterAgent
from execution.agents.base_agent import LLMError
from execution.config import config
```

## Error Handling

**Patterns:**
- Typed exception hierarchy: Base `LLMError`, specialized subclasses `ProviderError`, `AllProvidersFailedError`, `LLMNotConfiguredError`
- Structured exceptions with context: `ProviderError(provider_name, message_string, original_error=e)`
- Transient error detection: `_is_transient()` checks typed exceptions first, falls back to string matching
- Fail-closed verification: Empty claims fail quality gate (no assumptions, explicit fail status)
- All exceptions should be raised, not returned as error strings: `raise ProviderError(...)` not `return "error: ..."`

**Retry Strategy:**
- Uses `tenacity` library for exponential backoff with jitter
- Configuration in `base_agent.py`:
```python
retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=30) + wait_random(0, 2),
    retry=retry_if_exception(_is_transient),
    before_sleep=self._log_retry,
    reraise=True,
)
```
- Transient vs permanent errors distinguished by type (OpenAI, Google, Groq exception classes)

**Response Validation:**
- All LLM responses validated before returning: `_validate_response(response, provider_name)`
- Checks for non-empty, non-trivial content (min 20 chars)
- Raises `ProviderError` on invalid response (not returning empty/error string)

## Logging

**Framework:** `print()` for console output (no structured logging framework detected)

**Patterns:**
- Logging callback for retry attempts: `_log_retry()` called by tenacity before sleep
- Format: `[retry] Attempt N failed (exception), retrying...`
- Quality gate uses `_log()` method for verbose output
- No centralized logger; direct print statements throughout

**Example from base_agent.py:**
```python
@staticmethod
def _log_retry(retry_state):
    """Logging callback invoked before each retry sleep."""
    exc = retry_state.outcome.exception()
    attempt = retry_state.attempt_number
    print(f"[retry] Attempt {attempt} failed ({exc}), retrying...")
```

## Comments

**When to Comment:**
- LLM provider selection logic: Long decision tree with conditionals documented inline
- Complex exception handling: Explanation of why specific exception types are caught
- Investigation summaries: Technical decisions with trade-off analysis (e.g., LiteLLM investigation in `base_agent.py`)
- Configuration edge cases: Why certain settings are required or optional

**Commented Decision Log:**
- `base_agent.py` lines 6-27: LiteLLM investigation verdict with dealbreakers and future considerations
- Pattern: Problem statement, reasoning, dealbreakers, future criteria

## Function Design

**Size:**
- Typically 20-80 lines for agent methods
- Larger methods broken into helper methods prefixed with underscore: `_get_api_key()`, `_setup_client()`, `_validate_response()`
- Retry logic encapsulated in `_call_provider_with_retry()`

**Parameters:**
- Limited to 3-5 required parameters; rest use defaults or kwargs
- Configuration passed via constructor (agents initialized with role/persona)
- Keyword-only args for optional params: `temperature=0.7`, `system_prompt=None`

**Return Values:**
- Explicit return type hints required: `-> str`, `-> Dict`, `-> Optional[List]`
- Dataclass returns for complex results: `PanelVerdict`, `StyleScore`, `FactVerificationReport`
- Async variants return same types as sync (no special async-only returns)

## Module Design

**Exports:**
- Agent modules export agent class and supporting dataclasses: `from execution.agents import WriterAgent, CriticAgent`
- `execution/agents/__init__.py` defines `__all__` with public exports:
```python
__all__ = [
    'BaseAgent',
    'EditorAgent',
    'CriticAgent',
    'WriterAgent',
    'AdversarialPanelAgent',
    'PanelVerdict',
    'ExpertCritique',
    # ...
]
```

**Barrel Files:**
- Central package export in `execution/agents/__init__.py`
- Consolidates all agent imports for ease of use: `from execution.agents import WriterAgent`
- Not used for utility modules (utils/ imports are specific)

**Configuration Pattern:**
- Single config module: `execution/config.py`
- Pydantic-based nested structure: `config.paths`, `config.api`, `config.quality`, `config.models`, `config.voice`
- Singleton pattern: `config = GhostWriterConfig()` exported for module-wide use
- Validation method: `config.validate()` returns dict with issues/warnings
- Environment variable support: `.env` file loading via `pydantic_settings.BaseSettings`

## Agent Architecture

**Base Class Pattern:**
All agents inherit from `BaseAgent` in `execution/agents/base_agent.py`:
- Constructor signature: `__init__(self, role: str, persona: str, model: str = None, provider: str = None)`
- Core method: `call_llm(prompt, system_instruction=None, temperature=0.7, system_prompt=None) -> str`
- Async variant: `call_llm_async(prompt, **kwargs) -> str` uses `run_in_executor` for thread-based concurrency
- JSON parsing: `generate(prompt, expect_json=True, system_prompt=None)` auto-parses JSON responses

**Specialized Agents:**
- `WriterAgent`: Uses `FORBIDDEN_PHRASES`, `HOOK_PATTERNS`, `CTA_PATTERNS` for content generation
- `CriticAgent`: Uses `KILL_PHRASES` with compiled regex for detection
- `EditorAgent`: Implements `QUALITY_CHECKLIST` for review
- `StyleEnforcerAgent`: Scores across 5 dimensions (burstiness, lexical diversity, AI tells, authenticity, framework compliance)
- `AdversarialPanelAgent`: Multi-expert panel with dataclass-based verdicts

**Agent Initialization Pattern:**
```python
class SpecializedAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            role="Role Name",
            persona="""Detailed persona instructions...
            With formatting and style guidelines.""",
            model=config.models.DEFAULT_WRITER_MODEL
        )
```

## Pydantic Models

**Configuration Models:**
- `PathConfig`: Computed fields for paths (OUTPUT_DIR, TEMP_DIR, LOGS_DIR, etc.)
- `APIConfig`: Settings-based environment variable loading
- `QualityConfig`: Threshold values and escalation rules
- `ModelConfig`: Model selection and cost tiers
- `VoiceConfig`: Voice types and publication styles
- `GhostWriterConfig`: Root config combining all sections

**State Models:**
- `ArticleState`: Full pipeline state with 40+ fields (content, metadata, verification, quality scores)
- Dict-style access support: `state["key"]` and `state["key"] = value` for backward compatibility

**Data Models:**
- `ExpertCritique`: Expert's single critique (name, agency, score, verdict, failures, fixes)
- `PanelVerdict`: Aggregated panel verdict with computed fields
- `StyleScore`: 5-dimensional style scoring with individual metrics
- `FactVerificationReport`: Verification results with claims and pass/fail status

All Pydantic models use:
- `model_config` for settings: `arbitrary_types_allowed = True`, `extra = "allow"`
- `Field()` for defaults and documentation
- `@computed_field` for derived values (paths computed from env vars)

---

*Conventions analysis: 2026-02-09*
