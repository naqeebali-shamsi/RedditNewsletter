# Testing Patterns

**Analysis Date:** 2026-02-09

## Test Framework

**Runner:**
- pytest (no version constraint in requirements.txt, uses 7.x+)
- Config: No `pytest.ini`, `setup.cfg`, or `pyproject.toml` file for pytest in root
- Tests discoverable by standard pytest pattern: `test_*.py` and `*_test.py`

**Assertion Library:**
- pytest built-in assertions: `assert condition`, `assert x == y`
- Exception assertions: `pytest.raises(ValidationError)`

**Run Commands:**
```bash
pytest tests/test_config.py -v              # Run single test file with verbose output
pytest tests/test_provenance.py -v          # Run provenance tests
pytest                                      # Run all tests in tests/ and test_*.py
python tests/test_config.py                 # Test files also runnable as scripts
python test_integration.py                  # Standalone integration test runner
```

## Test File Organization

**Location:**
- Primary: `tests/` directory (8 test files)
  - `tests/test_config.py`
  - `tests/test_provenance.py`
  - `tests/test_tone_profiles.py`
  - `tests/test_tone_inference.py`
  - `tests/test_tone_integration.py`
  - `tests/test_user_preferences.py`
  - `tests/test_voice_utils.py`
  - `tests/test_hardening.py`

- Root level: `test_*.py` files (5 test files)
  - `test_integration.py` (end-to-end smoke tests)
  - `test_adversarial_review.py` (adversarial panel testing)
  - `test_dashboard_validation.py` (UI validation)
  - `test_fact_agent.py` (fact verification agent tests)
  - Mixed location: `execution/test_auxiliary_agents.py` (agent implementation tests)

**Naming:**
- Files: `test_<module>.py` (e.g., `test_config.py`, `test_provenance.py`)
- Classes: `Test<Feature>` (e.g., `TestPathConfig`, `TestConfigValidation`)
- Methods: `test_<behavior>` (e.g., `test_project_root_exists`, `test_empty_claims_fails_quality_gate`)

**Structure:**
```
tests/
├── __init__.py
├── test_config.py           # Configuration validation
├── test_provenance.py       # Provenance tracking and manifests
├── test_tone_profiles.py    # Tone profile data models
├── test_tone_inference.py   # Tone inference from samples
├── test_tone_integration.py # Tone system integration
├── test_user_preferences.py # User preference persistence
├── test_voice_utils.py      # Voice validation utilities
└── test_hardening.py        # Security and hardening validation
```

## Test Structure

**Suite Organization:**
Classes group related tests by feature/module:
```python
class TestPathConfig:
    """Test path configuration."""
    def test_project_root_exists(self):
        """PROJECT_ROOT should exist."""
    def test_project_root_is_absolute(self):
        """PROJECT_ROOT should be absolute path."""

class TestAPIConfig:
    """Test API configuration."""
    def test_has_key_method(self):
        """has_key should check API availability."""
```

**Patterns:**

1. **Setup/teardown via fixtures:**
```python
@pytest.fixture
def expert_profile():
    """Return the Expert Pragmatist preset."""
    return get_preset("Expert Pragmatist")

@pytest.fixture
def minimal_profile_data():
    """Minimal valid data for ToneProfile construction."""
    return {
        "name": "Test Profile",
        "description": "A test profile for unit tests.",
        "formality_level": 0.5,
        # ...
    }
```

2. **Explicit docstrings on all test methods:**
```python
def test_empty_claims_fails_quality_gate(self):
    """An article with zero extractable claims must NOT pass the gate."""
    from execution.agents.fact_verification_agent import FactVerificationReport

    report = FactVerificationReport(
        claims=[],
        results=[],
        verified_count=0,
        unverified_count=0,
        false_count=0,
        passes_quality_gate=False,
        summary="No claims extracted"
    )
    assert report.passes_quality_gate is False
```

3. **Isolated imports within tests:**
```python
def test_config_validation(self):
    """Validation should detect issues."""
    from execution.config import config
    result = config.validate()
    assert isinstance(result["valid"], bool)
```

4. **System path injection for root discovery:**
```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
```

## Mocking

**Framework:** Built-in `unittest.mock` via pytest fixtures (no external mocking library detected)

**Patterns:**

1. **Monkey-patching for agent testing:**
```python
def test_verify_article_empty_claims_returns_fail(self):
    """verify_article with no claims should fail-closed."""
    agent = FactVerificationAgent.__new__(FactVerificationAgent)
    agent.providers = [("mock", None)]
    agent.max_unverified = 1
    agent.min_verified = 3

    report = FactVerificationReport(...)
    assert report.passes_quality_gate is False
```

2. **Fixture-based temporary file creation:**
```python
@pytest.fixture
def custom_presets_file(tmp_path):
    """Create a temporary presets JSON file."""
    data = [{"name": "Custom Test", ...}]
    path = tmp_path / "test_presets.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return str(path)
```

3. **No complex mocking of LLM providers** - agents tested with actual configs where available

**What to Mock:**
- File system operations: Use `tmp_path` pytest fixture
- Agent initialization: Monkey-patch attributes for unit isolation
- Temporary files: Use pytest's `tmp_path` fixture

**What NOT to Mock:**
- LLM provider calls (use integration tests instead)
- Configuration loading (test actual config module)
- Pydantic models (test actual validation)
- Exception hierarchy (test actual exceptions)

## Fixtures and Factories

**Test Data Fixtures:**
- `expert_profile`: Returns pre-loaded tone preset
- `minimal_profile_data`: Dict with minimal required fields for ToneProfile
- `custom_presets_file`: Temporary JSON file with test presets
- `tmp_path`: pytest built-in for temporary directories

**Pattern from test_tone_profiles.py:**
```python
@pytest.fixture
def minimal_profile_data():
    return {
        "name": "Test Profile",
        "description": "A test profile for unit tests.",
        "formality_level": 0.5,
        "technical_depth": 0.5,
        "personality": "conversational",
        "hook_style": "problem_statement",
        "cta_style": "none",
    }

class TestToneProfileModel:
    def test_create_with_minimal_fields(self, minimal_profile_data):
        profile = ToneProfile(**minimal_profile_data)
        assert profile.name == "Test Profile"
```

**Location:**
- Fixtures defined in test files (no conftest.py for shared fixtures)
- Each test file is self-contained
- No factory classes; simple dict-based test data

## Coverage

**Requirements:** Not enforced (no coverage threshold in config)

**View Coverage:**
- No coverage tool configured or required
- No coverage reports in CI/CD

## Test Types

**Unit Tests:**
- Scope: Individual functions/classes in isolation
- Approach: Pytest with fixtures, focused on single responsibility
- Examples:
  - `test_config.py`: Configuration object creation and validation
  - `test_provenance.py`: ContentProvenance dataclass, ProvenanceTracker methods
  - `test_tone_profiles.py`: ToneProfile model validation and preset loading
  - `test_voice_utils.py`: Voice validation utility functions
- Pattern: One test class per module, one test method per behavior

**Integration Tests:**
- Scope: Multiple components working together
- Approach: Standalone Python scripts, can be run independently
- Location: Root level `test_*.py` files
- Examples:
  - `test_integration.py`: Full pipeline imports, agent initialization, module dependencies
  - `test_hardening.py`: Security validation, HTML sanitization, fail-closed verification
  - `test_adversarial_review.py`: Adversarial panel workflow
  - `test_fact_agent.py`: Fact verification agent with actual configuration
- Pattern: Tests in classes, also runnable with `python test_integration.py`

**E2E Tests:**
- Framework: Not used (no Selenium, Playwright, or similar)
- Stream: Not detected in codebase

## Common Patterns

**Async Testing:**
Not explicitly tested; async functions in codebase (`call_llm_async`, `infer_from_text`) not covered in test files. Async testing would require:
```python
@pytest.mark.asyncio
async def test_call_llm_async():
    agent = WriterAgent()
    result = await agent.call_llm_async("test prompt")
    assert isinstance(result, str)
```

**Error Testing:**
```python
def test_formality_level_validation_too_high(self, minimal_profile_data):
    """Formality level > 1.0 should raise ValidationError."""
    minimal_profile_data["formality_level"] = 1.5
    with pytest.raises(ValidationError):
        ToneProfile(**minimal_profile_data)

def test_empty_claims_fails_quality_gate(self):
    """Empty claims list should result in passes_quality_gate=False."""
    report = FactVerificationReport(
        claims=[],
        results=[],
        verified_count=0,
        unverified_count=0,
        false_count=0,
        passes_quality_gate=False,
        summary="No claims extracted"
    )
    assert report.passes_quality_gate is False
```

**Boundary Value Testing:**
```python
def test_threshold_values(self):
    """Quality thresholds should be sensible."""
    from execution.config import config
    assert 0 <= config.quality.PASS_THRESHOLD <= 10
    assert 0 <= config.quality.ESCALATION_THRESHOLD <= 10
    assert config.quality.ESCALATION_THRESHOLD <= config.quality.PASS_THRESHOLD
```

**Data Validation Testing:**
```python
def test_provider_aliases(self):
    """Provider aliases should work."""
    from execution.config import config
    if config.api.GOOGLE_API_KEY:
        assert config.api.has_key("google") == config.api.has_key("gemini")
```

## Fail-Closed Verification Testing

**Key Pattern from test_hardening.py:**

Tests verify that the system fails safely when assertions cannot be made:

```python
def test_empty_claims_fails_quality_gate(self):
    """An article with zero extractable claims must NOT pass the gate."""
    report = FactVerificationReport(
        claims=[],
        results=[],
        verified_count=0,
        unverified_count=0,
        false_count=0,
        passes_quality_gate=False,
        summary="No claims extracted"
    )
    assert report.passes_quality_gate is False
```

This is intentional: When fact verification cannot extract verifiable claims, the article fails the quality gate. The default state is rejection unless proven safe.

## Test Execution Patterns

**Per-file runners:**
All test files support both pytest and direct Python execution:
```python
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

**Integration test with detailed reporting:**
`test_integration.py` implements its own test framework with formatted output:
```python
def test_imports():
    """Test all critical imports work."""
    print("\n" + "="*60)
    print("TEST: Critical Imports")
    print("="*60)

    tests = [
        ("Config module", try_import_config()),
        ("Agents", try_import_agents()),
        # ...
    ]

    passed = sum(1 for _, ok, _ in tests if ok)
    print(f"\nSummary: {passed}/{len(tests)} imports successful")
```

---

*Testing analysis: 2026-02-09*
