"""
Tests for centralized configuration module.

Run with: pytest tests/test_config.py -v
"""

import os
import sys
from pathlib import Path
import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestPathConfig:
    """Test path configuration."""

    def test_project_root_exists(self):
        """PROJECT_ROOT should exist."""
        from execution.config import config
        assert config.paths.PROJECT_ROOT.exists()

    def test_project_root_is_absolute(self):
        """PROJECT_ROOT should be absolute path."""
        from execution.config import config
        assert config.paths.PROJECT_ROOT.is_absolute()

    def test_output_dir_created(self):
        """OUTPUT_DIR should be created on access."""
        from execution.config import config
        output_dir = config.paths.OUTPUT_DIR
        assert output_dir.exists()
        assert output_dir.is_dir()

    def test_temp_dir_created(self):
        """TEMP_DIR should be created on access."""
        from execution.config import config
        temp_dir = config.paths.TEMP_DIR
        assert temp_dir.exists()
        assert temp_dir.is_dir()

    def test_logs_dir_created(self):
        """LOGS_DIR should be created on access."""
        from execution.config import config
        logs_dir = config.paths.LOGS_DIR
        assert logs_dir.exists()
        assert logs_dir.is_dir()

    def test_path_types(self):
        """All paths should be Path objects."""
        from execution.config import config
        assert isinstance(config.paths.PROJECT_ROOT, Path)
        assert isinstance(config.paths.OUTPUT_DIR, Path)
        assert isinstance(config.paths.TEMP_DIR, Path)
        assert isinstance(config.paths.LOGS_DIR, Path)
        assert isinstance(config.paths.DIRECTIVES_DIR, Path)
        assert isinstance(config.paths.EXECUTION_DIR, Path)


class TestAPIConfig:
    """Test API configuration."""

    def test_has_key_method(self):
        """has_key should check API availability."""
        from execution.config import config

        # These should work regardless of actual key presence
        result = config.api.has_key("groq")
        assert isinstance(result, bool)

        result = config.api.has_key("unknown_provider")
        assert result is False

    def test_provider_aliases(self):
        """Provider aliases should work."""
        from execution.config import config

        # Test aliases resolve to same key
        if config.api.GOOGLE_API_KEY:
            assert config.api.has_key("google") == config.api.has_key("gemini")
        if config.api.ANTHROPIC_API_KEY:
            assert config.api.has_key("anthropic") == config.api.has_key("claude")
        if config.api.OPENAI_API_KEY:
            assert config.api.has_key("openai") == config.api.has_key("gpt")


class TestQualityConfig:
    """Test quality gate configuration."""

    def test_threshold_values(self):
        """Quality thresholds should be sensible."""
        from execution.config import config

        assert 0 <= config.quality.PASS_THRESHOLD <= 10
        assert 0 <= config.quality.ESCALATION_THRESHOLD <= 10
        assert config.quality.ESCALATION_THRESHOLD <= config.quality.PASS_THRESHOLD
        assert config.quality.MAX_ITERATIONS >= 1
        assert config.quality.MIN_VERIFIED_FACTS >= 0
        assert config.quality.MAX_UNVERIFIED_CLAIMS >= 0

    def test_verification_settings(self):
        """Verification settings should be boolean."""
        from execution.config import config

        assert isinstance(config.quality.FACT_VERIFICATION_REQUIRED, bool)
        assert isinstance(config.quality.MULTI_MODEL_REVIEW_REQUIRED, bool)
        assert isinstance(config.quality.HUMAN_REVIEW_REQUIRED_FOR_PUBLISH, bool)


class TestModelConfig:
    """Test model configuration."""

    def test_default_models_set(self):
        """Default models should be set."""
        from execution.config import config

        assert config.models.DEFAULT_WRITER_MODEL
        assert config.models.DEFAULT_CRITIC_MODEL
        assert config.models.DEFAULT_EDITOR_MODEL
        assert config.models.RESEARCH_MODEL_PRIMARY
        assert config.models.RESEARCH_MODEL_FALLBACK

    def test_panel_models_set(self):
        """Adversarial panel models should be set."""
        from execution.config import config

        assert config.models.ETHICS_REVIEWER_MODEL
        assert config.models.STRUCTURE_REVIEWER_MODEL
        assert config.models.FACT_REVIEWER_MODEL

    def test_tier_models(self):
        """Model tiers should have all providers."""
        from execution.config import config

        for tier in [config.models.TIER_SIMPLE, config.models.TIER_MEDIUM, config.models.TIER_COMPLEX]:
            assert "anthropic" in tier
            assert "openai" in tier
            assert "google" in tier


class TestVoiceConfig:
    """Test voice configuration."""

    def test_voice_types(self):
        """Voice types should be set."""
        from execution.config import config

        assert config.voice.VOICE_EXTERNAL
        assert config.voice.VOICE_INTERNAL
        assert config.voice.VOICE_EXTERNAL != config.voice.VOICE_INTERNAL

    def test_style_options(self):
        """Style options should be set."""
        from execution.config import config

        styles = [
            config.voice.STYLE_WSJ,
            config.voice.STYLE_BBC,
            config.voice.STYLE_CBC,
            config.voice.STYLE_CNN,
            config.voice.STYLE_MEDIUM,
        ]
        assert all(styles)
        assert len(set(styles)) == len(styles)  # All unique


class TestConfigValidation:
    """Test configuration validation."""

    def test_validate_returns_dict(self):
        """validate() should return a dict."""
        from execution.config import config

        result = config.validate()
        assert isinstance(result, dict)
        assert "valid" in result
        assert "issues" in result
        assert "warnings" in result
        assert "environment" in result

    def test_validate_valid_key(self):
        """valid key should be boolean."""
        from execution.config import config

        result = config.validate()
        assert isinstance(result["valid"], bool)

    def test_validate_issues_and_warnings_are_lists(self):
        """issues and warnings should be lists."""
        from execution.config import config

        result = config.validate()
        assert isinstance(result["issues"], list)
        assert isinstance(result["warnings"], list)


class TestConfigSingleton:
    """Test configuration singleton behavior."""

    def test_singleton_instance(self):
        """config should be a singleton."""
        from execution.config import config, get_config

        assert config is get_config()

    def test_convenience_exports(self):
        """Convenience exports should match config."""
        from execution.config import config, PROJECT_ROOT, OUTPUT_DIR, TEMP_DIR

        assert PROJECT_ROOT == config.paths.PROJECT_ROOT
        assert OUTPUT_DIR == config.paths.OUTPUT_DIR
        assert TEMP_DIR == config.paths.TEMP_DIR


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
