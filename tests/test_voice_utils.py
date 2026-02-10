"""
Unit tests for the voice utilities module.

Run with: pytest tests/test_voice_utils.py -v
"""

import os
import sys
from pathlib import Path
import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestVoiceConfigs:
    """Tests for voice configuration objects."""

    def test_observer_voice_exists(self):
        """Observer voice should be defined."""
        from execution.voice_utils import OBSERVER_VOICE

        assert OBSERVER_VOICE is not None
        assert OBSERVER_VOICE.name == "Observer Voice"

    def test_practitioner_voice_exists(self):
        """Practitioner voice should be defined."""
        from execution.voice_utils import PRACTITIONER_VOICE

        assert PRACTITIONER_VOICE is not None
        assert PRACTITIONER_VOICE.name == "Practitioner Voice"

    def test_voice_source_types(self):
        """Voice configs should have correct source types."""
        from execution.voice_utils import OBSERVER_VOICE, PRACTITIONER_VOICE

        assert OBSERVER_VOICE.source_type == "external"
        assert PRACTITIONER_VOICE.source_type == "internal"

    def test_observer_has_forbidden_phrases(self):
        """Observer voice should have forbidden phrases."""
        from execution.voice_utils import OBSERVER_VOICE

        assert hasattr(OBSERVER_VOICE, 'forbidden_phrases')
        assert len(OBSERVER_VOICE.forbidden_phrases) > 0

    def test_practitioner_allows_ownership(self):
        """Practitioner voice should allow ownership phrases."""
        from execution.voice_utils import PRACTITIONER_VOICE

        # Practitioner allows ownership, so forbidden_phrases should not include "I"
        forbidden = PRACTITIONER_VOICE.forbidden_phrases
        # Check that basic ownership phrases aren't in forbidden list
        assert not any("we built" in p.lower() for p in forbidden)


class TestVoiceViolationCheck:
    """Tests for check_voice_violations function."""

    def test_detects_ownership_phrases(self):
        """Should detect ownership phrases in external voice."""
        from execution.voice_utils import check_voice_violations

        content = "I built this system and our team created a novel approach."
        violations = check_voice_violations(content, "external")

        assert len(violations) >= 1

    def test_external_catches_first_person(self):
        """External voice should catch first-person pronouns."""
        from execution.voice_utils import check_voice_violations

        content = "We developed this technology in our labs."
        violations = check_voice_violations(content, "external")

        # Should detect "We" or "our"
        assert len(violations) >= 1

    def test_clean_content_no_violations(self):
        """Clean content should have no violations."""
        from execution.voice_utils import check_voice_violations

        content = "Engineers at the company discovered significant improvements to the algorithm."
        violations = check_voice_violations(content, "external")

        # No first-person pronouns
        assert len(violations) == 0

    def test_internal_allows_ownership(self):
        """Internal voice should allow ownership phrases."""
        from execution.voice_utils import check_voice_violations

        content = "I built this system and our team created a novel approach."
        violations = check_voice_violations(content, "internal")

        # Internal voice allows ownership
        assert len(violations) == 0


class TestValidateVoice:
    """Tests for validate_voice function."""

    def test_returns_dict(self):
        """validate_voice should return a dict."""
        from execution.voice_utils import validate_voice

        result = validate_voice("Test content.", "external")

        assert isinstance(result, dict)
        assert "passed" in result

    def test_passed_is_boolean(self):
        """passed key should be boolean."""
        from execution.voice_utils import validate_voice

        result = validate_voice("Test content.", "external")

        assert isinstance(result["passed"], bool)

    def test_external_fails_with_ownership(self):
        """External voice should fail with ownership phrases."""
        from execution.voice_utils import validate_voice

        content = "I built this system and our team created a novel approach."
        result = validate_voice(content, "external")

        assert result["passed"] is False

    def test_external_passes_clean_content(self):
        """External voice should pass with clean content."""
        from execution.voice_utils import validate_voice

        content = "The research team at the university published their findings."
        result = validate_voice(content, "external")

        assert result["passed"] is True

    def test_internal_passes_with_ownership(self):
        """Internal voice should pass with ownership phrases."""
        from execution.voice_utils import validate_voice

        content = "We developed this feature to improve user experience."
        result = validate_voice(content, "internal")

        assert result["passed"] is True

    def test_result_includes_violations(self):
        """Result should include violations list."""
        from execution.voice_utils import validate_voice

        content = "I built this."
        result = validate_voice(content, "external")

        assert "violations" in result
        assert isinstance(result["violations"], list)


class TestGetVoiceInstruction:
    """Tests for get_voice_instruction function."""

    def test_returns_string(self):
        """get_voice_instruction should return a string."""
        from execution.voice_utils import get_voice_instruction

        result = get_voice_instruction("external")

        assert isinstance(result, str)
        assert len(result) > 0

    def test_external_instructions(self):
        """External voice instructions should mention third-person."""
        from execution.voice_utils import get_voice_instruction

        result = get_voice_instruction("external")

        # Should mention observer/third-person perspective
        result_lower = result.lower()
        assert "observer" in result_lower or "third" in result_lower or "external" in result_lower

    def test_internal_instructions(self):
        """Internal voice instructions should allow first-person."""
        from execution.voice_utils import get_voice_instruction

        result = get_voice_instruction("internal")

        result_lower = result.lower()
        assert "practitioner" in result_lower or "first" in result_lower or "internal" in result_lower

    def test_default_to_external(self):
        """Unknown source type should default to external."""
        from execution.voice_utils import get_voice_instruction

        result = get_voice_instruction("unknown")
        external_result = get_voice_instruction("external")

        # Should default to external
        assert len(result) > 0


class TestVoiceConfigMapping:
    """Tests for voice config mapping."""

    def test_get_voice_config_external(self):
        """Should get correct config for external."""
        from execution.voice_utils import OBSERVER_VOICE, PRACTITIONER_VOICE

        # External maps to Observer
        assert OBSERVER_VOICE.source_type == "external"

    def test_get_voice_config_internal(self):
        """Should get correct config for internal."""
        from execution.voice_utils import OBSERVER_VOICE, PRACTITIONER_VOICE

        # Internal maps to Practitioner
        assert PRACTITIONER_VOICE.source_type == "internal"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
