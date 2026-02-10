"""
Unit tests for the provenance tracking module.

Run with: pytest tests/test_provenance.py -v
"""

import os
import sys
from pathlib import Path
import pytest
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestContentProvenance:
    """Tests for ContentProvenance dataclass."""

    def test_content_provenance_creation(self):
        """ContentProvenance should be creatable."""
        from execution.provenance import ContentProvenance
        from datetime import datetime, timezone

        provenance = ContentProvenance(
            content_id="gw-test-123",
            content_hash="abc123",
            created_at=datetime.now(timezone.utc).isoformat(),
            topic="Test Topic",
            source_type="external"
        )

        assert provenance.content_id == "gw-test-123"
        assert provenance.source_type == "external"

    def test_to_dict_method(self):
        """ContentProvenance should convert to dict."""
        from execution.provenance import ContentProvenance
        from datetime import datetime, timezone

        provenance = ContentProvenance(
            content_id="gw-test-123",
            content_hash="abc123",
            created_at=datetime.now(timezone.utc).isoformat(),
            topic="Test Topic",
            source_type="external"
        )

        result = provenance.to_dict()

        assert isinstance(result, dict)
        assert result["content_id"] == "gw-test-123"
        assert result["source_type"] == "external"


class TestProvenanceTracker:
    """Tests for ProvenanceTracker class."""

    def test_tracker_initialization(self):
        """Tracker should initialize properly."""
        from execution.provenance import ProvenanceTracker

        tracker = ProvenanceTracker()

        assert tracker is not None

    def test_start_tracking(self):
        """start_tracking should begin tracking session."""
        from execution.provenance import ProvenanceTracker

        tracker = ProvenanceTracker()
        tracker.start_tracking(
            topic="Test Topic",
            source_type="external",
            platform="medium"
        )

        # Tracker stores provenance object
        assert tracker._provenance is not None
        assert tracker._provenance.topic == "Test Topic"
        assert tracker._provenance.source_type == "external"

    def test_record_research(self):
        """record_research should add action."""
        from execution.provenance import ProvenanceTracker

        tracker = ProvenanceTracker()
        tracker.start_tracking(topic="Test", source_type="external")

        initial_actions = len(tracker._provenance.actions)
        tracker.record_research("TestAgent", model="test-model", facts_found=3)

        assert len(tracker._provenance.actions) > initial_actions

    def test_record_generation(self):
        """record_generation should add action."""
        from execution.provenance import ProvenanceTracker

        tracker = ProvenanceTracker()
        tracker.start_tracking(topic="Test", source_type="external")

        initial_actions = len(tracker._provenance.actions)
        tracker.record_generation("WriterAgent", word_count=500)

        assert len(tracker._provenance.actions) > initial_actions

    def test_record_verification(self):
        """record_verification should add action."""
        from execution.provenance import ProvenanceTracker

        tracker = ProvenanceTracker()
        tracker.start_tracking(topic="Test", source_type="external")

        initial_actions = len(tracker._provenance.actions)
        tracker.record_verification(passed=True, claims_verified=5)

        assert len(tracker._provenance.actions) > initial_actions

    def test_record_review(self):
        """record_review should add action."""
        from execution.provenance import ProvenanceTracker

        tracker = ProvenanceTracker()
        tracker.start_tracking(topic="Test", source_type="external")

        initial_actions = len(tracker._provenance.actions)
        tracker.record_review(score=8.5, passed=True)

        assert len(tracker._provenance.actions) > initial_actions

    def test_finalize_returns_provenance(self):
        """finalize should return ContentProvenance."""
        from execution.provenance import ProvenanceTracker, ContentProvenance

        tracker = ProvenanceTracker()
        tracker.start_tracking(topic="Test", source_type="external", platform="medium")
        tracker.record_research("TestAgent", model="test", facts_found=2)
        tracker.record_generation("WriterAgent", word_count=300)

        result = tracker.finalize("Test content here")

        assert isinstance(result, ContentProvenance)

    def test_finalize_generates_content_id(self):
        """finalize should generate content ID."""
        from execution.provenance import ProvenanceTracker

        tracker = ProvenanceTracker()
        tracker.start_tracking(topic="Test", source_type="external")

        result = tracker.finalize("Test content")

        assert result.content_id.startswith("gw-")

    def test_finalize_generates_content_hash(self):
        """finalize should generate content hash."""
        from execution.provenance import ProvenanceTracker

        tracker = ProvenanceTracker()
        tracker.start_tracking(topic="Test", source_type="external")

        result = tracker.finalize("Test content")

        # SHA-256 hash is 64 hex characters
        assert len(result.content_hash) == 64


class TestC2PAManifest:
    """Tests for C2PA manifest generation."""

    def test_generate_c2pa_manifest(self):
        """Should generate C2PA-like manifest."""
        from execution.provenance import ProvenanceTracker, generate_c2pa_manifest

        tracker = ProvenanceTracker()
        tracker.start_tracking(topic="Test", source_type="external")
        tracker.record_generation("Writer", word_count=100)
        provenance = tracker.finalize("Test content")

        manifest = generate_c2pa_manifest(provenance)

        assert isinstance(manifest, dict)
        assert "claim_generator" in manifest

    def test_c2pa_has_claim_info(self):
        """C2PA manifest should have claim information."""
        from execution.provenance import ProvenanceTracker, generate_c2pa_manifest

        tracker = ProvenanceTracker()
        tracker.start_tracking(topic="Test", source_type="external")
        provenance = tracker.finalize("Test")

        manifest = generate_c2pa_manifest(provenance)

        # C2PA uses @type: c2pa.claim format
        assert "@type" in manifest
        assert manifest["@type"] == "c2pa.claim"
        assert "assertions" in manifest


class TestSchemaOrgJsonLD:
    """Tests for Schema.org JSON-LD generation."""

    def test_generate_schema_org_jsonld(self):
        """Should generate Schema.org JSON-LD."""
        from execution.provenance import ProvenanceTracker, generate_schema_org_jsonld

        tracker = ProvenanceTracker()
        tracker.start_tracking(topic="Test Article", source_type="external")
        provenance = tracker.finalize("Test content")

        schema = generate_schema_org_jsonld(provenance, "Test Title", "Test Description")

        assert isinstance(schema, dict)
        assert "@type" in schema
        assert schema["@type"] == "Article"

    def test_schema_has_headline(self):
        """Schema should include headline."""
        from execution.provenance import ProvenanceTracker, generate_schema_org_jsonld

        tracker = ProvenanceTracker()
        tracker.start_tracking(topic="Test", source_type="external")
        provenance = tracker.finalize("Test")

        schema = generate_schema_org_jsonld(provenance, "My Headline", "Description")

        assert schema["headline"] == "My Headline"

    def test_schema_has_description(self):
        """Schema should include description."""
        from execution.provenance import ProvenanceTracker, generate_schema_org_jsonld

        tracker = ProvenanceTracker()
        tracker.start_tracking(topic="Test", source_type="external")
        provenance = tracker.finalize("Test")

        schema = generate_schema_org_jsonld(provenance, "Title", "My Description")

        assert schema["description"] == "My Description"


class TestInlineDisclosure:
    """Tests for inline disclosure generation."""

    def test_generate_inline_disclosure_brief(self):
        """Should generate brief disclosure."""
        from execution.provenance import ProvenanceTracker, generate_inline_disclosure

        tracker = ProvenanceTracker()
        tracker.start_tracking(topic="Test", source_type="external")
        provenance = tracker.finalize("Test")

        disclosure = generate_inline_disclosure(provenance, "brief")

        assert isinstance(disclosure, str)
        assert len(disclosure) > 0
        assert "AI" in disclosure

    def test_generate_inline_disclosure_detailed(self):
        """Should generate detailed disclosure."""
        from execution.provenance import ProvenanceTracker, generate_inline_disclosure

        tracker = ProvenanceTracker()
        tracker.start_tracking(topic="Test", source_type="external")
        tracker.record_generation("Writer", word_count=100)
        provenance = tracker.finalize("Test")

        disclosure = generate_inline_disclosure(provenance, "detailed")

        assert isinstance(disclosure, str)
        assert len(disclosure) > 0

    def test_disclosure_default_style(self):
        """Should default to brief style."""
        from execution.provenance import ProvenanceTracker, generate_inline_disclosure

        tracker = ProvenanceTracker()
        tracker.start_tracking(topic="Test", source_type="external")
        provenance = tracker.finalize("Test")

        disclosure = generate_inline_disclosure(provenance)

        assert isinstance(disclosure, str)
        assert "AI" in disclosure


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
