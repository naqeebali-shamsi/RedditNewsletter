"""
Content Provenance Module - C2PA, Schema.org, and AI Disclosure.

Implements content provenance tracking per industry standards:
- C2PA (Coalition for Content Provenance and Authenticity) manifest generation
- Schema.org JSON-LD structured data for search engines
- Inline AI disclosure text generation
- Full pipeline provenance tracking

Standards Reference:
- C2PA: https://c2pa.org/specifications/specifications/1.0/specs/C2PA_Specification.html
- Schema.org: https://schema.org/Article
- Google AI Content Guidelines: https://developers.google.com/search/docs/appearance/ai-content
"""

import json
import hashlib
import uuid
from datetime import datetime, timezone
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any
from pathlib import Path


@dataclass
class ProvenanceAction:
    """Single action in the provenance chain."""
    action_type: str  # "created", "modified", "verified", "reviewed", "approved"
    agent: str  # Agent name or "human"
    timestamp: str
    details: Dict[str, Any] = field(default_factory=dict)
    model: Optional[str] = None
    confidence: Optional[float] = None


@dataclass
class ContentProvenance:
    """Full provenance record for generated content."""
    # Identifiers
    content_id: str
    content_hash: str  # SHA-256 of final content

    # Creation metadata
    created_at: str
    created_by: str = "GhostWriter AI Pipeline"
    version: str = "3.0"

    # Source information
    source_type: str = "external"  # "external" or "internal"
    source_url: Optional[str] = None
    source_title: Optional[str] = None

    # Generation details
    topic: str = ""
    platform: str = "medium"
    word_count: int = 0

    # Quality metrics
    quality_score: float = 0.0
    fact_verification_passed: bool = False
    verified_claims_count: int = 0
    wsj_checklist_passed: bool = False

    # Models used
    models_used: List[str] = field(default_factory=list)

    # Action history
    actions: List[ProvenanceAction] = field(default_factory=list)

    # Human involvement
    human_reviewed: bool = False
    human_reviewer: Optional[str] = None
    human_review_timestamp: Optional[str] = None

    def add_action(self, action_type: str, agent: str, details: Dict = None, model: str = None):
        """Add an action to the provenance chain."""
        self.actions.append(ProvenanceAction(
            action_type=action_type,
            agent=agent,
            timestamp=datetime.now(timezone.utc).isoformat(),
            details=details or {},
            model=model
        ))

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "content_id": self.content_id,
            "content_hash": self.content_hash,
            "created_at": self.created_at,
            "created_by": self.created_by,
            "version": self.version,
            "source_type": self.source_type,
            "source_url": self.source_url,
            "source_title": self.source_title,
            "topic": self.topic,
            "platform": self.platform,
            "word_count": self.word_count,
            "quality_score": self.quality_score,
            "fact_verification_passed": self.fact_verification_passed,
            "verified_claims_count": self.verified_claims_count,
            "wsj_checklist_passed": self.wsj_checklist_passed,
            "models_used": self.models_used,
            "human_reviewed": self.human_reviewed,
            "human_reviewer": self.human_reviewer,
            "human_review_timestamp": self.human_review_timestamp,
            "actions": [asdict(a) for a in self.actions]
        }


def generate_content_hash(content: str) -> str:
    """Generate SHA-256 hash of content."""
    return hashlib.sha256(content.encode('utf-8')).hexdigest()


def generate_content_id() -> str:
    """Generate unique content ID."""
    return f"gw-{uuid.uuid4().hex[:12]}"


class ProvenanceTracker:
    """
    Tracks provenance through the entire generation pipeline.

    Usage:
        tracker = ProvenanceTracker()
        tracker.start_tracking(topic="AI News", source_type="external")
        tracker.record_research(agent="GeminiResearcher", model="gemini-2.0-flash")
        tracker.record_generation(agent="WriterAgent", model="llama-3.3-70b")
        tracker.record_verification(passed=True, claims=5)
        tracker.record_review(score=7.5, passed=True)
        tracker.finalize(content="...", human_reviewed=True)
        provenance = tracker.get_provenance()
    """

    def __init__(self):
        self._provenance: Optional[ContentProvenance] = None
        self._models: set = set()

    def start_tracking(
        self,
        topic: str = "",
        source_type: str = "external",
        source_url: str = None,
        source_title: str = None,
        platform: str = "medium"
    ):
        """Initialize provenance tracking for new content."""
        self._provenance = ContentProvenance(
            content_id=generate_content_id(),
            content_hash="",  # Set on finalize
            created_at=datetime.now(timezone.utc).isoformat(),
            source_type=source_type,
            source_url=source_url,
            source_title=source_title,
            topic=topic,
            platform=platform
        )
        self._models = set()

        self._provenance.add_action(
            "created",
            "ProvenanceTracker",
            {"topic": topic, "source_type": source_type}
        )

    def record_research(self, agent: str, model: str = None, facts_found: int = 0):
        """Record research phase."""
        if not self._provenance:
            return

        if model:
            self._models.add(model)

        self._provenance.add_action(
            "research",
            agent,
            {"facts_found": facts_found},
            model=model
        )

    def record_generation(self, agent: str, model: str = None, word_count: int = 0):
        """Record generation phase."""
        if not self._provenance:
            return

        if model:
            self._models.add(model)

        self._provenance.word_count = word_count
        self._provenance.add_action(
            "generated",
            agent,
            {"word_count": word_count},
            model=model
        )

    def record_verification(self, passed: bool, claims_verified: int = 0, false_claims: int = 0):
        """Record fact verification phase."""
        if not self._provenance:
            return

        self._provenance.fact_verification_passed = passed
        self._provenance.verified_claims_count = claims_verified

        self._provenance.add_action(
            "verified",
            "FactVerificationAgent",
            {
                "passed": passed,
                "claims_verified": claims_verified,
                "false_claims": false_claims
            }
        )

    def record_review(
        self,
        score: float,
        passed: bool,
        wsj_passed: bool = False,
        iteration: int = 1,
        models_used: List[str] = None
    ):
        """Record adversarial review phase."""
        if not self._provenance:
            return

        self._provenance.quality_score = score
        self._provenance.wsj_checklist_passed = wsj_passed

        if models_used:
            self._models.update(models_used)

        self._provenance.add_action(
            "reviewed",
            "AdversarialPanel",
            {
                "score": score,
                "passed": passed,
                "wsj_passed": wsj_passed,
                "iteration": iteration
            }
        )

    def record_revision(self, agent: str, model: str = None, iteration: int = 1):
        """Record revision phase."""
        if not self._provenance:
            return

        if model:
            self._models.add(model)

        self._provenance.add_action(
            "revised",
            agent,
            {"iteration": iteration},
            model=model
        )

    def record_human_review(self, reviewer: str, decision: str, notes: str = ""):
        """Record human review/approval."""
        if not self._provenance:
            return

        self._provenance.human_reviewed = True
        self._provenance.human_reviewer = reviewer
        self._provenance.human_review_timestamp = datetime.now(timezone.utc).isoformat()

        self._provenance.add_action(
            "human_reviewed",
            reviewer,
            {"decision": decision, "notes": notes}
        )

    def finalize(self, content: str, human_reviewed: bool = False) -> ContentProvenance:
        """Finalize provenance with content hash."""
        if not self._provenance:
            raise ValueError("No provenance tracking started")

        self._provenance.content_hash = generate_content_hash(content)
        self._provenance.word_count = len(content.split())
        self._provenance.models_used = list(self._models)
        self._provenance.human_reviewed = human_reviewed

        self._provenance.add_action(
            "finalized",
            "ProvenanceTracker",
            {"content_hash": self._provenance.content_hash[:16] + "..."}
        )

        return self._provenance

    def get_provenance(self) -> Optional[ContentProvenance]:
        """Get current provenance record."""
        return self._provenance


# ============================================================================
# Content Metadata Generation
# ============================================================================

def generate_content_metadata(provenance: ContentProvenance) -> Dict:
    """
    Generate content metadata manifest for provenance tracking.

    Based on C2PA (Coalition for Content Provenance and Authenticity) standards.
    This generates a simplified manifest that can be embedded or linked.

    Reference: https://c2pa.org/specifications/
    """
    manifest = {
        "@context": "https://c2pa.org/specifications/1.0",
        "@type": "c2pa.claim",
        "dc:title": provenance.topic or "AI-Generated Article",
        "dc:format": "text/markdown",
        "claim_generator": {
            "name": "GhostWriter AI Pipeline",
            "version": provenance.version,
            "url": "https://github.com/ghostwriter-ai"
        },
        "signature": {
            "algorithm": "sha256",
            "hash": provenance.content_hash
        },
        "assertions": [
            {
                "label": "c2pa.actions",
                "data": {
                    "actions": [
                        {
                            "action": "c2pa.created",
                            "when": provenance.created_at,
                            "softwareAgent": provenance.created_by,
                            "parameters": {
                                "ai_generated": True,
                                "ai_models": provenance.models_used
                            }
                        }
                    ]
                }
            },
            {
                "label": "c2pa.ai.training",
                "data": {
                    "ai_model_info": [
                        {"name": model, "type": "large_language_model"}
                        for model in provenance.models_used
                    ]
                }
            }
        ],
        "claim_metadata": {
            "content_id": provenance.content_id,
            "created_at": provenance.created_at,
            "source_type": provenance.source_type,
            "quality_score": provenance.quality_score,
            "fact_verified": provenance.fact_verification_passed,
            "human_reviewed": provenance.human_reviewed
        }
    }

    return manifest


# Backward compatibility alias
generate_c2pa_manifest = generate_content_metadata


# ============================================================================
# Schema.org JSON-LD Generation
# ============================================================================

def generate_schema_org_jsonld(
    provenance: ContentProvenance,
    title: str,
    description: str,
    author_name: str = "AI Writing Assistant",
    publisher_name: str = "GhostWriter",
    publisher_url: str = "https://ghostwriter.ai"
) -> Dict:
    """
    Generate Schema.org JSON-LD structured data for search engines.

    This follows Google's guidelines for AI-generated content attribution.
    Reference: https://schema.org/Article
    """
    jsonld = {
        "@context": "https://schema.org",
        "@type": "Article",
        "headline": title,
        "description": description,
        "articleBody": "",  # Can be filled with content if needed
        "datePublished": provenance.created_at,
        "dateModified": provenance.actions[-1].timestamp if provenance.actions else provenance.created_at,
        "author": {
            "@type": "Organization",
            "name": author_name,
            "description": "AI-assisted content creation"
        },
        "publisher": {
            "@type": "Organization",
            "name": publisher_name,
            "url": publisher_url
        },
        "isAccessibleForFree": True,
        "creativeWorkStatus": "Published",

        # AI-specific attributes (Schema.org extensions)
        "sdPublisher": {
            "@type": "Organization",
            "name": "GhostWriter AI Pipeline"
        },

        # Custom attributes for AI transparency
        "additionalProperty": [
            {
                "@type": "PropertyValue",
                "name": "ai_generated",
                "value": "true"
            },
            {
                "@type": "PropertyValue",
                "name": "ai_models_used",
                "value": ", ".join(provenance.models_used)
            },
            {
                "@type": "PropertyValue",
                "name": "fact_verified",
                "value": str(provenance.fact_verification_passed).lower()
            },
            {
                "@type": "PropertyValue",
                "name": "human_reviewed",
                "value": str(provenance.human_reviewed).lower()
            },
            {
                "@type": "PropertyValue",
                "name": "quality_score",
                "value": str(provenance.quality_score)
            }
        ]
    }

    # Add source attribution if external
    if provenance.source_type == "external" and provenance.source_url:
        jsonld["isBasedOn"] = {
            "@type": "CreativeWork",
            "url": provenance.source_url,
            "name": provenance.source_title or "Source Material"
        }

    return jsonld


# ============================================================================
# Inline Disclosure Generator
# ============================================================================

def generate_inline_disclosure(
    provenance: ContentProvenance,
    style: str = "full",
    include_models: bool = True
) -> str:
    """
    Generate inline AI disclosure text for content.

    Styles:
    - "full": Complete disclosure with all details
    - "brief": Short one-line disclosure
    - "footer": Footer-style disclosure
    - "byline": Byline-style attribution

    This follows best practices for AI content transparency.
    """
    models_str = ", ".join(provenance.models_used[:3]) if provenance.models_used else "AI"
    date_str = provenance.created_at[:10]  # YYYY-MM-DD

    if style == "full":
        disclosure = f"""---
**AI Transparency Disclosure**

This content was created with AI assistance using the following process:
- **Generation**: AI-assisted writing using {models_str}
- **Fact Verification**: {"Verified" if provenance.fact_verification_passed else "Not verified"} ({provenance.verified_claims_count} claims checked)
- **Quality Review**: Scored {provenance.quality_score}/10 by multi-model review panel
- **Human Review**: {"Yes" if provenance.human_reviewed else "Pending"}
- **Source Type**: {"Community-sourced (external)" if provenance.source_type == "external" else "Original research (internal)"}

Content ID: {provenance.content_id}
Generated: {date_str}
---"""

    elif style == "brief":
        human_note = " + human review" if provenance.human_reviewed else ""
        disclosure = f"AI-assisted content ({models_str}{human_note}) | Quality: {provenance.quality_score}/10"

    elif style == "footer":
        human_status = "human-reviewed" if provenance.human_reviewed else "AI-only"
        disclosure = f"""
---
*This article was created with AI assistance ({human_status}). Facts have been {"verified" if provenance.fact_verification_passed else "not independently verified"}. Content ID: {provenance.content_id}*
"""

    elif style == "byline":
        if provenance.human_reviewed:
            disclosure = f"By AI Writing Assistant with {provenance.human_reviewer or 'Editorial'} Review"
        else:
            disclosure = "By AI Writing Assistant"

    else:
        disclosure = f"AI-generated content (ID: {provenance.content_id})"

    return disclosure


# ============================================================================
# Provenance Export Functions
# ============================================================================

def export_provenance_json(provenance: ContentProvenance, filepath: str):
    """Export full provenance record to JSON file."""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(provenance.to_dict(), f, indent=2)


def export_c2pa_manifest(provenance: ContentProvenance, filepath: str):
    """Export C2PA manifest to JSON file."""
    manifest = generate_c2pa_manifest(provenance)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2)


def export_schema_org(
    provenance: ContentProvenance,
    filepath: str,
    title: str,
    description: str
):
    """Export Schema.org JSON-LD to file."""
    jsonld = generate_schema_org_jsonld(provenance, title, description)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(jsonld, f, indent=2)


# ============================================================================
# Test / Demo
# ============================================================================

if __name__ == "__main__":
    print("Testing Provenance Module")
    print("=" * 60)

    # Create tracker and simulate pipeline
    tracker = ProvenanceTracker()

    tracker.start_tracking(
        topic="AI Model Performance Benchmarks",
        source_type="external",
        source_url="https://reddit.com/r/MachineLearning/example",
        platform="medium"
    )

    tracker.record_research("GeminiResearcher", model="gemini-2.0-flash", facts_found=5)
    tracker.record_generation("WriterAgent", model="llama-3.3-70b-versatile", word_count=800)
    tracker.record_verification(passed=True, claims_verified=5, false_claims=0)
    tracker.record_review(score=7.8, passed=True, wsj_passed=True, models_used=["claude-sonnet-4", "gpt-4o"])
    tracker.record_human_review("editor@example.com", "approved", "Minor edits made")

    # Finalize
    test_content = "This is test content for provenance tracking."
    provenance = tracker.finalize(test_content, human_reviewed=True)

    # Display results
    print("\n1. Provenance Record:")
    print(f"   Content ID: {provenance.content_id}")
    print(f"   Content Hash: {provenance.content_hash[:32]}...")
    print(f"   Models Used: {provenance.models_used}")
    print(f"   Quality Score: {provenance.quality_score}/10")
    print(f"   Human Reviewed: {provenance.human_reviewed}")

    print("\n2. C2PA Manifest (excerpt):")
    c2pa = generate_c2pa_manifest(provenance)
    print(f"   Claim Generator: {c2pa['claim_generator']['name']}")
    print(f"   AI Models: {c2pa['assertions'][1]['data']['ai_model_info']}")

    print("\n3. Schema.org JSON-LD (excerpt):")
    schema = generate_schema_org_jsonld(provenance, "Test Article", "A test description")
    print(f"   Type: {schema['@type']}")
    print(f"   AI Generated: {schema['additionalProperty'][0]['value']}")

    print("\n4. Inline Disclosures:")
    print("\n   BRIEF:")
    print(f"   {generate_inline_disclosure(provenance, 'brief')}")

    print("\n   BYLINE:")
    print(f"   {generate_inline_disclosure(provenance, 'byline')}")

    print("\n   FOOTER:")
    print(generate_inline_disclosure(provenance, 'footer'))

    print("\n" + "=" * 60)
    print("Provenance module test complete!")
