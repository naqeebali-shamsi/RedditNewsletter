"""
Article State Schema - TypedDict schema for LangGraph state management.

This module defines the article state that flows through the generation pipeline,
including fact verification status for quality gate compliance.

Usage:
    from execution.article_state import ArticleState, VerificationState

Reference: PRD Section 7.2 - State Schema Definition
"""

from typing import TypedDict, List, Dict, Optional, Any
from dataclasses import dataclass, field, asdict
from enum import Enum
from datetime import datetime


class VerificationStatusEnum(Enum):
    """Verification status for the article."""
    PENDING = "pending"           # Not yet verified
    IN_PROGRESS = "in_progress"   # Verification running
    PASSED = "passed"             # Passed quality gate
    FAILED = "failed"             # Failed quality gate
    ESCALATED = "escalated"       # Requires human review


class ArticleState(TypedDict, total=False):
    """
    TypedDict schema for article state in LangGraph pipeline.

    This is the primary state object that flows through all pipeline nodes.
    Compatible with LangGraph StateGraph for checkpoint persistence.
    """
    # Core Content
    topic: str                      # Original topic/signal
    source_content: str             # Raw source content (Reddit post, etc.)
    source_type: str                # "external" or "internal"
    draft: str                      # Current draft content
    final_content: str              # Approved final content

    # Metadata
    article_id: str                 # Unique identifier
    created_at: str                 # ISO timestamp
    updated_at: str                 # ISO timestamp
    platform: str                   # "medium", "linkedin", etc.
    word_count: int

    # Research Phase
    research_facts: List[Dict]      # Facts from pre-generation research
    research_sources: List[Dict]    # Sources from research phase

    # Fact Verification (Post-generation)
    claims: List[Dict]              # Extracted claims from draft
    verification_results: List[Dict]  # Results per claim
    unverified_claim_count: int     # Count of unverified claims
    false_claim_count: int          # Count of false claims
    verification_status: str        # VerificationStatusEnum value
    verification_passed: bool       # Quick check flag

    # Quality Review
    quality_score: float            # Overall quality score (0-10)
    quality_passed: bool            # Whether passed quality gate
    review_iterations: int          # Number of revision iterations
    reviewer_feedback: List[Dict]   # Feedback from reviewers

    # Adversarial Panel
    panel_scores: Dict[str, float]  # Scores from each panel expert
    panel_verdict: str              # "approved", "revision_needed", "rejected"
    critical_failures: List[str]    # List of critical issues
    priority_fixes: List[str]       # Ordered list of fixes needed

    # Voice Validation
    voice_type: str                 # "Journalist Observer" or "Practitioner Owner"
    voice_violations: List[Dict]    # Voice rule violations found
    voice_validated: bool           # Whether voice is compliant

    # Provenance (C2PA)
    c2pa_manifest: Dict[str, Any]   # C2PA content provenance manifest
    ai_disclosure: str              # AI disclosure text

    # Workflow Control
    current_phase: str              # Current pipeline phase
    next_action: str                # Suggested next action
    requires_human_review: bool     # Flag for HITL
    human_decision: Optional[str]   # Human reviewer decision
    error_messages: List[str]       # Any errors encountered


@dataclass
class VerificationState:
    """
    Detailed verification state for quality gate integration.

    This dataclass provides a structured way to track verification progress
    and results through the pipeline.
    """
    status: VerificationStatusEnum = VerificationStatusEnum.PENDING
    claims_extracted: int = 0
    claims_verified: int = 0
    claims_partial: int = 0
    claims_unverified: int = 0
    claims_false: int = 0
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    verification_time_seconds: float = 0.0
    provider_used: str = ""
    detailed_results: List[Dict] = field(default_factory=list)
    summary: str = ""

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        result = asdict(self)
        result['status'] = self.status.value
        return result

    @property
    def passes_gate(self) -> bool:
        """Check if verification passes quality gate requirements."""
        # From config: MAX_UNVERIFIED_CLAIMS = 1, MIN_VERIFIED_FACTS = 3
        max_unverified = 1
        min_verified = 3

        return (
            self.claims_false == 0 and
            self.claims_unverified <= max_unverified and
            (self.claims_verified + self.claims_partial) >= min_verified
        )


@dataclass
class QualityGateInput:
    """
    Input for quality gate processing.

    Combines content with verification results for final quality check.
    """
    content: str
    topic: str
    source_type: str = "external"
    platform: str = "medium"
    verification_state: Optional[VerificationState] = None
    research_facts: List[Dict] = field(default_factory=list)

    def to_article_state(self) -> Dict:
        """Convert to ArticleState dict for LangGraph."""
        state: ArticleState = {
            "topic": self.topic,
            "draft": self.content,
            "source_type": self.source_type,
            "platform": self.platform,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "word_count": len(self.content.split()),
            "research_facts": self.research_facts,
            "verification_status": "pending",
            "verification_passed": False,
            "quality_passed": False,
            "requires_human_review": False,
            "error_messages": []
        }

        if self.verification_state:
            state["verification_status"] = self.verification_state.status.value
            state["verification_passed"] = self.verification_state.passes_gate
            state["unverified_claim_count"] = self.verification_state.claims_unverified
            state["false_claim_count"] = self.verification_state.claims_false
            state["claims"] = [
                {"text": r.get("claim", ""), "status": r.get("status", "unknown")}
                for r in self.verification_state.detailed_results
            ]
            state["verification_results"] = self.verification_state.detailed_results

        return state


def create_initial_state(
    topic: str,
    source_content: str = "",
    source_type: str = "external",
    platform: str = "medium"
) -> ArticleState:
    """
    Create initial article state for pipeline start.

    Args:
        topic: The article topic/signal
        source_content: Raw source content
        source_type: "external" or "internal"
        platform: Target platform

    Returns:
        ArticleState dict ready for pipeline processing
    """
    import uuid

    return ArticleState(
        # Core
        topic=topic,
        source_content=source_content,
        source_type=source_type,
        draft="",
        final_content="",

        # Metadata
        article_id=str(uuid.uuid4())[:8],
        created_at=datetime.now().isoformat(),
        updated_at=datetime.now().isoformat(),
        platform=platform,
        word_count=0,

        # Research
        research_facts=[],
        research_sources=[],

        # Verification
        claims=[],
        verification_results=[],
        unverified_claim_count=0,
        false_claim_count=0,
        verification_status="pending",
        verification_passed=False,

        # Quality
        quality_score=0.0,
        quality_passed=False,
        review_iterations=0,
        reviewer_feedback=[],

        # Panel
        panel_scores={},
        panel_verdict="pending",
        critical_failures=[],
        priority_fixes=[],

        # Voice
        voice_type="Journalist Observer" if source_type == "external" else "Practitioner Owner",
        voice_violations=[],
        voice_validated=False,

        # Provenance
        c2pa_manifest={},
        ai_disclosure="",

        # Workflow
        current_phase="init",
        next_action="research",
        requires_human_review=False,
        human_decision=None,
        error_messages=[]
    )


def update_verification_state(state: ArticleState, verification_report: Dict) -> ArticleState:
    """
    Update article state with verification results.

    Args:
        state: Current article state
        verification_report: Report from FactVerificationAgent.verify_article()

    Returns:
        Updated ArticleState
    """
    state["claims"] = verification_report.get("claims", [])
    state["verification_results"] = verification_report.get("results", [])
    state["unverified_claim_count"] = verification_report.get("unverified_count", 0)
    state["false_claim_count"] = verification_report.get("false_count", 0)
    state["verification_passed"] = verification_report.get("passes_quality_gate", False)
    state["verification_status"] = "passed" if state["verification_passed"] else "failed"
    state["updated_at"] = datetime.now().isoformat()

    # If verification failed, flag for human review
    if not state["verification_passed"]:
        state["requires_human_review"] = True
        state["next_action"] = "human_review"

    return state
