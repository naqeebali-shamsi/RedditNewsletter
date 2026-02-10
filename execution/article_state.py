"""
Article State Schema - Pydantic model for LangGraph state management.

This module defines the article state that flows through the generation pipeline,
including fact verification status for quality gate compliance.

Usage:
    from execution.article_state import ArticleState, VerificationState

Reference: PRD Section 7.2 - State Schema Definition
"""

from typing import List, Dict, Optional, Any, Iterator, Tuple
from enum import Enum
from execution.utils.datetime_utils import utc_now, utc_iso

from pydantic import BaseModel, Field


class VerificationStatusEnum(Enum):
    """Verification status for the article."""
    PENDING = "pending"           # Not yet verified
    IN_PROGRESS = "in_progress"   # Verification running
    PASSED = "passed"             # Passed quality gate
    FAILED = "failed"             # Failed quality gate
    ESCALATED = "escalated"       # Requires human review


class ArticleState(BaseModel):
    """
    Pydantic model for article state in LangGraph pipeline.

    This is the primary state object that flows through all pipeline nodes.
    Compatible with LangGraph StateGraph for checkpoint persistence.

    Supports dict-style access (state["key"], state["key"] = val) for
    backward compatibility with code that used the previous TypedDict schema.
    """
    model_config = {"extra": "allow"}

    # Core Content
    topic: str = ""
    source_content: str = ""
    source_type: str = "external"
    draft: str = ""
    final_content: str = ""

    # Metadata
    article_id: str = ""
    created_at: str = ""
    updated_at: str = ""
    platform: str = "medium"
    word_count: int = 0

    # Research Phase
    research_facts: List[Dict] = Field(default_factory=list)
    research_sources: List[Dict] = Field(default_factory=list)

    # Fact Verification (Post-generation)
    claims: List[Dict] = Field(default_factory=list)
    verification_results: List[Dict] = Field(default_factory=list)
    unverified_claim_count: int = 0
    false_claim_count: int = 0
    verification_status: str = "pending"
    verification_passed: bool = False

    # Quality Review
    quality_score: float = 0.0
    quality_passed: bool = False
    review_iterations: int = 0
    reviewer_feedback: List[Dict] = Field(default_factory=list)

    # Adversarial Panel
    panel_scores: Dict[str, float] = Field(default_factory=dict)
    panel_verdict: str = "pending"
    critical_failures: List[str] = Field(default_factory=list)
    priority_fixes: List[str] = Field(default_factory=list)

    # Voice Validation
    voice_type: str = "Journalist Observer"
    voice_violations: List[Dict] = Field(default_factory=list)
    voice_validated: bool = False

    # Tone Profile
    tone_profile_name: str = "Expert Pragmatist"

    # Provenance (C2PA)
    c2pa_manifest: Dict[str, Any] = Field(default_factory=dict)
    ai_disclosure: str = ""

    # Workflow Control
    current_phase: str = "init"
    next_action: str = "research"
    requires_human_review: bool = False
    human_decision: Optional[str] = None
    error_messages: List[str] = Field(default_factory=list)

    # Dict-style access for backward compatibility
    def __getitem__(self, key: str) -> Any:
        try:
            return getattr(self, key)
        except AttributeError:
            raise KeyError(key)

    def __setitem__(self, key: str, value: Any) -> None:
        setattr(self, key, value)

    def __contains__(self, key: str) -> bool:
        return hasattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def keys(self) -> list:
        return list(self.__class__.model_fields.keys()) + list(self.__pydantic_extra__.keys() if self.__pydantic_extra__ else [])

    def values(self) -> list:
        return [getattr(self, k) for k in self.keys()]

    def items(self) -> List[Tuple[str, Any]]:
        return [(k, getattr(self, k)) for k in self.keys()]

    def update(self, d: dict = None, **kwargs) -> None:
        """Update state from a dict and/or keyword arguments (dict-style compat)."""
        items = {}
        if d:
            items.update(d)
        items.update(kwargs)
        for k, v in items.items():
            setattr(self, k, v)

    def __iter__(self) -> Iterator[str]:
        return iter(self.keys())


class VerificationState(BaseModel):
    """
    Detailed verification state for quality gate integration.

    This provides a structured way to track verification progress
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
    detailed_results: List[Dict] = Field(default_factory=list)
    summary: str = ""

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        result = self.model_dump()
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


class QualityGateInput(BaseModel):
    """
    Input for quality gate processing.

    Combines content with verification results for final quality check.
    """
    content: str
    topic: str
    source_type: str = "external"
    platform: str = "medium"
    verification_state: Optional[VerificationState] = None
    research_facts: List[Dict] = Field(default_factory=list)

    def to_article_state(self) -> ArticleState:
        """Convert to ArticleState for LangGraph."""
        state = ArticleState(
            topic=self.topic,
            draft=self.content,
            source_type=self.source_type,
            platform=self.platform,
            created_at=utc_iso(),
            updated_at=utc_iso(),
            word_count=len(self.content.split()),
            research_facts=self.research_facts,
            verification_status="pending",
            verification_passed=False,
            quality_passed=False,
            requires_human_review=False,
            error_messages=[]
        )

        if self.verification_state:
            state.verification_status = self.verification_state.status.value
            state.verification_passed = self.verification_state.passes_gate
            state.unverified_claim_count = self.verification_state.claims_unverified
            state.false_claim_count = self.verification_state.claims_false
            state.claims = [
                {"text": r.get("claim", ""), "status": r.get("status", "unknown")}
                for r in self.verification_state.detailed_results
            ]
            state.verification_results = self.verification_state.detailed_results

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
        ArticleState ready for pipeline processing
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
        created_at=utc_iso(),
        updated_at=utc_iso(),
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
    state["updated_at"] = utc_iso()

    # If verification failed, flag for human review
    if not state["verification_passed"]:
        state["requires_human_review"] = True
        state["next_action"] = "human_review"

    return state
