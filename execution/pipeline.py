"""
LangGraph Pipeline Orchestrator - WSJ-Tier AI Writing Pipeline.

This module implements the article generation pipeline using LangGraph for
state management, checkpointing, and workflow orchestration.

Pipeline Phases:
1. RESEARCH: Gather facts from sources (Gemini/Perplexity)
2. GENERATE: Write draft using verified facts
3. VERIFY: Post-generation fact verification
4. REVIEW: Multi-model adversarial panel review
5. REVISE: Fix issues identified by panel (loop)
6. APPROVE: Human-in-the-loop approval (interrupt)

Usage:
    from execution.pipeline import create_pipeline, run_pipeline

    pipeline = create_pipeline()
    result = run_pipeline(pipeline, topic="AI News", source_content="...")
"""

import os
import sys
import threading
from pathlib import Path
from typing import Dict, Any, Optional, List, Literal, Annotated
from execution.utils.datetime_utils import utc_iso
from functools import wraps
import operator

from pydantic import Field

# LangGraph imports
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Local imports
from execution.article_state import ArticleState, create_initial_state, update_verification_state
from execution.config import config
from execution.provenance import (
    ProvenanceTracker, generate_c2pa_manifest, generate_schema_org_jsonld,
    generate_inline_disclosure, export_provenance_json
)

# Phase constants
PHASE_RESEARCH = "research"
PHASE_GENERATE = "generate"
PHASE_VERIFY = "verify"
PHASE_REVIEW = "review"
PHASE_REVISE = "revise"
PHASE_APPROVE = "approve"
PHASE_STYLE_CHECK = "style_check"
PHASE_DONE = "done"
PHASE_ESCALATE = "escalate"


# ============================================================================
# State Reducer Functions (for LangGraph state updates)
# ============================================================================

def merge_lists(left: list, right: list) -> list:
    """Merge two lists, appending new items."""
    return left + right


def merge_dicts(left: dict, right: dict) -> dict:
    """Merge two dicts, with right taking precedence."""
    return {**left, **right}


# ============================================================================
# Per-Node Timeout Support (cross-platform, threading-based)
# ============================================================================

class NodeTimeoutError(Exception):
    """Raised when a pipeline node exceeds its timeout."""
    def __init__(self, node_name: str, timeout_seconds: int):
        self.node_name = node_name
        self.timeout_seconds = timeout_seconds
        super().__init__(f"Node '{node_name}' timed out after {timeout_seconds}s")


def with_timeout(timeout_seconds: int):
    """Decorator to add timeout to pipeline nodes (cross-platform, threading-based)."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            result = [None]
            exception = [None]

            def target():
                try:
                    result[0] = func(*args, **kwargs)
                except Exception as e:
                    exception[0] = e

            thread = threading.Thread(target=target, daemon=True)
            thread.start()
            thread.join(timeout=timeout_seconds)

            if thread.is_alive():
                raise NodeTimeoutError(func.__name__, timeout_seconds)
            if exception[0]:
                raise exception[0]
            return result[0]
        return wrapper
    return decorator


# ============================================================================
# Pipeline State Definition (extends ArticleState for LangGraph)
# ============================================================================

class PipelineState(ArticleState):
    """
    Extended state for LangGraph pipeline with annotation support.

    This adds LangGraph-specific fields for graph execution control
    and keys written by pipeline nodes not covered by ArticleState.
    """
    # Graph control
    messages: Annotated[list, add_messages] = Field(default_factory=list)
    iteration_count: int = 0
    max_iterations: int = 3

    # Style check outputs
    style_score: Optional[float] = None
    style_result: Dict[str, Any] = Field(default_factory=dict)
    style_passed: bool = False
    style_error: str = ""

    # Escalation outputs
    escalation_codes: List[str] = Field(default_factory=list)
    escalation_reasons: List[str] = Field(default_factory=list)
    iterations_used: int = 0

    # Approval outputs
    approval_reason: str = ""
    review_reasons: List[str] = Field(default_factory=list)


# ============================================================================
# Node Functions (Pipeline Phases)
# ============================================================================

@with_timeout(180)  # 3 minutes for research
def research_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    RESEARCH PHASE: Gather verified facts before writing.

    Uses Gemini (primary) or Perplexity (fallback) for grounded research.
    """
    print(f"\n{'='*60}")
    print("PHASE: RESEARCH")
    print(f"{'='*60}")

    topic = state.get("topic", "")
    source_content = state.get("source_content", "")

    research_facts = []
    research_sources = []

    # Try Gemini first
    try:
        from execution.agents.gemini_researcher import GeminiResearchAgent
        researcher = GeminiResearchAgent()
        result = researcher.research(topic, source_content[:3000] if source_content else "")

        if result:
            research_facts = result.get("verified_facts", [])
            research_sources = result.get("sources", [])
            print(f"  Gemini research: {len(research_facts)} facts found")
    except Exception as e:
        print(f"  Gemini research failed: {e}")

    # Fallback to Perplexity
    if not research_facts:
        try:
            from execution.agents.perplexity_researcher import PerplexityResearchAgent
            researcher = PerplexityResearchAgent()
            result = researcher.research_topic(topic, source_content[:3000] if source_content else "")

            if result:
                research_facts = result.get("verified_facts", [])
                research_sources = result.get("perplexity_citations", [])
                print(f"  Perplexity research: {len(research_facts)} facts found")
        except Exception as e:
            print(f"  Perplexity research failed: {e}")

    return {
        "research_facts": research_facts,
        "research_sources": research_sources,
        "current_phase": PHASE_RESEARCH,
        "next_action": PHASE_GENERATE,
        "updated_at": utc_iso()
    }


@with_timeout(120)  # 2 minutes for generation
def generate_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    GENERATE PHASE: Write article draft using verified facts.

    Uses the Writer agent with fact constraints from research phase.
    """
    print(f"\n{'='*60}")
    print("PHASE: GENERATE")
    print(f"{'='*60}")

    topic = state.get("topic", "")
    source_content = state.get("source_content", "")
    source_type = state.get("source_type", "external")
    platform = state.get("platform", "medium")
    research_facts = state.get("research_facts", [])

    try:
        from execution.agents.writer import WriterAgent
        writer = WriterAgent()

        # Build fact constraints
        fact_sheet = _build_fact_sheet(research_facts)

        # Voice instruction based on source type
        voice = config.voice.VOICE_EXTERNAL if source_type == "external" else config.voice.VOICE_INTERNAL

        prompt = f"""Write a {platform} article on the following topic.

TOPIC: {topic}

SOURCE CONTENT:
{source_content[:2000] if source_content else "No source provided"}

{fact_sheet}

VOICE: {voice}
{"You are OBSERVING others' work. Never claim ownership." if source_type == "external" else "This is YOUR work. Ownership voice is appropriate."}

TARGET: {platform.upper()} platform
- Hook in first 2 sentences
- Specific, actionable insights
- Clear CTA at the end
- 800-1200 words

Write the article now. Output ONLY the article content."""

        # call_llm raises LLMError on failure (no more error strings)
        draft = writer.call_llm(prompt, temperature=0.7)

        # BaseAgent validates non-empty; also check minimum article length
        if len(draft.strip()) < 100:
            raise ValueError(f"Generated draft too short ({len(draft.strip())} chars)")

        # Clean up any meta-commentary
        if draft.startswith("Here") or draft.startswith("I've"):
            lines = draft.split('\n')
            draft = '\n'.join(lines[1:]) if len(lines) > 1 else draft

        print(f"  Generated draft: {len(draft)} chars, {len(draft.split())} words")

        return {
            "draft": draft.strip(),
            "word_count": len(draft.split()),
            "current_phase": PHASE_GENERATE,
            "next_action": PHASE_VERIFY,
            "updated_at": utc_iso()
        }

    except Exception as e:
        print(f"  Generation failed: {e}")
        return {
            "draft": "",
            "error_messages": state.get("error_messages", []) + [f"Generation failed: {e}"],
            "current_phase": PHASE_GENERATE,
            "next_action": PHASE_ESCALATE
        }


@with_timeout(300)  # 5 minutes for verification (many claims)
def verify_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    VERIFY PHASE: Post-generation fact verification.

    Extracts claims from draft and verifies each against web sources.
    """
    print(f"\n{'='*60}")
    print("PHASE: VERIFY")
    print(f"{'='*60}")

    draft = state.get("draft", "")
    topic = state.get("topic", "")

    if not draft:
        print("  No draft to verify")
        return {
            "verification_passed": False,
            "verification_status": "failed",
            "current_phase": PHASE_VERIFY,
            "next_action": PHASE_ESCALATE
        }

    try:
        from execution.agents.fact_verification_agent import FactVerificationAgent
        verifier = FactVerificationAgent()
        report = verifier.verify_article(draft, topic)

        print(f"  Claims found: {len(report.claims)}")
        print(f"  Verified: {report.verified_count}")
        print(f"  Unverified: {report.unverified_count}")
        print(f"  False: {report.false_count}")
        print(f"  Passes gate: {report.passes_quality_gate}")

        # Update state with verification results
        updated = update_verification_state(state.copy(), report.to_dict())

        # Determine next action
        if report.passes_quality_gate:
            next_action = PHASE_REVIEW
        elif report.false_count > 0:
            # False claims require escalation
            next_action = PHASE_ESCALATE
            updated["requires_human_review"] = True
        else:
            # Unverified claims can try revision
            next_action = PHASE_REVISE

        updated["current_phase"] = PHASE_VERIFY
        updated["next_action"] = next_action

        return updated

    except Exception as e:
        print(f"  Verification failed: {e}")
        return {
            "verification_passed": False,
            "verification_status": "error",
            "error_messages": state.get("error_messages", []) + [f"Verification failed: {e}"],
            "current_phase": PHASE_VERIFY,
            "next_action": PHASE_ESCALATE  # Do not proceed without verification
        }


@with_timeout(300)  # 5 minutes for review (10 experts)
def review_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    REVIEW PHASE: Multi-model adversarial panel review.

    Uses Claude (ethics), Gemini (accuracy), GPT-4o (structure) to review content.
    """
    print(f"\n{'='*60}")
    print("PHASE: REVIEW (Adversarial Panel)")
    print(f"{'='*60}")

    draft = state.get("draft", "")
    platform = state.get("platform", "medium")
    source_type = state.get("source_type", "external")
    iteration = state.get("review_iterations", 0) + 1

    if not draft:
        return {
            "quality_passed": False,
            "panel_verdict": "rejected",
            "current_phase": PHASE_REVIEW,
            "next_action": PHASE_ESCALATE
        }

    try:
        from execution.agents.adversarial_panel import AdversarialPanelAgent
        panel = AdversarialPanelAgent()

        verdict = panel.review_content(
            content=draft,
            platform=platform,
            iteration=iteration,
            source_type=source_type
        )

        print(f"  Panel verdict: {verdict.verdict}")
        print(f"  Average score: {verdict.average_score}/10")
        print(f"  Passed: {verdict.passed}")

        # Map panel verdict to state
        panel_scores = {}
        for expert_result in verdict.expert_scores:
            panel_scores[expert_result.get("expert", "unknown")] = expert_result.get("score", 0)

        next_action = PHASE_APPROVE if verdict.passed else PHASE_REVISE
        if iteration >= config.quality.MAX_ITERATIONS and not verdict.passed:
            next_action = PHASE_ESCALATE

        return {
            "quality_score": verdict.average_score,
            "quality_passed": verdict.passed,
            "panel_scores": panel_scores,
            "panel_verdict": verdict.verdict,
            "critical_failures": verdict.critical_failures,
            "priority_fixes": verdict.priority_fixes,
            "review_iterations": iteration,
            "reviewer_feedback": state.get("reviewer_feedback", []) + [verdict.to_dict()],
            "current_phase": PHASE_REVIEW,
            "next_action": next_action,
            "updated_at": utc_iso()
        }

    except Exception as e:
        print(f"  Review failed: {e}")
        return {
            "quality_passed": False,
            "panel_verdict": "error",
            "error_messages": state.get("error_messages", []) + [f"Review failed: {e}"],
            "current_phase": PHASE_REVIEW,
            "next_action": PHASE_ESCALATE
        }


@with_timeout(120)  # 2 minutes for revision
def revise_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    REVISE PHASE: Fix issues identified by panel or verification.

    Uses Writer agent to revise based on specific fix instructions.
    """
    print(f"\n{'='*60}")
    print("PHASE: REVISE")
    print(f"{'='*60}")

    draft = state.get("draft", "")
    platform = state.get("platform", "medium")
    source_type = state.get("source_type", "external")
    priority_fixes = state.get("priority_fixes", [])
    critical_failures = state.get("critical_failures", [])

    if not draft:
        return {
            "current_phase": PHASE_REVISE,
            "next_action": PHASE_ESCALATE
        }

    try:
        from execution.agents.writer import WriterAgent
        writer = WriterAgent()

        # Build fix instructions
        fix_instructions = _build_fix_instructions(priority_fixes, critical_failures)

        # Voice instruction
        if source_type == "external":
            voice_instruction = """
CRITICAL VOICE REQUIREMENT (External Source):
- You are an OBSERVER reporting on others' work
- FORBIDDEN: "I built", "we created", "our team", "my approach"
- USE: "teams found", "engineers discovered", "this approach"
"""
        else:
            voice_instruction = """
VOICE (Internal Source):
- This is your own work, use ownership voice naturally
"""

        prompt = f"""REVISION TASK

You are revising a {platform} draft that FAILED the quality gate.
{voice_instruction}

CURRENT DRAFT:
---
{draft}
---

{fix_instructions}

IMPORTANT:
- Output ONLY the revised content
- No explanations, no meta-commentary
- Preserve the core message while fixing the issues
- Make the content feel human-written, not AI-generated
"""

        # call_llm raises LLMError on failure (no more error strings)
        revised = writer.call_llm(prompt, temperature=0.7)

        # BaseAgent validates non-empty; also check minimum revision length
        if len(revised.strip()) < 100:
            raise ValueError(f"Revised draft too short ({len(revised.strip())} chars)")

        # Clean up
        if revised.startswith("Here") or revised.startswith("I've"):
            lines = revised.split('\n')
            revised = '\n'.join(lines[1:]) if len(lines) > 1 else revised

        print(f"  Revised draft: {len(revised)} chars")

        return {
            "draft": revised.strip(),
            "word_count": len(revised.split()),
            "current_phase": PHASE_REVISE,
            "next_action": PHASE_VERIFY,  # Re-verify after revision
            "updated_at": utc_iso()
        }

    except Exception as e:
        print(f"  Revision failed: {e}")
        return {
            "error_messages": state.get("error_messages", []) + [f"Revision failed: {e}"],
            "current_phase": PHASE_REVISE,
            "next_action": PHASE_ESCALATE
        }


def approve_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    APPROVE PHASE: Human-in-the-loop approval checkpoint.

    This is an interrupt point where human review is required.
    High-scoring content may be auto-approved based on config.
    """
    print(f"\n{'='*60}")
    print("PHASE: APPROVE (Human Review)")
    print(f"{'='*60}")

    draft = state.get("draft", "")
    quality_score = state.get("quality_score", 0)
    wsj_checklist = state.get("wsj_checklist", {})
    verification_result = state.get("verification_result", {})

    # Check for auto-approval conditions (high score + all checks passed)
    auto_approve_conditions = [
        quality_score >= config.quality.MIN_SCORE_FOR_AUTO_APPROVE,
        wsj_checklist.get("overall_passed", False) if wsj_checklist else True,
        verification_result.get("false_count", 0) == 0,
        not config.quality.HUMAN_REVIEW_REQUIRED_FOR_PUBLISH
    ]

    if all(auto_approve_conditions):
        print(f"  AUTO-APPROVED: Score {quality_score}/10 exceeds threshold")
        print(f"  WSJ: PASSED | Verification: PASSED")
        return {
            "requires_human_review": False,
            "final_content": draft,
            "human_decision": "auto_approved",
            "approval_reason": f"Score {quality_score}/10 with all quality checks passed",
            "current_phase": PHASE_DONE,
            "next_action": "complete",
            "updated_at": utc_iso()
        }

    # Human review required
    review_reasons = []
    if quality_score < config.quality.MIN_SCORE_FOR_AUTO_APPROVE:
        review_reasons.append(f"Score {quality_score} below auto-approve threshold {config.quality.MIN_SCORE_FOR_AUTO_APPROVE}")
    if wsj_checklist and not wsj_checklist.get("overall_passed", True):
        failed_cats = [k for k in ["inaccuracy", "unfairness", "disorganization", "poor_writing"]
                      if not wsj_checklist.get(k, {}).get("passed", True)]
        review_reasons.append(f"WSJ showstoppers failed: {', '.join(failed_cats)}")
    if config.quality.HUMAN_REVIEW_REQUIRED_FOR_PUBLISH:
        review_reasons.append("Human review required by configuration")

    print(f"  HUMAN REVIEW REQUIRED")
    print(f"  Score: {quality_score}/10")
    for reason in review_reasons:
        print(f"  - {reason}")

    return {
        "requires_human_review": True,
        "final_content": draft,
        "review_reasons": review_reasons,
        "current_phase": PHASE_APPROVE,
        "next_action": "await_human_decision",
        "updated_at": utc_iso()
    }


@with_timeout(60)  # 1 minute for style check
def style_check_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    STYLE CHECK PHASE: Quantitative voice fingerprinting.

    Runs the StyleEnforcerAgent to score content across 5 dimensions:
    burstiness, lexical diversity, AI-tell detection, authenticity markers,
    and framework compliance. Threshold: 80/100 for publish, <60 for revision.
    """
    print(f"\n{'='*60}")
    print("PHASE: STYLE CHECK - Voice Fingerprinting")
    print(f"{'='*60}")

    draft = state.get("final_content") or state.get("current_draft", "")
    if not draft:
        print("  No content to score, skipping style check")
        return {
            "style_score": None,
            "style_passed": True,
            "current_phase": PHASE_STYLE_CHECK,
            "updated_at": utc_iso()
        }

    try:
        from execution.agents.style_enforcer import StyleEnforcerAgent
        enforcer = StyleEnforcerAgent()
        result = enforcer.score(draft, content_type="article")

        total = result.total
        print(f"  Style Score: {total}/100")
        print(f"    Burstiness:          {result.burstiness_score}/100 (weight: 20%)")
        print(f"    Lexical Diversity:   {result.lexical_diversity_score}/100 (weight: 15%)")
        print(f"    AI-Tell Detection:   {result.ai_tell_score}/100 (weight: 25%)")
        print(f"    Authenticity:        {result.authenticity_score}/100 (weight: 25%)")
        print(f"    Framework Compliance:{result.framework_compliance_score}/100 (weight: 15%)")

        passed = result.passed
        if passed:
            print("  STYLE CHECK PASSED")
        else:
            print(f"  STYLE CHECK {'NEEDS REVISION' if result.needs_revision else 'FAILED'}")

        return {
            "style_score": total,
            "style_result": result.to_dict(),
            "style_passed": passed,
            "current_phase": PHASE_STYLE_CHECK,
            "updated_at": utc_iso()
        }
    except ImportError:
        print("  StyleEnforcerAgent not available — flagging for review")
        return {
            "style_score": None,
            "style_passed": False,
            "style_error": "StyleEnforcerAgent unavailable (missing dependency)",
            "current_phase": PHASE_STYLE_CHECK,
            "updated_at": utc_iso()
        }
    except Exception as e:
        print(f"  Style check error: {e}")
        return {
            "style_score": None,
            "style_passed": False,
            "style_error": f"Style check failed: {e}",
            "current_phase": PHASE_STYLE_CHECK,
            "updated_at": utc_iso()
        }


def route_after_style_check(state: Dict[str, Any]) -> str:
    """Route after style check phase."""
    if state.get("style_passed", True):
        return PHASE_APPROVE
    else:
        # Style failed — route to revision
        iteration = state.get("review_iterations", 0)
        if iteration >= config.quality.MAX_ITERATIONS:
            return PHASE_ESCALATE
        return PHASE_REVISE


def escalate_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    ESCALATE PHASE: Content requires human intervention.

    Called when max iterations reached, critical failures detected,
    or verification issues found. Provides detailed escalation report.
    """
    print(f"\n{'='*60}")
    print("PHASE: ESCALATE - HUMAN INTERVENTION REQUIRED")
    print(f"{'='*60}")

    # Gather escalation context
    errors = state.get("error_messages", [])
    critical = state.get("critical_failures", [])
    quality_score = state.get("quality_score", 0)
    iteration = state.get("review_iterations", 0)
    verification_result = state.get("verification_result", {})
    wsj_checklist = state.get("wsj_checklist", {})

    # Determine escalation reasons
    escalation_reasons = []
    escalation_codes = []

    if iteration >= config.quality.MAX_ITERATIONS:
        escalation_codes.append("MAX_ITERATIONS")
        escalation_reasons.append(config.quality.ESCALATION_REASONS["MAX_ITERATIONS"])

    if verification_result.get("false_count", 0) > 0:
        escalation_codes.append("FALSE_CLAIMS")
        escalation_reasons.append(config.quality.ESCALATION_REASONS["FALSE_CLAIMS"])

    if quality_score < config.quality.ESCALATION_THRESHOLD:
        escalation_codes.append("LOW_SCORE")
        escalation_reasons.append(config.quality.ESCALATION_REASONS["LOW_SCORE"])

    if wsj_checklist and not wsj_checklist.get("overall_passed", True):
        escalation_codes.append("CRITICAL_FAILURE")
        escalation_reasons.append(config.quality.ESCALATION_REASONS["CRITICAL_FAILURE"])

    if any("KILL_PHRASE" in str(f) for f in critical):
        escalation_codes.append("KILL_PHRASE")
        escalation_reasons.append(config.quality.ESCALATION_REASONS["KILL_PHRASE"])

    # If no specific reason, generic escalation
    if not escalation_reasons:
        escalation_reasons.append("Content did not meet quality standards after automated review")

    # Build detailed escalation report
    print(f"\n  ESCALATION CODES: {', '.join(escalation_codes)}")
    print(f"\n  REASONS:")
    for i, reason in enumerate(escalation_reasons, 1):
        print(f"    {i}. {reason}")

    print(f"\n  QUALITY SCORE: {quality_score}/10")
    print(f"  ITERATIONS USED: {iteration}/{config.quality.MAX_ITERATIONS}")

    if verification_result:
        print(f"  FACT VERIFICATION:")
        print(f"    Verified: {verification_result.get('verified_count', 0)}")
        print(f"    Unverified: {verification_result.get('unverified_count', 0)}")
        print(f"    FALSE: {verification_result.get('false_count', 0)}")

    if wsj_checklist:
        print(f"  WSJ SHOWSTOPPERS:")
        for key in ["inaccuracy", "unfairness", "disorganization", "poor_writing"]:
            if key in wsj_checklist:
                status = "PASS" if wsj_checklist[key].get("passed") else "FAIL"
                print(f"    {key.title()}: {status}")

    if critical:
        print(f"\n  CRITICAL FAILURES ({len(critical)}):")
        for i, failure in enumerate(critical[:5], 1):
            print(f"    {i}. {failure}")

    print(f"\n  ACTION REQUIRED: Human review and intervention")
    print("=" * 60)

    return {
        "requires_human_review": True,
        "panel_verdict": "escalated",
        "escalation_codes": escalation_codes,
        "escalation_reasons": escalation_reasons,
        "quality_score": quality_score,
        "iterations_used": iteration,
        "critical_failures": critical,
        "current_phase": PHASE_ESCALATE,
        "next_action": "await_human_decision",
        "updated_at": utc_iso()
    }


# ============================================================================
# Router Functions (Conditional Edges)
# ============================================================================

def route_after_verify(state: Dict[str, Any]) -> str:
    """Route after verification phase."""
    next_action = state.get("next_action", PHASE_REVIEW)

    if next_action == PHASE_ESCALATE:
        return PHASE_ESCALATE
    elif state.get("verification_passed", False):
        return PHASE_REVIEW
    else:
        iteration = state.get("review_iterations", 0)
        if iteration >= config.quality.MAX_ITERATIONS:
            return PHASE_ESCALATE
        return PHASE_REVISE


def route_after_review(state: Dict[str, Any]) -> str:
    """Route after review phase."""
    next_action = state.get("next_action", PHASE_APPROVE)

    if next_action == PHASE_ESCALATE:
        return PHASE_ESCALATE
    elif state.get("quality_passed", False):
        return PHASE_APPROVE
    else:
        iteration = state.get("review_iterations", 0)
        if iteration >= config.quality.MAX_ITERATIONS:
            return PHASE_ESCALATE
        return PHASE_REVISE


def route_after_revise(state: Dict[str, Any]) -> str:
    """Route after revision phase."""
    return PHASE_VERIFY  # Always re-verify after revision


# ============================================================================
# Helper Functions
# ============================================================================

def _build_fact_sheet(facts: list) -> str:
    """Build fact sheet string for writer prompt."""
    if not facts:
        return """
FACT SHEET: No verified facts available.
Write with general knowledge only. Avoid specific claims without sources.
"""

    lines = ["FACT SHEET (Use only these verified facts):"]
    for i, fact in enumerate(facts, 1):
        if isinstance(fact, dict):
            text = fact.get("fact", fact.get("text", str(fact)))
            source = fact.get("source_url", fact.get("source", ""))
            lines.append(f"  {i}. {text}")
            if source:
                lines.append(f"     Source: {source}")
        else:
            lines.append(f"  {i}. {fact}")

    return "\n".join(lines)


def _build_fix_instructions(priority_fixes: list, critical_failures: list) -> str:
    """Build fix instructions from review feedback."""
    lines = ["FIX THESE ISSUES (in order of priority):"]

    if critical_failures:
        lines.append("\nCRITICAL (Must fix):")
        for i, fix in enumerate(critical_failures, 1):
            lines.append(f"  {i}. {fix}")

    if priority_fixes:
        lines.append("\nPRIORITY FIXES:")
        for i, fix in enumerate(priority_fixes, 1):
            lines.append(f"  {i}. {fix}")

    if not critical_failures and not priority_fixes:
        lines.append("  - Improve overall quality and clarity")
        lines.append("  - Strengthen the opening hook")
        lines.append("  - Make the CTA more specific")

    return "\n".join(lines)


# ============================================================================
# Pipeline Builder
# ============================================================================

def create_pipeline(checkpointer: Optional[SqliteSaver] = None) -> StateGraph:
    """
    Create the LangGraph pipeline for article generation.

    Args:
        checkpointer: Optional SQLite checkpointer for persistence.
                      If None, a default SQLite checkpointer is created automatically.

    Returns:
        Compiled StateGraph ready for execution
    """
    # Default to SQLite checkpointing for crash resilience
    if checkpointer is None:
        try:
            import sqlite3
            db_path = config.paths.TEMP_DIR / ".pipeline_checkpoints.db"
            db_path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(str(db_path), check_same_thread=False)
            checkpointer = SqliteSaver(conn)
            print(f"  Checkpointing enabled: {db_path}")
        except Exception as e:
            print(f"  Checkpointing unavailable ({e}), running without persistence")
            checkpointer = None

    # Create graph
    builder = StateGraph(PipelineState)  # Typed state with validation

    # Add nodes
    builder.add_node(PHASE_RESEARCH, research_node)
    builder.add_node(PHASE_GENERATE, generate_node)
    builder.add_node(PHASE_VERIFY, verify_node)
    builder.add_node(PHASE_REVIEW, review_node)
    builder.add_node(PHASE_REVISE, revise_node)
    builder.add_node(PHASE_STYLE_CHECK, style_check_node)
    builder.add_node(PHASE_APPROVE, approve_node)
    builder.add_node(PHASE_ESCALATE, escalate_node)

    # Set entry point
    builder.set_entry_point(PHASE_RESEARCH)

    # Add edges
    builder.add_edge(PHASE_RESEARCH, PHASE_GENERATE)
    builder.add_edge(PHASE_GENERATE, PHASE_VERIFY)

    # Conditional edges
    builder.add_conditional_edges(
        PHASE_VERIFY,
        route_after_verify,
        {
            PHASE_REVIEW: PHASE_REVIEW,
            PHASE_REVISE: PHASE_REVISE,
            PHASE_ESCALATE: PHASE_ESCALATE
        }
    )

    builder.add_conditional_edges(
        PHASE_REVIEW,
        route_after_review,
        {
            PHASE_APPROVE: PHASE_STYLE_CHECK,  # Route through style check before approve
            PHASE_REVISE: PHASE_REVISE,
            PHASE_ESCALATE: PHASE_ESCALATE
        }
    )

    # Style check gates before final approval
    builder.add_conditional_edges(
        PHASE_STYLE_CHECK,
        route_after_style_check,
        {
            PHASE_APPROVE: PHASE_APPROVE,
            PHASE_REVISE: PHASE_REVISE,
            PHASE_ESCALATE: PHASE_ESCALATE
        }
    )

    builder.add_edge(PHASE_REVISE, PHASE_VERIFY)  # Loop back to verify
    builder.add_edge(PHASE_APPROVE, END)
    builder.add_edge(PHASE_ESCALATE, END)

    # Compile
    if checkpointer:
        return builder.compile(checkpointer=checkpointer)
    return builder.compile()


def create_pipeline_with_sqlite(db_path: str = ".ghostwriter_checkpoints.db") -> StateGraph:
    """
    Create pipeline with SQLite checkpointing for persistence.

    Args:
        db_path: Path to SQLite database file

    Returns:
        Compiled StateGraph with checkpoint support
    """
    from langgraph.checkpoint.sqlite import SqliteSaver
    import sqlite3

    conn = sqlite3.connect(db_path, check_same_thread=False)
    checkpointer = SqliteSaver(conn)

    return create_pipeline(checkpointer)


def run_pipeline(
    pipeline: StateGraph,
    topic: str,
    source_content: str = "",
    source_type: str = "external",
    platform: str = "medium",
    thread_id: Optional[str] = None,
    source_url: str = None,
    track_provenance: bool = True
) -> Dict[str, Any]:
    """
    Run the pipeline to generate an article with provenance tracking.

    Args:
        pipeline: Compiled StateGraph
        topic: Article topic
        source_content: Source material
        source_type: "external" or "internal"
        platform: Target platform
        thread_id: Optional thread ID for checkpointing
        source_url: URL of source material (for provenance)
        track_provenance: Enable provenance tracking (default: True)

    Returns:
        Final article state with provenance data
    """
    # Initialize provenance tracker
    tracker = None
    if track_provenance:
        tracker = ProvenanceTracker()
        tracker.start_tracking(
            topic=topic,
            source_type=source_type,
            source_url=source_url,
            platform=platform
        )

    # Create initial state
    initial_state = create_initial_state(
        topic=topic,
        source_content=source_content,
        source_type=source_type,
        platform=platform
    )

    # Run pipeline (thread_id required when checkpointing is enabled)
    config_dict = {}
    if not thread_id:
        import uuid
        thread_id = str(uuid.uuid4())[:8]
    config_dict["configurable"] = {"thread_id": thread_id}

    final_state = dict(initial_state)  # Start with copy of initial state
    for state in pipeline.stream(initial_state, config=config_dict):
        for node_name, node_state in state.items():
            # Merge with special handling for list fields (append, don't replace)
            for key, value in node_state.items():
                if key == "error_messages" and isinstance(value, list):
                    existing = final_state.get("error_messages", [])
                    new_errors = [e for e in value if e not in existing]
                    final_state["error_messages"] = existing + new_errors
                elif key == "reviewer_feedback" and isinstance(value, list):
                    existing = final_state.get("reviewer_feedback", [])
                    final_state["reviewer_feedback"] = existing + [v for v in value if v not in existing]
                else:
                    final_state[key] = value
            print(f"  [{node_name}] -> {node_state.get('next_action', 'unknown')}")

            # Track provenance for each phase
            if tracker:
                if node_name == PHASE_RESEARCH:
                    tracker.record_research(
                        "ResearchAgent",
                        model=node_state.get("research_model", config.models.RESEARCH_MODEL_PRIMARY),
                        facts_found=len(node_state.get("research_facts", []))
                    )
                elif node_name == PHASE_GENERATE:
                    tracker.record_generation(
                        "WriterAgent",
                        model=node_state.get("writer_model", config.models.DEFAULT_WRITER_MODEL),
                        word_count=node_state.get("word_count", 0)
                    )
                elif node_name == PHASE_VERIFY:
                    verification = node_state.get("verification_result", {})
                    tracker.record_verification(
                        passed=node_state.get("verification_passed", False),
                        claims_verified=verification.get("verified_count", 0),
                        false_claims=verification.get("false_count", 0)
                    )
                elif node_name == PHASE_REVIEW:
                    tracker.record_review(
                        score=node_state.get("quality_score", 0),
                        passed=node_state.get("quality_passed", False),
                        wsj_passed=node_state.get("wsj_checklist", {}).get("overall_passed", False),
                        iteration=node_state.get("review_iterations", 1)
                    )
                elif node_name == PHASE_REVISE:
                    tracker.record_revision(
                        "WriterAgent",
                        iteration=node_state.get("review_iterations", 1)
                    )

    # Finalize provenance
    if tracker and final_state:
        final_content = final_state.get("final_content", final_state.get("draft", ""))
        human_reviewed = final_state.get("human_reviewed", False)
        provenance = tracker.finalize(final_content, human_reviewed=human_reviewed)

        # Add provenance to final state
        final_state["provenance"] = provenance.to_dict()
        final_state["provenance_disclosure"] = generate_inline_disclosure(provenance, "brief")
        final_state["content_id"] = provenance.content_id

    return final_state


# ============================================================================
# CLI Entry Point
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="GhostWriter Pipeline")
    parser.add_argument("--topic", "-t", required=True, help="Article topic")
    parser.add_argument("--source", "-s", default="", help="Source content")
    parser.add_argument("--source-type", choices=["external", "internal"], default="external")
    parser.add_argument("--platform", "-p", choices=["medium", "linkedin"], default="medium")
    parser.add_argument("--output", "-o", help="Output file path")
    parser.add_argument("--checkpoint", "-c", help="SQLite checkpoint database path")

    args = parser.parse_args()

    # Create pipeline
    if args.checkpoint:
        pipeline = create_pipeline_with_sqlite(args.checkpoint)
    else:
        pipeline = create_pipeline()

    print("\n" + "=" * 70)
    print("GHOSTWRITER PIPELINE")
    print("=" * 70)
    print(f"Topic: {args.topic}")
    print(f"Platform: {args.platform}")
    print(f"Source Type: {args.source_type}")
    print("=" * 70)

    # Run pipeline
    result = run_pipeline(
        pipeline,
        topic=args.topic,
        source_content=args.source,
        source_type=args.source_type,
        platform=args.platform
    )

    # Output results
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    print(f"Final Phase: {result.get('current_phase', 'unknown')}")
    print(f"Quality Passed: {result.get('quality_passed', False)}")
    print(f"Verification Passed: {result.get('verification_passed', False)}")
    print(f"Requires Human Review: {result.get('requires_human_review', False)}")
    print(f"Word Count: {result.get('word_count', 0)}")

    # Show provenance info
    if result.get("content_id"):
        print(f"Content ID: {result.get('content_id')}")
        print(f"Provenance: {result.get('provenance_disclosure', 'N/A')}")

    # Save output
    final_content = result.get("final_content", result.get("draft", ""))
    if final_content:
        if args.output:
            output_path = Path(args.output)
        else:
            from execution.config import OUTPUT_DIR
            content_id = result.get("content_id", result.get("article_id", "unknown"))
            output_path = OUTPUT_DIR / f"article_{content_id}.md"

        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Add disclosure footer to content
        disclosure = result.get("provenance_disclosure", "")
        content_with_disclosure = final_content
        if disclosure:
            content_with_disclosure = f"{final_content}\n\n---\n*{disclosure}*\n"

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content_with_disclosure)

        print(f"Output saved: {output_path}")

        # Export provenance files
        if result.get("provenance"):
            provenance_dir = output_path.parent / "provenance"
            provenance_dir.mkdir(exist_ok=True)

            content_id = result.get("content_id", "unknown")

            # Export full provenance JSON
            provenance_path = provenance_dir / f"{content_id}_provenance.json"
            with open(provenance_path, "w", encoding="utf-8") as f:
                json.dump(result["provenance"], f, indent=2)
            print(f"Provenance: {provenance_path}")

            # Export C2PA manifest
            from execution.provenance import ContentProvenance
            prov_data = result["provenance"]
            prov_obj = ContentProvenance(
                content_id=prov_data["content_id"],
                content_hash=prov_data["content_hash"],
                created_at=prov_data["created_at"],
                topic=prov_data.get("topic", ""),
                platform=prov_data.get("platform", "medium"),
                quality_score=prov_data.get("quality_score", 0),
                fact_verification_passed=prov_data.get("fact_verification_passed", False),
                verified_claims_count=prov_data.get("verified_claims_count", 0),
                models_used=prov_data.get("models_used", []),
                human_reviewed=prov_data.get("human_reviewed", False)
            )

            c2pa_path = provenance_dir / f"{content_id}_c2pa.json"
            c2pa_manifest = generate_c2pa_manifest(prov_obj)
            with open(c2pa_path, "w", encoding="utf-8") as f:
                json.dump(c2pa_manifest, f, indent=2)
            print(f"C2PA Manifest: {c2pa_path}")

            # Export Schema.org JSON-LD
            schema_path = provenance_dir / f"{content_id}_schema.json"
            schema_jsonld = generate_schema_org_jsonld(
                prov_obj,
                title=args.topic,
                description=f"AI-generated article about {args.topic}"
            )
            with open(schema_path, "w", encoding="utf-8") as f:
                json.dump(schema_jsonld, f, indent=2)
            print(f"Schema.org: {schema_path}")

    print("=" * 70)
