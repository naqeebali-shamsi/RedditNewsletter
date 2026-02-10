# GhostWriter → WSJ-Tier AI Writing Agency
## Comprehensive Transformation Roadmap

> **Mission**: Build the world's first and best SOTA AI-powered writing agency at par with The Wall Street Journal, BBC, CNN, and CBC Radio.

---

## Executive Summary

### Current State Assessment

| Metric | Score | Gap to WSJ-Tier |
|--------|-------|-----------------|
| **Production Readiness** | 45/100 | Critical gaps in verification, HITL |
| **Fact Verification** | Optional | Must be mandatory + multi-source |
| **Adversarial Review** | Theatrical | Need true multi-model critique |
| **Human Oversight** | Flag only | Need full approval workflows |
| **Content Provenance** | None | Need C2PA + audit trail |
| **Test Coverage** | ~5% | Need 80%+ |

### Target Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    WSJ-TIER AI WRITING AGENCY                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    LAYER 1: DISCOVERY                        │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │   │
│  │  │ Signal      │  │ Topic       │  │ Audience            │  │   │
│  │  │ Detection   │──│ Validation  │──│ Analysis            │  │   │
│  │  └─────────────┘  └─────────────┘  └─────────────────────┘  │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              ↓                                      │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    LAYER 2: RESEARCH                         │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │   │
│  │  │ Gemini      │  │ Perplexity  │  │ Domain Expert       │  │   │
│  │  │ (Grounded)  │──│ (Citations) │──│ (RAG Knowledge)     │  │   │
│  │  └─────────────┘  └─────────────┘  └─────────────────────┘  │   │
│  │                   ↓ Fact Sheet with Sources                  │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              ↓                                      │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    LAYER 3: PRODUCTION                       │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │   │
│  │  │ Editor      │──│ Writer      │──│ Voice Transform     │  │   │
│  │  │ (Structure) │  │ (Draft)     │  │ (WSJ/BBC/CBC)       │  │   │
│  │  └─────────────┘  └─────────────┘  └─────────────────────┘  │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              ↓                                      │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                LAYER 4: VERIFICATION (Multi-Model)           │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │   │
│  │  │ Claude      │  │ GPT-4       │  │ Gemini              │  │   │
│  │  │ (Ethics)    │──│ (Structure) │──│ (Facts)             │  │   │
│  │  └─────────────┘  └─────────────┘  └─────────────────────┘  │   │
│  │                   ↓ Unanimous or Escalate                    │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              ↓                                      │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    LAYER 5: HUMAN GATE                       │   │
│  │  ┌─────────────────────────────────────────────────────────┐│   │
│  │  │  Review Queue │ Approve │ Reject │ Request Revision    ││   │
│  │  │  ───────────────────────────────────────────────────── ││   │
│  │  │  C2PA Manifest │ Provenance │ Audit Trail │ Timestamp  ││   │
│  │  └─────────────────────────────────────────────────────────┘│   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              ↓                                      │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    LAYER 6: PUBLISHING                       │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │   │
│  │  │ Medium      │  │ Newsletter  │  │ Distribution        │  │   │
│  │  │ Formatter   │──│ Optimizer   │──│ Tracker             │  │   │
│  │  └─────────────┘  └─────────────┘  └─────────────────────┘  │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Phase 1: Critical Fixes (Foundation)
**Timeline: Immediate**
**Goal: Raise production readiness from 45/100 to 65/100**

### 1.1 Fix Hardcoded Windows Paths

**Problem**: Paths like `n:/RedditNews/drafts` break cross-platform deployment.

**Files to Fix**:
- `execution/generate_medium_full.py` (line 27)
- `app.py` (line 1186)
- Various agent files

**Solution**:
```python
# Before
OUTPUT_DIR = Path("n:/RedditNews/drafts")

# After
from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = Path(os.getenv("GHOSTWRITER_OUTPUT_DIR", PROJECT_ROOT / "drafts"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
```

**Create** `execution/config.py`:
```python
"""Centralized configuration management."""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = Path(os.getenv("GHOSTWRITER_OUTPUT_DIR", PROJECT_ROOT / "drafts"))
TEMP_DIR = Path(os.getenv("GHOSTWRITER_TEMP_DIR", PROJECT_ROOT / ".tmp"))

# API Keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Quality thresholds
PASS_THRESHOLD = 7.0
MAX_ITERATIONS = 3
FACT_VERIFICATION_REQUIRED = True

# Model configuration
DEFAULT_WRITER_MODEL = os.getenv("DEFAULT_WRITER_MODEL", "llama-3.3-70b-versatile")
DEFAULT_CRITIC_MODEL = os.getenv("DEFAULT_CRITIC_MODEL", "llama-3.3-70b-versatile")
```

### 1.2 Make Fact Verification MANDATORY

**Problem**: Current code skips fact verification if API key is missing.

```python
# CURRENT (app.py lines 1282-1309) - DANGEROUS
if perplexity_available and fact_researcher:
    fact_sheet = fact_researcher.research_topic(topic, source_content)
else:
    status_callback("Skipping fact research (no Perplexity API key)")
    # Writer has NO constraints - can hallucinate freely!
```

**Solution**: Create verification chain with fallbacks.

```python
# execution/verification/fact_chain.py
"""Mandatory fact verification with fallback chain."""

class FactVerificationChain:
    """Never skip fact verification - use fallbacks."""

    def __init__(self):
        self.verifiers = []
        self._init_verifiers()

    def _init_verifiers(self):
        """Initialize verifiers in priority order."""
        # Priority 1: Gemini with Google Search
        try:
            from execution.agents.gemini_researcher import GeminiResearchAgent
            self.verifiers.append(("Gemini", GeminiResearchAgent()))
        except Exception:
            pass

        # Priority 2: Perplexity Sonar Pro
        try:
            from execution.agents.perplexity_researcher import PerplexityResearchAgent
            self.verifiers.append(("Perplexity", PerplexityResearchAgent()))
        except Exception:
            pass

        # Priority 3: Claude with web search MCP
        try:
            from execution.agents.claude_researcher import ClaudeResearchAgent
            self.verifiers.append(("Claude", ClaudeResearchAgent()))
        except Exception:
            pass

        if not self.verifiers:
            raise RuntimeError(
                "CRITICAL: No fact verification available. "
                "At least one of GOOGLE_API_KEY, PERPLEXITY_API_KEY, or "
                "ANTHROPIC_API_KEY must be set. "
                "WSJ-tier journalism requires fact verification."
            )

    def verify(self, topic: str, source_content: str) -> dict:
        """Verify facts using available verifiers."""
        for name, verifier in self.verifiers:
            try:
                result = verifier.research_topic(topic, source_content)
                result["verifier_used"] = name
                return result
            except Exception as e:
                continue

        # Should never reach here due to __init__ check
        raise RuntimeError("All fact verifiers failed")
```

### 1.3 Integrate TechnicalSupervisor into Main Pipeline

**Problem**: TechnicalSupervisor exists but isn't integrated.

**Solution**: Add to quality gate.

```python
# execution/quality_gate.py - Enhanced

from execution.agents.adversarial_panel import AdversarialPanelAgent
from execution.agents.technical_supervisor import TechnicalSupervisor

class QualityGate:
    def __init__(self):
        self.panel = AdversarialPanelAgent()
        self.tech_supervisor = TechnicalSupervisor()

    def review(self, article: str, metadata: dict) -> QualityGateResult:
        """Run full quality review including technical supervision."""
        # Panel review (creativity, engagement, voice)
        panel_result = self.panel.review(article)

        # Technical review (facts, structure, SEO)
        tech_result = self.tech_supervisor.review(article, metadata)

        # Combine scores (weighted)
        combined_score = (
            panel_result.score * 0.6 +
            tech_result.score * 0.4
        )

        # Must pass BOTH thresholds
        passes = (
            panel_result.score >= 7.0 and
            tech_result.score >= 7.0
        )

        return QualityGateResult(
            passes=passes,
            score=combined_score,
            panel_feedback=panel_result.feedback,
            tech_feedback=tech_result.feedback,
            requires_human_review=panel_result.escalate or tech_result.escalate
        )
```

### 1.4 Add Structured Logging

**Problem**: Print statements scattered throughout; no audit trail.

**Solution**:
```python
# execution/logging_config.py
import logging
import json
from datetime import datetime
from pathlib import Path

def setup_logging(run_id: str = None):
    """Configure structured logging for audit trail."""
    run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create logs directory
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    # JSON formatter for structured logs
    class JSONFormatter(logging.Formatter):
        def format(self, record):
            log_entry = {
                "timestamp": datetime.utcnow().isoformat(),
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
                "run_id": run_id,
            }
            if hasattr(record, "agent"):
                log_entry["agent"] = record.agent
            if hasattr(record, "action"):
                log_entry["action"] = record.action
            if hasattr(record, "metrics"):
                log_entry["metrics"] = record.metrics
            return json.dumps(log_entry)

    # File handler (JSON for parsing)
    file_handler = logging.FileHandler(log_dir / f"run_{run_id}.jsonl")
    file_handler.setFormatter(JSONFormatter())

    # Console handler (human readable)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
    ))

    # Configure root logger
    root_logger = logging.getLogger("ghostwriter")
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    return root_logger, run_id
```

---

## Phase 2: Architecture Modernization
**Timeline: 2-3 weeks after Phase 1**
**Goal: Raise production readiness from 65/100 to 80/100**

### 2.1 LangGraph State Machine

**Why LangGraph**:
- Node-level caching (50% cost reduction)
- `interrupt_before` for human gates
- LangGraph Studio for debugging
- Built-in state persistence

**Implementation**:
```python
# execution/pipeline/langgraph_orchestrator.py
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from typing import TypedDict, Annotated, List
import operator

class ArticleState(TypedDict):
    """State passed through the pipeline."""
    topic: str
    source_content: str
    source_type: str  # external/internal

    # Research phase
    fact_sheet: dict
    research_sources: List[str]

    # Production phase
    outline: str
    draft: str
    voice_transformed: str

    # Verification phase
    panel_scores: List[dict]
    tech_score: float
    combined_score: float

    # Human review
    requires_human_review: bool
    human_approved: bool
    human_feedback: str

    # Iteration tracking
    iteration: int
    revision_history: Annotated[List[str], operator.add]

    # Final output
    final_article: str
    c2pa_manifest: dict

def create_pipeline():
    """Create the LangGraph pipeline."""
    workflow = StateGraph(ArticleState)

    # Add nodes
    workflow.add_node("research", research_node)
    workflow.add_node("outline", outline_node)
    workflow.add_node("write", write_node)
    workflow.add_node("voice_transform", voice_transform_node)
    workflow.add_node("panel_review", panel_review_node)
    workflow.add_node("tech_review", tech_review_node)
    workflow.add_node("human_gate", human_gate_node)
    workflow.add_node("revise", revise_node)
    workflow.add_node("finalize", finalize_node)

    # Define edges
    workflow.add_edge("research", "outline")
    workflow.add_edge("outline", "write")
    workflow.add_edge("write", "voice_transform")
    workflow.add_edge("voice_transform", "panel_review")
    workflow.add_edge("panel_review", "tech_review")

    # Conditional routing after tech review
    workflow.add_conditional_edges(
        "tech_review",
        route_after_review,
        {
            "pass": "human_gate",
            "revise": "revise",
            "fail": END
        }
    )

    # Human gate with interrupt
    workflow.add_conditional_edges(
        "human_gate",
        route_human_decision,
        {
            "approved": "finalize",
            "revise": "revise",
            "rejected": END
        }
    )

    workflow.add_edge("revise", "write")
    workflow.add_edge("finalize", END)

    workflow.set_entry_point("research")

    # Compile with SQLite checkpointing
    memory = SqliteSaver.from_conn_string(":memory:")
    return workflow.compile(
        checkpointer=memory,
        interrupt_before=["human_gate"]  # Pause for human review
    )

def route_after_review(state: ArticleState) -> str:
    """Route based on review scores."""
    if state["combined_score"] >= 7.0:
        if state["requires_human_review"]:
            return "pass"  # Goes to human gate
        return "pass"
    elif state["iteration"] < 3:
        return "revise"
    else:
        return "fail"

def human_gate_node(state: ArticleState) -> ArticleState:
    """Human review gate with LangGraph interrupt."""
    from langgraph.types import interrupt

    # Present article for review
    review_data = {
        "article": state["voice_transformed"],
        "score": state["combined_score"],
        "panel_feedback": state["panel_scores"],
        "sources": state["research_sources"]
    }

    # This will pause execution until human responds
    human_decision = interrupt(review_data)

    return {
        **state,
        "human_approved": human_decision.get("approved", False),
        "human_feedback": human_decision.get("feedback", "")
    }
```

### 2.2 True Multi-Model Adversarial Review

**Problem**: Current panel uses same model with different prompts (theatrical adversarialism).

**Solution**: Use different model families for genuine adversarial perspectives.

```python
# execution/agents/multi_model_panel.py
"""True adversarial review using different model families."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional
import asyncio

@dataclass
class ReviewResult:
    model: str
    perspective: str
    score: float
    critique: str
    kill_phrases_found: List[str]
    suggestions: List[str]

class BaseReviewer(ABC):
    """Base class for model-specific reviewers."""

    @abstractmethod
    async def review(self, article: str, context: dict) -> ReviewResult:
        pass

class ClaudeEthicsReviewer(BaseReviewer):
    """Claude for ethics, bias, and tone analysis."""

    def __init__(self):
        from anthropic import AsyncAnthropic
        self.client = AsyncAnthropic()

    async def review(self, article: str, context: dict) -> ReviewResult:
        response = await self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2000,
            messages=[{
                "role": "user",
                "content": f"""You are a senior ethics editor at The Wall Street Journal.

Review this article for:
1. Ethical concerns (privacy, harm, fairness)
2. Bias detection (political, cultural, demographic)
3. Tone appropriateness (matches claimed voice/source type)
4. Attribution integrity (claims properly sourced)

Article:
{article}

Source type: {context.get('source_type', 'external')}

Respond in JSON:
{{
    "score": 0-10,
    "ethical_concerns": [],
    "bias_detected": [],
    "tone_issues": [],
    "attribution_problems": [],
    "overall_critique": "",
    "suggestions": []
}}"""
            }]
        )
        # Parse and return ReviewResult
        ...

class GPT4StructureReviewer(BaseReviewer):
    """GPT-4 for structure, flow, and engagement analysis."""

    def __init__(self):
        from openai import AsyncOpenAI
        self.client = AsyncOpenAI()

    async def review(self, article: str, context: dict) -> ReviewResult:
        response = await self.client.chat.completions.create(
            model="gpt-4o",
            messages=[{
                "role": "system",
                "content": "You are a senior editor at BBC News specializing in story structure."
            }, {
                "role": "user",
                "content": f"""Analyze this article's structure:

1. Lead effectiveness (does it hook the reader?)
2. Nut graph clarity (is the "so what" clear?)
3. Body organization (logical flow?)
4. Transitions quality
5. Ending impact (memorable kicker?)
6. Kill phrases (clichés, weak CTAs, buzzwords)

Article:
{article}

Return JSON with score 0-10 and detailed feedback."""
            }]
        )
        ...

class GeminiFactReviewer(BaseReviewer):
    """Gemini for fact verification against research."""

    def __init__(self):
        import google.generativeai as genai
        self.model = genai.GenerativeModel("gemini-2.0-flash")

    async def review(self, article: str, context: dict) -> ReviewResult:
        fact_sheet = context.get("fact_sheet", {})

        response = await self.model.generate_content_async(f"""
You are a fact-checker at CNN's Facts First unit.

Compare this article against the verified fact sheet.
Flag any claims that:
1. Contradict verified facts
2. Aren't supported by the research
3. Exaggerate or distort findings
4. Use weasel words to avoid verification

Article:
{article}

Fact Sheet:
{fact_sheet}

Return JSON with score 0-10 and list of issues.""")
        ...

class MultiModelPanel:
    """Orchestrate multi-model adversarial review."""

    def __init__(self):
        self.reviewers = [
            ClaudeEthicsReviewer(),
            GPT4StructureReviewer(),
            GeminiFactReviewer()
        ]

    async def review(self, article: str, context: dict) -> dict:
        """Run all reviewers in parallel."""
        results = await asyncio.gather(*[
            r.review(article, context) for r in self.reviewers
        ])

        # Unanimous pass required for auto-approval
        scores = [r.score for r in results]
        min_score = min(scores)
        avg_score = sum(scores) / len(scores)

        # Any reviewer can trigger human escalation
        requires_human = any(r.score < 6.0 for r in results)

        return {
            "passes": min_score >= 7.0,
            "requires_human_review": requires_human,
            "min_score": min_score,
            "avg_score": avg_score,
            "results": [r.__dict__ for r in results]
        }
```

### 2.3 Cascaded Routing for Cost Optimization

**Problem**: Using expensive models for everything wastes money.

**Solution**: Screen with cheap models, escalate complex cases.

```python
# execution/routing/cascaded_router.py
"""Cost-optimized model routing."""

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional

class Complexity(Enum):
    SIMPLE = "simple"      # Haiku/GPT-4o-mini
    MEDIUM = "medium"      # Sonnet/GPT-4o
    COMPLEX = "complex"    # Opus/GPT-4

@dataclass
class RoutingDecision:
    complexity: Complexity
    model: str
    reasoning: str
    estimated_cost: float

class CascadedRouter:
    """Route tasks to appropriate model tier."""

    TIER_MODELS = {
        Complexity.SIMPLE: {
            "anthropic": "claude-3-haiku-20240307",
            "openai": "gpt-4o-mini",
            "google": "gemini-2.0-flash"
        },
        Complexity.MEDIUM: {
            "anthropic": "claude-sonnet-4-20250514",
            "openai": "gpt-4o",
            "google": "gemini-2.0-pro"
        },
        Complexity.COMPLEX: {
            "anthropic": "claude-opus-4-20250514",
            "openai": "gpt-4",
            "google": "gemini-2.0-ultra"
        }
    }

    async def classify_task(self, task: str, context: dict) -> Complexity:
        """Use cheap model to classify task complexity."""
        classifier_prompt = f"""Classify this writing task's complexity:

Task: {task}
Context: {context}

Respond with ONE word: simple, medium, or complex

- simple: Straightforward edits, summaries, formatting
- medium: Original writing, analysis, moderate research
- complex: Investigative pieces, technical deep-dives, sensitive topics"""

        # Use cheapest model for classification
        response = await self._quick_classify(classifier_prompt)

        try:
            return Complexity(response.strip().lower())
        except ValueError:
            return Complexity.MEDIUM  # Default to medium

    async def route(
        self,
        task: str,
        context: dict,
        provider: str = "anthropic"
    ) -> RoutingDecision:
        """Route task to appropriate model."""
        complexity = await self.classify_task(task, context)
        model = self.TIER_MODELS[complexity][provider]

        costs = {
            Complexity.SIMPLE: 0.001,
            Complexity.MEDIUM: 0.015,
            Complexity.COMPLEX: 0.075
        }

        return RoutingDecision(
            complexity=complexity,
            model=model,
            reasoning=f"Task classified as {complexity.value}",
            estimated_cost=costs[complexity]
        )
```

---

## Phase 3: WSJ-Tier Quality Systems
**Timeline: 3-4 weeks after Phase 2**
**Goal: Raise production readiness from 80/100 to 95/100**

### 3.1 WSJ Four Showstoppers Implementation

Based on WSJ's actual editorial standards:

```python
# execution/verification/wsj_showstoppers.py
"""WSJ Four Showstoppers quality framework."""

from dataclasses import dataclass
from typing import List, Optional
from enum import Enum

class ShowstopperResult(Enum):
    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"

@dataclass
class ShowstopperCheck:
    name: str
    result: ShowstopperResult
    details: str
    suggestions: List[str]

class WSJShowstoppers:
    """
    WSJ's Four Showstoppers:
    1. Attribution - Are sources stated explicitly?
    2. Quality of Sources - First-hand expertise?
    3. Tone - Bias-free language?
    4. No Surprises - Subjects given chance to comment?
    """

    def check_attribution(self, article: str, fact_sheet: dict) -> ShowstopperCheck:
        """Every claim must have clear attribution."""
        unattributed_claims = self._find_unattributed_claims(article, fact_sheet)

        if not unattributed_claims:
            return ShowstopperCheck(
                name="Attribution",
                result=ShowstopperResult.PASS,
                details="All claims properly attributed",
                suggestions=[]
            )
        elif len(unattributed_claims) <= 2:
            return ShowstopperCheck(
                name="Attribution",
                result=ShowstopperResult.WARN,
                details=f"Found {len(unattributed_claims)} weakly attributed claims",
                suggestions=[f"Strengthen attribution for: {c}" for c in unattributed_claims]
            )
        else:
            return ShowstopperCheck(
                name="Attribution",
                result=ShowstopperResult.FAIL,
                details=f"CRITICAL: {len(unattributed_claims)} claims lack attribution",
                suggestions=[f"Must add source for: {c}" for c in unattributed_claims]
            )

    def check_source_quality(self, article: str, sources: List[dict]) -> ShowstopperCheck:
        """Sources must be first-hand with relevant expertise."""
        issues = []

        for source in sources:
            # Check if source is first-hand
            if source.get("hand") == "second":
                issues.append(f"{source['name']} is second-hand source")

            # Check expertise relevance
            if not source.get("expertise_relevant"):
                issues.append(f"{source['name']} lacks relevant expertise")

        if not issues:
            return ShowstopperCheck(
                name="Source Quality",
                result=ShowstopperResult.PASS,
                details="All sources are first-hand with relevant expertise",
                suggestions=[]
            )
        elif len(issues) <= 2:
            return ShowstopperCheck(
                name="Source Quality",
                result=ShowstopperResult.WARN,
                details=f"Some source quality concerns",
                suggestions=issues
            )
        else:
            return ShowstopperCheck(
                name="Source Quality",
                result=ShowstopperResult.FAIL,
                details="CRITICAL: Major source quality issues",
                suggestions=issues
            )

    def check_tone(self, article: str) -> ShowstopperCheck:
        """Language must be neutral, free of loaded modifiers."""
        loaded_words = self._find_loaded_language(article)
        bias_indicators = self._detect_bias_patterns(article)

        total_issues = len(loaded_words) + len(bias_indicators)

        if total_issues == 0:
            return ShowstopperCheck(
                name="Tone",
                result=ShowstopperResult.PASS,
                details="Neutral tone maintained throughout",
                suggestions=[]
            )
        elif total_issues <= 3:
            return ShowstopperCheck(
                name="Tone",
                result=ShowstopperResult.WARN,
                details=f"Minor tone issues detected",
                suggestions=[f"Remove loaded word: {w}" for w in loaded_words]
            )
        else:
            return ShowstopperCheck(
                name="Tone",
                result=ShowstopperResult.FAIL,
                details="CRITICAL: Significant bias detected",
                suggestions=[f"Rewrite for neutrality: {b}" for b in bias_indicators]
            )

    def check_no_surprises(self, article: str, subjects: List[str]) -> ShowstopperCheck:
        """Subjects being criticized must have had chance to respond."""
        # This requires metadata about who was contacted
        uncontacted = [s for s in subjects if not s.get("contacted")]

        if not uncontacted:
            return ShowstopperCheck(
                name="No Surprises",
                result=ShowstopperResult.PASS,
                details="All subjects given opportunity to respond",
                suggestions=[]
            )
        else:
            return ShowstopperCheck(
                name="No Surprises",
                result=ShowstopperResult.FAIL,
                details=f"CRITICAL: {len(uncontacted)} subjects not contacted",
                suggestions=[f"Must contact: {s['name']}" for s in uncontacted]
            )

    def run_all(self, article: str, context: dict) -> dict:
        """Run all four showstopper checks."""
        results = [
            self.check_attribution(article, context.get("fact_sheet", {})),
            self.check_source_quality(article, context.get("sources", [])),
            self.check_tone(article),
            self.check_no_surprises(article, context.get("subjects", []))
        ]

        # Any FAIL = overall fail
        overall_pass = not any(r.result == ShowstopperResult.FAIL for r in results)

        return {
            "passes": overall_pass,
            "checks": [r.__dict__ for r in results],
            "fail_count": sum(1 for r in results if r.result == ShowstopperResult.FAIL),
            "warn_count": sum(1 for r in results if r.result == ShowstopperResult.WARN)
        }
```

### 3.2 C2PA Content Provenance

**Why C2PA**: EU AI Act Article 50 requires provenance tracking. Major platforms (Meta, YouTube) already use C2PA.

```python
# execution/provenance/c2pa_manifest.py
"""C2PA Content Credentials for provenance tracking."""

import json
import hashlib
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List, Optional

@dataclass
class AgentAction:
    """Record of an agent's action on content."""
    agent_name: str
    agent_type: str  # researcher, writer, reviewer, etc.
    model_used: str
    action: str  # researched, drafted, reviewed, revised, approved
    timestamp: str
    input_hash: str
    output_hash: str
    metadata: dict

@dataclass
class HumanAction:
    """Record of human intervention."""
    reviewer_id: str  # Anonymized
    action: str  # approved, rejected, revised
    timestamp: str
    feedback: Optional[str]

@dataclass
class C2PAManifest:
    """Content Credentials manifest for AI-generated content."""

    # Content identification
    content_hash: str
    content_type: str  # article, newsletter, social
    created_at: str

    # Source provenance
    source_urls: List[str]
    source_type: str  # external, internal

    # Agent chain
    agent_actions: List[AgentAction]

    # Human oversight
    human_actions: List[HumanAction]
    final_approval: Optional[HumanAction]

    # Verification results
    fact_verification: dict
    quality_scores: dict

    # Disclosure statement
    ai_disclosure: str = (
        "This content was created with AI assistance. "
        "All facts were verified by research agents and reviewed by human editors. "
        "See full provenance chain for details."
    )

    def to_json(self) -> str:
        """Serialize to JSON for embedding."""
        return json.dumps(asdict(self), indent=2)

    def to_html_meta(self) -> str:
        """Generate HTML meta tags for embedding."""
        return f"""
<!-- C2PA Content Credentials -->
<meta name="c2pa:claim" content="{self.content_hash}">
<meta name="c2pa:created" content="{self.created_at}">
<meta name="c2pa:ai-disclosure" content="{self.ai_disclosure}">
<meta name="c2pa:human-approved" content="{bool(self.final_approval)}">
<meta name="c2pa:provenance" content='{self.to_json()}'>
"""

class ProvenanceTracker:
    """Track content provenance throughout pipeline."""

    def __init__(self):
        self.actions: List[AgentAction] = []
        self.human_actions: List[HumanAction] = []
        self.sources: List[str] = []

    def record_agent_action(
        self,
        agent_name: str,
        agent_type: str,
        model: str,
        action: str,
        input_content: str,
        output_content: str,
        metadata: dict = None
    ):
        """Record an agent's action."""
        self.actions.append(AgentAction(
            agent_name=agent_name,
            agent_type=agent_type,
            model_used=model,
            action=action,
            timestamp=datetime.utcnow().isoformat(),
            input_hash=hashlib.sha256(input_content.encode()).hexdigest()[:16],
            output_hash=hashlib.sha256(output_content.encode()).hexdigest()[:16],
            metadata=metadata or {}
        ))

    def record_human_action(
        self,
        reviewer_id: str,
        action: str,
        feedback: str = None
    ):
        """Record human intervention."""
        self.human_actions.append(HumanAction(
            reviewer_id=hashlib.sha256(reviewer_id.encode()).hexdigest()[:8],
            action=action,
            timestamp=datetime.utcnow().isoformat(),
            feedback=feedback
        ))

    def build_manifest(
        self,
        final_content: str,
        content_type: str,
        source_type: str,
        fact_verification: dict,
        quality_scores: dict
    ) -> C2PAManifest:
        """Build final C2PA manifest."""
        return C2PAManifest(
            content_hash=hashlib.sha256(final_content.encode()).hexdigest(),
            content_type=content_type,
            created_at=datetime.utcnow().isoformat(),
            source_urls=self.sources,
            source_type=source_type,
            agent_actions=self.actions,
            human_actions=self.human_actions,
            final_approval=self.human_actions[-1] if self.human_actions else None,
            fact_verification=fact_verification,
            quality_scores=quality_scores
        )
```

### 3.3 Human-in-the-Loop Review UI

**Problem**: Escalation flag exists but no UI for humans to review.

**Solution**: Streamlit-based review queue.

```python
# execution/ui/review_queue.py
"""Human-in-the-loop review interface."""

import streamlit as st
from datetime import datetime
from pathlib import Path
import json

def render_review_queue():
    """Render the human review queue."""
    st.title("Editorial Review Queue")

    # Load pending reviews
    pending = load_pending_reviews()

    if not pending:
        st.success("No articles pending review!")
        return

    st.warning(f"{len(pending)} articles awaiting review")

    for review in pending:
        with st.expander(f"📄 {review['title']} (Score: {review['score']:.1f})"):
            render_review_card(review)

def render_review_card(review: dict):
    """Render a single review card."""
    col1, col2 = st.columns([2, 1])

    with col1:
        # Article preview
        st.subheader("Article Draft")
        st.markdown(review["article"][:2000] + "...")

        # Expandable full article
        with st.expander("View Full Article"):
            st.markdown(review["article"])

    with col2:
        # Scores and feedback
        st.subheader("Quality Assessment")

        # Panel scores
        for result in review.get("panel_results", []):
            score_color = "green" if result["score"] >= 7 else "orange" if result["score"] >= 5 else "red"
            st.markdown(f"**{result['model']}**: :{score_color}[{result['score']:.1f}/10]")
            with st.expander(f"Feedback from {result['model']}"):
                st.write(result["critique"])

        # Fact verification status
        st.subheader("Fact Verification")
        facts = review.get("fact_sheet", {})
        verified = len(facts.get("verified_facts", []))
        unverified = len(facts.get("unverified_claims", []))
        st.metric("Verified Facts", verified)
        st.metric("Unverified Claims", unverified, delta=-unverified if unverified else None)

        # Sources
        st.subheader("Sources")
        for source in review.get("sources", []):
            st.markdown(f"- [{source['title']}]({source['url']})")

    # Review actions
    st.divider()
    st.subheader("Editorial Decision")

    col_a, col_b, col_c = st.columns(3)

    with col_a:
        if st.button("✅ Approve", key=f"approve_{review['id']}", type="primary"):
            approve_article(review["id"])
            st.success("Article approved for publication!")
            st.rerun()

    with col_b:
        if st.button("📝 Request Revision", key=f"revise_{review['id']}"):
            feedback = st.text_area("Revision feedback", key=f"feedback_{review['id']}")
            if st.button("Submit Feedback", key=f"submit_{review['id']}"):
                request_revision(review["id"], feedback)
                st.info("Revision requested")
                st.rerun()

    with col_c:
        if st.button("❌ Reject", key=f"reject_{review['id']}", type="secondary"):
            reject_article(review["id"])
            st.warning("Article rejected")
            st.rerun()

def approve_article(review_id: str):
    """Approve article and record in provenance."""
    # Update status
    # Record human action in C2PA manifest
    # Move to publication queue
    pass

def request_revision(review_id: str, feedback: str):
    """Request revision with feedback."""
    # Update status
    # Record human action
    # Trigger revision pipeline
    pass

def reject_article(review_id: str):
    """Reject article."""
    # Update status
    # Record human action
    # Archive with reason
    pass
```

### 3.4 Agent-Lightning Integration

From REPORT.md: Use Agent-Lightning for continuous improvement via RL.

```python
# execution/optimization/agent_lightning.py
"""Agent-Lightning integration for RL-based optimization."""

# Note: This is a conceptual implementation based on REPORT.md
# Actual implementation depends on Agent-Lightning's released API

class AgentLightningWrapper:
    """Wrap agents with Agent-Lightning optimization."""

    def __init__(self, agent, agent_name: str):
        self.agent = agent
        self.agent_name = agent_name
        self.store = LightningStore()

    async def execute(self, prompt: str, context: dict) -> str:
        """Execute with instrumentation for learning."""
        import agl  # Agent-Lightning library

        # Record the prompt
        span_id = agl.emit_prompt(
            agent=self.agent_name,
            prompt=prompt,
            context=context
        )

        # Execute the agent
        result = await self.agent.execute(prompt, context)

        # Record the output
        agl.emit_response(
            span_id=span_id,
            response=result
        )

        return result

    def record_reward(self, span_id: str, score: float, feedback: str = None):
        """Record reward signal from quality gate or human review."""
        import agl

        agl.emit_reward(
            span_id=span_id,
            reward=score,  # Normalize to 0-1
            feedback=feedback
        )

class AgentLightningTrainer:
    """Background trainer for continuous improvement."""

    def __init__(self, store: LightningStore):
        self.store = store

    async def train_iteration(self):
        """Run one training iteration."""
        import agl

        # Fetch recent spans with rewards
        spans = await self.store.get_recent_spans(limit=100)

        # Train using APO (Automatic Prompt Optimization)
        optimizer = agl.APOOptimizer(
            algorithm="verl",  # Reinforcement Learning
            learning_rate=0.01
        )

        # Update prompt templates based on rewards
        updates = await optimizer.optimize(spans)

        # Apply updates to agent prompts
        for agent_name, new_prompts in updates.items():
            await self.apply_prompt_update(agent_name, new_prompts)

    async def apply_prompt_update(self, agent_name: str, prompts: dict):
        """Apply optimized prompts to agent."""
        # Save to prompt registry
        # Next agent execution uses updated prompts
        pass
```

---

## Phase 4: Production Hardening
**Timeline: 4-5 weeks after Phase 3**
**Goal: Reach 95/100+ production readiness**

### 4.1 Comprehensive Test Suite

```python
# tests/test_quality_gate.py
"""Tests for quality gate system."""

import pytest
from unittest.mock import Mock, patch
from execution.quality_gate import QualityGate, QualityGateResult
from execution.verification.fact_chain import FactVerificationChain

class TestQualityGate:
    """Test quality gate functionality."""

    @pytest.fixture
    def quality_gate(self):
        return QualityGate()

    def test_passes_high_quality_article(self, quality_gate):
        """Article meeting all criteria should pass."""
        article = """
        # Great Title Here

        Strong hook opening that grabs attention.

        According to Dr. Jane Smith, professor at MIT, "This finding
        represents a significant breakthrough."

        The research, published in Nature, shows...
        """
        context = {
            "fact_sheet": {"verified_facts": ["breakthrough confirmed"]},
            "sources": [{"name": "Dr. Jane Smith", "hand": "first", "expertise_relevant": True}]
        }

        result = quality_gate.review(article, context)
        assert result.passes
        assert result.score >= 7.0

    def test_fails_kill_phrase_article(self, quality_gate):
        """Article with kill phrases should fail."""
        article = """
        What's been your experience with AI?

        In this article, we'll explore the game-changing potential...
        """

        result = quality_gate.review(article, {})
        assert not result.passes
        assert "kill phrase" in result.feedback.lower()

    def test_escalates_borderline_article(self, quality_gate):
        """Borderline articles should trigger human review."""
        article = "Decent article but some concerns..."

        with patch.object(quality_gate.panel, 'review') as mock_review:
            mock_review.return_value = Mock(score=6.5, escalate=True, feedback="")
            result = quality_gate.review(article, {})
            assert result.requires_human_review

class TestFactVerificationChain:
    """Test mandatory fact verification."""

    def test_raises_if_no_verifiers(self):
        """Should raise if no API keys available."""
        with patch.dict('os.environ', {}, clear=True):
            with pytest.raises(RuntimeError, match="No fact verification available"):
                FactVerificationChain()

    def test_falls_back_on_failure(self):
        """Should try next verifier if first fails."""
        chain = FactVerificationChain()
        # Mock first verifier to fail
        chain.verifiers[0] = ("Failing", Mock(side_effect=Exception("API error")))

        result = chain.verify("test topic", "test content")
        assert result["verifier_used"] != "Failing"

# tests/test_wsj_showstoppers.py
class TestWSJShowstoppers:
    """Test WSJ Four Showstoppers."""

    def test_attribution_check_passes_sourced_claims(self):
        pass  # Implementation

    def test_attribution_check_fails_unsourced_claims(self):
        pass

    def test_tone_check_detects_loaded_language(self):
        pass

    def test_no_surprises_requires_subject_contact(self):
        pass
```

### 4.2 Cost Tracking Dashboard

```python
# execution/monitoring/cost_tracker.py
"""Track and report LLM costs."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List
import json

@dataclass
class ModelCall:
    model: str
    provider: str
    input_tokens: int
    output_tokens: int
    timestamp: str
    agent: str
    cost: float

@dataclass
class RunCosts:
    run_id: str
    total_cost: float
    calls: List[ModelCall]
    by_agent: Dict[str, float] = field(default_factory=dict)
    by_model: Dict[str, float] = field(default_factory=dict)

class CostTracker:
    """Track LLM API costs."""

    # Pricing per 1M tokens (as of 2025)
    PRICING = {
        "claude-opus-4-20250514": {"input": 15.0, "output": 75.0},
        "claude-sonnet-4-20250514": {"input": 3.0, "output": 15.0},
        "claude-3-haiku-20240307": {"input": 0.25, "output": 1.25},
        "gpt-4o": {"input": 2.5, "output": 10.0},
        "gpt-4o-mini": {"input": 0.15, "output": 0.6},
        "gemini-2.0-flash": {"input": 0.075, "output": 0.3},
        "gemini-2.0-pro": {"input": 1.25, "output": 5.0},
        "llama-3.3-70b-versatile": {"input": 0.59, "output": 0.79},  # Groq
    }

    def __init__(self, run_id: str):
        self.run_id = run_id
        self.calls: List[ModelCall] = []

    def record_call(
        self,
        model: str,
        provider: str,
        input_tokens: int,
        output_tokens: int,
        agent: str
    ):
        """Record a model API call."""
        pricing = self.PRICING.get(model, {"input": 0, "output": 0})
        cost = (
            (input_tokens / 1_000_000) * pricing["input"] +
            (output_tokens / 1_000_000) * pricing["output"]
        )

        self.calls.append(ModelCall(
            model=model,
            provider=provider,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            timestamp=datetime.utcnow().isoformat(),
            agent=agent,
            cost=cost
        ))

    def get_summary(self) -> RunCosts:
        """Get cost summary for this run."""
        by_agent = {}
        by_model = {}

        for call in self.calls:
            by_agent[call.agent] = by_agent.get(call.agent, 0) + call.cost
            by_model[call.model] = by_model.get(call.model, 0) + call.cost

        return RunCosts(
            run_id=self.run_id,
            total_cost=sum(c.cost for c in self.calls),
            calls=self.calls,
            by_agent=by_agent,
            by_model=by_model
        )

    def save_report(self, path: str):
        """Save cost report to file."""
        summary = self.get_summary()
        with open(path, "w") as f:
            json.dump({
                "run_id": summary.run_id,
                "total_cost": f"${summary.total_cost:.4f}",
                "by_agent": {k: f"${v:.4f}" for k, v in summary.by_agent.items()},
                "by_model": {k: f"${v:.4f}" for k, v in summary.by_model.items()},
                "call_count": len(summary.calls)
            }, f, indent=2)
```

### 4.3 Refactor app.py (1,964 lines → modular)

```
app.py (1,964 lines) →
├── app.py (main Streamlit entry, ~200 lines)
├── ui/
│   ├── __init__.py
│   ├── dashboard.py (~150 lines)
│   ├── review_queue.py (~200 lines)
│   ├── signal_browser.py (~150 lines)
│   ├── article_editor.py (~200 lines)
│   └── analytics.py (~150 lines)
├── handlers/
│   ├── __init__.py
│   ├── generation_handler.py (~200 lines)
│   ├── review_handler.py (~150 lines)
│   └── publishing_handler.py (~150 lines)
└── state/
    ├── __init__.py
    └── session_state.py (~100 lines)
```

---

## Implementation Checklist

### Phase 1: Critical Fixes ✅ → 65/100
- [ ] Create `execution/config.py` for centralized configuration
- [ ] Fix all hardcoded paths in `generate_medium_full.py`, `app.py`
- [ ] Create `FactVerificationChain` with mandatory verification
- [ ] Integrate `TechnicalSupervisor` into quality gate
- [ ] Add structured JSON logging with `logging_config.py`
- [ ] Update `.env.example` with all required variables

### Phase 2: Architecture Modernization → 80/100
- [ ] Implement LangGraph state machine (`langgraph_orchestrator.py`)
- [ ] Create multi-model adversarial panel (`multi_model_panel.py`)
- [ ] Implement cascaded routing (`cascaded_router.py`)
- [ ] Add SQLite checkpointing for state persistence
- [ ] Create `interrupt_before` human gates

### Phase 3: WSJ-Tier Quality → 95/100
- [ ] Implement WSJ Four Showstoppers (`wsj_showstoppers.py`)
- [ ] Add C2PA provenance tracking (`c2pa_manifest.py`)
- [ ] Build human review UI (`review_queue.py`)
- [ ] Integrate Agent-Lightning for continuous learning
- [ ] Add voice transformation for BBC/CBC/CNN styles

### Phase 4: Production Hardening → 95/100+
- [ ] Write comprehensive test suite (80%+ coverage)
- [ ] Implement cost tracking dashboard
- [ ] Refactor `app.py` into modular components
- [ ] Add Prometheus metrics for monitoring
- [ ] Create deployment documentation
- [ ] Security audit (API key handling, input validation)

---

## Success Metrics

| Metric | Current | Phase 1 | Phase 2 | Phase 3 | Phase 4 |
|--------|---------|---------|---------|---------|---------|
| Production Readiness | 45/100 | 65/100 | 80/100 | 95/100 | 95/100+ |
| Test Coverage | ~5% | 30% | 50% | 70% | 80%+ |
| Fact Verification | Optional | Mandatory | Multi-source | + Human verify | + Continuous |
| Human Oversight | Flag only | Queue UI | Full workflow | + Analytics | + ML triage |
| Provenance | None | Logging | Audit trail | C2PA | + SynthID |
| Cost Tracking | None | Basic | Per-agent | Dashboard | Optimization |

---

## References

### Editorial Standards Researched
- [WSJ Four Showstoppers](https://worldpressinstitute.org/)
- [BBC Editorial Guidelines](https://www.bbc.co.uk/editorialguidelines/)
- [CBC Journalistic Standards](https://cbc.radio-canada.ca/en/vision/governance/journalistic-standards-and-practices)
- [CNN Facts First](https://www.cnncreativemarketing.com/project/cnn_factsfirst/)
- [AP AI Guidelines](https://reutersinstitute.politics.ox.ac.uk/)

### Technical References
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [C2PA Specification](https://c2pa.org/)
- [EU AI Act Article 50](https://artificialintelligenceact.eu/article/50/)
- [Agent-Lightning Framework](https://github.com/microsoft/agent-lightning)
- [SynthID by DeepMind](https://deepmind.google/models/synthid/)

---

*Generated by the Expert Panel Summit*
*WSJ | BBC | CNN | CBC Radio | FirstPost*
