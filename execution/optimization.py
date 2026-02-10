"""
Optimization Module - Agent Feedback Loop and Performance Tracking.

Implements optimization features for the writing pipeline:
- Agent performance feedback collection
- Score-based reward signals
- Prompt refinement tracking
- Cost monitoring and optimization

Note: Full RL integration (Agent-Lightning) is a future enhancement.
This module provides the foundation for feedback-driven optimization.
"""

import json
import os
from datetime import datetime, timezone
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any
from pathlib import Path
import sqlite3


@dataclass
class AgentPerformance:
    """Performance metrics for a single agent run."""
    agent_name: str
    model: str
    task_type: str  # "research", "generation", "verification", "review"

    # Quality metrics
    quality_score: float = 0.0
    passed: bool = False

    # Cost metrics
    input_tokens: int = 0
    output_tokens: int = 0
    estimated_cost: float = 0.0

    # Timing
    duration_seconds: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    # Context
    content_id: str = ""
    iteration: int = 1

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class FeedbackSignal:
    """Feedback signal for agent optimization."""
    content_id: str
    signal_type: str  # "reward", "penalty", "neutral"
    signal_value: float  # -1.0 to 1.0

    # Context
    agent_name: str = ""
    reason: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    # Human feedback
    human_provided: bool = False
    reviewer: str = ""


class OptimizationTracker:
    """
    Tracks agent performance and feedback for optimization.

    Stores data in SQLite for persistence across sessions.
    """

    def __init__(self, db_path: str = None):
        if db_path is None:
            db_path = str(Path(__file__).parent.parent / ".tmp" / "optimization.db")

        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Initialize database tables."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS agent_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    agent_name TEXT,
                    model TEXT,
                    task_type TEXT,
                    quality_score REAL,
                    passed INTEGER,
                    input_tokens INTEGER,
                    output_tokens INTEGER,
                    estimated_cost REAL,
                    duration_seconds REAL,
                    timestamp TEXT,
                    content_id TEXT,
                    iteration INTEGER
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS feedback_signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    content_id TEXT,
                    signal_type TEXT,
                    signal_value REAL,
                    agent_name TEXT,
                    reason TEXT,
                    timestamp TEXT,
                    human_provided INTEGER,
                    reviewer TEXT
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS cost_tracking (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    content_id TEXT,
                    total_cost REAL,
                    token_count INTEGER,
                    model_breakdown TEXT,
                    timestamp TEXT
                )
            """)

    def record_performance(self, perf: AgentPerformance):
        """Record agent performance metrics."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO agent_performance
                (agent_name, model, task_type, quality_score, passed,
                 input_tokens, output_tokens, estimated_cost, duration_seconds,
                 timestamp, content_id, iteration)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                perf.agent_name, perf.model, perf.task_type, perf.quality_score,
                1 if perf.passed else 0, perf.input_tokens, perf.output_tokens,
                perf.estimated_cost, perf.duration_seconds, perf.timestamp,
                perf.content_id, perf.iteration
            ))

    def record_feedback(self, signal: FeedbackSignal):
        """Record feedback signal."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO feedback_signals
                (content_id, signal_type, signal_value, agent_name, reason,
                 timestamp, human_provided, reviewer)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                signal.content_id, signal.signal_type, signal.signal_value,
                signal.agent_name, signal.reason, signal.timestamp,
                1 if signal.human_provided else 0, signal.reviewer
            ))

    def record_cost(self, content_id: str, total_cost: float, token_count: int, model_breakdown: Dict):
        """Record cost tracking data."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO cost_tracking
                (content_id, total_cost, token_count, model_breakdown, timestamp)
                VALUES (?, ?, ?, ?, ?)
            """, (
                content_id, total_cost, token_count,
                json.dumps(model_breakdown),
                datetime.now(timezone.utc).isoformat()
            ))

    def get_agent_stats(self, agent_name: str = None, days: int = 30) -> Dict:
        """Get aggregated agent statistics."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row

            if agent_name:
                rows = conn.execute("""
                    SELECT
                        agent_name,
                        model,
                        task_type,
                        COUNT(*) as run_count,
                        AVG(quality_score) as avg_score,
                        SUM(CASE WHEN passed = 1 THEN 1 ELSE 0 END) as pass_count,
                        AVG(duration_seconds) as avg_duration,
                        SUM(estimated_cost) as total_cost
                    FROM agent_performance
                    WHERE agent_name = ?
                      AND datetime(timestamp) > datetime('now', ?)
                    GROUP BY agent_name, model, task_type
                """, (agent_name, f"-{days} days")).fetchall()
            else:
                rows = conn.execute("""
                    SELECT
                        agent_name,
                        model,
                        task_type,
                        COUNT(*) as run_count,
                        AVG(quality_score) as avg_score,
                        SUM(CASE WHEN passed = 1 THEN 1 ELSE 0 END) as pass_count,
                        AVG(duration_seconds) as avg_duration,
                        SUM(estimated_cost) as total_cost
                    FROM agent_performance
                    WHERE datetime(timestamp) > datetime('now', ?)
                    GROUP BY agent_name, model, task_type
                """, (f"-{days} days",)).fetchall()

            return [dict(row) for row in rows]

    def get_feedback_summary(self, content_id: str = None) -> Dict:
        """Get feedback summary."""
        with sqlite3.connect(self.db_path) as conn:
            if content_id:
                result = conn.execute("""
                    SELECT
                        signal_type,
                        COUNT(*) as count,
                        AVG(signal_value) as avg_value
                    FROM feedback_signals
                    WHERE content_id = ?
                    GROUP BY signal_type
                """, (content_id,)).fetchall()
            else:
                result = conn.execute("""
                    SELECT
                        signal_type,
                        COUNT(*) as count,
                        AVG(signal_value) as avg_value
                    FROM feedback_signals
                    GROUP BY signal_type
                """).fetchall()

            return {row[0]: {"count": row[1], "avg_value": row[2]} for row in result}

    def get_cost_summary(self, days: int = 30) -> Dict:
        """Get cost summary over time period."""
        with sqlite3.connect(self.db_path) as conn:
            result = conn.execute("""
                SELECT
                    SUM(total_cost) as total_cost,
                    SUM(token_count) as total_tokens,
                    COUNT(*) as content_count
                FROM cost_tracking
                WHERE datetime(timestamp) > datetime('now', ?)
            """, (f"-{days} days",)).fetchone()

            return {
                "total_cost": result[0] or 0,
                "total_tokens": result[1] or 0,
                "content_count": result[2] or 0,
                "avg_cost_per_article": (result[0] or 0) / max(1, result[2] or 1)
            }


# ============================================================================
# Cost Estimation Utilities
# ============================================================================

# Approximate token costs per 1M tokens (as of 2025)
MODEL_COSTS = {
    # Google
    "gemini-2.0-flash": {"input": 0.075, "output": 0.30},
    "gemini-2.0-flash-exp": {"input": 0.0, "output": 0.0},  # Free tier
    "gemini-1.5-pro": {"input": 1.25, "output": 5.00},

    # Anthropic
    "claude-sonnet-4-20250514": {"input": 3.00, "output": 15.00},
    "claude-3-5-sonnet-20241022": {"input": 3.00, "output": 15.00},
    "claude-3-haiku-20240307": {"input": 0.25, "output": 1.25},

    # OpenAI
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4-turbo": {"input": 10.00, "output": 30.00},

    # Groq (Llama)
    "llama-3.3-70b-versatile": {"input": 0.59, "output": 0.79},
    "llama-3.1-8b-instant": {"input": 0.05, "output": 0.08},

    # Perplexity (includes search)
    "llama-3.1-sonar-large-128k-online": {"input": 1.00, "output": 1.00},
}


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Estimate cost for a model call."""
    costs = MODEL_COSTS.get(model, {"input": 0.0, "output": 0.0})
    input_cost = (input_tokens / 1_000_000) * costs["input"]
    output_cost = (output_tokens / 1_000_000) * costs["output"]
    return round(input_cost + output_cost, 6)


def estimate_article_cost(models_used: List[str], avg_input_tokens: int = 2000, avg_output_tokens: int = 1000) -> float:
    """Estimate total cost for article generation."""
    total = 0
    for model in models_used:
        total += estimate_cost(model, avg_input_tokens, avg_output_tokens)
    return total


# ============================================================================
# Feedback Generation
# ============================================================================

def generate_quality_feedback(
    content_id: str,
    quality_score: float,
    passed: bool,
    agent_name: str = ""
) -> FeedbackSignal:
    """Generate feedback signal from quality score."""
    if passed and quality_score >= 8.0:
        signal_type = "reward"
        signal_value = min(1.0, (quality_score - 7.0) / 3.0)
        reason = f"High quality score: {quality_score}/10"
    elif passed:
        signal_type = "neutral"
        signal_value = 0.0
        reason = f"Passed with score: {quality_score}/10"
    else:
        signal_type = "penalty"
        signal_value = max(-1.0, (quality_score - 7.0) / 7.0)
        reason = f"Failed quality gate: {quality_score}/10"

    return FeedbackSignal(
        content_id=content_id,
        signal_type=signal_type,
        signal_value=signal_value,
        agent_name=agent_name,
        reason=reason,
        human_provided=False
    )


def generate_human_feedback(
    content_id: str,
    decision: str,
    reviewer: str,
    notes: str = ""
) -> FeedbackSignal:
    """Generate feedback signal from human decision."""
    signal_map = {
        "approved": ("reward", 1.0),
        "approved_with_edits": ("reward", 0.5),
        "revision_requested": ("penalty", -0.3),
        "rejected": ("penalty", -1.0),
        "escalated": ("penalty", -0.5)
    }

    signal_type, signal_value = signal_map.get(decision, ("neutral", 0.0))

    return FeedbackSignal(
        content_id=content_id,
        signal_type=signal_type,
        signal_value=signal_value,
        reason=f"Human decision: {decision}. {notes}",
        human_provided=True,
        reviewer=reviewer
    )


# ============================================================================
# Test / Demo
# ============================================================================

if __name__ == "__main__":
    print("Testing Optimization Module")
    print("=" * 60)

    # Initialize tracker
    tracker = OptimizationTracker()

    # Record sample performance
    perf = AgentPerformance(
        agent_name="WriterAgent",
        model="llama-3.3-70b-versatile",
        task_type="generation",
        quality_score=7.8,
        passed=True,
        input_tokens=1500,
        output_tokens=800,
        duration_seconds=12.5,
        content_id="gw-test123"
    )
    perf.estimated_cost = estimate_cost(perf.model, perf.input_tokens, perf.output_tokens)

    tracker.record_performance(perf)
    print(f"Recorded performance: {perf.agent_name} - {perf.quality_score}/10")

    # Record feedback
    feedback = generate_quality_feedback(
        "gw-test123", 7.8, True, "WriterAgent"
    )
    tracker.record_feedback(feedback)
    print(f"Recorded feedback: {feedback.signal_type} ({feedback.signal_value})")

    # Record cost
    tracker.record_cost(
        "gw-test123",
        perf.estimated_cost,
        perf.input_tokens + perf.output_tokens,
        {"llama-3.3-70b-versatile": perf.estimated_cost}
    )

    # Get stats
    print("\nAgent Stats:")
    stats = tracker.get_agent_stats()
    for stat in stats:
        print(f"  {stat['agent_name']}: {stat['run_count']} runs, avg score: {stat['avg_score']:.1f}")

    print("\nCost Summary:")
    costs = tracker.get_cost_summary()
    print(f"  Total Cost: ${costs['total_cost']:.4f}")
    print(f"  Total Tokens: {costs['total_tokens']}")
    print(f"  Avg per Article: ${costs['avg_cost_per_article']:.4f}")

    print("\nCost Estimation Examples:")
    for model in ["gemini-2.0-flash-exp", "claude-sonnet-4-20250514", "gpt-4o", "llama-3.3-70b-versatile"]:
        cost = estimate_cost(model, 2000, 1000)
        print(f"  {model}: ${cost:.4f}")

    print("\n" + "=" * 60)
    print("Optimization module test complete!")
