# Feature: Pipeline Orchestration — Upgrade Plan

**Agent**: Engineering Upgrade Agent
**Date**: 2026-02-08
**Constraint**: No rewrites. Incremental fixes only.

---

## Upgrade 1: Use Typed State in StateGraph

**Priority**: HIGH
**Effort**: S (one-line change + testing)
**Regression Risk**: Medium (may surface existing state bugs)

### What to Change

**In `pipeline.py:773`:**

```python
# BEFORE:
builder = StateGraph(dict)  # Using dict for flexibility

# AFTER:
builder = StateGraph(PipelineState)  # Typed state with validation
```

`PipelineState` is already defined at line 75. This change enables:
- Key validation (misspelled keys raise `KeyError`)
- Type hints for all state fields
- The `messages` field already has `Annotated[list, add_messages]` reducer

**Testing required**: Run pipeline end-to-end. Fix any nodes that write keys not in `PipelineState`. May need to add missing keys to the TypedDict.

### Additional Changes

Add any missing keys to `ArticleState` that nodes currently write:
- Check `style_result`, `escalation_codes`, `escalation_reasons`, `review_reasons`, `approval_reason`, etc.
- Any key written by any node that isn't in ArticleState needs to be added

---

## Upgrade 2: Fix State Merge in run_pipeline

**Priority**: HIGH
**Effort**: S
**Regression Risk**: Low

### What to Change

**In `pipeline.py:903-907`:**

```python
# BEFORE (lossy merge):
final_state = None
for state in pipeline.stream(initial_state, config=config_dict):
    for node_name, node_state in state.items():
        final_state = {**initial_state, **node_state} if final_state is None else {**final_state, **node_state}

# AFTER (proper accumulation):
final_state = dict(initial_state)  # Start with copy of initial state
for state in pipeline.stream(initial_state, config=config_dict):
    for node_name, node_state in state.items():
        # Merge with special handling for list fields (append, don't replace)
        for key, value in node_state.items():
            if key == "error_messages" and isinstance(value, list):
                existing = final_state.get("error_messages", [])
                # Only add new errors (avoid duplicates from node re-reading state)
                new_errors = [e for e in value if e not in existing]
                final_state["error_messages"] = existing + new_errors
            elif key == "reviewer_feedback" and isinstance(value, list):
                existing = final_state.get("reviewer_feedback", [])
                final_state["reviewer_feedback"] = existing + [v for v in value if v not in existing]
            else:
                final_state[key] = value
        print(f"  [{node_name}] -> {node_state.get('next_action', 'unknown')}")
```

**Better long-term**: Once using `StateGraph(PipelineState)`, define proper reducers:
```python
error_messages: Annotated[list, merge_lists] = []
reviewer_feedback: Annotated[list, merge_lists] = []
```
Then LangGraph handles accumulation automatically.

---

## Upgrade 3: Add Per-Node Timeouts

**Priority**: HIGH
**Effort**: S
**Regression Risk**: Low

### What to Change

Create a timeout decorator usable by all nodes:

```python
import signal
import threading
from functools import wraps

class NodeTimeoutError(Exception):
    """Raised when a pipeline node exceeds its timeout."""
    def __init__(self, node_name: str, timeout_seconds: int):
        self.node_name = node_name
        self.timeout_seconds = timeout_seconds
        super().__init__(f"Node '{node_name}' timed out after {timeout_seconds}s")

def with_timeout(timeout_seconds: int):
    """Decorator to add timeout to pipeline nodes (cross-platform)."""
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

            thread = threading.Thread(target=target)
            thread.start()
            thread.join(timeout=timeout_seconds)

            if thread.is_alive():
                raise NodeTimeoutError(func.__name__, timeout_seconds)
            if exception[0]:
                raise exception[0]
            return result[0]
        return wrapper
    return decorator
```

Apply to each node:
```python
@with_timeout(180)  # 3 minutes for research
def research_node(state): ...

@with_timeout(120)  # 2 minutes for generation
def generate_node(state): ...

@with_timeout(300)  # 5 minutes for verification (many claims)
def verify_node(state): ...

@with_timeout(300)  # 5 minutes for review (10 experts)
def review_node(state): ...

@with_timeout(120)  # 2 minutes for revision
def revise_node(state): ...
```

**On timeout**: Node should return escalation state with timeout error in `error_messages`.

---

## Upgrade 4: Add Per-Run Cost Tracking

**Priority**: HIGH
**Effort**: M
**Regression Risk**: None (additive)

### What to Change

1. Create `execution/cost_tracker.py`:

```python
from dataclasses import dataclass, field
from typing import List, Dict

# Approximate rates per 1M tokens (update as needed)
RATES = {
    "groq": {"input": 0.59, "output": 0.79},
    "gemini": {"input": 0.075, "output": 0.30},
    "openai": {"input": 2.50, "output": 10.00},
}

@dataclass
class CostEntry:
    node: str
    provider: str
    input_tokens: int
    output_tokens: int
    cost_usd: float

@dataclass
class RunCostTracker:
    budget_usd: float = 5.00  # Default budget cap
    entries: List[CostEntry] = field(default_factory=list)

    @property
    def total_cost(self) -> float:
        return sum(e.cost_usd for e in self.entries)

    @property
    def over_budget(self) -> bool:
        return self.total_cost > self.budget_usd

    def record(self, node, provider, input_tokens, output_tokens):
        rate = RATES.get(provider, RATES["gemini"])
        cost = (input_tokens * rate["input"] + output_tokens * rate["output"]) / 1_000_000
        self.entries.append(CostEntry(node, provider, input_tokens, output_tokens, cost))
        if self.over_budget:
            raise BudgetExceededError(self.total_cost, self.budget_usd)

    def summary(self) -> str:
        return f"Total: ${self.total_cost:.4f} ({len(self.entries)} calls, budget: ${self.budget_usd})"
```

2. Pass tracker through state or as a global per-run
3. After each `call_llm()`, record token usage (requires BaseAgent Upgrade 5)
4. Include cost summary in pipeline output and dashboard display

---

## Upgrade 5: Add LLM Output Validation in Pipeline Nodes

**Priority**: HIGH
**Effort**: S
**Regression Risk**: Low

### What to Change

Add a validation check after `call_llm()` in `generate_node`:

```python
# In generate_node, after line 189:
draft = writer.call_llm(prompt, temperature=0.7)

# NEW: Validate the draft is actual content
if not draft or len(draft.strip()) < 100:
    raise ValueError(f"Generated draft too short ({len(draft)} chars)")

error_prefixes = ["Error:", "Groq Error:", "Gemini Error:", "OpenAI Error:"]
if any(draft.strip().startswith(p) for p in error_prefixes):
    raise ValueError(f"LLM returned error string: {draft[:100]}")
```

This catches the "error article" scenario at the earliest point before it enters the pipeline. Apply similar validation in `revise_node`.

---

## Upgrade 6: Fix Error Routing in verify_node

**Priority**: CRITICAL (ship-blocking)
**Effort**: S
**Regression Risk**: Low

### What to Change

**In `pipeline.py:275`:**

```python
# BEFORE:
"next_action": PHASE_REVIEW  # Continue anyway but flag

# AFTER:
"next_action": PHASE_ESCALATE  # Do not proceed without verification
```

Verification failure must NOT route to REVIEW. Unverified content must not proceed through the pipeline.

---

## Upgrade 7: Fix Style Check Fail-Open Default

**Priority**: HIGH
**Effort**: S
**Regression Risk**: Low

### What to Change

**In `pipeline.py:550-565`:**

```python
# BEFORE: Missing dependency = pass
except ImportError:
    return {"style_passed": True, ...}

# AFTER: Missing dependency = needs review
except ImportError:
    print("  StyleEnforcerAgent not available — flagging for review")
    return {
        "style_score": None,
        "style_passed": False,
        "style_error": "StyleEnforcerAgent unavailable (missing dependency)",
        ...
    }
```

Same fix for the general `Exception` handler.

---

## Upgrade 8: Enable Checkpointing by Default

**Priority**: MEDIUM
**Effort**: S
**Regression Risk**: Low

### What to Change

In `create_pipeline()`, add a default SQLite checkpointer:

```python
def create_pipeline(checkpointer=None):
    if checkpointer is None:
        # Default to SQLite checkpointing
        import sqlite3
        db_path = config.paths.TEMP_DIR / ".pipeline_checkpoints.db"
        conn = sqlite3.connect(str(db_path), check_same_thread=False)
        checkpointer = SqliteSaver(conn)

    builder = StateGraph(PipelineState)
    ...
    return builder.compile(checkpointer=checkpointer)
```

This provides:
- Pipeline state persistence across crashes
- Resume capability for interrupted runs
- Audit trail of intermediate states

---

## Upgrade 9: Dashboard Calls LangGraph Pipeline (Consolidation Phase 1)

**Priority**: MEDIUM (but high architectural impact)
**Effort**: L
**Regression Risk**: Medium

### What to Change

**Phase 1 (minimum viable):**
1. In `app.py:run_full_pipeline()`, replace the inline pipeline with:
   ```python
   from execution.pipeline import create_pipeline, run_pipeline
   pipeline = create_pipeline()
   result = run_pipeline(pipeline, topic=topic, source_content=source,
                         source_type=source_type, platform="medium")
   ```
2. Map `result` dict to the dashboard's expected format
3. Keep the specialist pipeline as a post-processing step if needed

**Phase 2 (full consolidation):**
- Move specialist pipeline into LangGraph nodes
- Remove `generate_medium_full.py` entirely
- Single pipeline, multiple entry points

### Why This is Effort L

The dashboard's `run_full_pipeline()` has significant logic differences:
- Topic research integration (TopicResearchAgent)
- GitHub commit analysis path
- Progress callbacks for Streamlit UI
- Image generation coordination
- Different specialist ordering

Reconciling these requires careful feature parity analysis.

---

## Implementation Order

```
1. [CRITICAL] Upgrade 6: Fix verify_node error routing (one-line fix)
2. [HIGH]     Upgrade 5: Add LLM output validation (catches error articles)
3. [HIGH]     Upgrade 7: Fix style check fail-open (one-line fix)
4. [HIGH]     Upgrade 1: Use typed state (one-line change + testing)
5. [HIGH]     Upgrade 2: Fix state merge (prevent error loss)
6. [HIGH]     Upgrade 3: Add per-node timeouts (prevent zombie pipelines)
7. [HIGH]     Upgrade 4: Add cost tracking (budget visibility)
8. [MEDIUM]   Upgrade 8: Enable checkpointing by default (crash resilience)
9. [MEDIUM]   Upgrade 9: Dashboard consolidation (architecture cleanup)
```

## Estimated Total Effort: 5-7 days for a single engineer (Upgrades 1-8: 3 days; Upgrade 9: 2-4 days)

## Files Modified

| File | Changes |
|------|---------|
| `execution/pipeline.py` | Upgrades 1-3, 5-8: typed state, merge fix, timeouts, validation, routing, style, checkpoint |
| `execution/cost_tracker.py` | Upgrade 4: New file for cost tracking |
| `execution/agents/base_agent.py` | Upgrade 4: Token count extraction (dependency on Feature 1 Upgrade 5) |
| `app.py` | Upgrade 9: Replace inline pipeline with `run_pipeline()` call |
| `execution/generate_medium_full.py` | Upgrade 9 Phase 2: Deprecate or convert to thin wrapper |

## Cross-Feature Dependencies

| Upgrade | Depends On |
|---------|------------|
| Upgrade 4 (cost tracking) | Feature 1 Upgrade 5 (token counting in BaseAgent) |
| Upgrade 5 (output validation) | Feature 1 Upgrade 1 (exceptions in BaseAgent) makes this partially redundant |
| Upgrade 9 (consolidation) | Feature 2 fixes (fail-closed verification) should be done first |
