# Feature: Pipeline Orchestration — Research & Adversarial Critique

**Agents**: Technique Research + Red Team
**Date**: 2026-02-08

---

## Research: Industry Standard Patterns for Pipeline Orchestration

### R1: Typed State in LangGraph (LangGraph's Own Best Practice)

**The problem**: `StateGraph(dict)` bypasses LangGraph's type safety.

**LangGraph's recommended pattern:**
```python
class MyState(TypedDict):
    messages: Annotated[list, add_messages]
    draft: str
    ...

builder = StateGraph(MyState)  # Typed state
```

This enables:
- Compile-time key validation (misspelled keys raise errors)
- Reducer functions for list/dict fields (proper accumulation)
- State schema documentation as code

**GhostWriter already has** `PipelineState(ArticleState)` defined at `pipeline.py:75` — it's just not being USED. This is a one-line fix.

### R2: Per-Node Timeouts (Standard LangGraph Pattern)

**The problem**: No timeouts on pipeline nodes. A hung API call freezes everything.

**Industry solutions:**
- **Python `asyncio.wait_for()`**: Native timeout for async operations
- **`tenacity` with timeout**: Retry decorator includes timeout
- **LangGraph node wrapper**: Decorator that wraps node functions in `signal.alarm()` (Unix) or `threading.Timer()` (cross-platform)
- **LLM library timeouts**: OpenAI SDK supports `timeout=` parameter directly

**Pattern**: Each node gets a configurable timeout (default 120s for generation, 60s per verification claim, 30s per panel expert). On timeout, raise a timeout exception → route to escalation.

### R3: Cost Tracking and Budget Caps (Production Standard)

**The problem**: Zero cost visibility. Users discover costs on their API bill.

**Industry solutions:**
- **LiteLLM `completion_cost()`**: Per-call cost estimation from token usage
- **Helicone proxy**: Transparent cost tracking per request
- **LangSmith**: Request-level cost tracking in traces
- **Custom**: Extract tokens from API responses, multiply by per-token rates

**Pattern**: Track cumulative cost per pipeline run. Display running total. Abort if budget exceeded. Show cost in final output.

**Approximate token rates (as of 2026):**
| Provider | Input $/1M tok | Output $/1M tok |
|----------|---------------|-----------------|
| Groq (Llama 3.3 70B) | $0.59 | $0.79 |
| Gemini Flash | $0.075 | $0.30 |
| GPT-4o | $2.50 | $10.00 |
| Claude Sonnet | $3.00 | $15.00 |

### R4: Pipeline Consolidation (Software Engineering 101)

**The problem**: Three implementations of the same workflow.

**Industry solutions:**
- **Single pipeline, multiple entry points**: One canonical pipeline; CLI, dashboard, and API call the same pipeline with different configuration
- **Dagster/Airflow pattern**: Pipeline definition is separate from execution environment
- **Feature flags, not separate code**: Behavioral differences controlled by configuration, not code duplication

**Pattern**: Consolidate to LangGraph pipeline as canonical implementation. Dashboard calls `run_pipeline()` from `pipeline.py`. Legacy script becomes a thin wrapper. Delete duplicate code.

### R5: Idempotent Pipeline Runs (Data Engineering Standard)

**The problem**: Re-running the same topic generates a new article with new API calls and new costs.

**Industry solutions:**
- **Dagster**: Asset-based materialization with caching
- **Airflow**: Task-level caching with `depends_on_past`
- **Custom**: Hash inputs to create run ID; cache intermediate outputs by run ID

**Pattern**: Hash (topic + source_content + source_type + date) to create a deterministic run ID. Cache research results, draft, verification results. Re-running with same inputs reuses cached outputs.

### R6: Observability (LangGraph Ecosystem)

**The problem**: No tracing, no logging, no visibility into what happened during a run.

**Industry solutions:**
- **LangSmith**: Native LangGraph integration, traces every node execution
- **Helicone**: Transparent LLM proxy with logging
- **OpenTelemetry**: Standard observability framework
- **Custom structured logging**: JSON logs per node with timing, token count, cost

**Pattern**: At minimum, add structured JSON logging per node (start time, end time, tokens, cost, result summary). LangSmith integration for production.

---

## Red Team: Trust-Breaking Attack Scenarios

### Attack 1: "The Error Article" (SEVERITY: CRITICAL)

**Vector**: BaseAgent error string + pipeline processes it as content
**Steps**:
1. Writer's Groq key hits rate limit
2. `call_llm()` returns `"Groq Error: 429 Too Many Requests"`
3. `generate_node` assigns error string as `draft`
4. Meta-commentary check (`startswith("Here")`) doesn't catch it
5. `verify_node` receives error string as "article"
6. Fact extractor finds 0 claims → auto-pass (Feature 2 bug)
7. `review_node` sends error string to adversarial panel → low score → REVISE
8. `revise_node` generates real content from revision prompt (not original topic)
9. New content passes because it was actually written (just wrong topic/context)
10. Published article has nothing to do with the original source material

**Likelihood**: HIGH under load
**Impact**: Published content disconnected from source. Complete trust loss.

### Attack 2: "The Cost Bomb" (SEVERITY: HIGH)

**Vector**: Revision loop with multi-model panel
**Steps**:
1. User generates article. Panel gives 6.5/10.
2. Iteration 1: REVISE → VERIFY (15 claims × Gemini) → REVIEW (10 experts × 3 models)
3. Score: 6.8/10. Still below 7.0 threshold.
4. Iteration 2: Same cycle. Score: 6.9/10.
5. Iteration 3: Same cycle. Score: 6.95/10. Max iterations hit → ESCALATE.
6. Total: 3 full cycles × (15 verify calls + 10 review calls + 1 revise call) = 78 API calls
7. At ~$0.10/call average = ~$7.80 for an article that wasn't even published
8. User sees nothing until their API bill

**Likelihood**: MEDIUM (depends on content quality and model leniency)
**Impact**: $5-$15 wasted per failed article. No warning.

### Attack 3: "The Zombie Pipeline" (SEVERITY: HIGH)

**Vector**: No timeouts on API calls
**Steps**:
1. Gemini research API hangs (network issue, overloaded server)
2. `research_node` blocks indefinitely waiting for response
3. LangGraph waits indefinitely (no node-level timeout)
4. Dashboard shows "Processing..." spinner forever
5. User refreshes browser → Streamlit session state lost
6. Pipeline continues running in background (Python thread/process)
7. Background pipeline completes, writes output to file nobody reads
8. API credits consumed for a lost run

**Likelihood**: MEDIUM (API hangs happen occasionally)
**Impact**: Lost user time, wasted API credits, frustrated user

### Attack 4: "The State Corruption" (SEVERITY: MEDIUM)

**Vector**: Untyped state + merge strategy
**Steps**:
1. `verify_node` returns `{"error_messages": ["Verification failed: timeout"]}`
2. `review_node` also fails, returns `{"error_messages": ["Review failed: rate limit"]}`
3. Merge in `run_pipeline()`: `{**final_state, **node_state}`
4. Final `error_messages` = `["Review failed: rate limit"]` — verification error LOST
5. Escalation report only shows review error, masking the verification failure
6. Human reviewer doesn't know verification was skipped

**Likelihood**: MEDIUM (multiple nodes failing in same run)
**Impact**: Lost error context makes debugging and human review harder

### Attack 5: "The Pipeline Roulette" (SEVERITY: HIGH)

**Vector**: Three pipelines with different behavior
**Steps**:
1. User tests article generation via CLI → uses LangGraph pipeline → passes
2. User demonstrates same topic via Dashboard → uses inline pipeline → fails
3. (Or vice versa) Different quality gates, different agent ordering, different thresholds
4. "It worked on my machine" but fails in the demo

**Likelihood**: HIGH (anyone using both CLI and dashboard will encounter this)
**Impact**: Unpredictable behavior. Erodes user trust in consistency.

### Attack 6: "The Lost Article" (SEVERITY: MEDIUM)

**Vector**: No default checkpointing
**Steps**:
1. Pipeline runs for 5 minutes generating article
2. Reaches REVIEW phase. Panel gives 7.5/10. Passes.
3. Style check runs. Passes.
4. APPROVE phase starts. Python process crashes (OOM, kill signal, power loss).
5. No checkpointing was enabled (default). All state lost.
6. 5 minutes of API calls ($3-$5) wasted. Article gone.
7. User must re-run from scratch.

**Likelihood**: LOW (crashes are rare but not impossible)
**Impact**: Lost work and wasted API credits

---

## Summary Verdict

| Dimension | Rating | Notes |
|-----------|--------|-------|
| Architecture | D | Three separate pipelines; untyped state graph |
| Reliability | D | No timeouts, no retries, error swallowing |
| Cost control | F | Zero visibility, zero caps, zero tracking |
| Consistency | F | Three pipelines produce different results |
| Observability | F | No tracing, no structured logging |
| Resilience | D | No checkpointing by default; state lost on crash |
| Error handling | D | Errors swallowed; verify crash → continue to review |
