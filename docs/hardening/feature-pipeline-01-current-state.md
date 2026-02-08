# Feature: Pipeline Orchestration — Current State

**Agent**: Feature Deconstruction Agent
**Date**: 2026-02-08
**Files Analyzed**:
- `execution/pipeline.py` (1083 lines) — LangGraph pipeline
- `execution/generate_medium_full.py` (499 lines) — Legacy CLI pipeline
- `execution/article_state.py` (279 lines) — State schema
- `execution/config.py` (267 lines) — Configuration
- `app.py:1157-1240` — Dashboard inline pipeline

---

## Architecture Summary

### THREE pipelines exist (not two as dossier reported)

| Pipeline | File | Invoked By | State Management |
|----------|------|------------|------------------|
| LangGraph | `pipeline.py` | CLI (`python pipeline.py --topic X`) | `StateGraph(dict)` with stream |
| Legacy | `generate_medium_full.py` | CLI (`python generate_medium_full.py`) | Local variables, sequential |
| Dashboard | `app.py:run_full_pipeline()` | Streamlit UI button | `st.session_state` + inline |

**The dashboard does NOT call either pipeline.** It has its own 600+ line inline implementation that reimplements the full workflow with different agents, different ordering, and different quality gate behavior.

### LangGraph Pipeline Graph Structure

```
RESEARCH → GENERATE → VERIFY
                        ↓ (conditional)
              ┌─────── REVIEW ←──────── REVISE ←─┐
              │          ↓ (conditional)          │
              │     STYLE_CHECK                   │
              │          ↓ (conditional)          │
              │       APPROVE → END               │
              │          or                       │
              └──→ ESCALATE → END                 │
                        ↑                         │
                        └─── (max iterations) ────┘
```

### State Flow

```
create_initial_state() → ArticleState (~30 fields)
  → Each node returns partial dict with changed fields
  → run_pipeline() merges: final_state = {**final_state, **node_state}
  → Final state used for provenance, output
```

---

## Critical Findings

### F1: StateGraph Uses `dict` Instead of Typed State (HIGH)

**Location**: `pipeline.py:773`

```python
builder = StateGraph(dict)  # Using dict for flexibility
```

`PipelineState` is defined at line 75-84 (extending `ArticleState`) but NEVER USED for the graph. The graph operates on raw `dict`, which means:
- No type validation on state keys
- No validation on state values
- Any node can write any key with any type
- Misspelled keys create new fields silently (e.g., `"verfication_passed"`)
- LangGraph's built-in state validation is entirely bypassed

### F2: State Merge Strategy Silently Overwrites Values (HIGH)

**Location**: `pipeline.py:904-907`

```python
for state in pipeline.stream(initial_state, config=config_dict):
    for node_name, node_state in state.items():
        final_state = {**initial_state, **node_state} if final_state is None else {**final_state, **node_state}
```

**Problems:**
1. On first merge: `{**initial_state, **node_state}` — initial_state has 30+ fields, node_state has 5-8 fields. Works correctly.
2. On subsequent merges: `{**final_state, **node_state}` — node_state overwrites ALL matching keys.
3. If a revision node sets `error_messages: [new_error]`, it REPLACES the accumulated error list instead of appending.
4. This is because nodes return `state.get("error_messages", []) + [new_error]` — but `state` is the INPUT state, not final_state. If multiple nodes accumulate errors, only the last one's errors survive.

### F3: Three Pipelines, Zero Shared Implementation (CRITICAL)

| Capability | LangGraph (`pipeline.py`) | Legacy (`generate_medium_full.py`) | Dashboard (`app.py`) |
|-----------|--------------------------|-------------------------------------|---------------------|
| Research | Gemini → Perplexity fallback | Gemini → Perplexity fallback | Perplexity only |
| Generation | WriterAgent | WriterAgent | WriterAgent |
| Verification | FactVerificationAgent | GeminiResearcher.verify_draft | QualityGate.verify_facts |
| Review | AdversarialPanelAgent | None (Editor review only) | QualityGate.process |
| Specialists | None | 5 sequential specialists | Inline specialist calls |
| Style Check | StyleEnforcerAgent (node) | None | Via QualityGate |
| Provenance | Full provenance tracking | None | None |
| Cost control | None | None | None |
| Error handling | Exception → escalate | Exception → print, continue | Exception → st.error() |

**Bugs fixed in one pipeline don't propagate to the others.** The "no claims = pass" fix (Feature 2) needs to be applied in THREE places.

### F4: No Per-Node Timeouts (HIGH)

**Zero mentions of "timeout"** in `pipeline.py`. Each node makes one or more LLM API calls with no timeout protection:

| Node | API Calls | Risk |
|------|-----------|------|
| research_node | Gemini + Perplexity (2 calls) | Hang on slow research |
| generate_node | WriterAgent.call_llm (1 call) | Hang on generation |
| verify_node | FactVerificationAgent (5-15 calls per claim) | Hang on verification |
| review_node | AdversarialPanel (10+ expert calls) | Hang on panel |
| revise_node | WriterAgent.call_llm (1 call) | Hang on revision |
| style_check_node | StyleEnforcer (1 call) | Hang on style |

A single hung API call freezes the entire pipeline indefinitely.

### F5: No Cost Controls or Budget Limits (HIGH)

No token counting. No per-run budget cap. No per-node cost tracking.

**Worst-case cost estimate per pipeline run:**
- Research: 1 Gemini call (~$0.01) + 1 Perplexity call (~$0.05)
- Generate: 1 Groq/Gemini call (~$0.01-$0.10)
- Verify: 5-15 claims × 1 call each (~$0.05-$1.50)
- Review: 10 experts × 1 call each (~$0.10-$2.00)
- Style: 1 call (~$0.01)
- **Per iteration**: ~$0.22-$3.66
- **Max iterations (3)**: ~$0.66-$11.00
- **Plus revision loops**: Each REVISE → VERIFY → REVIEW cycle adds another iteration
- **Theoretical maximum**: 3 full loops = ~$2.00-$33.00 per article

No warning is given at any threshold.

### F6: Error Swallowing in Multiple Nodes

Every node has the same pattern:

```python
try:
    # Do work
except Exception as e:
    print(f"  Node failed: {e}")
    return {
        "error_messages": state.get("error_messages", []) + [f"Failed: {e}"],
        "next_action": PHASE_ESCALATE  # or sometimes PHASE_REVIEW
    }
```

**The verify_node is the worst offender** (line 275): verification crash routes to `PHASE_REVIEW` instead of `PHASE_ESCALATE`, so the pipeline continues without verification.

### F7: Style Check Silently Passes on Error

**Location**: `pipeline.py:550-565`

```python
except ImportError:
    return {"style_passed": True, ...}  # Missing dependency = pass
except Exception as e:
    return {"style_passed": True, ...}  # Any error = pass
```

Both `ImportError` and general exceptions default to `style_passed: True`. If the style enforcer is broken, content auto-passes.

### F8: Dashboard Pipeline Has No Quality Gate Loop

The dashboard's `run_full_pipeline()` at `app.py:1176` calls `QualityGate.process()` which does its own review loop, but:
- It's a completely separate code path from the LangGraph pipeline
- It uses `WriterAgent.call_llm()` for revision (inherits error-string bug)
- Its revision output is not re-verified (only re-reviewed)

### F9: LangGraph Checkpointing is Available but Not Default

**Location**: `pipeline.py:834-850`

`create_pipeline_with_sqlite()` exists but is only used when `--checkpoint` CLI flag is passed. The default `create_pipeline()` has no checkpointing, meaning:
- Pipeline state is lost on crash
- No resume capability
- No audit trail of intermediate states

### F10: Legacy Pipeline Has No Revision Loop Control

**Location**: `generate_medium_full.py:177-257`

```python
MAX_REVISIONS = 2
for attempt in range(MAX_REVISIONS + 1):
    ...
```

Hardcoded `MAX_REVISIONS = 2` (not from config). No way to change this without editing code. Different from LangGraph pipeline's `config.quality.MAX_ITERATIONS = 3`.

---

## Data Flow for "Error Article" Attack (Cross-Feature)

```
1. pipeline.py:generate_node() calls writer.call_llm()
2. BaseAgent returns "Groq Error: rate limit" (Feature 1 bug)
3. generate_node line 192: if draft.startswith("Here") or draft.startswith("I've") → FALSE
4. draft = "Groq Error: rate limit".strip() → stored in state
5. verify_node: FactVerificationAgent.verify_article("Groq Error: rate limit")
6. 0 claims extracted → passes_quality_gate=True (Feature 2 bug)
7. review_node: AdversarialPanel.review_content("Groq Error: rate limit")
8. Panel experts score a 6-word error string → low scores → REVISE
9. revise_node: writer.call_llm(revision_prompt + "Groq Error: rate limit")
10. If call_llm works now: writer generates actual content from the revision prompt
11. New draft passes verification → passes review → published
12. BUT: the published article has NO relation to the original topic
    (it was generated from revision instructions, not from the source content)
```

## Entry Points into Pipeline Execution

| Entry | Invokes | Has Provenance | Has Quality Gate | Has Checkpointing |
|-------|---------|----------------|------------------|-------------------|
| CLI: `pipeline.py` | LangGraph pipeline | Yes | Yes (panel + style) | Optional |
| CLI: `generate_medium_full.py` | Legacy pipeline | No | Editor review only | No |
| Dashboard: `app.py` | Inline pipeline | No | QualityGate class | No |
