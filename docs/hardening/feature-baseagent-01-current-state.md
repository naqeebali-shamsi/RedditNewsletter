# Feature: BaseAgent & Multi-Provider Architecture — Current State

**Agent**: Feature Deconstruction Agent
**Date**: 2026-02-08
**Files Analyzed**: `execution/agents/base_agent.py` (130 lines)

---

## Architecture Summary

`BaseAgent` is the foundation class for ALL 12 agents in the system:

| Agent | File | call_llm Usage |
|-------|------|----------------|
| WriterAgent | writer.py | 3 call sites |
| EditorAgent | editor.py | 2 call sites |
| CriticAgent | critic.py | 3 call sites |
| SpecialistAgent | specialist.py | 1 call site |
| AdversarialPanelAgent | adversarial_panel.py | 4 call sites |
| FactResearchAgent | fact_researcher.py | 2 call sites |
| TopicResearchAgent | topic_researcher.py | 1 call site |
| CommitAnalysisAgent | commit_analyzer.py | 1 call site |
| VisualsAgent | visuals.py | 1 call site |
| TechnicalSupervisorAgent | technical_supervisor.py | 1 call site |
| CopywriterAgent | copywriter_agent.py | inherits |
| OriginalThoughtAgent | original_thought_agent.py | inherits |

**Total call_llm invocations across codebase: 25+**

---

## Critical Findings

### F1: Error Strings as Return Values (CRITICAL)

Four return paths in `call_llm()` return error strings instead of raising exceptions:

```
base_agent.py:80  → "Error: LLM client not configured."
base_agent.py:98  → f"Groq Error: {str(e)}"
base_agent.py:113 → f"Gemini Error: {str(e)}"
base_agent.py:127 → f"OpenAI Error: {str(e)}"
base_agent.py:129 → "Error: Unsupported client type."
```

**Blast radius**: 25+ call sites across 12 agents. NONE check for error prefixes.

**Propagation path** (worst case):
1. `call_llm()` returns `"Groq Error: rate limit exceeded"`
2. `pipeline.py:189` assigns this to `draft`
3. `pipeline.py:192-194` meta-commentary check passes (doesn't start with "Here" or "I've")
4. `pipeline.py:196` prints: "Generated draft: 34 chars, 6 words"
5. Pipeline continues to verification with an error string as the article

### F2: Global Provider Selection — No Per-Agent Control

Provider priority is determined by environment variables at init time:
```
Groq → Vertex → Gemini (API key) → OpenAI
```

ALL agents get the same provider. The adversarial panel (which needs multi-model diversity) can't request specific providers per-expert. If Groq is configured, ALL 10 adversarial experts use Groq.

### F3: No Retry Logic

Single attempt per `call_llm()` call. No backoff. No fallback to alternative providers on transient failure. A single rate limit hit = immediate error string return.

### F4: No Response Validation

`call_llm()` returns raw text with zero validation:
- No check for empty responses
- No check for truncated responses
- No check for refusal responses ("I can't help with that")
- No minimum length check
- No format validation for structured output requests

### F5: No Token Counting or Cost Visibility

No `tiktoken`, no API response token fields, no per-call cost tracking, no cumulative budgeting. Users discover costs only on their API bill.

### F6: Silent Import Failures

```python
try:
    from google import genai
except ImportError:
    genai = None
```

If the google-genai package isn't installed, `genai` is silently `None`. Client setup falls through to `client_type = "unknown"`. No warning is logged.

---

## Hardcoded Assumptions

1. Groq uses OpenAI-compatible API (via `openai.OpenAI` with custom base_url)
2. Default model is `gemini-2.0-flash-exp` (may be deprecated/removed)
3. Default temperature is 0.7 for all calls
4. No timeout on API calls (default library timeout, potentially infinite)
5. Vertex AI fallback logic checks for `GOOGLE_CLOUD_PROJECT` but uses API key path too

---

## Data Flow Diagram

```
User triggers pipeline
  → pipeline.py creates WriterAgent(BaseAgent)
    → BaseAgent.__init__() checks env vars → picks provider
    → BaseAgent.call_llm(prompt) → makes API call
      → SUCCESS: returns raw text string
      → FAILURE: returns "Error: ..." string (SAME TYPE AS SUCCESS)
    → pipeline.py receives string → treats as valid draft
      → Continues to verification, review, style check
      → Error string may be "published" as article
```
