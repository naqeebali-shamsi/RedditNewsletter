# Feature: BaseAgent & Multi-Provider — Upgrade Plan

**Agent**: Engineering Upgrade Agent
**Date**: 2026-02-08
**Constraint**: No rewrites. Incremental fixes only.

---

## Upgrade 1: Raise Exceptions Instead of Error Strings

**Priority**: CRITICAL (ship-blocking)
**Effort**: M (touches all 25+ call sites)
**Regression Risk**: Medium

### What to Change

**In `base_agent.py`:**

1. Define custom exception classes at the top of the file:
   ```python
   class LLMError(Exception):
       """Base exception for LLM call failures."""
       pass

   class LLMProviderError(LLMError):
       """Raised when the LLM provider returns an error."""
       def __init__(self, provider: str, original_error: Exception):
           self.provider = provider
           self.original_error = original_error
           # Sanitize: don't include raw error string (may contain keys)
           super().__init__(f"{provider} call failed: {type(original_error).__name__}")

   class LLMNotConfiguredError(LLMError):
       """Raised when no LLM provider is configured."""
       pass
   ```

2. Replace each `return f"... Error: {str(e)}"` with `raise LLMProviderError(provider, e)`
3. Replace `return "Error: LLM client not configured."` with `raise LLMNotConfiguredError()`

**In callers (25+ sites):**
- Add `try/except LLMError` where the call result is used
- In `pipeline.py` node functions: catch `LLMError`, set `next_action = PHASE_ESCALATE`, add error to `error_messages`
- In `generate_medium_full.py`: catch and log, then abort the specialist step
- In `adversarial_panel.py`: catch and skip the failed expert, note degradation

### Migration Strategy
- Change BaseAgent first
- Run test suite to find all broken callers
- Fix callers one-by-one with proper try/except
- Each caller fix is a separate, reviewable commit

---

## Upgrade 2: Add Retry with Exponential Backoff

**Priority**: HIGH
**Effort**: S
**Regression Risk**: Low

### What to Change

1. Add `tenacity` to `requirements.txt`
2. In `base_agent.py`, wrap the API call inside each provider block with retry logic:
   ```python
   from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type

   @retry(
       wait=wait_exponential(multiplier=1, min=1, max=10),
       stop=stop_after_attempt(3),
       retry=retry_if_exception_type((Exception,)),
       reraise=True
   )
   def _call_provider(self, ...):
       ...
   ```
3. Only retry on transient errors (rate limit, timeout, server error). Don't retry on auth errors or invalid requests.

---

## Upgrade 3: Add Response Validation

**Priority**: HIGH
**Effort**: S
**Regression Risk**: Low

### What to Change

Add a `_validate_response(self, response: str) -> str` method in `BaseAgent`:

```python
def _validate_response(self, response: str) -> str:
    if not response or not response.strip():
        raise LLMProviderError(self.provider, ValueError("Empty response"))
    if len(response.strip()) < 20:
        raise LLMProviderError(self.provider, ValueError(f"Response too short: {len(response)} chars"))
    # Check for refusal patterns
    refusal_patterns = ["I cannot", "I can't help", "I'm unable to"]
    if any(response.strip().startswith(p) for p in refusal_patterns):
        raise LLMProviderError(self.provider, ValueError("Model refused request"))
    return response.strip()
```

Call this at the end of `call_llm()` before returning.

---

## Upgrade 4: Add Per-Agent Provider Override

**Priority**: HIGH
**Effort**: S
**Regression Risk**: Low

### What to Change

1. Add optional `provider` parameter to `BaseAgent.__init__()`:
   ```python
   def __init__(self, role, persona, model="gemini-2.0-flash-exp", provider=None):
       ...
       if provider:
           self.provider = provider
       else:
           self._get_api_key()  # existing auto-detect logic
   ```

2. In `AdversarialPanelAgent`, use this to create per-expert agents with specific providers
3. This allows the adversarial panel to enforce multi-model diversity

---

## Upgrade 5: Add Token Counting

**Priority**: MEDIUM
**Effort**: M
**Regression Risk**: None

### What to Change

1. Add `tiktoken` to requirements
2. In `call_llm()`, after successful response:
   - Extract token count from API response metadata (OpenAI/Groq provide this)
   - For Gemini, use response.usage_metadata
   - Store in instance variable: `self.last_call_tokens = {...}`
3. Add `get_usage_summary()` method that returns cumulative token usage
4. The pipeline can call this to track per-run costs

---

## Upgrade 6: Add Call Timeouts

**Priority**: MEDIUM
**Effort**: S
**Regression Risk**: Low

### What to Change

1. Add `timeout` parameter to `call_llm()` (default: 120 seconds)
2. For Groq/OpenAI: pass `timeout=timeout` to `client.chat.completions.create()`
3. For Gemini: use `request_options` with timeout
4. On timeout: raise `LLMProviderError(provider, TimeoutError(...))`

---

## Implementation Order

```
1. [CRITICAL] Upgrade 1: Exceptions (must be first — everything depends on this)
2. [HIGH]     Upgrade 3: Response validation (catches bad output at source)
3. [HIGH]     Upgrade 2: Retry logic (reduces transient failures)
4. [HIGH]     Upgrade 4: Per-agent provider override (enables multi-model panel)
5. [MEDIUM]   Upgrade 6: Timeouts (prevents zombie calls)
6. [MEDIUM]   Upgrade 5: Token counting (cost visibility)
```

## Estimated Total Effort: 2-3 days for a single engineer

## Files Modified

| File | Changes |
|------|---------|
| `execution/agents/base_agent.py` | Exception classes, retry, validation, timeout, provider override |
| `execution/pipeline.py` | try/except LLMError in generate_node, verify_node, review_node, revise_node |
| `execution/generate_medium_full.py` | try/except LLMError in 5+ call sites |
| `execution/quality_gate.py` | try/except LLMError in revision and polish calls |
| `execution/agents/adversarial_panel.py` | try/except LLMError, per-expert provider selection |
| `execution/agents/writer.py` | try/except LLMError in 3 call sites |
| `execution/agents/critic.py` | try/except LLMError in 3 call sites |
| `execution/agents/editor.py` | try/except LLMError in 2 call sites |
| `execution/agents/specialist.py` | try/except LLMError in 1 call site |
| `execution/agents/fact_researcher.py` | try/except LLMError in 2 call sites |
| `execution/agents/topic_researcher.py` | try/except LLMError in 1 call site |
| `execution/agents/commit_analyzer.py` | try/except LLMError in 1 call site |
| `execution/agents/visuals.py` | try/except LLMError in 1 call site |
| `execution/agents/technical_supervisor.py` | try/except LLMError in 1 call site |
| `requirements.txt` | Add `tenacity`, `tiktoken` |
