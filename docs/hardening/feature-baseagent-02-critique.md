# Feature: BaseAgent & Multi-Provider — Research & Adversarial Critique

**Agents**: Technique Research + Red Team
**Date**: 2026-02-08

---

## Research: Industry Standard Patterns

### R1: Exception-Based Error Handling (Universal Standard)

Every production LLM wrapper raises typed exceptions:
- **LangChain**: `OutputParserException`, `LLMChainError`, custom retry decorators
- **OpenAI SDK**: Raises `APIError`, `RateLimitError`, `AuthenticationError`
- **LiteLLM**: Provider-agnostic exceptions with retry + fallback built in
- **Instructor**: Structured output with `InstructorRetryException`

**Pattern**: Return types should ONLY contain valid domain data. Errors flow through exception hierarchy. This is not controversial — it's Programming 101.

### R2: Retry with Exponential Backoff (Standard Infrastructure)

- **tenacity** library: Decorator-based retry with configurable backoff
  ```python
  @retry(wait=wait_exponential(min=1, max=60), stop=stop_after_attempt(3))
  ```
- **LiteLLM**: Built-in retry with provider-specific backoff
- **Production pattern**: 3 retries, exponential backoff (1s → 2s → 4s), jitter

### R3: Provider Fallback Chains

- **LiteLLM**: `fallbacks=["gpt-4o", "claude-3-opus", "gemini-pro"]`
- **Custom**: Try primary, catch exception, try secondary
- **Production pattern**: Per-agent or per-task provider selection, not global

### R4: Response Validation

- **Guardrails AI**: Schema validation on LLM output
- **Instructor**: Pydantic model enforcement with retry on validation failure
- **NeMo Guardrails**: Input/output filtering
- **Minimum**: Check for empty, too-short, error-prefixed, refusal patterns

### R5: Token Counting & Cost Tracking

- **tiktoken**: OpenAI-compatible token counting
- **LiteLLM**: `completion_cost()` per-call cost estimation
- **Helicone/LangSmith**: Request-level cost tracking and analytics
- **Production pattern**: Per-request cost logging, per-run budget caps, alerts

---

## Red Team: Trust-Breaking Attack Scenarios

### Attack 1: "The Error Article" (SEVERITY: CRITICAL)

**Vector**: Any API rate limit, timeout, or authentication failure
**Steps**:
1. Groq API key is rate-limited (normal under load)
2. `call_llm()` returns `"Groq Error: 429 Too Many Requests"`
3. Pipeline treats this as the article draft
4. Fact verification finds 0 verifiable claims in error string → passes
5. Style enforcer: low burstiness (1 sentence) but passes minimum thresholds depending on config
6. Error string is stored as final article

**Likelihood**: HIGH — rate limits happen regularly under production load
**Impact**: Published error string as content. Customer-visible. Unrecoverable trust damage.

### Attack 2: "The Leaked Credentials" (SEVERITY: HIGH)

**Vector**: Verbose error messages from providers
**Steps**:
1. OpenAI returns `AuthenticationError: Invalid API key sk-proj-abc...xyz`
2. `call_llm()` returns `f"OpenAI Error: {str(e)}"` which includes the partial key
3. If this string appears in dashboard, logs, or generated content → key leakage

**Likelihood**: MEDIUM — depends on provider error verbosity
**Impact**: API key exposure in user-facing output

### Attack 3: "The Single-Model Adversarial Panel" (SEVERITY: HIGH)

**Vector**: Global provider selection
**Steps**:
1. User sets up Groq API key (fastest, cheapest)
2. All 12 agents use Groq, including adversarial panel's 10 experts
3. "Multi-model review" becomes "same model, same biases, 10 times"
4. Quality scores are inflated and correlated (same model = same blind spots)

**Likelihood**: HIGH — Groq is the first provider checked
**Impact**: Adversarial review provides false confidence. Quality gate is theater.

### Attack 4: "The Silent Degradation" (SEVERITY: HIGH)

**Vector**: Missing or expired API keys
**Steps**:
1. Google API key expires. Provider falls back to OpenAI.
2. No warning logged. No notification to user.
3. Model quality changes. Costs change. Behavior changes.
4. User doesn't notice until outputs look different or bill arrives.

**Likelihood**: HIGH — API keys expire regularly
**Impact**: Invisible quality shift and unexpected costs

### Attack 5: "The Timeout Death" (SEVERITY: MEDIUM)

**Vector**: No timeouts on API calls
**Steps**:
1. Gemini API hangs (network issue, overloaded)
2. `call_llm()` blocks indefinitely (default httpx/requests timeout)
3. Pipeline freezes. Dashboard shows "Processing..." forever.
4. User refreshes → session state lost → pipeline still running in background

**Likelihood**: MEDIUM — API hangs are uncommon but happen
**Impact**: Zombie pipeline consuming resources, lost user session

---

## Summary Verdict

| Dimension | Rating | Notes |
|-----------|--------|-------|
| Error handling | F | Error strings as returns = no error handling |
| Resilience | F | No retry, no fallback, no timeout |
| Observability | F | No logging, no cost tracking, no health monitoring |
| Security | D | Potential credential leakage in error messages |
| Architecture | D | Global provider selection undermines multi-model design |
