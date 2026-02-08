# Feature: Fact Verification System — Current State

**Agent**: Feature Deconstruction Agent
**Date**: 2026-02-08
**Files Analyzed**:
- `execution/agents/fact_verification_agent.py` (655 lines)
- `execution/agents/gemini_researcher.py` (447 lines)
- `execution/agents/perplexity_researcher.py` (406 lines)
- `execution/quality_gate.py` (461 lines)
- `execution/pipeline.py:216-276` (verify_node)
- `execution/config.py:111-131` (QualityConfig)

---

## Architecture Summary

Three-tier verification system with provider fallback chain:

```
Article Draft
  → FactVerificationAgent.verify_article()
    → Step 1: _extract_claims(content[:6000])
    |    Providers: Gemini → Perplexity → Groq
    |    Output: List[Claim] via regex JSON parsing
    |
    → Step 2: _verify_claim() per claim
    |    Providers: Gemini (grounding) → Perplexity (search)
    |    Output: VerificationResult per claim
    |
    → Step 3: Quality gate logic
    |    passes = (unverified <= 1) AND (false == 0) AND (verified+partial >= 3)
    |
    → FactVerificationReport(passes_quality_gate=bool)
```

**Providers:**

| Provider | Role | Web Search? | Fallback Position |
|----------|------|-------------|-------------------|
| Gemini (GeminiResearchAgent) | Extraction + Verification | Yes (Google Search Grounding) | Primary |
| Perplexity (PerplexityResearchAgent) | Extraction + Verification | Yes (Sonar Pro native search) | Secondary |
| Groq (GroqWrapper) | Extraction ONLY | No | Tertiary (extraction only) |

**Two integration points:**
1. `pipeline.py:verify_node()` — LangGraph pipeline path
2. `quality_gate.py:QualityGate.verify_facts()` — Quality gate path

---

## Critical Findings

### F1: "No Claims = Auto-Pass" Bypass (CRITICAL)

**Location**: `fact_verification_agent.py:242-251`

```python
if not claims:
    return FactVerificationReport(
        ...
        passes_quality_gate=True,  # No claims = nothing to verify
    )
```

If claim extraction fails for ANY reason (provider timeout, JSON parse error, article too abstract, content too short), 0 claims are extracted, and the article auto-passes verification. This is the single biggest safety bypass in the system.

**Trigger conditions:**
- All three providers fail for extraction → `_extract_claims()` returns `[]` → auto-pass
- LLM returns malformed JSON → `_parse_claims()` returns `[]` → auto-pass
- Article content is abstract/opinion (no extractable claims) → `[]` → auto-pass
- Article is an error string like "Groq Error: rate limit" → `[]` → auto-pass

### F2: Content Truncation to 6000 Chars

**Location**: `fact_verification_agent.py:287`

```python
prompt = self.CLAIM_EXTRACTOR_PROMPT.format(content=content[:6000])
```

Only the first ~1000-1200 words are sent for claim extraction. For a typical 1500-2000 word Medium article, the last 30-50% of the article is NEVER verified. Claims in the second half get a free pass.

### F3: Circular LLM-Verifies-LLM Reasoning

The entire verification chain is:
1. LLM (Writer) generates claims
2. LLM (Claim Extractor) identifies which claims to check
3. LLM (Verifier) checks those claims using web search

Step 2 is the weak link — the extraction LLM decides what IS and IS NOT a checkable claim. If the extraction model doesn't recognize a false statement as a "claim" (e.g., a subtle technical inaccuracy phrased as common knowledge), it's never sent for verification.

### F4: Fragile Regex-Based JSON Parsing

**Location**: `fact_verification_agent.py:340-361` and `477-540`

Two separate JSON parsing methods, both using string splitting:

```python
if "```json" in response:
    response = response.split("```json")[1].split("```")[0]
```

**Failure modes:**
- LLM returns JSON without code fences → may fail
- LLM returns markdown with multiple code blocks → splits wrong
- LLM returns truncated JSON → `json.loads()` fails → returns `[]`
- Nested code fences → splits at wrong boundary

The fallback in `_parse_verification_result()` (line 526-532) uses keyword matching:
```python
if "verified" in response_lower or "confirmed" in response_lower:
    status = VerificationStatus.PARTIALLY_VERIFIED
```

This would mark "this claim is NOT verified" as PARTIALLY_VERIFIED because the word "verified" appears.

### F5: Silent Provider Degradation

**Location**: `fact_verification_agent.py:189-226`

Provider initialization catches all exceptions and continues:
```python
except Exception as e:
    print(f"Gemini provider not available: {e}")  # Just prints
```

If Gemini fails to init → falls back to Perplexity.
If Perplexity fails → falls back to Groq.
Groq CAN'T do web search verification — only claim extraction.

If only Groq is available, `_verify_claim()` at line 370 routes to `_verify_with_perplexity()` for non-Gemini providers, which calls `provider.verify_draft()`. But the GroqWrapper doesn't HAVE a `verify_draft()` method — this would raise `AttributeError` for every claim, caught at line 372, resulting in ALL claims being `UNVERIFIED`.

### F6: Verification Exception Swallowing in Pipeline

**Location**: `pipeline.py:268-276`

```python
except Exception as e:
    return {
        ...
        "next_action": PHASE_REVIEW  # Continue anyway but flag
    }
```

If the entire verification system crashes, the pipeline continues to REVIEW phase. It sets `verification_passed: False` and `verification_status: "error"`, but nobody downstream checks these fields.

### F7: QualityGate Also Swallows Errors

**Location**: `quality_gate.py:186-190`

```python
except Exception as e:
    self._log(f"   Warning: Fact verification failed: {e}")
    verification_result = {"passed": True, ...}  # Defaults to PASSED
```

If verification throws an exception in the quality gate path, it silently defaults to `passed: True`. The most critical safety gate in the system defaults to "safe" on error.

### F8: Gemini Researcher Truncates Drafts to 4000 Chars

**Location**: `gemini_researcher.py:195` and `perplexity_researcher.py:179`

```python
draft[:4000]  # Both providers truncate
```

Even less content is sent for verification than the 6000-char extraction limit. Verification sees less context than extraction.

### F9: Self-Reported Confidence (Uncalibrated)

**Location**: `fact_verification_agent.py:58, 166`

The LLM reports its own confidence (0.0-1.0) and this value is stored as-is. No calibration, no validation. A model claiming 0.95 confidence is not actually 95% accurate. This number is meaningless but looks authoritative in the report.

---

## Quality Gate Thresholds (from config.py)

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `MAX_UNVERIFIED_CLAIMS` | 1 | Only 1 unverified claim allowed |
| `MIN_VERIFIED_FACTS` | 3 | At least 3 verified facts required |
| `false_claims == 0` | hardcoded | Zero tolerance for false claims |

**Interaction with F1**: When 0 claims are extracted, `unverified=0` (≤1 passes), `false=0` (passes), but `verified+partial=0` which is `< 3`. However, the check on line 249 returns BEFORE the gate logic — the early return auto-passes regardless.

---

## Data Flow for "Rubber Stamp" Attack

```
1. Writer generates article with subtle false claims
2. FactVerificationAgent._extract_claims(content[:6000])
3. LLM fails to extract claims from abstract prose → returns malformed JSON
4. _parse_claims() catches json.loads exception → returns []
5. verify_article() line 242: `if not claims:` → TRUE
6. Returns FactVerificationReport(passes_quality_gate=True)
7. Pipeline continues with "verified" article
8. Published with false claims
```

## Entry Points into Verification

| Entry Point | File | Error Behavior |
|-------------|------|----------------|
| `verify_node()` | pipeline.py:216 | Exception → continues to REVIEW anyway |
| `QualityGate.verify_facts()` | quality_gate.py:88 | Exception → defaults passed=True |
| `QualityGate.process()` | quality_gate.py:188 | Exception → defaults passed=True |
| `verify_article_facts()` | fact_verification_agent.py:597 | Direct call, exception propagates |
