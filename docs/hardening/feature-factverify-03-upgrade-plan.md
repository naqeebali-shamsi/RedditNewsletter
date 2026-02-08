# Feature: Fact Verification System — Upgrade Plan

**Agent**: Engineering Upgrade Agent
**Date**: 2026-02-08
**Constraint**: No rewrites. Incremental fixes only.

---

## Upgrade 1: Fix "No Claims = Pass" Bypass

**Priority**: CRITICAL (ship-blocking)
**Effort**: S
**Regression Risk**: Low

### What to Change

**In `fact_verification_agent.py`, replace lines 242-251:**

```python
# BEFORE (dangerous):
if not claims:
    return FactVerificationReport(
        passes_quality_gate=True,  # Auto-pass!
        summary="No verifiable claims found in content."
    )

# AFTER (fail-closed):
if not claims:
    return FactVerificationReport(
        claims=[],
        results=[],
        verified_count=0,
        unverified_count=0,
        false_count=0,
        passes_quality_gate=False,  # Fail-closed
        summary="NEEDS REVIEW: No verifiable claims could be extracted. "
                "This may indicate extraction failure or highly abstract content. "
                "Human review required."
    )
```

**Also fix `quality_gate.py` lines 95-104 and 188-190:**

```python
# BEFORE: Missing verifier defaults to passed=True
if not self.fact_verifier:
    return {"passed": True, ...}

# AFTER: Missing verifier = needs review
if not self.fact_verifier:
    return {
        "passed": False,
        "summary": "Fact verification unavailable (no provider). Content requires human review.",
        ...
    }
```

```python
# BEFORE: Exception defaults to passed=True
except Exception as e:
    verification_result = {"passed": True, ...}

# AFTER: Exception = needs review
except Exception as e:
    verification_result = {"passed": False,
        "summary": f"Verification failed ({e}). Content requires human review.",
        ...}
```

**Also fix `pipeline.py` line 275:**

```python
# BEFORE: Verification crash continues to REVIEW
"next_action": PHASE_REVIEW  # Continue anyway but flag

# AFTER: Verification crash → ESCALATE
"next_action": PHASE_ESCALATE  # Do not continue without verification
```

### Files Modified
- `execution/agents/fact_verification_agent.py` (lines 242-251)
- `execution/quality_gate.py` (lines 95-104, 188-190)
- `execution/pipeline.py` (line 275)

---

## Upgrade 2: Use Structured Output for Claim Extraction

**Priority**: HIGH
**Effort**: M
**Regression Risk**: Low

### What to Change

1. Add `instructor` to `requirements.txt`
2. Define Pydantic models for claim extraction and verification:

```python
from pydantic import BaseModel, Field
from typing import List, Literal

class ExtractedClaim(BaseModel):
    text: str = Field(description="The exact claim text from the article")
    type: Literal["statistic", "technical_spec", "quote", "date", "comparison", "general"]
    context: str = Field(description="The sentence containing the claim")
    why_verify: str = Field(description="What needs verification")

class ClaimExtractionResult(BaseModel):
    claims: List[ExtractedClaim] = Field(min_length=0, max_length=20)

class VerificationOutput(BaseModel):
    status: Literal["verified", "partial", "unverified", "false", "not_checkable"]
    sources: List[dict] = []
    explanation: str
    correction: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)
```

3. In `_extract_claims()`, replace the raw API call + regex parsing with:
   - For Gemini: Use `response_schema` in `GenerateContentConfig`
   - For Perplexity/Groq: Use `instructor.from_openai()` wrapper

4. Remove `_parse_claims()` entirely (no longer needed)
5. Remove the keyword-matching fallback in `_parse_verification_result()`

### Migration Strategy
- Add Pydantic models alongside existing code
- Switch one provider at a time (Gemini first, then Perplexity)
- Keep Groq as-is (extraction only, already simpler)
- Remove old parsing code after all providers migrated

---

## Upgrade 3: Add Verification Provider Health Check

**Priority**: HIGH
**Effort**: S
**Regression Risk**: None

### What to Change

Add a `check_health()` method to `FactVerificationAgent`:

```python
def check_health(self) -> dict:
    """Check which providers are available and their capabilities."""
    health = {
        "providers_available": len(self.providers),
        "has_web_search": False,
        "extraction_only_providers": [],
        "full_providers": [],
        "warnings": []
    }

    for name, provider in self.providers:
        if name in ("gemini", "perplexity"):
            health["full_providers"].append(name)
            health["has_web_search"] = True
        elif name == "groq":
            health["extraction_only_providers"].append(name)
            health["warnings"].append(
                "Groq available for extraction only (no web search). "
                "Verification quality may be degraded."
            )

    if not health["has_web_search"]:
        health["warnings"].append(
            "CRITICAL: No web search provider available. "
            "Verification will be unreliable."
        )

    if len(health["full_providers"]) < 2:
        health["warnings"].append(
            f"Only {len(health['full_providers'])} verification provider(s) available. "
            "Multi-source verification not possible."
        )

    return health
```

Call this at pipeline start and include warnings in pipeline output and dashboard.

---

## Upgrade 4: Remove Content Truncation

**Priority**: HIGH
**Effort**: S
**Regression Risk**: Low

### What to Change

**In `fact_verification_agent.py:287`:**

```python
# BEFORE:
prompt = self.CLAIM_EXTRACTOR_PROMPT.format(content=content[:6000])

# AFTER - chunk if needed:
MAX_CHUNK_SIZE = 12000  # ~2000 words, covers typical articles

if len(content) <= MAX_CHUNK_SIZE:
    chunks = [content]
else:
    # Split into overlapping chunks at paragraph boundaries
    chunks = self._split_into_chunks(content, MAX_CHUNK_SIZE, overlap=500)

all_claims = []
for chunk in chunks:
    prompt = self.CLAIM_EXTRACTOR_PROMPT.format(content=chunk)
    chunk_claims = self._extract_claims_from_text(prompt)
    all_claims.extend(chunk_claims)

# Deduplicate claims by text similarity
claims = self._deduplicate_claims(all_claims)
```

Similarly fix truncation in `gemini_researcher.py:195` and `perplexity_researcher.py:179`.

---

## Upgrade 5: Add Minimum Claim Count Validation

**Priority**: HIGH
**Effort**: S
**Regression Risk**: Low

### What to Change

After claim extraction, validate that a reasonable number of claims were found:

```python
def verify_article(self, content: str, topic: str = "") -> FactVerificationReport:
    claims = self._extract_claims(content)

    # Estimate expected claim density
    word_count = len(content.split())
    min_expected_claims = max(2, word_count // 300)  # ~1 claim per 300 words

    if len(claims) < min_expected_claims and word_count > 200:
        # Suspiciously few claims — likely extraction failure
        return FactVerificationReport(
            claims=claims,
            results=[],
            passes_quality_gate=False,
            summary=f"NEEDS REVIEW: Only {len(claims)} claims extracted from "
                    f"{word_count}-word article (expected >= {min_expected_claims}). "
                    "Possible extraction failure."
        )

    # ... rest of verification
```

---

## Upgrade 6: Add Verification Audit Logging

**Priority**: MEDIUM
**Effort**: M
**Regression Risk**: None

### What to Change

Add structured logging for every verification run:

```python
import logging
import json
from datetime import datetime

verification_logger = logging.getLogger("ghostwriter.verification")

def _log_verification_run(self, report: FactVerificationReport,
                          providers_used: list, duration_ms: float):
    verification_logger.info(json.dumps({
        "timestamp": datetime.now().isoformat(),
        "total_claims": len(report.claims),
        "verified": report.verified_count,
        "unverified": report.unverified_count,
        "false": report.false_count,
        "passed": report.passes_quality_gate,
        "providers_used": providers_used,
        "duration_ms": duration_ms,
        "extraction_provider": self._last_extraction_provider,
    }))
```

Store logs in `logs/verification/` for post-hoc analysis and calibration.

---

## Upgrade 7: Add Back-Reference Validation for Claims

**Priority**: MEDIUM
**Effort**: S
**Regression Risk**: None

### What to Change

After claim extraction, validate that each extracted claim actually appears (or closely matches) text in the article:

```python
def _validate_claims_against_source(self, claims: List[Claim], content: str) -> List[Claim]:
    """Ensure extracted claims actually come from the article."""
    validated = []
    content_lower = content.lower()

    for claim in claims:
        # Check if key phrases from the claim appear in the source
        claim_words = set(claim.text.lower().split())
        significant_words = {w for w in claim_words if len(w) > 4}

        if not significant_words:
            validated.append(claim)  # Short claims, give benefit of doubt
            continue

        # At least 60% of significant words should appear in source
        found = sum(1 for w in significant_words if w in content_lower)
        if found / len(significant_words) >= 0.6:
            validated.append(claim)
        else:
            # LLM hallucinated a claim not in the article
            print(f"  Dropping hallucinated claim: {claim.text[:60]}...")

    return validated
```

This catches the case where the extraction LLM invents claims that aren't actually in the article (LLM hallucination during extraction).

---

## Implementation Order

```
1. [CRITICAL] Upgrade 1: Fix fail-open bypasses (must be first)
2. [HIGH]     Upgrade 5: Minimum claim count validation (quick safety net)
3. [HIGH]     Upgrade 3: Provider health check (visibility)
4. [HIGH]     Upgrade 4: Remove content truncation (coverage)
5. [HIGH]     Upgrade 2: Structured output (reliability)
6. [MEDIUM]   Upgrade 7: Back-reference validation (accuracy)
7. [MEDIUM]   Upgrade 6: Audit logging (observability)
```

## Estimated Total Effort: 3-4 days for a single engineer

## Files Modified

| File | Changes |
|------|---------|
| `execution/agents/fact_verification_agent.py` | Upgrades 1, 2, 4, 5, 6, 7 — fail-closed logic, structured output, chunking, validation, logging |
| `execution/quality_gate.py` | Upgrade 1 — fail-closed defaults for missing verifier and exceptions |
| `execution/pipeline.py` | Upgrade 1 — verification crash → ESCALATE instead of REVIEW |
| `execution/agents/gemini_researcher.py` | Upgrade 4 — increase truncation limit |
| `execution/agents/perplexity_researcher.py` | Upgrade 4 — increase truncation limit |
| `requirements.txt` | Add `instructor`, `pydantic` (if not present) |

## Dependencies on Feature 1 (BaseAgent)

- Upgrade 2 (structured output) benefits from BaseAgent's exception handling fix
- Provider health check should integrate with BaseAgent's per-agent provider override
- These can be implemented in parallel, but testing requires both
