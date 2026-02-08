# Feature: Fact Verification System — Research & Adversarial Critique

**Agents**: Technique Research + Red Team
**Date**: 2026-02-08

---

## Research: Industry Standard Patterns for Fact Verification

### R1: Structured Output for Claim Extraction (Standard Practice)

**The problem**: Regex-based JSON parsing of LLM output is fragile and fails silently.

**Industry solutions:**
- **Instructor library** (`instructor` + Pydantic): Forces LLM to return valid JSON matching a Pydantic schema. Automatic retry on validation failure. Used by most production LLM apps.
- **OpenAI structured outputs**: `response_format={"type": "json_schema", ...}` guarantees JSON schema compliance at the API level.
- **Gemini `response_schema`**: `GenerateContentConfig(response_schema=...)` enforces JSON schema.

**Pattern**: Define Pydantic models for `ClaimList` and `VerificationResult`. Use `instructor` or native structured output. Never regex-parse LLM output.

### R2: Fail-Closed Verification Gates (Critical Safety Pattern)

**The problem**: "No claims = pass" and "error = pass" are fail-open patterns.

**Industry standard**: Safety-critical systems use **fail-closed** defaults. If the safety check can't run, the item is BLOCKED, not approved.

- **Medical device software (IEC 62304)**: Validation failure = rejection
- **Financial compliance**: KYC check timeout = transaction blocked
- **Content moderation (Meta, Google)**: Classifier error = content held for review

**Pattern**: If verification cannot run or extracts 0 claims, return `needs_review` status, not `pass`. Require human approval for the exception path.

### R3: Natural Language Inference (NLI) for Claim Detection

**The problem**: LLM-based claim extraction misses subtle claims and is unreliable.

**Industry solutions:**
- **ClaimBuster**: Trained NLI model that scores sentences 0-1 for "check-worthiness"
- **Full Fact (Google-funded)**: Sentence-level claim detection classifier
- **FEVER dataset**: Standard benchmark for fact verification systems

**Pattern**: Use a fine-tuned NLI model (e.g., DeBERTa on FEVER) to identify check-worthy sentences, THEN use LLM+search to verify. Separates detection from verification.

**Pragmatic alternative for now**: Use the LLM for extraction but enforce minimum claim count + validate that extracted claims actually appear in the article text (back-reference check).

### R4: Source Quality Scoring

**The problem**: All verification sources are treated equally. A random blog has the same weight as an official documentation page.

**Industry solutions:**
- **Google Fact Check API**: Provides ClaimReview ratings from fact-checking organizations
- **Domain authority scoring**: Common in SEO but applicable to source reliability
- **Snopes methodology**: Rate sources by reputation tier (official > academic > news > blog > forum)

**Pattern**: Classify verification sources into tiers and require at least one high-authority source for "verified" status.

### R5: Verification Provenance and Audit Trail

**The problem**: No record of which claims were checked, which sources were used, or which provider performed the verification.

**Industry solutions:**
- **AP Fact Check**: Every verified claim has a paper trail
- **Reuters**: Audit log of verification steps
- **LangSmith**: Traces every LLM call with inputs, outputs, and metadata

**Pattern**: Log every verification attempt: provider used, search queries, sources found, confidence score, latency. Store as structured data for post-hoc audit.

### R6: Human-in-the-Loop Sampling

**The problem**: No way to know if automated verification is actually correct over time.

**Industry solutions:**
- **Active learning**: Randomly sample verified claims for human review. Use disagreements to improve the system.
- **Reuters verification desk**: Automated tools flag, humans confirm
- **Wikipedia**: Bot-assisted but human-approved for policy articles

**Pattern**: Flag 1-2 claims per article for human spot-check. Track agreement rate over time as a calibration metric.

---

## Red Team: Trust-Breaking Attack Scenarios

### Attack 1: "The Rubber Stamp" (SEVERITY: CRITICAL)

**Vector**: Abstract article with no extractable statistical claims
**Steps**:
1. Writer generates opinion piece about "Why Kubernetes is overrated"
2. Article contains claims like "most teams don't need Kubernetes" (opinion, not extractable)
3. Article also contains "Kubernetes requires 3x more engineering time" (false, but phrased as common knowledge)
4. Claim extractor finds 0 extractable claims (no statistics, no specs)
5. `verify_article()` line 242: `if not claims:` → auto-pass
6. Article published with false "3x engineering time" claim

**Likelihood**: HIGH — many articles are opinion-heavy with embedded claims
**Impact**: False claims published as verified content

### Attack 2: "The Truncation Exploit" (SEVERITY: HIGH)

**Vector**: Long article with false claims placed after char 6000
**Steps**:
1. Article has 2000 words (~12,000 chars)
2. First 6000 chars contain 5 easily verifiable claims → all pass
3. Last 6000 chars contain 3 false claims → never extracted (truncated away)
4. Verification passes with 5/5 verified, 0 false
5. Published article has 3 unchecked false claims in the second half

**Likelihood**: MEDIUM — depends on article structure, but long articles are common
**Impact**: False claims in second half of articles are never caught

### Attack 3: "The Grade Inflation Cascade" (SEVERITY: HIGH)

**Vector**: Provider fallback degrades verification quality invisibly
**Steps**:
1. Google API key expires. Gemini provider fails to init.
2. `_setup_providers()` catches exception, prints warning, continues
3. Only Perplexity and Groq remain
4. Perplexity has rate limit → verification falls back to Groq
5. Groq CAN'T do web search → `_verify_claim()` calls `_verify_with_perplexity()` on GroqWrapper
6. GroqWrapper has no `verify_draft()` method → `AttributeError`
7. Exception caught → ALL claims return `UNVERIFIED`
8. BUT: only 5 claims were extracted, so `unverified=5` which is `> max_unverified(1)` → FAILS

**Actually**: This scenario DOES fail the gate, which is correct behavior. BUT the user gets zero useful information — just "5 unverified claims" with no explanation.

**Revised attack**: Gemini API key works but is rate-limited intermittently.
1. Claim extraction succeeds (Gemini, before rate limit)
2. First 2 claim verifications succeed (Gemini)
3. Gemini rate-limited → falls back to Perplexity for remaining claims
4. Perplexity has less accurate search → marks claims as "partial" instead of "verified"
5. Score: verified=2, partial=2, unverified=1 → passes gate (verified+partial=4 >= 3)
6. But the "partial" verdicts are lower confidence than Gemini would have given
7. User doesn't know verification quality degraded mid-run

**Likelihood**: MEDIUM
**Impact**: Inconsistent verification quality within same article, no visibility

### Attack 4: "The Keyword False Positive" (SEVERITY: MEDIUM)

**Vector**: Fallback keyword-matching in `_parse_verification_result()`
**Steps**:
1. Gemini returns verification response with malformed JSON
2. JSON parsing fails → enters fallback path at line 526
3. Response contains: "This claim is NOT verified by any authoritative source"
4. Keyword scan: `"verified" in response_lower` → TRUE
5. Status assigned: `PARTIALLY_VERIFIED`
6. A claim explicitly flagged as NOT verified gets marked as partially verified

**Likelihood**: MEDIUM — depends on JSON parsing failure frequency
**Impact**: Incorrectly verified claims boost the passing score

### Attack 5: "The Hallucinated Source" (SEVERITY: HIGH)

**Vector**: Gemini grounding search returns loosely-related snippets
**Steps**:
1. Article claims "GPT-5 has 10 trillion parameters"
2. Gemini searches, finds article: "OpenAI reportedly working on GPT-5"
3. No article confirms the 10T parameter count
4. But Gemini's verification LLM interprets the search result as partial confirmation
5. Returns: `status: "partial"`, confidence: 0.6, source: unrelated article
6. Claim counted as partially verified, contributes to passing score

**Likelihood**: HIGH — LLM interpretation of search results is unreliable
**Impact**: False claims backed by unrelated sources appear "verified"

### Attack 6: "The Error String Article" (SEVERITY: CRITICAL — crosses from Feature 1)

**Vector**: BaseAgent error string flows to verification
**Steps**:
1. Writer's `call_llm()` returns "Groq Error: rate limit exceeded" (Feature 1 bug)
2. This error string becomes the article "draft"
3. `verify_article("Groq Error: rate limit exceeded")`
4. Claim extractor: content[:6000] = "Groq Error: rate limit exceeded"
5. LLM finds 0 verifiable claims in an error string
6. `if not claims:` → auto-pass
7. Error string is "verified" by the fact checker

**Likelihood**: HIGH (when combined with Feature 1 BaseAgent bug)
**Impact**: Maximum — error strings published as verified content

---

## Summary Verdict

| Dimension | Rating | Notes |
|-----------|--------|-------|
| Safety defaults | F | Fail-open on 3 separate paths (no claims, exceptions, no provider) |
| Extraction reliability | D | Regex JSON parsing, 6000 char truncation, circular LLM reasoning |
| Verification accuracy | C | Web search grounding works when available; hallucination risk exists |
| Provider resilience | D | Silent fallback, no health monitoring, Groq can't verify |
| Observability | F | No audit trail, no verification logging, no calibration data |
| Confidence calibration | F | Self-reported LLM confidence, uncalibrated, misleading |
