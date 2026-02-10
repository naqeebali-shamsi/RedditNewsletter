#!/usr/bin/env python3
"""
Test Auxiliary Agents in Isolation

Tests the fact-checking and research agents that are NOT yet in the main pipeline.
Purpose: Understand what they catch before deciding on integration.

Usage:
    python execution/test_auxiliary_agents.py
"""

import sys
import io
from pathlib import Path

# Fix Windows console encoding for Unicode
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

# ============================================================================
# TEST CONTENT - A draft with INTENTIONAL problems for agents to catch
# ============================================================================

PROBLEMATIC_DRAFT = """
# Why RAG is Dead: The Rise of Long-Context Models

The AI landscape shifted dramatically when teams discovered that RAG pipelines were causing
a 40% increase in latency without proportional accuracy gains. Studies show that most
enterprises are now abandoning RAG entirely.

## The Numbers Don't Lie

Our benchmarks revealed that Gemini 2.0's 2M token context window processes documents
3x faster than traditional RAG approaches. The H100's 80GB HBM3 memory running at
3.35TB/s bandwidth makes this possible, achieving 1,979 TFLOPS of FP16 performance.

Engineers have found that switching from chunked retrieval to full-context injection
reduced their error rates by 67%. One company reported saving $50,000/month on
infrastructure after the migration.

## Code Example

Here's how simple it is:

```python
def process_with_context(docs):
    combined = " ".join(docs)
    response = model.generate(combined, max_tokens=100000
    return response
```

## Practical Implications

Research indicates that by 2025, 80% of production RAG systems will be replaced by
long-context models. This aligns with what we're seeing across the industry.

The lesson is clear: if you're still building RAG pipelines, you're building technical debt.

What's been your experience with long-context models? Drop a comment below!
"""

# ============================================================================
# TEST 1: TechnicalSupervisorAgent
# ============================================================================

def test_technical_supervisor():
    """Test the BS detector on problematic content."""
    print("\n" + "=" * 70)
    print("TEST 1: TechnicalSupervisorAgent (BS Detector)")
    print("=" * 70)

    from execution.agents.technical_supervisor import TechnicalSupervisorAgent

    supervisor = TechnicalSupervisorAgent()

    print("\nRunning technical review on draft with known issues...")
    print("Looking for: fabricated stats, phantom evidence, broken code\n")

    # Test the validation method (pattern-based)
    print("[1a] Pattern-based validation...")
    validation_result = supervisor.validate(PROBLEMATIC_DRAFT)

    print(f"Score: {validation_result['score']}/100")
    print(f"Passed: {validation_result['passed']}")
    print(f"Critical issues: {validation_result['critical_count']}")
    print(f"Major issues: {validation_result['major_count']}")
    print("\nSummary:")
    print(validation_result['summary'])

    # Test the review_draft method (generates feedback string)
    print("\n[1b] Full review (for revision instructions)...")
    result = supervisor.review_draft(PROBLEMATIC_DRAFT)

    print("-" * 70)
    print("SUPERVISOR VERDICT:")
    print("-" * 70)
    print(result)

    return result


# ============================================================================
# TEST 2: GeminiResearchAgent
# ============================================================================

def test_gemini_researcher():
    """Test Gemini's ability to fact-check claims."""
    print("\n" + "=" * 70)
    print("TEST 2: GeminiResearchAgent (Google Search Grounding)")
    print("=" * 70)

    from execution.agents.gemini_researcher import GeminiResearchAgent

    researcher = GeminiResearchAgent()

    # Test 1: Verify the entire draft
    print("\n[Test 2a] Fact-checking entire draft via verify_draft()...")
    verification_result = researcher.verify_draft(
        draft=PROBLEMATIC_DRAFT,
        topic="RAG vs Long-Context Models"
    )

    print("-" * 70)
    print("GEMINI VERIFICATION RESULT:")
    print("-" * 70)

    # Pretty print the result
    import json
    if isinstance(verification_result, dict):
        print(f"Verified Claims: {len(verification_result.get('verified_claims', []))}")
        print(f"False Claims: {len(verification_result.get('false_claims', []))}")
        print(f"Unverifiable Claims: {len(verification_result.get('unverifiable_claims', []))}")
        print(f"Suspicious Claims: {len(verification_result.get('suspicious_claims', []))}")
        print(f"Overall Score: {verification_result.get('overall_accuracy_score', 'N/A')}/100")
        print(f"Recommendation: {verification_result.get('recommendation', 'N/A')}")

        if verification_result.get('revision_instructions'):
            print("\nRevision Instructions:")
            print(verification_result['revision_instructions'])
    else:
        print(verification_result)

    return verification_result


# ============================================================================
# TEST 3: FactResearchAgent (limited - no web search)
# ============================================================================

def test_fact_researcher_limited():
    """Test FactResearchAgent's analysis capabilities (without web search)."""
    print("\n" + "=" * 70)
    print("TEST 3: FactResearchAgent (Analysis Only - No Web Search)")
    print("=" * 70)

    from execution.agents.fact_researcher import FactResearchAgent

    researcher = FactResearchAgent()

    print("\nAsking agent to identify what needs verification...")
    print("(No actual web search - just planning phase)\n")

    # Test the planning phase only
    topic = "RAG vs Long-Context Models performance comparison"
    result = researcher.research(topic, source_content=PROBLEMATIC_DRAFT, web_search_func=None)

    print("-" * 70)
    print("FACT RESEARCHER ANALYSIS:")
    print("-" * 70)

    if isinstance(result, dict):
        print(f"Verified Facts: {len(result.get('verified_facts', []))}")
        print(f"Unverified Claims: {len(result.get('unverified_claims', []))}")
        print(f"General Knowledge: {len(result.get('general_knowledge', []))}")
        print(f"Unknowns: {len(result.get('unknowns', []))}")

        if result.get('writer_constraints'):
            print("\nWriter Constraints Generated:")
            print(result['writer_constraints'][:1000])
    else:
        print(result)

    return result


# ============================================================================
# TEST 4: PerplexityResearchAgent (Sonar Pro with web search)
# ============================================================================

def test_perplexity_researcher():
    """Test Perplexity Sonar Pro for grounded fact verification."""
    print("\n" + "=" * 70)
    print("TEST 4: PerplexityResearchAgent (Sonar Pro)")
    print("=" * 70)

    import os
    if not os.getenv("PERPLEXITY_API_KEY"):
        print("\n[SKIPPED] PERPLEXITY_API_KEY not found in environment")
        return None

    from execution.agents.perplexity_researcher import PerplexityResearchAgent

    researcher = PerplexityResearchAgent()

    # Test: Verify the draft
    print("\n[Test 4a] Fact-checking draft with Perplexity Sonar Pro...")
    verification_result = researcher.verify_draft(
        draft=PROBLEMATIC_DRAFT,
        topic="RAG vs Long-Context Models"
    )

    print("-" * 70)
    print("PERPLEXITY VERIFICATION RESULT:")
    print("-" * 70)

    if isinstance(verification_result, dict):
        print(f"Verified Claims: {len(verification_result.get('verified_claims', []))}")
        print(f"False Claims: {len(verification_result.get('false_claims', []))}")
        print(f"Unverifiable Claims: {len(verification_result.get('unverifiable_claims', []))}")
        print(f"Suspicious Claims: {len(verification_result.get('suspicious_claims', []))}")
        print(f"Overall Score: {verification_result.get('overall_accuracy_score', 'N/A')}/100")
        print(f"Recommendation: {verification_result.get('recommendation', 'N/A')}")

        if verification_result.get('revision_instructions'):
            print("\nRevision Instructions:")
            print(verification_result['revision_instructions'])
    else:
        print(verification_result)

    return verification_result


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "=" * 70)
    print("AUXILIARY AGENTS TEST SUITE")
    print("=" * 70)
    print("\nTesting agents that catch fabricated stats, phantom evidence, etc.")
    print("Using a draft with INTENTIONAL problems:\n")

    problems_in_draft = [
        "- '40% increase in latency' - fabricated stat",
        "- 'Studies show' - phantom evidence",
        "- '3x faster' - unverified claim",
        "- '67% reduced error rates' - fabricated stat",
        "- '$50,000/month savings' - unverified specific",
        "- 'Research indicates 80%' - phantom evidence",
        "- Broken Python code (missing parenthesis)",
        "- 'What's been your experience?' - weak CTA",
        "- H100 specs - needs verification",
    ]

    for p in problems_in_draft:
        print(f"  {p}")

    print("\n" + "-" * 70)
    input("\nPress Enter to start tests (uses API credits)...")

    results = {}

    # Test 1: Technical Supervisor (pattern-based + LLM)
    try:
        results['supervisor'] = test_technical_supervisor()
    except Exception as e:
        print(f"\n[X] TechnicalSupervisor failed: {e}")
        import traceback
        traceback.print_exc()
        results['supervisor'] = None

    # Test 2: Gemini Researcher (Google Search grounding)
    try:
        results['gemini'] = test_gemini_researcher()
    except Exception as e:
        print(f"\n[X] GeminiResearcher failed: {e}")
        import traceback
        traceback.print_exc()
        results['gemini'] = None

    # Test 3: Fact Researcher (LLM analysis, no web search)
    try:
        results['fact'] = test_fact_researcher_limited()
    except Exception as e:
        print(f"\n[X] FactResearcher failed: {e}")
        import traceback
        traceback.print_exc()
        results['fact'] = None

    # Test 4: Perplexity Researcher (Sonar Pro with web search)
    try:
        results['perplexity'] = test_perplexity_researcher()
    except Exception as e:
        print(f"\n[X] PerplexityResearcher failed: {e}")
        import traceback
        traceback.print_exc()
        results['perplexity'] = None

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    for name, result in results.items():
        if result is None:
            status = "[FAILED/SKIPPED]"
        else:
            status = "[COMPLETED]"
        print(f"  {name}: {status}")

    print("\n" + "=" * 70)
    print("\nNOTE: To add Perplexity support, add PERPLEXITY_API_KEY to your .env file")
    print("Get an API key at: https://www.perplexity.ai/settings/api")
    print("=" * 70)


if __name__ == "__main__":
    main()
