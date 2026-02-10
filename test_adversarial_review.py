#!/usr/bin/env python3
"""
USER-TEST-2: Adversarial Review System Validation

Tests the complete adversarial review system including:
1. Multi-model adversarial panel (Claude/Gemini/GPT-4o routing)
2. WSJ Four Showstoppers checklist
3. Voice validation (external vs internal)
4. Escalation logic
5. Quality gate integration
"""

import sys
import io

# Fix Windows console encoding for Unicode support
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from execution.agents.adversarial_panel import AdversarialPanelAgent, PanelVerdict
from execution.voice_utils import validate_voice, check_voice_violations, get_voice_instruction
from execution.config import config

# Test content with known issues
TEST_CONTENT_WITH_ISSUES = """
# How I Built a Revolutionary AI System

In this article, we'll explore the latest trends in AI. I built this system over 6 months
and our team created a novel approach to prompt engineering.

The key findings show that AI performance is "very good" in most cases. We discovered
that using more tokens generally helps...

## What's been your experience?

Drop a comment below and share your thoughts! #AIEngineering #MachineLearning #LLMOps #ProductionAI
"""

TEST_CONTENT_CLEAN = """
# Engineering Teams Report 40% Latency Improvements with Novel Caching

A growing number of engineering teams are finding that strategic caching layers
can dramatically reduce API latency. Recent benchmarks from three major tech
companies show consistent 35-45% improvements.

## Key Technical Insights

The approach involves three core components:
1. Edge caching for static responses (reduces 60% of requests)
2. Tiered memory caching for dynamic queries
3. Predictive prefetching based on user patterns

According to published case studies, teams implementing this architecture
reported the following metrics:
- Response time: 200ms -> 120ms (40% reduction)
- Cache hit rate: 78% average
- Infrastructure cost: 25% reduction

## Implementation Considerations

Engineers evaluating this approach should consider their specific traffic patterns.
The technique works best for read-heavy workloads with predictable access patterns.

For teams interested in exploring these optimizations, the open-source reference
implementation provides a solid starting point for experimentation.
"""


def test_wsj_showstoppers():
    """Test WSJ Four Showstoppers checklist."""
    print("\n" + "="*60)
    print("TEST 1: WSJ Four Showstoppers")
    print("="*60)

    panel = AdversarialPanelAgent(multi_model=False)  # Use single model for testing

    # Test problematic content
    print("\nTesting problematic content...")
    wsj_result = panel.check_wsj_showstoppers(TEST_CONTENT_WITH_ISSUES, "AI Engineering")

    print(f"Overall Passed: {wsj_result.get('overall_passed', False)}")
    print(f"Weighted Score: {wsj_result.get('weighted_score', 0)}/10")

    for key in ["inaccuracy", "unfairness", "disorganization", "poor_writing"]:
        if key in wsj_result:
            cat = wsj_result[key]
            status = "PASS" if cat.get("passed") else "FAIL"
            print(f"  {key.title()}: {status} ({cat.get('score', 0)}/10)")
            if cat.get("issues"):
                for issue in cat["issues"][:2]:
                    print(f"    - {issue}")

    # Test clean content
    print("\nTesting clean content...")
    wsj_clean = panel.check_wsj_showstoppers(TEST_CONTENT_CLEAN, "AI Engineering")
    print(f"Overall Passed: {wsj_clean.get('overall_passed', False)}")
    print(f"Weighted Score: {wsj_clean.get('weighted_score', 0)}/10")

    return wsj_result, wsj_clean


def test_voice_validation():
    """Test voice validation for external vs internal sources."""
    print("\n" + "="*60)
    print("TEST 2: Voice Validation")
    print("="*60)

    # Test external source (should fail)
    print("\nExternal source (observer voice) - problematic content:")
    result_ext = validate_voice(TEST_CONTENT_WITH_ISSUES, "external")
    print(f"  Passed: {result_ext['passed']}")
    print(f"  Score: {result_ext['score']}/10")
    print(f"  Violations: {len(result_ext['violations'])}")
    for phrase, context, line in result_ext['violations'][:3]:
        print(f"    Line {line}: '{phrase}'")
    print(f"  Recommendation: {result_ext['recommendation']}")

    # Test internal source (should pass)
    print("\nInternal source (practitioner voice) - same content:")
    result_int = validate_voice(TEST_CONTENT_WITH_ISSUES, "internal")
    print(f"  Passed: {result_int['passed']}")
    print(f"  Score: {result_int['score']}/10")

    # Test clean content with external source
    print("\nExternal source - clean content:")
    result_clean = validate_voice(TEST_CONTENT_CLEAN, "external")
    print(f"  Passed: {result_clean['passed']}")
    print(f"  Score: {result_clean['score']}/10")

    return result_ext, result_int, result_clean


def test_adversarial_panel():
    """Test full adversarial panel review."""
    print("\n" + "="*60)
    print("TEST 3: Adversarial Panel Review")
    print("="*60)

    panel = AdversarialPanelAgent(multi_model=False)  # Single model for faster testing

    # Review problematic content
    print("\nReviewing problematic content (external source)...")
    print("(This may take 30-60 seconds for LLM calls)\n")

    verdict_bad = panel.review_content(
        content=TEST_CONTENT_WITH_ISSUES,
        platform="medium",
        source_type="external",
        include_wsj_check=True,
        topic="AI Engineering"
    )

    print(panel.format_verdict_summary(verdict_bad))

    # Check verdict details
    print(f"\nVerdict Details:")
    print(f"  Average Score: {verdict_bad.average_score}/10")
    print(f"  Passed: {verdict_bad.passed}")
    print(f"  Verdict: {verdict_bad.verdict}")
    print(f"  Expert Scores: {verdict_bad.expert_scores}")

    return verdict_bad


def test_kill_phrases():
    """Test kill phrase detection."""
    print("\n" + "="*60)
    print("TEST 4: Kill Phrase Detection")
    print("="*60)

    panel = AdversarialPanelAgent(multi_model=False)

    # Check for kill phrases
    hits = panel._check_kill_phrases(TEST_CONTENT_WITH_ISSUES)

    print(f"Kill phrases found: {len(hits)}")
    for code, message, phrase in hits:
        print(f"  [{code}] {phrase}")
        print(f"    -> {message}")

    return hits


def test_escalation_config():
    """Test escalation configuration."""
    print("\n" + "="*60)
    print("TEST 5: Escalation Configuration")
    print("="*60)

    print(f"Pass Threshold: {config.quality.PASS_THRESHOLD}")
    print(f"Escalation Threshold: {config.quality.ESCALATION_THRESHOLD}")
    print(f"Max Iterations: {config.quality.MAX_ITERATIONS}")
    print(f"Auto-Approve Threshold: {config.quality.MIN_SCORE_FOR_AUTO_APPROVE}")

    print("\nEscalation Reasons:")
    for code, reason in config.quality.ESCALATION_REASONS.items():
        print(f"  {code}: {reason[:50]}...")


def run_all_tests():
    """Run all USER-TEST-2 validation tests."""
    print("\n" + "="*60)
    print("USER-TEST-2: Adversarial Review System Validation")
    print("="*60)

    results = {
        "wsj": None,
        "voice": None,
        "panel": None,
        "kill_phrases": None,
        "config": True
    }

    try:
        # Test 1: WSJ Showstoppers
        wsj_bad, wsj_clean = test_wsj_showstoppers()
        results["wsj"] = not wsj_bad.get("overall_passed", True) and wsj_clean.get("overall_passed", False)
        print(f"\nWSJ Test: {'PASS' if results['wsj'] else 'INCONCLUSIVE'}")

    except Exception as e:
        print(f"\nWSJ Test FAILED: {e}")
        results["wsj"] = False

    try:
        # Test 2: Voice Validation
        voice_ext, voice_int, voice_clean = test_voice_validation()
        results["voice"] = (
            not voice_ext["passed"] and  # External with violations should fail
            voice_int["passed"] and      # Internal allows ownership
            voice_clean["passed"]        # Clean content should pass
        )
        print(f"\nVoice Test: {'PASS' if results['voice'] else 'FAIL'}")

    except Exception as e:
        print(f"\nVoice Test FAILED: {e}")
        results["voice"] = False

    try:
        # Test 3: Kill Phrases
        kill_hits = test_kill_phrases()
        results["kill_phrases"] = len(kill_hits) >= 3  # Expected: weak CTA, hashtag spam, boring opener
        print(f"\nKill Phrase Test: {'PASS' if results['kill_phrases'] else 'FAIL'}")

    except Exception as e:
        print(f"\nKill Phrase Test FAILED: {e}")
        results["kill_phrases"] = False

    try:
        # Test 4: Escalation Config
        test_escalation_config()
        results["config"] = True
        print(f"\nConfig Test: PASS")

    except Exception as e:
        print(f"\nConfig Test FAILED: {e}")
        results["config"] = False

    try:
        # Test 5: Full Panel Review (optional - expensive LLM calls)
        verdict = test_adversarial_panel()
        # Problematic content should not pass
        results["panel"] = not verdict.passed and verdict.average_score < 8.0
        print(f"\nPanel Test: {'PASS' if results['panel'] else 'INCONCLUSIVE'}")

    except Exception as e:
        print(f"\nPanel Test FAILED: {e}")
        results["panel"] = False

    # Summary
    print("\n" + "="*60)
    print("USER-TEST-2 SUMMARY")
    print("="*60)

    all_passed = all(v for v in results.values() if v is not None)
    for test, passed in results.items():
        status = "PASS" if passed else ("FAIL" if passed is False else "SKIP")
        print(f"  {test.upper()}: {status}")

    print(f"\nOVERALL: {'PASS' if all_passed else 'REVIEW NEEDED'}")

    if all_passed:
        print("\nAdversarial review system is working correctly.")
        print("Ready to proceed to Phase 3 tasks.")
    else:
        print("\nSome tests need review. Check output above.")

    return all_passed


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
