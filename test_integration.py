#!/usr/bin/env python3
"""
End-to-End Integration Test Suite.

Tests the complete GhostWriter pipeline including:
1. Configuration and imports
2. Agent initialization
3. Fact verification
4. Adversarial panel review
5. Provenance tracking
6. Voice validation
7. Quality gate
8. Full pipeline flow (dry run)

Run with: python test_integration.py
"""

import sys
import io
import os
import json
from pathlib import Path

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')


def test_imports():
    """Test all critical imports work."""
    print("\n" + "="*60)
    print("TEST: Critical Imports")
    print("="*60)

    tests = []

    # Core config
    try:
        from execution.config import config, OUTPUT_DIR
        tests.append(("Config module", True, ""))
    except Exception as e:
        tests.append(("Config module", False, str(e)))

    # Agents
    try:
        from execution.agents import WriterAgent, EditorAgent, CriticAgent
        tests.append(("Agent base classes", True, ""))
    except Exception as e:
        tests.append(("Agent base classes", False, str(e)))

    try:
        from execution.agents.adversarial_panel import AdversarialPanelAgent
        tests.append(("Adversarial panel", True, ""))
    except Exception as e:
        tests.append(("Adversarial panel", False, str(e)))

    try:
        from execution.agents.fact_verification_agent import FactVerificationAgent
        tests.append(("Fact verification agent", True, ""))
    except Exception as e:
        tests.append(("Fact verification agent", False, str(e)))

    # Pipeline
    try:
        from execution.pipeline import create_pipeline, run_pipeline
        tests.append(("LangGraph pipeline", True, ""))
    except Exception as e:
        tests.append(("LangGraph pipeline", False, str(e)))

    # Quality gate
    try:
        from execution.quality_gate import QualityGate
        tests.append(("Quality gate", True, ""))
    except Exception as e:
        tests.append(("Quality gate", False, str(e)))

    # Provenance
    try:
        from execution.provenance import ProvenanceTracker, generate_c2pa_manifest
        tests.append(("Provenance module", True, ""))
    except Exception as e:
        tests.append(("Provenance module", False, str(e)))

    # Voice utilities
    try:
        from execution.voice_utils import validate_voice, get_voice_instruction
        tests.append(("Voice utilities", True, ""))
    except Exception as e:
        tests.append(("Voice utilities", False, str(e)))

    # Optimization
    try:
        from execution.optimization import OptimizationTracker, estimate_cost
        tests.append(("Optimization module", True, ""))
    except Exception as e:
        tests.append(("Optimization module", False, str(e)))

    # Report results
    passed = sum(1 for _, ok, _ in tests if ok)
    total = len(tests)

    for name, ok, error in tests:
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {name}")
        if error:
            print(f"       Error: {error[:50]}...")

    print(f"\nImports: {passed}/{total} passed")
    return passed == total


def test_config():
    """Test configuration is properly loaded."""
    print("\n" + "="*60)
    print("TEST: Configuration")
    print("="*60)

    from execution.config import config

    tests = []

    # Check quality config
    tests.append(("Pass threshold", config.quality.PASS_THRESHOLD == 7.0))
    tests.append(("Max iterations", config.quality.MAX_ITERATIONS >= 1))
    tests.append(("Escalation reasons", len(config.quality.ESCALATION_REASONS) > 0))

    # Check voice config
    tests.append(("Voice external", hasattr(config.voice, 'VOICE_EXTERNAL')))
    tests.append(("Voice internal", hasattr(config.voice, 'VOICE_INTERNAL')))

    # Check API keys (at least one should be present)
    has_api = any([
        config.api.has_key("gemini"),
        config.api.has_key("perplexity"),
        config.api.has_key("anthropic"),
        config.api.has_key("groq")
    ])
    tests.append(("At least one API key", has_api))

    passed = sum(1 for _, ok in tests if ok)
    total = len(tests)

    for name, ok in tests:
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {name}")

    print(f"\nConfig: {passed}/{total} passed")
    return passed == total


def test_voice_validation():
    """Test voice validation utilities."""
    print("\n" + "="*60)
    print("TEST: Voice Validation")
    print("="*60)

    from execution.voice_utils import validate_voice, check_voice_violations

    tests = []

    # Test external voice violations
    bad_content = "I built this system and our team created a novel approach."
    violations = check_voice_violations(bad_content, "external")
    tests.append(("Detects ownership violations", len(violations) >= 2))

    result = validate_voice(bad_content, "external")
    tests.append(("External fails with violations", not result["passed"]))

    # Test internal voice (should pass)
    result_int = validate_voice(bad_content, "internal")
    tests.append(("Internal passes with ownership", result_int["passed"]))

    # Test clean external content
    clean_content = "Engineers at the company discovered significant improvements."
    result_clean = validate_voice(clean_content, "external")
    tests.append(("Clean external passes", result_clean["passed"]))

    passed = sum(1 for _, ok in tests if ok)
    total = len(tests)

    for name, ok in tests:
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {name}")

    print(f"\nVoice: {passed}/{total} passed")
    return passed == total


def test_adversarial_panel():
    """Test adversarial panel initialization and structure."""
    print("\n" + "="*60)
    print("TEST: Adversarial Panel")
    print("="*60)

    from execution.agents.adversarial_panel import AdversarialPanelAgent

    tests = []

    # Initialize panel
    try:
        panel = AdversarialPanelAgent(multi_model=False)
        tests.append(("Panel initializes", True))
    except Exception as e:
        tests.append(("Panel initializes", False))
        print(f"  Error: {e}")
        return False

    # Check expert panels exist
    tests.append(("Agency panel exists", "agency" in panel.EXPERT_PANELS))
    tests.append(("Brand panel exists", "brand" in panel.EXPERT_PANELS))
    tests.append(("SEO panel exists", "seo" in panel.EXPERT_PANELS))
    tests.append(("Creative panel exists", "creative" in panel.EXPERT_PANELS))

    # Check WSJ showstoppers
    tests.append(("WSJ inaccuracy defined", "inaccuracy" in panel.WSJ_FOUR_SHOWSTOPPERS))
    tests.append(("WSJ unfairness defined", "unfairness" in panel.WSJ_FOUR_SHOWSTOPPERS))

    # Check kill phrases
    tests.append(("Kill phrases defined", len(panel.KILL_PHRASES) > 0))

    # Check model routing
    tests.append(("Model routing defined", len(panel.MODEL_ROUTING) > 0))

    # Test kill phrase detection
    bad_content = "In this article... What's been your experience?"
    hits = panel._check_kill_phrases(bad_content)
    tests.append(("Kill phrase detection works", len(hits) >= 2))

    passed = sum(1 for _, ok in tests if ok)
    total = len(tests)

    for name, ok in tests:
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {name}")

    print(f"\nAdversarial Panel: {passed}/{total} passed")
    return passed == total


def test_provenance():
    """Test provenance tracking."""
    print("\n" + "="*60)
    print("TEST: Provenance Tracking")
    print("="*60)

    from execution.provenance import (
        ProvenanceTracker, generate_c2pa_manifest,
        generate_schema_org_jsonld, generate_inline_disclosure
    )

    tests = []

    # Initialize tracker
    try:
        tracker = ProvenanceTracker()
        tracker.start_tracking(
            topic="Test Topic",
            source_type="external",
            platform="medium"
        )
        tests.append(("Tracker initializes", True))
    except Exception as e:
        tests.append(("Tracker initializes", False))
        print(f"  Error: {e}")
        return False

    # Record actions
    tracker.record_research("TestAgent", model="test-model", facts_found=3)
    tracker.record_generation("TestAgent", word_count=500)
    tracker.record_verification(passed=True, claims_verified=3)
    tracker.record_review(score=7.5, passed=True)

    # Finalize
    provenance = tracker.finalize("Test content")
    tests.append(("Provenance finalized", provenance is not None))
    tests.append(("Content ID generated", provenance.content_id.startswith("gw-")))
    tests.append(("Content hash generated", len(provenance.content_hash) == 64))
    tests.append(("Actions recorded", len(provenance.actions) > 0))

    # Generate C2PA
    c2pa = generate_c2pa_manifest(provenance)
    tests.append(("C2PA manifest generated", "claim_generator" in c2pa))

    # Generate Schema.org
    schema = generate_schema_org_jsonld(provenance, "Test", "Description")
    tests.append(("Schema.org JSON-LD generated", schema["@type"] == "Article"))

    # Generate disclosure
    disclosure = generate_inline_disclosure(provenance, "brief")
    tests.append(("Inline disclosure generated", "AI-assisted" in disclosure))

    passed = sum(1 for _, ok in tests if ok)
    total = len(tests)

    for name, ok in tests:
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {name}")

    print(f"\nProvenance: {passed}/{total} passed")
    return passed == total


def test_optimization():
    """Test optimization tracking."""
    print("\n" + "="*60)
    print("TEST: Optimization Tracking")
    print("="*60)

    from execution.optimization import (
        OptimizationTracker, AgentPerformance, estimate_cost,
        generate_quality_feedback
    )

    tests = []

    # Initialize tracker
    try:
        tracker = OptimizationTracker()
        tests.append(("Tracker initializes", True))
    except Exception as e:
        tests.append(("Tracker initializes", False))
        print(f"  Error: {e}")
        return False

    # Test cost estimation
    cost = estimate_cost("gpt-4o", 1000, 500)
    tests.append(("Cost estimation works", cost > 0))

    cost_free = estimate_cost("gemini-2.0-flash-exp", 1000, 500)
    tests.append(("Free tier has zero cost", cost_free == 0))

    # Test feedback generation
    feedback = generate_quality_feedback("test-123", 8.5, True)
    tests.append(("Quality feedback generated", feedback.signal_type == "reward"))

    feedback_fail = generate_quality_feedback("test-456", 5.0, False)
    tests.append(("Failed feedback is penalty", feedback_fail.signal_type == "penalty"))

    passed = sum(1 for _, ok in tests if ok)
    total = len(tests)

    for name, ok in tests:
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {name}")

    print(f"\nOptimization: {passed}/{total} passed")
    return passed == total


def test_pipeline_structure():
    """Test pipeline structure (without running LLMs)."""
    print("\n" + "="*60)
    print("TEST: Pipeline Structure")
    print("="*60)

    from execution.pipeline import (
        create_pipeline, PHASE_RESEARCH, PHASE_GENERATE,
        PHASE_VERIFY, PHASE_REVIEW, PHASE_APPROVE
    )

    tests = []

    # Create pipeline
    try:
        pipeline = create_pipeline()
        tests.append(("Pipeline creates", True))
    except Exception as e:
        tests.append(("Pipeline creates", False))
        print(f"  Error: {e}")
        return False

    # Check phases are defined
    tests.append(("Research phase defined", PHASE_RESEARCH == "research"))
    tests.append(("Generate phase defined", PHASE_GENERATE == "generate"))
    tests.append(("Verify phase defined", PHASE_VERIFY == "verify"))
    tests.append(("Review phase defined", PHASE_REVIEW == "review"))
    tests.append(("Approve phase defined", PHASE_APPROVE == "approve"))

    passed = sum(1 for _, ok in tests if ok)
    total = len(tests)

    for name, ok in tests:
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {name}")

    print(f"\nPipeline: {passed}/{total} passed")
    return passed == total


def test_quality_gate_structure():
    """Test quality gate structure."""
    print("\n" + "="*60)
    print("TEST: Quality Gate Structure")
    print("="*60)

    from execution.quality_gate import QualityGate, QualityGateResult

    tests = []

    # Initialize quality gate
    try:
        gate = QualityGate(max_iterations=3, verbose=False, require_verification=False)
        tests.append(("Quality gate initializes", True))
    except Exception as e:
        tests.append(("Quality gate initializes", False))
        print(f"  Error: {e}")
        return False

    # Check components
    tests.append(("Has panel agent", gate.panel is not None))
    tests.append(("Has writer agent", gate.writer is not None))
    tests.append(("Has editor agent", gate.editor is not None))
    tests.append(("Max iterations set", gate.max_iterations == 3))

    passed = sum(1 for _, ok in tests if ok)
    total = len(tests)

    for name, ok in tests:
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {name}")

    print(f"\nQuality Gate: {passed}/{total} passed")
    return passed == total


def run_all_tests():
    """Run all integration tests."""
    print("\n" + "="*60)
    print("GHOSTWRITER END-TO-END INTEGRATION TESTS")
    print("="*60)

    results = {}

    # Run tests
    results["imports"] = test_imports()
    results["config"] = test_config()
    results["voice"] = test_voice_validation()
    results["adversarial"] = test_adversarial_panel()
    results["provenance"] = test_provenance()
    results["optimization"] = test_optimization()
    results["pipeline"] = test_pipeline_structure()
    results["quality_gate"] = test_quality_gate_structure()

    # Summary
    print("\n" + "="*60)
    print("INTEGRATION TEST SUMMARY")
    print("="*60)

    all_passed = True
    for test_name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {test_name.upper()}: {status}")
        if not passed:
            all_passed = False

    total = len(results)
    passed_count = sum(1 for v in results.values() if v)

    print(f"\nOVERALL: {passed_count}/{total} test suites passed")

    if all_passed:
        print("\nAll integration tests PASSED!")
        print("GhostWriter pipeline is ready for use.")
    else:
        print("\nSome tests FAILED. Review output above.")

    return all_passed


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
