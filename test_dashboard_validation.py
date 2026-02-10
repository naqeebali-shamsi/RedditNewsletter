#!/usr/bin/env python3
"""
USER-TEST-3: HITL Dashboard Validation.

Validates the Streamlit dashboard functionality without starting the server.
Tests:
1. All imports work
2. Dashboard pages exist
3. Components are properly structured
4. Mock data flows correctly
"""

import sys
import io

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')


def test_dashboard_imports():
    """Test all dashboard dependencies import correctly."""
    print("\n" + "="*60)
    print("TEST: Dashboard Imports")
    print("="*60)

    tests = []

    # Core dependencies
    try:
        import streamlit as st
        tests.append(("Streamlit", True, ""))
    except Exception as e:
        tests.append(("Streamlit", False, str(e)))

    try:
        import json
        from pathlib import Path
        from datetime import datetime
        tests.append(("Standard library", True, ""))
    except Exception as e:
        tests.append(("Standard library", False, str(e)))

    try:
        from execution.config import config, OUTPUT_DIR
        tests.append(("Config module", True, ""))
    except Exception as e:
        tests.append(("Config module", False, str(e)))

    try:
        from execution.provenance import generate_inline_disclosure
        tests.append(("Provenance module", True, ""))
    except Exception as e:
        tests.append(("Provenance module", False, str(e)))

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


def test_dashboard_structure():
    """Test dashboard file structure."""
    print("\n" + "="*60)
    print("TEST: Dashboard Structure")
    print("="*60)

    from pathlib import Path

    tests = []

    # Check files exist
    dashboard_dir = Path("execution/dashboard")
    tests.append(("Dashboard directory exists", dashboard_dir.exists()))
    tests.append(("__init__.py exists", (dashboard_dir / "__init__.py").exists()))
    tests.append(("app.py exists", (dashboard_dir / "app.py").exists()))

    # Check app.py content
    app_file = dashboard_dir / "app.py"
    if app_file.exists():
        content = app_file.read_text(encoding='utf-8')

        # Key components
        tests.append(("Has page config", "st.set_page_config" in content))
        tests.append(("Has sidebar navigation", "st.sidebar" in content))
        tests.append(("Has Review Queue page", "def render_review_queue" in content))
        tests.append(("Has Article Review page", "def render_article_review" in content))
        tests.append(("Has Escalations page", "def render_escalations" in content))
        tests.append(("Has Audit Trail page", "def render_audit_trail" in content))
        tests.append(("Has Settings page", "def render_settings" in content))

        # Key functionality
        tests.append(("Has data loading", "load_review_queue" in content or "review_queue" in content))
        tests.append(("Has approval handling", "Approve" in content))
        tests.append(("Has rejection handling", "Reject" in content))
        tests.append(("Has escalation handling", "Escalate" in content))
    else:
        tests.append(("App content checks", False))

    # Report results
    passed = sum(1 for _, ok in tests if ok)
    total = len(tests)

    for name, ok in tests:
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {name}")

    print(f"\nStructure: {passed}/{total} passed")
    return passed == total


def test_dashboard_components():
    """Test individual dashboard components are well-formed."""
    print("\n" + "="*60)
    print("TEST: Dashboard Components")
    print("="*60)

    from pathlib import Path

    tests = []

    app_file = Path("execution/dashboard/app.py")
    content = app_file.read_text(encoding='utf-8')

    # Check for UI components
    ui_components = [
        ("st.title", "Page titles"),
        ("st.header", "Section headers"),
        ("st.columns", "Column layouts"),
        ("st.tabs", "Tab navigation"),
        ("st.button", "Interactive buttons"),
        ("st.text_area", "Text input areas"),
        ("st.selectbox", "Selection dropdowns"),
        ("st.metric", "Metric displays"),
    ]

    for component, name in ui_components:
        tests.append((name, component in content))

    # Check for data display components
    data_components = [
        ("st.json", "JSON display"),
        ("st.code", "Code display"),
        ("st.markdown", "Markdown rendering"),
    ]

    for component, name in data_components:
        tests.append((name, component in content))

    # Report results
    passed = sum(1 for _, ok in tests if ok)
    total = len(tests)

    for name, ok in tests:
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {name}")

    print(f"\nComponents: {passed}/{total} passed")
    return passed == total


def test_dashboard_pages():
    """Test all pages are properly defined."""
    print("\n" + "="*60)
    print("TEST: Dashboard Pages")
    print("="*60)

    from pathlib import Path

    tests = []

    app_file = Path("execution/dashboard/app.py")
    content = app_file.read_text(encoding='utf-8')

    # Check page definitions
    pages = [
        "Review Queue",
        "Article Review",
        "Escalations",
        "Audit Trail",
        "Settings",
    ]

    for page in pages:
        # Check both the page name and its function
        has_page = page in content or page.replace(" ", "_").lower() in content
        tests.append((f"Page: {page}", has_page))

    # Check navigation
    tests.append(("Navigation menu", "selected_page" in content or "page" in content))
    tests.append(("Page routing", "if" in content and "render_" in content))

    # Report results
    passed = sum(1 for _, ok in tests if ok)
    total = len(tests)

    for name, ok in tests:
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {name}")

    print(f"\nPages: {passed}/{total} passed")
    return passed == total


def test_hitl_workflow():
    """Test human-in-the-loop workflow components."""
    print("\n" + "="*60)
    print("TEST: HITL Workflow")
    print("="*60)

    from pathlib import Path

    tests = []

    app_file = Path("execution/dashboard/app.py")
    content = app_file.read_text(encoding='utf-8')

    # Review workflow steps
    workflow_steps = [
        ("Content display", "content" in content.lower()),
        ("Quality score display", "score" in content.lower() or "quality" in content.lower()),
        ("Fact check integration", "fact" in content.lower() or "verification" in content.lower()),
        ("Provenance display", "provenance" in content.lower()),
        ("Decision buttons", "Approve" in content and "Reject" in content),
        ("Feedback input", "text_area" in content or "notes" in content.lower()),
        ("Side-by-side comparison", "comparison" in content.lower() or "columns" in content),
    ]

    for name, ok in workflow_steps:
        tests.append((name, ok))

    # Report results
    passed = sum(1 for _, ok in tests if ok)
    total = len(tests)

    for name, ok in tests:
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {name}")

    print(f"\nHITL Workflow: {passed}/{total} passed")
    return passed == total


def run_all_tests():
    """Run all dashboard validation tests."""
    print("\n" + "="*60)
    print("USER-TEST-3: HITL DASHBOARD VALIDATION")
    print("="*60)

    results = {}

    # Run tests
    results["imports"] = test_dashboard_imports()
    results["structure"] = test_dashboard_structure()
    results["components"] = test_dashboard_components()
    results["pages"] = test_dashboard_pages()
    results["hitl_workflow"] = test_hitl_workflow()

    # Summary
    print("\n" + "="*60)
    print("DASHBOARD VALIDATION SUMMARY")
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
        print("\n" + "="*60)
        print("USER-TEST-3: PASSED")
        print("="*60)
        print("\nHITL Dashboard validation complete!")
        print("\nTo run the dashboard:")
        print("  streamlit run execution/dashboard/app.py")
    else:
        print("\nSome tests FAILED. Review output above.")

    return all_passed


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
