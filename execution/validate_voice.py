#!/usr/bin/env python3
"""
Voice Compliance Validator for Generated Content.

Scans generated articles for forbidden pronouns based on source type.
Use after generation, before publishing.

Usage:
    python validate_voice.py --file drafts/article.md --source-type external
    python validate_voice.py --file drafts/article.md --source-type internal  # Always passes
"""

import re
import argparse
import sys
from pathlib import Path
from typing import List, Tuple, Dict

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from execution.prompts.voice_templates import FORBIDDEN_PATTERNS_EXTERNAL, REPLACEMENTS


def validate_external_voice(content: str) -> List[Dict]:
    """
    Validate content against external voice rules.

    Args:
        content: The article content to validate

    Returns:
        List of violation dictionaries with line_number, match, pattern_name, suggestion
    """
    violations = []
    lines = content.split('\n')

    for line_num, line in enumerate(lines, 1):
        # Skip metadata lines (starting with * or #)
        if line.strip().startswith('*') and ('Voice:' in line or 'Signal:' in line):
            continue
        if line.strip().startswith('**Generated'):
            continue

        for pattern, pattern_name in FORBIDDEN_PATTERNS_EXTERNAL:
            matches = re.finditer(pattern, line)
            for match in matches:
                # Get suggestion if available
                suggestion = REPLACEMENTS.get(pattern_name, f"Rephrase to remove '{pattern_name}'")

                violations.append({
                    'line_number': line_num,
                    'line_content': line.strip(),
                    'match': match.group(),
                    'pattern_name': pattern_name,
                    'suggestion': suggestion,
                    'position': match.start()
                })

    return violations


def format_violation_report(violations: List[Dict], filepath: str) -> str:
    """Format violations into a readable report."""
    if not violations:
        return f"[PASSED] No voice violations found in {filepath}"

    report = []
    report.append(f"\n[WARNING] VOICE VIOLATIONS FOUND: {len(violations)} issues in {filepath}\n")
    report.append("=" * 70)

    # Group by line number
    by_line = {}
    for v in violations:
        ln = v['line_number']
        if ln not in by_line:
            by_line[ln] = []
        by_line[ln].append(v)

    for line_num in sorted(by_line.keys()):
        line_violations = by_line[line_num]
        report.append(f"\nLine {line_num}:")
        report.append(f"  | {line_violations[0]['line_content'][:80]}...")

        for v in line_violations:
            report.append(f"  +-- Found: \"{v['match']}\" (forbidden: {v['pattern_name']})")
            report.append(f"  +-- Suggestion: {v['suggestion']}")

    report.append("\n" + "=" * 70)
    report.append(f"\nTotal: {len(violations)} violations across {len(by_line)} lines")
    report.append("\nRun with --fix to see suggested corrections (manual review required)")

    return "\n".join(report)


def suggest_fixes(content: str, violations: List[Dict]) -> str:
    """
    Generate content with suggested fixes highlighted.

    Note: This provides suggestions but doesn't auto-fix, as context matters.
    """
    if not violations:
        return content

    lines = content.split('\n')
    suggestions = []

    suggestions.append("=" * 70)
    suggestions.append("SUGGESTED FIXES (Review before applying)")
    suggestions.append("=" * 70 + "\n")

    # Group by line
    by_line = {}
    for v in violations:
        ln = v['line_number']
        if ln not in by_line:
            by_line[ln] = []
        by_line[ln].append(v)

    for line_num in sorted(by_line.keys()):
        original = lines[line_num - 1]
        suggestions.append(f"Line {line_num}:")
        suggestions.append(f"  ORIGINAL:  {original}")

        # Apply simple replacements (for review)
        fixed = original
        for v in by_line[line_num]:
            # Simple word replacement (context may need adjustment)
            if v['pattern_name'] in REPLACEMENTS:
                replacement = REPLACEMENTS[v['pattern_name']].split(' / ')[0]  # Take first suggestion
                fixed = re.sub(rf'\b{re.escape(v["match"])}\b', replacement, fixed, count=1)

        suggestions.append(f"  SUGGESTED: {fixed}")
        suggestions.append("")

    return "\n".join(suggestions)


def main():
    parser = argparse.ArgumentParser(
        description='Validate voice compliance in generated articles'
    )
    parser.add_argument('--file', '-f', required=True,
                       help='Path to the markdown file to validate')
    parser.add_argument('--source-type', '-s', choices=['external', 'internal'],
                       default='external',
                       help='Source type for validation rules (default: external)')
    parser.add_argument('--fix', action='store_true',
                       help='Show suggested fixes for violations')
    parser.add_argument('--strict', action='store_true',
                       help='Exit with error code if violations found')
    parser.add_argument('--json', action='store_true',
                       help='Output violations as JSON')
    parser.add_argument('--score', action='store_true',
                       help='Run style enforcement scoring (5-dimension breakdown)')

    args = parser.parse_args()

    # Read file
    filepath = Path(args.file)
    if not filepath.exists():
        print(f"Error: File not found: {filepath}")
        sys.exit(1)

    content = filepath.read_text(encoding='utf-8')

    # Internal sources always pass (ownership voice allowed)
    if args.source_type == 'internal':
        print(f"[PASSED] Internal source - ownership voice allowed")
        print(f"   File: {filepath}")
        sys.exit(0)

    # Validate external voice
    violations = validate_external_voice(content)

    # Output format
    if args.json:
        import json
        result = {
            'file': str(filepath),
            'source_type': args.source_type,
            'passed': len(violations) == 0,
            'violation_count': len(violations),
            'violations': violations
        }
        print(json.dumps(result, indent=2))
    else:
        # Human-readable report
        report = format_violation_report(violations, str(filepath))
        print(report)

        if args.fix and violations:
            print("\n")
            print(suggest_fixes(content, violations))

    # Style scoring (new)
    if hasattr(args, 'score') and args.score:
        try:
            from execution.agents.style_enforcer import StyleEnforcerAgent
            enforcer = StyleEnforcerAgent()
            result = enforcer.score(content, content_type='article')
            if args.json:
                import json as json_mod
                print(json_mod.dumps(result.to_dict(), indent=2))
            else:
                print(enforcer.format_report(result))
        except ImportError:
            print("Style scoring requires: pip install lexicalrichness nltk")
        except Exception as e:
            print(f"Style scoring error: {e}")

    # Exit code
    if args.strict and violations:
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
