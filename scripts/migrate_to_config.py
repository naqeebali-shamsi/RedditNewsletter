#!/usr/bin/env python3
"""
Migration Script for GhostWriter Configuration.

Scans the codebase for hardcoded paths and other configuration issues,
then generates a report and optionally applies fixes.

Usage:
    python scripts/migrate_to_config.py --scan      # Scan only, report issues
    python scripts/migrate_to_config.py --fix       # Apply automatic fixes
    python scripts/migrate_to_config.py --verify    # Verify migration complete
"""

import os
import re
import argparse
from pathlib import Path
from typing import List, Dict, Tuple


# Project root
PROJECT_ROOT = Path(__file__).parent.parent

# Patterns to detect
HARDCODED_PATH_PATTERNS = [
    r'Path\(["\']n:/RedditNews',
    r'Path\(["\']N:/RedditNews',
    r'Path\(["\']N:\\\\RedditNews',
    r'["\']n:/RedditNews',
    r'["\']N:/RedditNews',
    r'["\']/home/.*/RedditNews',  # Linux paths
]

# Files to scan
SCAN_EXTENSIONS = ['.py', '.yaml', '.yml', '.json', '.toml']

# Files to exclude
EXCLUDE_DIRS = [
    '.git', '__pycache__', '.venv', 'venv', 'node_modules',
    '.taskmaster/tasks', 'docs'  # Exclude generated files
]

EXCLUDE_FILES = [
    'migrate_to_config.py',  # This file
    'generate_tasks.py',     # Task generation script
]


def scan_file(filepath: Path) -> List[Dict]:
    """Scan a single file for hardcoded paths."""
    issues = []

    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
    except Exception as e:
        return [{"file": str(filepath), "error": str(e)}]

    for i, line in enumerate(lines, 1):
        for pattern in HARDCODED_PATH_PATTERNS:
            matches = re.findall(pattern, line, re.IGNORECASE)
            if matches:
                issues.append({
                    "file": str(filepath.relative_to(PROJECT_ROOT)),
                    "line": i,
                    "content": line.strip()[:100],
                    "pattern": pattern,
                    "match": matches[0]
                })

    return issues


def scan_codebase() -> List[Dict]:
    """Scan entire codebase for hardcoded paths."""
    all_issues = []

    for root, dirs, files in os.walk(PROJECT_ROOT):
        # Exclude directories
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]

        for filename in files:
            if filename in EXCLUDE_FILES:
                continue

            filepath = Path(root) / filename
            if filepath.suffix in SCAN_EXTENSIONS:
                issues = scan_file(filepath)
                all_issues.extend(issues)

    return all_issues


def check_config_usage() -> Dict:
    """Check if config module is properly used across codebase."""
    results = {
        "files_using_config": [],
        "files_not_using_config": [],
        "total_python_files": 0
    }

    for root, dirs, files in os.walk(PROJECT_ROOT):
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]

        for filename in files:
            if filename.endswith('.py') and filename not in EXCLUDE_FILES:
                filepath = Path(root) / filename
                results["total_python_files"] += 1

                try:
                    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()

                    if 'from execution.config import' in content or \
                       'from execution import config' in content:
                        results["files_using_config"].append(
                            str(filepath.relative_to(PROJECT_ROOT))
                        )
                    elif 'Path(' in content:
                        # File uses paths but not config
                        results["files_not_using_config"].append(
                            str(filepath.relative_to(PROJECT_ROOT))
                        )
                except Exception:
                    pass

    return results


def apply_fixes(issues: List[Dict]) -> Tuple[int, int]:
    """Apply automatic fixes for common patterns."""
    fixed = 0
    failed = 0

    # Group issues by file
    files_to_fix = {}
    for issue in issues:
        filepath = PROJECT_ROOT / issue["file"]
        if filepath not in files_to_fix:
            files_to_fix[filepath] = []
        files_to_fix[filepath].append(issue)

    for filepath, file_issues in files_to_fix.items():
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()

            original = content

            # Apply replacements
            content = re.sub(
                r'Path\(["\']n:/RedditNews/drafts["\']\)',
                'OUTPUT_DIR',
                content,
                flags=re.IGNORECASE
            )
            content = re.sub(
                r'Path\(["\']n:/RedditNews/\.env["\']\)',
                'PROJECT_ROOT / ".env"',
                content,
                flags=re.IGNORECASE
            )
            content = re.sub(
                r'Path\(["\']n:/RedditNews/\.tmp["\']\)',
                'TEMP_DIR',
                content,
                flags=re.IGNORECASE
            )

            # Add import if needed
            if content != original and 'from execution.config import' not in content:
                # Add import after other imports
                import_line = '\nfrom execution.config import config, OUTPUT_DIR, PROJECT_ROOT, TEMP_DIR\n'
                if 'import ' in content:
                    lines = content.split('\n')
                    last_import = 0
                    for i, line in enumerate(lines):
                        if line.startswith('import ') or line.startswith('from '):
                            last_import = i
                    lines.insert(last_import + 1, import_line.strip())
                    content = '\n'.join(lines)

            if content != original:
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(content)
                fixed += 1
                print(f"  Fixed: {filepath.relative_to(PROJECT_ROOT)}")
            else:
                failed += 1

        except Exception as e:
            failed += 1
            print(f"  Failed: {filepath} - {e}")

    return fixed, failed


def verify_migration() -> bool:
    """Verify migration is complete."""
    issues = scan_codebase()

    if issues:
        print("\n" + "=" * 60)
        print("VERIFICATION FAILED")
        print("=" * 60)
        print(f"\nFound {len(issues)} hardcoded path(s):")
        for issue in issues[:10]:
            print(f"  - {issue['file']}:{issue['line']}")
            print(f"    {issue['content'][:60]}...")
        if len(issues) > 10:
            print(f"  ... and {len(issues) - 10} more")
        return False

    print("\n" + "=" * 60)
    print("VERIFICATION PASSED")
    print("=" * 60)
    print("\nNo hardcoded paths found in codebase.")

    # Show config usage
    usage = check_config_usage()
    print(f"\nConfig module usage:")
    print(f"  Files using config: {len(usage['files_using_config'])}")
    print(f"  Files not using config (but use paths): {len(usage['files_not_using_config'])}")

    return True


def main():
    parser = argparse.ArgumentParser(description="GhostWriter Configuration Migration Tool")
    parser.add_argument("--scan", action="store_true", help="Scan codebase for issues")
    parser.add_argument("--fix", action="store_true", help="Apply automatic fixes")
    parser.add_argument("--verify", action="store_true", help="Verify migration complete")

    args = parser.parse_args()

    if not any([args.scan, args.fix, args.verify]):
        args.scan = True  # Default to scan

    if args.scan:
        print("\n" + "=" * 60)
        print("SCANNING CODEBASE FOR HARDCODED PATHS")
        print("=" * 60)

        issues = scan_codebase()

        if issues:
            print(f"\nFound {len(issues)} potential issue(s):\n")
            for issue in issues:
                if "error" in issue:
                    print(f"  ERROR: {issue['file']} - {issue['error']}")
                else:
                    print(f"  {issue['file']}:{issue['line']}")
                    print(f"    Pattern: {issue['pattern']}")
                    print(f"    Content: {issue['content']}")
                    print()
        else:
            print("\nNo hardcoded paths found!")

        # Show config usage
        print("\n" + "-" * 40)
        usage = check_config_usage()
        print(f"Config module adoption:")
        print(f"  Total Python files: {usage['total_python_files']}")
        print(f"  Using config module: {len(usage['files_using_config'])}")
        print(f"  Not using config (but use paths): {len(usage['files_not_using_config'])}")

    if args.fix:
        print("\n" + "=" * 60)
        print("APPLYING AUTOMATIC FIXES")
        print("=" * 60)

        issues = scan_codebase()
        if issues:
            fixed, failed = apply_fixes(issues)
            print(f"\nResults: {fixed} fixed, {failed} failed")
        else:
            print("\nNo issues to fix!")

    if args.verify:
        success = verify_migration()
        exit(0 if success else 1)


if __name__ == "__main__":
    main()
