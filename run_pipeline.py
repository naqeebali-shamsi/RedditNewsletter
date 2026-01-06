#!/usr/bin/env python3
"""
Master orchestration script for the Reddit Newsletter Pipeline.

This script runs the complete workflow:
1. Fetch posts from Reddit RSS feeds
2. Evaluate posts for Signal vs Noise
3. Generate content drafts

Usage:
    python run_pipeline.py              # Run with defaults (S+ tier, 50 evaluations, 10 drafts)
    python run_pipeline.py --all        # Fetch from all tiers
    python run_pipeline.py --quick      # Quick mode (20 evaluations, 5 drafts)
"""

import subprocess
import sys
import argparse
from pathlib import Path

# Script paths
EXECUTION_DIR = Path(__file__).parent / "execution"
FETCH_SCRIPT = EXECUTION_DIR / "fetch_reddit.py"
EVALUATE_SCRIPT = EXECUTION_DIR / "evaluate_posts.py"
GENERATE_SCRIPT = EXECUTION_DIR / "generate_drafts.py"


def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n{'='*70}")
    print(f"🔄 {description}")
    print(f"{'='*70}\n")
    
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=False,
            text=True
        )
        print(f"\n✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ {description} failed with error code {e.returncode}")
        return False
    except Exception as e:
        print(f"\n✗ {description} failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Run the complete Reddit Newsletter pipeline'
    )
    parser.add_argument('--all', action='store_true',
                       help='Fetch from all S+ and S tier subreddits (default: S+ only)')
    parser.add_argument('--quick', action='store_true',
                       help='Quick mode: fewer posts and drafts')
    parser.add_argument('--skip-fetch', action='store_true',
                       help='Skip fetching, only evaluate and generate')
    parser.add_argument('--skip-generate', action='store_true',
                       help='Fetch and evaluate only, skip draft generation')
    
    args = parser.parse_args()
    
    # Determine parameters
    if args.quick:
        eval_limit = 20
        draft_limit = 5
    else:
        eval_limit = 50
        draft_limit = 10
    
    print(f"""
╔════════════════════════════════════════════════════════════════════╗
║                   Reddit Newsletter Pipeline                       ║
║                     Autonomous Content Generation                  ║
╚════════════════════════════════════════════════════════════════════╝

Mode: {'Quick' if args.quick else 'Standard'}
Tiers: {'All (S+ and S)' if args.all else 'S+ only'}
Eval Limit: {eval_limit}
Draft Limit: {draft_limit}
    """)
    
    success_count = 0
    total_steps = 3
    
    # Step 1: Fetch
    if not args.skip_fetch:
        fetch_cmd = [sys.executable, str(FETCH_SCRIPT)]
        if args.all:
            fetch_cmd.append('--all')
        
        if run_command(fetch_cmd, "Step 1/3: Fetching Reddit Posts"):
            success_count += 1
        else:
            print("\n⚠️  Fetch failed, but continuing with existing data...")
    else:
        print("\n⏭️  Skipping fetch step (using existing data)")
        total_steps -= 1
    
    # Step 2: Evaluate
    evaluate_cmd = [sys.executable, str(EVALUATE_SCRIPT), '--limit', str(eval_limit)]
    
    if run_command(evaluate_cmd, "Step 2/3: Evaluating Posts (Signal vs Noise)"):
        success_count += 1
    else:
        print("\n❌ Evaluation failed. Cannot continue to draft generation.")
        sys.exit(1)
    
    # Step 3: Generate
    if not args.skip_generate:
        generate_cmd = [
            sys.executable, str(GENERATE_SCRIPT),
            '--platform', 'both',
            '--limit', str(draft_limit)
        ]
        
        if run_command(generate_cmd, "Step 3/3: Generating Content Drafts"):
            success_count += 1
    else:
        print("\n⏭️  Skipping draft generation")
        total_steps -= 1
    
    # Summary
    print(f"\n{'='*70}")
    print(f"📊 Pipeline Summary")
    print(f"{'='*70}")
    print(f"Completed: {success_count}/{total_steps} steps")
    
    if success_count == total_steps:
        print("\n✅ Pipeline completed successfully!")
        print("\n📁 Next steps:")
        print("   1. Check .tmp/drafts/ for generated content")
        print("   2. Review and edit drafts as needed")
        print("   3. Publish to LinkedIn and Medium")
    else:
        print("\n⚠️  Pipeline completed with some failures.")
    
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
