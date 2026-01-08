#!/usr/bin/env python3
"""
Quality Gate - Adversarial Review Loop.

Implements the REVIEW <-> FIX loop that ensures all content passes
the adversarial expert panel before being approved for publication.

Flow:
1. Draft enters quality gate
2. Adversarial panel reviews and scores
3. If score < 7.0: Generate fix instructions, send to writer for revision
4. Loop until pass OR max iterations reached
5. Output: Approved draft OR escalation for human review

Usage:
    python quality_gate.py --input draft.md --platform medium
    python quality_gate.py --input draft.md --platform linkedin --max-iterations 5
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Optional, List, Tuple

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from execution.agents.adversarial_panel import AdversarialPanelAgent, PanelVerdict
from execution.agents.writer import WriterAgent
from execution.agents.editor import EditorAgent


@dataclass
class QualityGateResult:
    """Result of the quality gate process."""
    passed: bool
    final_score: float
    iterations_used: int
    max_iterations: int
    final_content: str
    revision_history: List[dict]  # Score progression
    escalated: bool  # True if max iterations hit without passing
    timestamp: str


class QualityGate:
    """
    Orchestrates the adversarial review loop.

    Coordinates between:
    - AdversarialPanelAgent (review)
    - WriterAgent (revision)
    - EditorAgent (final polish)
    """

    def __init__(self, max_iterations: int = 3, verbose: bool = True):
        self.max_iterations = max_iterations
        self.verbose = verbose

        # Initialize agents
        self.panel = AdversarialPanelAgent()
        self.writer = WriterAgent()
        self.editor = EditorAgent()

    def _log(self, message: str):
        """Print if verbose mode."""
        if self.verbose:
            print(message)

    def process(
        self,
        content: str,
        platform: str = "medium",
        source_context: Optional[str] = None,
        source_type: str = "external"
    ) -> QualityGateResult:
        """
        Process content through the quality gate.

        Args:
            content: Draft content to review
            platform: 'linkedin' or 'medium'
            source_context: Optional context about the source material
            source_type: 'external' (observer voice) or 'internal' (owner voice)

        Returns:
            QualityGateResult with final content and metadata
        """

        revision_history = []
        current_content = content
        iteration = 0

        voice_label = "Observer (external)" if source_type == "external" else "Practitioner (internal)"

        self._log("\n" + "=" * 60)
        self._log("QUALITY GATE INITIATED")
        self._log("=" * 60)
        self._log(f"Platform: {platform}")
        self._log(f"Voice: {voice_label}")
        self._log(f"Max Iterations: {self.max_iterations}")
        self._log(f"Content Length: {len(content)} chars")
        self._log("=" * 60 + "\n")

        while iteration < self.max_iterations:
            iteration += 1

            self._log(f"\n{'─' * 40}")
            self._log(f"ITERATION {iteration}/{self.max_iterations}")
            self._log(f"{'─' * 40}")

            # Step 1: Review with adversarial panel (voice-aware)
            self._log("\n🔍 Running adversarial panel review...")
            verdict = self.panel.review_content(
                content=current_content,
                platform=platform,
                iteration=iteration,
                source_type=source_type
            )

            # Log verdict summary
            self._log(self.panel.format_verdict_summary(verdict))

            # Record in history
            revision_history.append({
                "iteration": iteration,
                "score": verdict.average_score,
                "passed": verdict.passed,
                "critical_failures": verdict.critical_failures,
                "priority_fixes": verdict.priority_fixes
            })

            # Step 2: Check if passed
            if verdict.passed:
                self._log("\n✅ QUALITY GATE PASSED!")
                self._log(f"   Final Score: {verdict.average_score}/10")

                # Final polish by editor
                self._log("\n✨ Running final editor polish...")
                final_content = self._final_polish(current_content, platform)

                return QualityGateResult(
                    passed=True,
                    final_score=verdict.average_score,
                    iterations_used=iteration,
                    max_iterations=self.max_iterations,
                    final_content=final_content,
                    revision_history=revision_history,
                    escalated=False,
                    timestamp=datetime.now().isoformat()
                )

            # Step 3: Generate fix instructions and revise
            if iteration < self.max_iterations:
                self._log(f"\n⚠️ Score {verdict.average_score}/10 - Below threshold (7.0)")
                self._log("📝 Generating revision instructions...")

                fix_instructions = self.panel.generate_fix_instructions(verdict)

                self._log("✍️ Writer revising content...")
                current_content = self._revise_content(
                    current_content,
                    fix_instructions,
                    platform,
                    source_type
                )

                self._log(f"   Revised content: {len(current_content)} chars")
            else:
                self._log(f"\n❌ Max iterations reached without passing")

        # Escalation - max iterations hit
        self._log("\n" + "=" * 60)
        self._log("⚠️ ESCALATION: Quality gate could not pass content")
        self._log(f"   Best Score Achieved: {max(h['score'] for h in revision_history)}/10")
        self._log("   Content requires human review")
        self._log("=" * 60)

        return QualityGateResult(
            passed=False,
            final_score=revision_history[-1]["score"] if revision_history else 0,
            iterations_used=self.max_iterations,
            max_iterations=self.max_iterations,
            final_content=current_content,
            revision_history=revision_history,
            escalated=True,
            timestamp=datetime.now().isoformat()
        )

    def _revise_content(
        self,
        content: str,
        fix_instructions: str,
        platform: str,
        source_type: str = "external"
    ) -> str:
        """Have the writer agent revise content based on fix instructions."""

        # Voice context for revision
        if source_type == "external":
            voice_instruction = """
CRITICAL VOICE REQUIREMENT (External Source):
- You are an OBSERVER reporting on others' work
- FORBIDDEN: "I built", "we created", "our team", "my approach"
- USE: "teams found", "engineers discovered", "this approach"
- "I" ONLY for observations: "I noticed", "I've observed"
"""
        else:
            voice_instruction = """
VOICE (Internal Source):
- This is your own work, use ownership voice naturally
- "I", "we", "our" are appropriate
"""

        revision_prompt = f"""REVISION TASK

You are revising a {platform} draft that FAILED the quality gate.
{voice_instruction}

CURRENT DRAFT:
---
{content}
---

{fix_instructions}

IMPORTANT:
- Output ONLY the revised content
- No explanations, no meta-commentary
- Preserve the core message while fixing the issues
- Make the content feel human-written, not AI-generated
- STRICTLY follow the voice requirements above
"""

        revised = self.writer.call_llm(revision_prompt, temperature=0.7)

        # Clean up any meta-commentary that slipped through
        if revised.startswith("Here") or revised.startswith("I've"):
            lines = revised.split('\n')
            # Skip first line if it's meta
            if len(lines) > 1:
                revised = '\n'.join(lines[1:])

        return revised.strip()

    def _final_polish(self, content: str, platform: str) -> str:
        """Final editor polish after passing quality gate."""

        polish_prompt = f"""FINAL POLISH

This {platform} content has passed the quality gate. Your job is final polish:

1. Fix any remaining typos or grammatical issues
2. Ensure smooth transitions between sections
3. Verify the opening hook is strong
4. Confirm the CTA is specific and compelling
5. Remove any redundant phrases

CONTENT:
---
{content}
---

Output ONLY the polished content. No explanations."""

        polished = self.editor.call_llm(polish_prompt, temperature=0.3)

        # Clean up
        if polished.startswith("Here") or polished.startswith("I've"):
            lines = polished.split('\n')
            if len(lines) > 1:
                polished = '\n'.join(lines[1:])

        return polished.strip()


def main():
    parser = argparse.ArgumentParser(description="Quality Gate - Adversarial Review Loop")
    parser.add_argument("--input", "-i", required=True, help="Input draft file path")
    parser.add_argument("--output", "-o", help="Output file path (default: input_approved.md)")
    parser.add_argument("--platform", "-p", choices=["linkedin", "medium"], default="medium",
                       help="Target platform")
    parser.add_argument("--max-iterations", "-m", type=int, default=3,
                       help="Maximum revision iterations (default: 3)")
    parser.add_argument("--quiet", "-q", action="store_true", help="Suppress verbose output")
    parser.add_argument("--json", "-j", action="store_true", help="Output result as JSON")

    args = parser.parse_args()

    # Read input file
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)

    with open(input_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Run quality gate
    gate = QualityGate(
        max_iterations=args.max_iterations,
        verbose=not args.quiet
    )

    result = gate.process(
        content=content,
        platform=args.platform
    )

    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        suffix = "_approved" if result.passed else "_escalated"
        output_path = input_path.parent / f"{input_path.stem}{suffix}{input_path.suffix}"

    # Write output
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(result.final_content)

    # Output result summary
    if args.json:
        result_dict = asdict(result)
        result_dict['output_path'] = str(output_path)
        print(json.dumps(result_dict, indent=2))
    else:
        print(f"\n{'=' * 60}")
        print(f"QUALITY GATE RESULT")
        print(f"{'=' * 60}")
        print(f"Status: {'PASSED' if result.passed else 'ESCALATED'}")
        print(f"Final Score: {result.final_score}/10")
        print(f"Iterations: {result.iterations_used}/{result.max_iterations}")
        print(f"Output: {output_path}")

        if result.revision_history:
            print(f"\nScore Progression:")
            for h in result.revision_history:
                status = "✓" if h['passed'] else "✗"
                print(f"  Iteration {h['iteration']}: {h['score']}/10 {status}")

        print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
