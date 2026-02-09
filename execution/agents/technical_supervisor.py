"""
Technical Quality Supervisor Agent

This agent acts as a technical BS detector, catching:
1. Fabricated statistics without citations
2. Code that doesn't parse/run
3. Causally impossible claims
4. Domain mismatches (solution doesn't fit problem)
5. Surface-level insights dressed as depth
6. Phantom evidence ("studies show" without source)

Works BEFORE style/voice refinement to reject bad content early.
"""

from .base_agent import BaseAgent
from execution.config import config
import re
import ast
from typing import List, Dict, Tuple, Optional


class TechnicalSupervisorAgent(BaseAgent):
    """
    Technical BS Detector - catches fabricated stats, broken code, and shallow content.

    Philosophy: Good technical content must be:
    1. Factually verifiable (no made-up stats)
    2. Technically accurate (code must work, claims must be causal)
    3. Domain-appropriate (MLOps problems need MLOps solutions)
    4. Beyond obvious (not just "best practices 101")
    """

    # Patterns that indicate fabricated statistics
    STAT_PATTERNS = [
        r'\b(\d{1,3})%\s+(increase|decrease|reduction|improvement|faster|slower|more|less|better|worse)',
        r'\b(\d{1,3})x\s+(faster|slower|more|better|improvement)',
        r'\b(doubled|tripled|halved)\s+\w+',
        r'(\d+)%\s+of\s+(teams|engineers|companies|organizations)',
        r'(significant|substantial|dramatic)\s+(increase|decrease|improvement|reduction)',
    ]

    # Phrases that indicate phantom evidence
    PHANTOM_EVIDENCE = [
        "studies show",
        "research indicates",
        "according to a study",
        "experts agree",
        "it's been proven",
        "data shows",
        "statistics show",
        "a notable case study",
        "one company found",
        "teams have found",
        "engineers have discovered",
        "engineers have found",
        "engineers have observed",
        "according to .* documentation",  # Fake citations
        "has been successfully implemented in several",
    ]

    # MADE-UP METRICS - These terms don't exist in ML
    FAKE_METRICS = [
        (r'parameters?\s+per\s+second', "parameters per second",
         "'Parameters per second' is not a real ML metric. Use FLOPS, tokens/sec, or samples/sec."),
        (r'weights?\s+per\s+second', "weights per second",
         "'Weights per second' is not a real metric. Weights are static."),
        (r'neurons?\s+per\s+second', "neurons per second",
         "'Neurons per second' is not a real metric."),
    ]

    # SUSPICIOUS LARGE NUMBERS - likely fabricated specifics
    SUSPICIOUS_SPECS = [
        (r'\b(\d+)\s*trillion\s*(parameters?|tokens?|ops?)', "trillion-scale claim",
         "Trillion-scale claims need verification. Only a few models (GPT-4, etc.) are this scale."),
        (r'\b(\d+)\s*TB\s*(of\s+)?(storage|memory|RAM|VRAM)', "TB storage claim",
         "Multi-TB storage claims are suspicious. 70B model = ~140GB in FP16."),
        (r'\$\s*(\d{2,3}),?000', "large dollar amount",
         "Specific cost claims need real quotes/sources."),
    ]

    # HARDWARE that needs verification (easy to get wrong)
    HARDWARE_CLAIMS = [
        r'(MI\d+|A100|H100|V100|RTX\s*\d+|TPU)',  # GPU/accelerator names
        r'(TFLOPS|PFLOPS|teraflops|petaflops)',    # Performance units
        r'(NVLink|PCIe|InfiniBand)',               # Interconnects
    ]

    # Domain-specific required concepts (if topic matches, these should appear)
    DOMAIN_REQUIREMENTS = {
        "mlops": [
            "training-serving skew", "feature store", "model registry",
            "data drift", "experiment tracking", "model versioning",
            "canary", "rollback", "shadow deployment", "A/B test"
        ],
        "ml_deployment": [
            "inference latency", "batch vs real-time", "model serving",
            "GPU scheduling", "cold start", "autoscaling", "monitoring"
        ],
        "llm": [
            "prompt engineering", "context window", "token", "fine-tuning",
            "RAG", "embedding", "hallucination", "guardrails"
        ],
        "architecture": [
            "trade-off", "latency", "throughput", "consistency",
            "availability", "partition tolerance", "scaling"
        ],
    }

    # Causally impossible claims (X cannot cause Y)
    CAUSAL_NONSENSE_PATTERNS = [
        # Code organization cannot affect model accuracy
        (r"(clean architecture|dependency injection|code structure|refactoring).*?(accuracy|precision|recall|F1|AUC)",
         "Code organization cannot affect model accuracy metrics"),
        # Architecture patterns cannot affect training convergence
        (r"(design pattern|architecture|layering).*?(convergence|loss|gradient)",
         "Software architecture cannot affect training dynamics"),
    ]

    def __init__(self):
        super().__init__(
            role="Technical Quality Supervisor",
            persona="""You are a ruthless technical fact-checker with deep expertise in:
- Machine Learning systems and MLOps
- Software architecture (and its actual limitations)
- Statistical reasoning and causal inference
- Production engineering

Your job is NOT to rewrite content. Your job is to IDENTIFY SPECIFIC FAILURES:

1. FABRICATED_STATS: Any percentage or number without a verifiable source
   - "40% improvement" with no citation = REJECT
   - "In our testing, latency dropped from 200ms to 50ms" = OK (specific, owned claim)

2. INVALID_CODE: Code that would not run
   - Syntax errors, undefined variables, wrong API usage = REJECT
   - Must be copy-paste runnable or clearly pseudocode

3. CAUSAL_NONSENSE: Claims where X cannot possibly cause Y
   - "Clean Architecture improved model accuracy" = REJECT (impossible)
   - "Clean Architecture improved deployment speed" = OK (plausible)

4. DOMAIN_MISMATCH: Solution doesn't address actual problem space
   - MLOps article that never mentions data drift, feature stores = REJECT
   - Must address REAL practitioner concerns, not tangential topics

5. SHALLOW_INSIGHT: Table-stakes advice dressed as revelation
   - "Separate concerns for testability" = everyone knows this
   - Must provide insight beyond obvious best practices

6. PHANTOM_EVIDENCE: Vague appeals to authority
   - "Studies show..." without citation = REJECT
   - "According to Google's 2023 MLOps report..." = OK (verifiable)

Return a structured assessment with SPECIFIC line-by-line failures.""",
            model=config.models.DEFAULT_CRITIC_MODEL
        )

    def _extract_code_blocks(self, text: str) -> List[Tuple[str, str]]:
        """Extract code blocks with their language identifier."""
        pattern = r'```(\w+)?\n(.*?)```'
        matches = re.findall(pattern, text, re.DOTALL)
        return [(lang or 'unknown', code.strip()) for lang, code in matches]

    def _validate_python_syntax(self, code: str) -> Tuple[bool, Optional[str]]:
        """Check if Python code is syntactically valid.

        Uses ast.parse for runtime syntax checking of code blocks in generated
        articles. This is intentional — pyflakes/pylint are dev-time linters
        and should not be added as a runtime dependency.
        """
        try:
            ast.parse(code)
            return True, None
        except SyntaxError as e:
            return False, f"Line {e.lineno}: {e.msg}"

    def _find_uncited_stats(self, text: str) -> List[Dict]:
        """Find statistics that lack citations."""
        violations = []
        lines = text.split('\n')

        for i, line in enumerate(lines, 1):
            for pattern in self.STAT_PATTERNS:
                if re.search(pattern, line, re.IGNORECASE):
                    # Check if line has citation markers
                    has_citation = any(marker in line.lower() for marker in [
                        'according to', 'source:', 'cited', 'reported by',
                        'found that', 'published', '[', '(20'  # year citations
                    ])
                    if not has_citation:
                        match = re.search(pattern, line, re.IGNORECASE)
                        violations.append({
                            "line": i,
                            "type": "FABRICATED_STATS",
                            "text": line.strip()[:100],
                            "match": match.group(0) if match else "",
                            "fix": "Add citation source or remove statistic"
                        })
        return violations

    def _find_phantom_evidence(self, text: str) -> List[Dict]:
        """Find vague appeals to authority without sources."""
        violations = []
        lines = text.split('\n')

        for i, line in enumerate(lines, 1):
            line_lower = line.lower()
            for phrase in self.PHANTOM_EVIDENCE:
                if phrase in line_lower:
                    # Check if followed by actual citation
                    has_source = any(marker in line_lower for marker in [
                        'university', 'journal', 'conference', 'report',
                        '202', '201', 'published', 'arxiv', 'paper'
                    ])
                    if not has_source:
                        violations.append({
                            "line": i,
                            "type": "PHANTOM_EVIDENCE",
                            "text": line.strip()[:100],
                            "phrase": phrase,
                            "fix": "Add specific source or rephrase as opinion"
                        })
        return violations

    def _check_causal_validity(self, text: str) -> List[Dict]:
        """Check for causally impossible claims."""
        violations = []
        text_lower = text.lower()

        for pattern, reason in self.CAUSAL_NONSENSE_PATTERNS:
            if re.search(pattern, text_lower):
                match = re.search(pattern, text_lower)
                violations.append({
                    "type": "CAUSAL_NONSENSE",
                    "match": match.group(0) if match else "",
                    "reason": reason,
                    "fix": "Remove or correct the causal claim"
                })
        return violations

    def _check_fake_metrics(self, text: str) -> List[Dict]:
        """Check for made-up ML metrics that don't exist."""
        violations = []
        text_lower = text.lower()

        for pattern, name, explanation in self.FAKE_METRICS:
            if re.search(pattern, text_lower, re.IGNORECASE):
                match = re.search(pattern, text_lower, re.IGNORECASE)
                violations.append({
                    "type": "FAKE_METRIC",
                    "match": match.group(0) if match else name,
                    "reason": explanation,
                    "fix": "Remove this fake metric or replace with real ML metrics (FLOPS, tokens/sec, etc.)"
                })
        return violations

    def _check_suspicious_specs(self, text: str) -> List[Dict]:
        """Check for suspiciously specific technical claims likely to be fabricated."""
        violations = []

        for pattern, name, explanation in self.SUSPICIOUS_SPECS:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                violations.append({
                    "type": "SUSPICIOUS_SPEC",
                    "match": match.group(0),
                    "name": name,
                    "reason": explanation,
                    "fix": "Remove specific claim or cite verifiable source"
                })
        return violations

    def _check_hardware_claims(self, text: str) -> List[Dict]:
        """Flag hardware claims that need verification (not auto-fail, but warning)."""
        warnings = []
        lines = text.split('\n')

        for i, line in enumerate(lines, 1):
            for pattern in self.HARDWARE_CLAIMS:
                if re.search(pattern, line, re.IGNORECASE):
                    warnings.append({
                        "type": "HARDWARE_CLAIM_WARNING",
                        "line": i,
                        "text": line.strip()[:100],
                        "reason": "Hardware specs are easy to get wrong. Verify against official documentation.",
                        "fix": "Verify spec accuracy or remove specific numbers"
                    })
                    break  # One warning per line
        return warnings

    def _check_code_blocks(self, text: str) -> List[Dict]:
        """Validate all code blocks."""
        violations = []
        code_blocks = self._extract_code_blocks(text)

        for lang, code in code_blocks:
            if lang.lower() in ['python', 'py', 'python3']:
                valid, error = self._validate_python_syntax(code)
                if not valid:
                    violations.append({
                        "type": "INVALID_CODE",
                        "language": lang,
                        "error": error,
                        "code_preview": code[:200],
                        "fix": "Fix syntax error or mark as pseudocode"
                    })

                # Also check for undefined references (basic check)
                undefined = self._check_undefined_refs(code)
                if undefined:
                    violations.append({
                        "type": "INVALID_CODE",
                        "language": lang,
                        "error": f"Potentially undefined: {', '.join(undefined)}",
                        "fix": "Define or import missing references"
                    })
        return violations

    def _check_undefined_refs(self, code: str) -> List[str]:
        """Basic check for obviously undefined references."""
        # This is a simple heuristic, not a full static analysis
        undefined = []

        # Common patterns of undefined usage
        patterns = [
            (r'(\w+)\(\)', r'class \1|def \1|import \1|\1 ='),  # Function calls
            (r'= (\w+)\(', r'class \1|def \1|import \1|from .* import \1'),  # Assignments from calls
        ]

        lines = code.split('\n')
        defined = set()

        # Collect definitions
        for line in lines:
            # imports
            if 'import ' in line:
                match = re.search(r'import (\w+)', line)
                if match:
                    defined.add(match.group(1))
                match = re.search(r'from .* import (\w+)', line)
                if match:
                    defined.add(match.group(1))
            # class/def
            match = re.search(r'(class|def) (\w+)', line)
            if match:
                defined.add(match.group(2))
            # assignments
            match = re.search(r'^(\w+)\s*=', line)
            if match:
                defined.add(match.group(1))

        # Check for obviously undefined calls
        for line in lines:
            # Look for CapitalizedWord() that's not defined
            matches = re.findall(r'\b([A-Z]\w+)\s*\(', line)
            for match in matches:
                if match not in defined and match not in ['True', 'False', 'None', 'Exception', 'Container', 'List', 'Dict', 'Optional']:
                    undefined.append(match)

        return list(set(undefined))

    def validate(self, text: str, topic: str = "") -> Dict:
        """
        Run all technical validation checks.

        Returns:
            {
                "passed": bool,
                "score": int (0-100),
                "violations": [...],
                "summary": str
            }
        """
        all_violations = []
        hardware_warnings = []

        # 1. Check for fabricated stats
        all_violations.extend(self._find_uncited_stats(text))

        # 2. Check for phantom evidence
        all_violations.extend(self._find_phantom_evidence(text))

        # 3. Check causal validity
        all_violations.extend(self._check_causal_validity(text))

        # 4. Check code blocks
        all_violations.extend(self._check_code_blocks(text))

        # 5. Check for FAKE metrics (e.g., "parameters per second" - doesn't exist)
        all_violations.extend(self._check_fake_metrics(text))

        # 6. Check for suspicious specs (trillion-scale claims, TB storage, etc.)
        all_violations.extend(self._check_suspicious_specs(text))

        # 7. Check hardware claims (warnings only - for review)
        hardware_warnings = self._check_hardware_claims(text)

        # Calculate score
        # CRITICAL: Fake metrics and suspicious specs are as bad as invalid code
        critical_count = len([v for v in all_violations if v["type"] in [
            "INVALID_CODE", "CAUSAL_NONSENSE", "FAKE_METRIC", "SUSPICIOUS_SPEC"
        ]])
        major_count = len([v for v in all_violations if v["type"] in ["FABRICATED_STATS", "PHANTOM_EVIDENCE"]])

        score = 100 - (critical_count * 25) - (major_count * 10)
        score = max(0, score)

        passed = score >= 60 and critical_count == 0

        return {
            "passed": passed,
            "score": score,
            "violations": all_violations,
            "warnings": hardware_warnings,
            "critical_count": critical_count,
            "major_count": major_count,
            "summary": self._generate_summary(all_violations, score, passed, hardware_warnings)
        }

    def _generate_summary(self, violations: List[Dict], score: int, passed: bool, warnings: List[Dict] = None) -> str:
        """Generate human-readable summary."""
        if not violations and not warnings:
            return "PASSED: No technical issues detected."

        summary = f"{'PASSED' if passed else 'FAILED'} (Score: {score}/100)\n\n"

        # Group by type
        by_type = {}
        for v in violations:
            vtype = v["type"]
            if vtype not in by_type:
                by_type[vtype] = []
            by_type[vtype].append(v)

        for vtype, items in by_type.items():
            severity = "CRITICAL" if vtype in ["INVALID_CODE", "CAUSAL_NONSENSE", "FAKE_METRIC", "SUSPICIOUS_SPEC"] else "MAJOR"
            summary += f"## {vtype} [{severity}] ({len(items)} issues)\n"
            for item in items[:3]:  # Show max 3 per type
                if "line" in item:
                    summary += f"  - Line {item['line']}: {item.get('match', item.get('phrase', ''))}\n"
                elif "reason" in item:
                    summary += f"  - {item.get('match', '')}: {item.get('reason', '')}\n"
                else:
                    summary += f"  - {item.get('match', item.get('error', ''))}\n"
                summary += f"    Fix: {item['fix']}\n"
            if len(items) > 3:
                summary += f"  ... and {len(items) - 3} more\n"
            summary += "\n"

        # Add hardware warnings (not failures, just flags for review)
        if warnings:
            summary += f"## HARDWARE_CLAIMS [WARNING] ({len(warnings)} items to verify)\n"
            for w in warnings[:3]:
                summary += f"  - Line {w.get('line', '?')}: {w.get('text', '')[:60]}...\n"
                summary += f"    Note: {w.get('reason', '')}\n"
            if len(warnings) > 3:
                summary += f"  ... and {len(warnings) - 3} more\n"
            summary += "\n"

        return summary

    def review_draft(self, draft: str, topic: str = "") -> str:
        """
        Review a draft and return actionable feedback.

        This is the main interface called by the pipeline.
        """
        result = self.validate(draft, topic)

        if result["passed"]:
            return f"APPROVED (Score: {result['score']}/100)\n\nNo critical technical issues."

        # Generate detailed feedback for revision
        feedback = f"REVISE REQUIRED (Score: {result['score']}/100)\n\n"
        feedback += "The following technical issues must be fixed:\n\n"
        feedback += result["summary"]
        feedback += "\n\nINSTRUCTIONS FOR REVISION:\n"
        feedback += "1. Remove or cite all uncited statistics\n"
        feedback += "2. Fix or remove broken code examples\n"
        feedback += "3. Remove causally impossible claims\n"
        feedback += "4. Replace vague 'studies show' with specific sources or rephrase as opinion\n"
        feedback += "5. REMOVE fake ML metrics like 'parameters per second' - use real metrics (FLOPS, tokens/sec, samples/sec)\n"
        feedback += "6. REMOVE trillion-scale claims and multi-TB storage claims unless citing official documentation\n"
        feedback += "7. VERIFY all hardware specs (GPU names, TFLOPS, etc.) against official sources\n"

        return feedback

    def deep_review(self, draft: str, topic: str = "") -> str:
        """
        Use LLM for deeper semantic analysis beyond pattern matching.

        Catches:
        - Domain mismatches
        - Shallow insights
        - Missing real-world concerns
        """
        prompt = f"""Analyze this technical article for SUBSTANTIVE issues (not style):

ARTICLE:
{draft[:4000]}

TOPIC CONTEXT: {topic}

Check for:

1. DOMAIN_MISMATCH: Does the solution actually address the stated problem?
   - If about MLOps: Does it mention data drift, feature stores, model registry, training-serving skew?
   - If about deployment: Does it mention real deployment concerns (latency, scaling, rollback)?
   - Or is it tangentially related?

2. SHALLOW_INSIGHT: Is this just "best practices 101" dressed up?
   - "Separate concerns" - everyone knows this
   - "Use dependency injection" - basic stuff
   - What would a SENIOR engineer learn from this?

3. MISSING_REAL_CONCERNS: What would a practitioner actually worry about that's not covered?
   - For ML: reproducibility, experiment tracking, model versioning
   - For deployment: cold starts, GPU scheduling, canary releases

4. TECHNICAL_ACCURACY: Are the technical claims accurate?
   - Are the code patterns actually recommended?
   - Are the trade-offs mentioned?

Return a structured assessment:
- DOMAIN_FIT: [Good/Partial/Poor] - explanation
- DEPTH_LEVEL: [Senior/Mid/Junior] - what level engineer would find this useful?
- MISSING_CONCERNS: [list of 3-5 real concerns not addressed]
- ACCURACY_ISSUES: [list any technical inaccuracies]
- VERDICT: [APPROVED/NEEDS_DEPTH/REJECT]
"""
        return self.call_llm(prompt, temperature=0.3)
