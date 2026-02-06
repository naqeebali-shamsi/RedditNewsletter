#!/usr/bin/env python3
"""
Style Enforcement Agent - Quantitative voice fingerprinting.

Scores content across 5 dimensions to ensure it matches
the TheWritingStack's unique voice signature.

Usage:
    from execution.agents.style_enforcer import StyleEnforcerAgent

    enforcer = StyleEnforcerAgent()
    result = enforcer.score(text)
    print(f"Score: {result['total']}/100")
"""

import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict

# Try to import optional dependencies
try:
    from lexicalrichness import LexicalRichness
    HAS_LEXICAL = True
except ImportError:
    HAS_LEXICAL = False

try:
    import nltk
    from nltk.tokenize import sent_tokenize
    HAS_NLTK = True
except ImportError:
    HAS_NLTK = False


@dataclass
class StyleScore:
    """Result of style enforcement scoring."""
    total: float  # 0-100 composite score
    passed: bool  # total >= 80
    needs_revision: bool  # 60 <= total < 80
    rejected: bool  # total < 60

    # Individual dimension scores (0-100 each)
    burstiness_score: float  # 20% weight
    lexical_diversity_score: float  # 15% weight
    ai_tell_score: float  # 25% weight
    authenticity_score: float  # 25% weight
    framework_compliance_score: float  # 15% weight

    # Raw metrics
    burstiness_ratio: float
    avg_sentence_length: float
    sentence_length_std: float
    vocd_score: Optional[float]
    ttr: Optional[float]

    # Violations
    ai_tells_found: List[Dict[str, Any]]
    war_story_keywords_found: List[str]
    specific_metrics_found: List[str]

    # Framework checks
    has_contrast_hook: bool
    has_tradeoff: bool
    paragraph_violations: List[Dict[str, Any]]

    def to_dict(self) -> dict:
        return asdict(self)


class StyleEnforcerAgent:
    """Quantitative style enforcement for TheWritingStack voice."""

    # Default forbidden phrases (AI tells)
    DEFAULT_FORBIDDEN = [
        "in this post, we will explore",
        "furthermore, it is important",
        "in conclusion, we have seen",
        "it is worth mentioning",
        "as we can see from the above",
        "transitioning to",
        "let's dive in",
        "without further ado",
        "in today's fast-paced world",
        "it is important to note",
        "it goes without saying",
        "needless to say",
        "at the end of the day",
        "in this article",
        "as mentioned earlier",
        "moving forward",
    ]

    # War story keywords that indicate authenticity
    DEFAULT_WAR_STORY_KEYWORDS = [
        "i built", "i broke", "we encountered", "pager duty",
        "context leak", "we built", "we broke", "i debugged",
        "i discovered", "we learned", "i spent", "we saw",
        "the gotcha", "in production", "i watched",
    ]

    # Hedging phrases (weaker AI tells)
    HEDGING_PHRASES = [
        "may potentially", "could possibly", "might be considered",
        "appears to indicate", "it seems that", "arguably",
        "to some extent", "in some cases",
    ]

    def __init__(self, profile_path: Optional[str] = None):
        """
        Initialize with optional voice profile.

        Args:
            profile_path: Path to voice_profile.json for baseline comparison
        """
        self.profile = None
        if profile_path:
            path = Path(profile_path)
            if path.exists():
                with open(path, 'r', encoding='utf-8') as f:
                    self.profile = json.load(f)

        # Load profile defaults
        if self.profile:
            self.forbidden_phrases = self.profile.get("forbidden_phrases", self.DEFAULT_FORBIDDEN)
            markers = self.profile.get("required_markers", {})
            self.war_story_keywords = markers.get("war_story_keywords", self.DEFAULT_WAR_STORY_KEYWORDS)
        else:
            self.forbidden_phrases = self.DEFAULT_FORBIDDEN
            self.war_story_keywords = self.DEFAULT_WAR_STORY_KEYWORDS

        # Ensure NLTK data is available
        if HAS_NLTK:
            try:
                nltk.data.find('tokenizers/punkt_tab')
            except LookupError:
                try:
                    nltk.download('punkt_tab', quiet=True)
                except Exception:
                    pass

    def _tokenize_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        if HAS_NLTK:
            try:
                return sent_tokenize(text)
            except Exception:
                pass
        # Fallback: simple regex
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip() and len(s.strip()) > 5]

    def _calculate_burstiness(self, text: str) -> tuple:
        """
        Calculate sentence length variation (burstiness).

        Returns: (burstiness_ratio, avg_length, std_dev, sentence_lengths)
        Human writing: 0.3-0.6 burstiness
        AI writing: 0.15-0.25 burstiness
        """
        sentences = self._tokenize_sentences(text)
        if len(sentences) < 3:
            return (0.0, 0.0, 0.0, [])

        lengths = [len(s.split()) for s in sentences]

        import statistics
        mean_len = statistics.mean(lengths)
        std_dev = statistics.stdev(lengths) if len(lengths) > 1 else 0

        burstiness = std_dev / mean_len if mean_len > 0 else 0

        return (burstiness, mean_len, std_dev, lengths)

    def _calculate_lexical_diversity(self, text: str) -> tuple:
        """
        Calculate lexical richness metrics.

        Returns: (vocd_score, ttr)
        """
        if not HAS_LEXICAL:
            # Fallback: simple TTR
            words = text.lower().split()
            if not words:
                return (None, None)
            ttr = len(set(words)) / len(words)
            return (None, ttr)

        try:
            lex = LexicalRichness(text)
            ttr = lex.ttr
            try:
                vocd = lex.vocd(ntokens=50, within_sample=100, iterations=3)
            except Exception:
                vocd = None
            return (vocd, ttr)
        except Exception:
            words = text.lower().split()
            ttr = len(set(words)) / len(words) if words else 0
            return (None, ttr)

    def _detect_ai_tells(self, text: str) -> List[Dict[str, Any]]:
        """Scan for forbidden AI-generated phrases."""
        text_lower = text.lower()
        violations = []

        lines = text.split('\n')
        for phrase in self.forbidden_phrases:
            if phrase.lower() in text_lower:
                # Find line numbers
                for i, line in enumerate(lines):
                    if phrase.lower() in line.lower():
                        violations.append({
                            "phrase": phrase,
                            "line": i + 1,
                            "context": line.strip()[:100],
                            "severity": "critical"
                        })

        # Also check hedging phrases (less severe)
        for phrase in self.HEDGING_PHRASES:
            if phrase.lower() in text_lower:
                for i, line in enumerate(lines):
                    if phrase.lower() in line.lower():
                        violations.append({
                            "phrase": phrase,
                            "line": i + 1,
                            "context": line.strip()[:100],
                            "severity": "warning"
                        })

        return violations

    def _detect_authenticity_markers(self, text: str) -> tuple:
        """
        Detect war story keywords and specific metrics.

        Returns: (keywords_found, metrics_found)
        """
        text_lower = text.lower()

        keywords_found = []
        for kw in self.war_story_keywords:
            if kw.lower() in text_lower:
                keywords_found.append(kw)

        # Detect specific metrics/numbers
        metrics = re.findall(
            r'\d+%|\d+ms|\d+x|\d+ hours?|\d+ days?|\d+ minutes?|\$\d+|\d+K|\d+M',
            text
        )

        return (keywords_found, metrics)

    def _check_framework_compliance(self, text: str, content_type: str = "article") -> dict:
        """Check compliance with 5-Pillar framework."""

        # Check for contrast hook (first paragraph challenges status quo)
        first_para = text.split('\n\n')[0] if '\n\n' in text else text[:500]
        contrast_indicators = ['vs', 'but', 'not', 'instead', 'stop', 'wrong', 'mistake', 'myth']
        has_contrast = any(ind in first_para.lower() for ind in contrast_indicators)

        # Check for tradeoff mentions
        tradeoff_indicators = ['tradeoff', 'trade-off', 'cost:', 'downside', 'vs.', 'versus', 'at the expense']
        has_tradeoff = any(ind in text.lower() for ind in tradeoff_indicators)

        # Check paragraph lengths
        limits = {'linkedin': 3, 'article': 5, 'longform': 6}
        max_lines = limits.get(content_type, 5)

        paragraphs = [p for p in text.split('\n\n') if p.strip() and not p.strip().startswith('#')]
        para_violations = []
        for i, para in enumerate(paragraphs):
            lines = para.count('\n') + 1
            word_count = len(para.split())
            # Approximate line count from word count (about 15 words per line)
            approx_lines = max(lines, word_count // 15)
            if approx_lines > max_lines:
                para_violations.append({
                    "paragraph": i + 1,
                    "approx_lines": approx_lines,
                    "limit": max_lines,
                    "word_count": word_count
                })

        return {
            "has_contrast_hook": has_contrast,
            "has_tradeoff": has_tradeoff,
            "paragraph_violations": para_violations
        }

    def score(self, text: str, content_type: str = "article") -> StyleScore:
        """
        Score content across all 5 dimensions.

        Args:
            text: Content to score
            content_type: 'linkedin', 'article', or 'longform'

        Returns:
            StyleScore with composite score and breakdown
        """
        # 1. Burstiness (20% weight)
        burstiness, avg_len, std_dev, _ = self._calculate_burstiness(text)
        if burstiness >= 0.4:
            burstiness_score = 100
        elif burstiness >= 0.3:
            burstiness_score = 80
        elif burstiness >= 0.2:
            burstiness_score = 60
        else:
            burstiness_score = max(20, burstiness * 250)  # Linear scale below 0.2

        # 2. Lexical Diversity (15% weight)
        vocd, ttr = self._calculate_lexical_diversity(text)
        if vocd is not None:
            if vocd >= 60:
                lex_score = 100
            elif vocd >= 45:
                lex_score = 80
            elif vocd >= 30:
                lex_score = 60
            else:
                lex_score = 40
        elif ttr is not None:
            # Fallback scoring based on TTR
            if ttr >= 0.6:
                lex_score = 90
            elif ttr >= 0.45:
                lex_score = 70
            else:
                lex_score = 50
        else:
            lex_score = 50  # Default if can't calculate

        # 3. AI-Tell Detection (25% weight)
        ai_tells = self._detect_ai_tells(text)
        critical_tells = [t for t in ai_tells if t["severity"] == "critical"]
        warning_tells = [t for t in ai_tells if t["severity"] == "warning"]

        if not ai_tells:
            ai_tell_score = 100
        elif not critical_tells:
            # Only warnings
            ai_tell_score = max(60, 100 - len(warning_tells) * 10)
        else:
            # Critical tells: heavy penalty
            ai_tell_score = max(0, 80 - len(critical_tells) * 25 - len(warning_tells) * 5)

        # 4. Authenticity Markers (25% weight)
        keywords_found, metrics_found = self._detect_authenticity_markers(text)

        auth_score = 0
        # War story keywords (up to 50 points)
        if len(keywords_found) >= 3:
            auth_score += 50
        elif len(keywords_found) >= 2:
            auth_score += 40
        elif len(keywords_found) >= 1:
            auth_score += 25

        # Specific metrics (up to 50 points)
        if len(metrics_found) >= 5:
            auth_score += 50
        elif len(metrics_found) >= 3:
            auth_score += 40
        elif len(metrics_found) >= 1:
            auth_score += 25

        auth_score = min(100, auth_score)

        # 5. Framework Compliance (15% weight)
        framework = self._check_framework_compliance(text, content_type)

        fw_score = 0
        if framework["has_contrast_hook"]:
            fw_score += 40
        if framework["has_tradeoff"]:
            fw_score += 30
        # Paragraph compliance
        para_penalty = len(framework["paragraph_violations"]) * 10
        fw_score += max(0, 30 - para_penalty)
        fw_score = min(100, fw_score)

        # Composite score (weighted)
        total = (
            burstiness_score * 0.20 +
            lex_score * 0.15 +
            ai_tell_score * 0.25 +
            auth_score * 0.25 +
            fw_score * 0.15
        )

        return StyleScore(
            total=round(total, 1),
            passed=total >= 80,
            needs_revision=60 <= total < 80,
            rejected=total < 60,
            burstiness_score=round(burstiness_score, 1),
            lexical_diversity_score=round(lex_score, 1),
            ai_tell_score=round(ai_tell_score, 1),
            authenticity_score=round(auth_score, 1),
            framework_compliance_score=round(fw_score, 1),
            burstiness_ratio=round(burstiness, 3),
            avg_sentence_length=round(avg_len, 1),
            sentence_length_std=round(std_dev, 1),
            vocd_score=round(vocd, 1) if vocd else None,
            ttr=round(ttr, 3) if ttr else None,
            ai_tells_found=ai_tells,
            war_story_keywords_found=keywords_found,
            specific_metrics_found=metrics_found,
            has_contrast_hook=framework["has_contrast_hook"],
            has_tradeoff=framework["has_tradeoff"],
            paragraph_violations=framework["paragraph_violations"],
        )

    def format_report(self, result: StyleScore) -> str:
        """Format a human-readable style enforcement report."""
        status = "PASS" if result.passed else ("NEEDS REVISION" if result.needs_revision else "REJECTED")

        lines = [
            f"# Style Enforcement Report",
            f"",
            f"## Overall Score: {result.total}/100 - {status}",
            f"",
            f"### Dimension Breakdown",
            f"| Dimension | Score | Weight | Weighted |",
            f"|-----------|-------|--------|----------|",
            f"| Burstiness | {result.burstiness_score} | 20% | {result.burstiness_score * 0.20:.1f} |",
            f"| Lexical Diversity | {result.lexical_diversity_score} | 15% | {result.lexical_diversity_score * 0.15:.1f} |",
            f"| AI-Tell Detection | {result.ai_tell_score} | 25% | {result.ai_tell_score * 0.25:.1f} |",
            f"| Authenticity Markers | {result.authenticity_score} | 25% | {result.authenticity_score * 0.25:.1f} |",
            f"| Framework Compliance | {result.framework_compliance_score} | 15% | {result.framework_compliance_score * 0.15:.1f} |",
            f"",
            f"### Raw Metrics",
            f"- Burstiness ratio: {result.burstiness_ratio} (target: 0.4-0.8)",
            f"- Avg sentence length: {result.avg_sentence_length} words",
            f"- Sentence length std dev: {result.sentence_length_std}",
        ]

        if result.vocd_score:
            lines.append(f"- VOCD score: {result.vocd_score} (target: 60-90)")
        if result.ttr:
            lines.append(f"- Type-Token Ratio: {result.ttr}")

        if result.ai_tells_found:
            lines.extend([
                f"",
                f"### AI Tells Found ({len(result.ai_tells_found)})",
            ])
            for tell in result.ai_tells_found:
                lines.append(f"- [{tell['severity'].upper()}] Line {tell['line']}: \"{tell['phrase']}\"")

        if result.war_story_keywords_found:
            lines.extend([
                f"",
                f"### Authenticity Markers Found",
                f"- War story keywords: {', '.join(result.war_story_keywords_found)}",
                f"- Specific metrics: {len(result.specific_metrics_found)} found",
            ])

        if result.paragraph_violations:
            lines.extend([
                f"",
                f"### Paragraph Violations",
            ])
            for v in result.paragraph_violations:
                lines.append(f"- Paragraph {v['paragraph']}: ~{v['approx_lines']} lines (limit: {v['limit']})")

        return "\n".join(lines)


# CLI entry point
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Style Enforcement Scoring")
    parser.add_argument("file", help="Path to article/content file")
    parser.add_argument("--profile", help="Path to voice_profile.json")
    parser.add_argument("--type", choices=["linkedin", "article", "longform"],
                       default="article", help="Content type")
    parser.add_argument("--json", action="store_true", help="Output as JSON")

    args = parser.parse_args()

    content = Path(args.file).read_text(encoding='utf-8')
    enforcer = StyleEnforcerAgent(profile_path=args.profile)
    result = enforcer.score(content, content_type=args.type)

    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print(enforcer.format_report(result))
