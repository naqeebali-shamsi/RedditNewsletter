"""
Voice Utilities - Source type to voice mapping and validation.

Provides utilities for determining appropriate voice/tone based on
content source type:
- External (Reddit, news, community): Observer voice - no ownership claims
- Internal (own work, case studies): Practitioner voice - ownership appropriate
"""

from dataclasses import dataclass
from typing import List, Tuple
import re


@dataclass
class VoiceConfig:
    """Configuration for a voice type."""
    name: str
    source_type: str
    allowed_first_person: List[str]
    forbidden_phrases: List[str]
    example_good: str
    example_bad: str


# Voice configurations
OBSERVER_VOICE = VoiceConfig(
    name="Observer Voice",
    source_type="external",
    allowed_first_person=[
        "I noticed",
        "I've observed",
        "I've seen",
        "I've been tracking",
        "I found it interesting",
        "I think",
        "I believe"
    ],
    forbidden_phrases=[
        "I built",
        "I created",
        "I developed",
        "we built",
        "we created",
        "we developed",
        "our team built",
        "our team created",
        "my team built",
        "my approach was",
        "we discovered",
        "our implementation",
        "my implementation"
    ],
    example_good="Engineers at the company discovered that caching reduced latency by 40%.",
    example_bad="We discovered that our caching approach reduced latency by 40%."
)

PRACTITIONER_VOICE = VoiceConfig(
    name="Practitioner Voice",
    source_type="internal",
    allowed_first_person=[
        "I built",
        "I created",
        "we built",
        "we created",
        "our team",
        "my approach",
        "we discovered",
        "our implementation"
    ],
    forbidden_phrases=[],  # No restrictions for internal content
    example_good="We built a caching layer that reduced our API latency by 40%.",
    example_bad="N/A - Practitioner voice allows ownership claims."
)

VOICE_MAP = {
    "external": OBSERVER_VOICE,
    "internal": PRACTITIONER_VOICE
}


def get_voice_config(source_type: str) -> VoiceConfig:
    """Get voice configuration for source type."""
    return VOICE_MAP.get(source_type, OBSERVER_VOICE)


def check_voice_violations(content: str, source_type: str) -> List[Tuple[str, str, int]]:
    """
    Check content for voice violations.

    Returns list of (phrase, context, line_number) tuples for each violation.
    """
    voice = get_voice_config(source_type)
    violations = []

    if not voice.forbidden_phrases:
        return violations

    lines = content.split('\n')
    for line_num, line in enumerate(lines, 1):
        line_lower = line.lower()
        for phrase in voice.forbidden_phrases:
            if phrase.lower() in line_lower:
                # Get context (surrounding text)
                start = max(0, line_lower.find(phrase.lower()) - 20)
                end = min(len(line), line_lower.find(phrase.lower()) + len(phrase) + 20)
                context = line[start:end]
                violations.append((phrase, f"...{context}...", line_num))

    return violations


def get_voice_instruction(source_type: str, detailed: bool = False) -> str:
    """
    Get voice instruction prompt for LLM.

    Args:
        source_type: 'external' or 'internal'
        detailed: Include examples and full explanation

    Returns:
        Voice instruction string for LLM prompts
    """
    voice = get_voice_config(source_type)

    if source_type == "external":
        basic = """VOICE REQUIREMENT (External Source - Observer Voice):
You are REPORTING on others' work, not claiming ownership.
- FORBIDDEN: "I built", "we created", "our team", "my approach"
- ALLOWED: "I noticed", "I've observed", "I think", "I believe"
- USE: "teams found", "engineers discovered", "this approach"
- Any ownership claims are INSTANT FAILURES."""

        if detailed:
            basic += f"""

GOOD EXAMPLE: {voice.example_good}
BAD EXAMPLE: {voice.example_bad}

FORBIDDEN PHRASES:
{chr(10).join(f'- "{p}"' for p in voice.forbidden_phrases[:5])}"""

    else:
        basic = """VOICE (Internal Source - Practitioner Voice):
This is YOUR OWN work. Ownership voice is appropriate.
- "I", "we", "our", "my" are all acceptable
- Speak from experience and authority
- Share your learnings and approach directly."""

    return basic


def validate_voice(content: str, source_type: str) -> dict:
    """
    Validate content voice and return detailed report.

    Returns:
        dict with 'passed', 'violations', 'score', 'recommendation'
    """
    violations = check_voice_violations(content, source_type)

    if not violations:
        return {
            "passed": True,
            "violations": [],
            "score": 10,
            "recommendation": "Voice is appropriate for source type."
        }

    # Score based on violation count
    score = max(1, 10 - len(violations) * 2)
    passed = len(violations) == 0

    # Build recommendations
    recommendations = []
    unique_phrases = set(v[0] for v in violations)
    for phrase in unique_phrases:
        if phrase.lower().startswith("i ") or phrase.lower().startswith("we "):
            recommendations.append(f'Replace "{phrase}" with third-person attribution')
        elif "our" in phrase.lower() or "my" in phrase.lower():
            recommendations.append(f'Remove ownership claim "{phrase}" - use "the" instead')

    return {
        "passed": passed,
        "violations": violations,
        "score": score,
        "recommendation": "; ".join(recommendations[:3]) if recommendations else "Remove ownership claims."
    }


def rewrite_for_voice(text: str, source_type: str) -> str:
    """
    Simple rule-based rewriting to fix common voice violations.

    For external sources, converts ownership claims to third-person.
    """
    if source_type != "external":
        return text

    # Simple replacements for common violations
    replacements = [
        (r'\bI built\b', 'The team built'),
        (r'\bWe built\b', 'The team built'),
        (r'\bI created\b', 'Engineers created'),
        (r'\bWe created\b', 'Engineers created'),
        (r'\bI developed\b', 'Developers created'),
        (r'\bWe developed\b', 'The team developed'),
        (r'\bour team\b', 'the team'),
        (r'\bOur team\b', 'The team'),
        (r'\bmy approach\b', 'this approach'),
        (r'\bMy approach\b', 'This approach'),
        (r'\bour implementation\b', 'the implementation'),
        (r'\bOur implementation\b', 'The implementation'),
        (r'\bwe discovered\b', 'engineers discovered'),
        (r'\bWe discovered\b', 'Engineers discovered'),
    ]

    result = text
    for pattern, replacement in replacements:
        result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)

    return result


# Quick test
if __name__ == "__main__":
    test_content = """
    I've been tracking AI developments and I built a system that processes 1000 requests per second.
    Our team created a novel approach to caching. We discovered significant improvements.
    """

    print("Testing voice utilities:")
    print(f"\nViolations in external source:")
    violations = check_voice_violations(test_content, "external")
    for phrase, context, line in violations:
        print(f"  Line {line}: '{phrase}' in '{context}'")

    print(f"\nVoice validation:")
    result = validate_voice(test_content, "external")
    print(f"  Passed: {result['passed']}")
    print(f"  Score: {result['score']}/10")
    print(f"  Recommendation: {result['recommendation']}")

    print(f"\nRewritten text:")
    print(rewrite_for_voice(test_content, "external"))
