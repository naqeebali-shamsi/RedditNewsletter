from .base_agent import BaseAgent
from execution.config import config
import re
from typing import List, Tuple


class CriticAgent(BaseAgent):
    """
    Enhanced Critic with kill-phrase detection and multi-lens critique.

    Combines the skepticism of a Senior Staff Engineer with the quality
    standards from the adversarial expert panel consensus.
    """

    # Kill phrases that trigger instant rejection
    KILL_PHRASES = [
        ("...", "PLACEHOLDER", "Incomplete placeholder text"),
        ("What's been your experience", "WEAK_CTA", "Generic weak call-to-action"),
        ("This aligns with what I'm seeing", "TEMPLATE", "Template boilerplate"),
        ("In this article", "BORING_OPENER", "Generic opener"),
        ("As AI engineering matures", "CLICHE", "Overused closing cliche"),
        ("Drop a comment below", "WEAK_CTA", "Weak engagement ask"),
        ("<!-- SC_OFF", "HTML_ARTIFACT", "Raw HTML leaked into content"),
        ("<div class=", "HTML_ARTIFACT", "Raw HTML leaked into content"),
        ("Contains some technical content", "METADATA_LEAK", "Internal metadata exposed"),
        ("contains 2 technical keywords", "METADATA_LEAK", "Internal scoring exposed"),
        ("🚀 Interesting insight from r/", "TEMPLATE_OPENER", "Same opener every post"),
        ("#AIEngineering #MachineLearning #LLMOps #ProductionAI", "HASHTAG_SPAM", "Same hashtags every post"),
    ]

    # Split kill phrases: word-boundary-safe vs literal substring
    # Phrases starting/ending with non-word chars cannot use \b
    _WORD_KILL_PHRASES = [
        (phrase, code, desc) for phrase, code, desc in KILL_PHRASES
        if re.search(r'^\w', phrase) and re.search(r'\w$', phrase)
    ]
    _LITERAL_KILL_PHRASES = [
        (phrase, code, desc) for phrase, code, desc in KILL_PHRASES
        if not (re.search(r'^\w', phrase) and re.search(r'\w$', phrase))
    ]

    # Compiled regex for word-boundary-safe kill phrases
    _KILL_PHRASE_RE = re.compile(
        r'\b(?:' + '|'.join(re.escape(p) for p, _, _ in _WORD_KILL_PHRASES) + r')\b',
        re.IGNORECASE
    ) if _WORD_KILL_PHRASES else None

    # Lookup from lowered phrase to (code, description) for kill phrase metadata
    _KILL_PHRASE_META = {phrase.lower(): (code, desc) for phrase, code, desc in KILL_PHRASES}

    def __init__(self):
        super().__init__(
            role="Senior Staff Engineer & Quality Critic",
            persona="""You are the cynical Senior Staff Engineer who has reviewed thousands of blog posts.
You've also internalized the standards from world-class copywriting agencies:
- Digital Commerce Partners (conversion focus)
- AWISEE (data-driven storytelling)
- Feldman Creative (extreme clarity)
- Apple (elegant simplicity)
- Google (SEO and structure)

Your job is DESTRUCTION of bad content. You look for:

TECHNICAL SINS:
- Claims without evidence
- Logic gaps in arguments
- Missing trade-offs ("nothing is free")
- Tutorial-speak instead of wisdom
- Vague generalities instead of specifics

COPYWRITING SINS:
- Weak hooks that don't stop the scroll
- Generic CTAs ("What do you think?")
- Template phrases repeated across posts
- Placeholder text ("...")
- HTML artifacts from scraping
- Internal metadata leaked into content

CREATIVE SINS:
- No memorable moments
- Nothing quotable/shareable
- Same structure as every other post
- AI-generated sameness
- Zero personality or voice

You score content 1-10 and provide SPECIFIC failures.
If content has ANY kill-phrase violations, automatic score cap of 4.""",
            model=config.models.DEFAULT_CRITIC_MODEL
        )

    def _check_kill_phrases(self, text: str) -> list:
        """Check for instant-failure phrases using word-boundary matching."""
        violations = []
        # Word-boundary regex for phrases that start/end with word chars
        if self._KILL_PHRASE_RE:
            for match in self._KILL_PHRASE_RE.finditer(text):
                matched = match.group()
                code, description = self._KILL_PHRASE_META[matched.lower()]
                violations.append({
                    "code": code,
                    "phrase": matched,
                    "description": description
                })
        # Literal substring match for phrases with special chars (e.g. "...", HTML tags)
        text_lower = text.lower()
        for phrase, code, description in self._LITERAL_KILL_PHRASES:
            if phrase.lower() in text_lower:
                violations.append({
                    "code": code,
                    "phrase": phrase,
                    "description": description
                })
        return violations

    def critique_outline(self, outline):
        """Critique an article outline before drafting begins."""

        prompt = f"""
Critique this article outline with extreme prejudice:

{outline}

EVALUATION CRITERIA:

1. **HOOK POTENTIAL** (Will the planned opening stop someone from scrolling?)
   - Does it promise curiosity, transformation, or bold claim?
   - Or is it generic "In this article..." territory?

2. **STRUCTURAL LOGIC** (Does the flow make sense?)
   - Does each section build on the previous?
   - Is there a clear narrative arc?

3. **SPECIFICITY PLAN** (Where will the numbers/examples come from?)
   - Are there planned data points?
   - Or will it be vague hand-waving?

4. **DIFFERENTIATION** (Why would anyone read THIS vs. 100 other articles on the topic?)
   - What's the unique angle?
   - What's the contrarian take?

5. **ACTIONABILITY** (Will readers DO something differently?)
   - Are there planned takeaways?
   - Or just information without application?

Be HARSH. Better to catch problems now than after drafting.

Format your response as:
SCORE: X/10
VERDICT: [Brief verdict]
FAILURES:
- [Specific failure 1]
- [Specific failure 2]
RECOMMENDATIONS:
- [Specific fix 1]
- [Specific fix 2]
"""
        return self.call_llm(prompt)

    def critique_section(self, section_text):
        """Critique a drafted section."""

        # First, check for kill phrases
        violations = self._check_kill_phrases(section_text)

        violation_warning = ""
        if violations:
            violation_warning = f"""
⚠️ KILL-PHRASE VIOLATIONS DETECTED (Auto-cap score at 4):
{chr(10).join(f"- [{v['code']}] Found: '{v['phrase']}' - {v['description']}" for v in violations)}

"""

        prompt = f"""{violation_warning}
Critique this content section:

---
{section_text}
---

CRITIQUE LENSES:

1. **TECHNICAL ACCURACY**
   - Are claims verifiable?
   - Are trade-offs acknowledged?
   - Would a senior engineer roll their eyes?

2. **COPYWRITING QUALITY**
   - Does it hook immediately?
   - Is every paragraph earning its place?
   - Is the CTA specific and compelling?

3. **SPECIFICITY**
   - Count the specific numbers/metrics
   - Count the vague words (many, some, often, significant)
   - Ratio should favor specifics

4. **MEMORABILITY**
   - Is there ONE line someone would screenshot?
   - Is there a metaphor that clicks?
   - Or is it forgettable commodity content?

5. **VOICE CONSISTENCY**
   - Does it sound like one person wrote it?
   - Any jarring tonal shifts?

SCORE: X/10
{"(CAPPED AT 4 DUE TO KILL-PHRASE VIOLATIONS)" if violations else ""}

VERDICT: [2-5 word summary]

SPECIFIC FAILURES:
- [Failure with exact quote from text]
- [Another failure]

FIX INSTRUCTIONS:
- [Concrete fix 1]
- [Concrete fix 2]
"""
        return self.call_llm(prompt)

    def critique_full_draft(self, draft: str, platform: str = "medium", source_type: str = "external") -> dict:
        """
        Comprehensive critique of a full draft.

        Args:
            draft: The content to critique
            platform: 'linkedin' or 'medium'
            source_type: 'external' (observer voice) or 'internal' (owner voice)

        Returns dict with score, violations, failures, and fixes.
        """

        # Check kill phrases first
        violations = self._check_kill_phrases(draft)

        violation_section = ""
        if violations:
            violation_section = f"""
⚠️ AUTOMATIC FAILURES (Kill-phrase violations):
{chr(10).join(f"- [{v['code']}] '{v['phrase']}' - {v['description']}" for v in violations)}

These MUST be fixed. Score capped at 4 until resolved.
"""

        # Voice context for LLM-based validation
        if source_type == "external":
            voice_section = """
**CRITICAL: VOICE/OWNERSHIP CHECK** (External Source)
This content is sourced from Reddit/external community. The author did NOT build this.
- INSTANT FAILURE if you find: "I built", "we created", "our team", "my approach", "we discovered"
- ACCEPTABLE: "teams found", "engineers discovered", "this approach", "the implementation"
- "I" ONLY allowed for observations: "I noticed", "I've observed", "I find this interesting"
- Scan the ENTIRE draft for ownership claims. Even ONE instance is a critical failure.
"""
        else:
            voice_section = """
**VOICE NOTE** (Internal Source - Author's Own Work)
Ownership voice ("I built", "we created", "our team") is appropriate for this content.
"""

        prompt = f"""
You are reviewing a {platform} draft for publication readiness.

{violation_section}
{voice_section}

DRAFT:
---
{draft}
---

COMPREHENSIVE CRITIQUE:

**1. HOOK TEST** (First 10 words)
- Does it create curiosity, tension, or promise transformation?
- Or is it generic/boring?
Score: X/10

**2. SPECIFICITY TEST** (Count numbers vs. vague words)
- List specific numbers/metrics found:
- List vague words found (many, some, significant, etc.):
Score: X/10

**3. MEMORABLE MOMENT TEST**
- Quote the most shareable line (if any):
- If none exists, note "NO MEMORABLE MOMENT"
Score: X/10

**4. COMPLETENESS TEST**
- Any placeholder text ("...")?
- Any incomplete sections?
- Any HTML/metadata artifacts?
Score: X/10

**5. VOICE CONSISTENCY TEST**
- Does it sound like one person throughout?
- Any jarring tonal shifts?
Score: X/10

**6. VOICE/OWNERSHIP TEST** (CRITICAL for external sources)
- List any ownership claims found ("I built", "we created", "our team", etc.):
- Are ownership claims appropriate for the source type?
- If external source: ANY ownership claim = automatic failure
Score: X/10

**7. CTA QUALITY TEST**
- Quote the CTA:
- Is it specific or generic?
Score: X/10

**OVERALL SCORE**: X/10 {"(CAPPED DUE TO VIOLATIONS)" if violations else ""}

**VERDICT**: [APPROVED / REVISE / REJECT]

**TOP 5 ISSUES** (in priority order):
1. [Most critical issue]
2. [Second issue]
3. [Third issue]
4. [Fourth issue]
5. [Fifth issue]

**FIX INSTRUCTIONS** (specific and actionable):
1. [How to fix issue 1]
2. [How to fix issue 2]
3. [How to fix issue 3]
"""

        response = self.call_llm(prompt, temperature=0.3)

        # Parse score from response
        score_match = re.search(r'\*\*OVERALL SCORE\*\*:\s*(\d+(?:\.\d+)?)', response)
        score = float(score_match.group(1)) if score_match else 5.0

        # Cap score if violations exist
        if violations and score > 4:
            score = 4.0

        return {
            "score": score,
            "violations": violations,
            "critique": response,
            "passed": score >= 7.0 and not violations
        }
