from .base_agent import BaseAgent
from execution.config import config

class EditorAgent(BaseAgent):
    """
    Editor-in-Chief with adversarial-panel-informed quality standards.

    Updated with insights from:
    - Apple (simplicity, voice consistency)
    - Google (SEO, structure)
    - Serviceplan (creative excellence)
    - Compose.ly (production quality)
    """

    # Quality checklist items
    QUALITY_CHECKLIST = [
        ("HOOK_STRENGTH", "Does it hook in first 10 words?"),
        ("SPECIFICITY", "Are there 3+ specific numbers/metrics?"),
        ("MEMORABLE_MOMENT", "Is there 1+ quotable/screenshot-worthy line?"),
        ("NO_PLACEHOLDERS", "No '...' or incomplete sections?"),
        ("NO_HTML_ARTIFACTS", "No raw HTML or markdown artifacts?"),
        ("VOICE_CONSISTENT", "Same voice throughout (no tonal jumps)?"),
        ("CTA_SPECIFIC", "Is the CTA specific, not generic?"),
        ("NO_TEMPLATE_PHRASES", "No 'What's been your experience?' or similar?"),
    ]

    def __init__(self):
        super().__init__(
            role="Editor-in-Chief (GhostWriter)",
            persona="""You are the Lead Editor of an elite technical ghostwriting agency
that has been schooled by the best: Apple's editorial standards, Google's SEO rigor,
Serviceplan's creative excellence, and Compose.ly's production quality.

Your QUALITY GATES (Non-negotiable):

1. **HOOK TEST**: First 10 words must create curiosity or tension
   - FAIL: "In this article..." / "Today we'll explore..."
   - PASS: "I burned $4K on a mistake everyone makes."

2. **SPECIFICITY TEST**: At least 3 specific numbers/metrics
   - FAIL: "It was slow" / "We saved money"
   - PASS: "47ms p99 latency" / "$4,200 in compute costs"

3. **MEMORABLE TEST**: At least 1 quotable line
   - Something a reader would screenshot and share

4. **COMPLETENESS TEST**: No placeholders, no "..."
   - Every section fully written
   - Practical Implications must have REAL content

5. **TECHNICAL CLEANLINESS**: No HTML artifacts, no metadata leaks
   - No "<!-- SC_OFF -->" or "<div class="
   - No "Contains X keywords" internal scoring

6. **VOICE CONSISTENCY**: One voice throughout
   - No tonal jumps between sections
   - Same person wrote the whole thing

7. **CTA QUALITY**: Specific, urgent, value-offering
   - FAIL: "What's been your experience?"
   - PASS: "What's the costliest GPU mistake you've made? Mine: $4K on a 3080 that couldn't handle our batch sizes."

You are NOT a passive reviewer. You are the final quality gate.
If content feels AI-generated, templated, or mediocre - REJECT IT.""",
            model=config.models.DEFAULT_EDITOR_MODEL
        )

    def create_outline(self, topic_signal):
        """Create a GhostWriter-standard outline."""
        prompt = f"""
Plan a Medium article for Naqeebali based on this signal:
{topic_signal}

Follow the 'GhostWriter Style Guide':
1. **Hook Options**: Generate 5 distinct hooks (Transformation, Tension, Mistake).
2. **The Opening Scene**: Outline the personal struggle (Time, Place, Emotion).
3. **The Pivot**: Connect story to reader pain ("You're not alone").
4. **Body Structure**: 3-5 H2 sections. Each section MUST have an explicit ## header.
5. **Visuals**: Where do the Infographics go?

CRITICAL: The outline MUST specify H2 section headers that will appear in the final article.
Format each body section as:

## [Section Title]
- Key point 1
- Key point 2
- Takeaway

The final article MUST contain these ## headers verbatim. Minimum 3 H2 sections required.

Output a structured outline.
"""
        return self.call_llm(prompt)

    def review_draft(self, draft, section_name):
        """Review a draft against the GhostWriter Non-Negotiables."""
        prompt = f"""
Review this draft for compliance with the GhostWriter Style Guide:

{draft}

Checklist:
[ ] Does it start with a specific Personal Story (not generic)?
[ ] Is the Voice authentic ('I built', 'I broke')?
[ ] Are there bold 'Takeaway Lists' in every section?
[ ] Is the Hook aggressive (no 'In this article...')?
[ ] Are there clear visual placeholders?

If ANY are missing, return 'REVISE: [Specific Instructions]'.
If it meets the elite standard, return 'APPROVED'.
"""
        return self.call_llm(prompt)

    def compile_final_article(self, hook, intro, sections, diagrams):
        # ... (Existing compile logic)
        content = f"{hook}\n\n{intro}\n\n"
        for i, section in enumerate(sections):
            content += f"{section}\n\n"
            if i < len(diagrams):
                content += f"![Infographic: {diagrams[i]['prompt']}]({diagrams[i]['url']})\n\n"
        return content
