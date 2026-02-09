"""
Copywriter Agent - Makes content compelling, persuasive, and memorable.

The difference between content that gets read and content that gets ignored
is often not the IDEAS but the PRESENTATION.

This agent transforms dry, informative content into compelling copy that:
- Hooks readers immediately
- Maintains attention throughout
- Drives action at the end
- Sounds like a human with personality
"""

from .base_agent import BaseAgent
from typing import Dict, List, Optional
import json


class CopywriterAgent(BaseAgent):
    """
    Transforms content into compelling, persuasive copy.

    This is NOT about:
    - Making things "catchy" with clickbait
    - Adding fluff to hit word counts
    - Generic marketing speak

    This IS about:
    - Clarity that respects the reader
    - Hooks that deliver on their promise
    - Voice that feels human and authentic
    - Structure that maintains engagement
    """

    # Copywriting frameworks
    FRAMEWORKS = {
        "aida": {
            "name": "AIDA",
            "structure": ["Attention", "Interest", "Desire", "Action"],
            "description": "Classic persuasion funnel"
        },
        "pas": {
            "name": "PAS",
            "structure": ["Problem", "Agitation", "Solution"],
            "description": "Problem-focused approach"
        },
        "storytelling": {
            "name": "Story Arc",
            "structure": ["Hook", "Context", "Conflict", "Resolution", "Takeaway"],
            "description": "Narrative structure"
        },
        "contrarian": {
            "name": "Contrarian",
            "structure": ["Common Belief", "Why It's Wrong", "Better Way", "Proof", "Action"],
            "description": "Challenge then prove"
        },
        "tutorial": {
            "name": "Tutorial",
            "structure": ["Promise", "Prerequisites", "Steps", "Verification", "Next Steps"],
            "description": "How-to structure"
        }
    }

    # Voice styles
    VOICE_STYLES = {
        "authoritative": {
            "description": "Expert speaking with confidence",
            "traits": ["direct", "confident", "knowledgeable", "decisive"],
            "avoid": ["hedging", "excessive caveats", "passive voice"]
        },
        "conversational": {
            "description": "Smart friend explaining over coffee",
            "traits": ["warm", "relatable", "uses 'you'", "occasional humor"],
            "avoid": ["stiffness", "jargon without explanation", "condescension"]
        },
        "provocative": {
            "description": "Challenging conventional wisdom",
            "traits": ["bold", "questioning", "slightly irreverent", "confident"],
            "avoid": ["being offensive", "shock for shock's sake", "unsubstantiated claims"]
        },
        "analytical": {
            "description": "Data-driven and logical",
            "traits": ["precise", "evidence-based", "structured", "thorough"],
            "avoid": ["emotional appeals", "vague claims", "unsupported opinions"]
        },
        "storyteller": {
            "description": "Engaging narrative voice",
            "traits": ["vivid", "uses examples", "builds tension", "memorable"],
            "avoid": ["being dry", "burying the lede", "excessive abstraction"]
        }
    }

    # Static persona text — stable across calls, ideal for prompt caching
    PERSONA_TEXT = """You are a senior copywriter who makes ideas irresistible.

Your superpower is taking good ideas and making them STICK.

You understand that:
- The first sentence must earn the second
- Every paragraph must justify its existence
- Clarity beats cleverness
- Specifics beat generalities
- Stories beat statistics (but use both)
- The reader's time is sacred

You never:
- Use clickbait that doesn't deliver
- Add fluff to hit word counts
- Sacrifice clarity for style
- Use jargon to sound smart
- Bore the reader with obvious points

You always:
- Lead with the most compelling point
- Use concrete examples
- Write like you talk (but better)
- Respect the reader's intelligence
- End with clear next steps

Your writing should make readers think:
"I couldn't stop reading" not "I had to finish this"."""

    def __init__(self, model: str = "claude-sonnet-4-20250514"):
        """Initialize with Claude for nuanced writing."""
        super().__init__(
            role="Senior Copywriter",
            persona=self.PERSONA_TEXT,
            model=model
        )
        # Pre-built static system prompt for caching
        self._static_system_prompt = (
            f"You are the {self.role}.\nPersona: {self.persona}\n"
        )

    def craft_hook(
        self,
        topic: str,
        main_point: str,
        target_audience: str,
        hook_style: str = "contrarian"
    ) -> Dict:
        """
        Create a compelling opening hook.

        The hook is the most important part - if it fails,
        nothing else matters.
        """
        prompt = f"""TOPIC: {topic}
MAIN POINT: {main_point}
AUDIENCE: {target_audience}
STYLE: {hook_style}

Create 5 different opening hooks for this content. Each should:
1. Be immediately compelling (reader MUST continue)
2. Promise value that the content delivers
3. Fit the style requested
4. Be specific, not generic

Hook styles to try:
- Contrarian: Challenge what reader believes
- Question: Ask something they can't ignore
- Statistic: Lead with surprising data
- Story: Start in the middle of action
- Direct: State the bold claim immediately

Return JSON:
{{
    "hooks": [
        {{
            "type": "contrarian/question/statistic/story/direct",
            "hook": "The actual opening (2-3 sentences max)",
            "why_it_works": "Brief explanation",
            "best_for": "When this hook is most effective"
        }}
    ],
    "recommended": 0,
    "recommendation_reason": "Why this hook is best for this content"
}}"""

        return self.generate(prompt, expect_json=True,
                             system_prompt=self._static_system_prompt)

    def apply_framework(
        self,
        content: str,
        framework: str = "aida",
        preserve_facts: bool = True
    ) -> Dict:
        """
        Restructure content using a copywriting framework.

        Takes existing content and reshapes it for maximum impact.
        """
        framework_config = self.FRAMEWORKS.get(framework, self.FRAMEWORKS["aida"])

        prompt = f"""ORIGINAL CONTENT:
{content}

FRAMEWORK: {framework_config['name']}
STRUCTURE: {' → '.join(framework_config['structure'])}
DESCRIPTION: {framework_config['description']}

Restructure this content using the {framework_config['name']} framework.

Requirements:
1. Keep ALL factual information (don't lose any substance)
2. Reorganize for maximum persuasive impact
3. Each section should flow naturally into the next
4. The reader should feel compelled to continue

Return JSON:
{{
    "restructured_content": "The full rewritten content",
    "section_breakdown": {{
        "{framework_config['structure'][0]}": "What this section accomplishes",
        ...
    }},
    "key_changes": ["What you changed and why"],
    "word_count_original": X,
    "word_count_new": Y
}}"""

        return self.generate(prompt, expect_json=True,
                             system_prompt=self._static_system_prompt)

    def inject_voice(
        self,
        content: str,
        voice_style: str = "conversational",
        intensity: str = "medium"
    ) -> str:
        """
        Transform content to have a distinct voice.

        Takes generic content and gives it personality.
        """
        style_config = self.VOICE_STYLES.get(voice_style, self.VOICE_STYLES["conversational"])

        prompt = f"""ORIGINAL CONTENT:
{content}

TARGET VOICE: {style_config['description']}
TRAITS: {', '.join(style_config['traits'])}
AVOID: {', '.join(style_config['avoid'])}
INTENSITY: {intensity} (how strongly to apply the voice)

Rewrite this content with the specified voice.

The voice should feel:
- Natural, not forced
- Consistent throughout
- Appropriate for the subject matter
- Like a real person wrote it

Keep the substance. Change the style.

Return the rewritten content only."""

        return self.generate(prompt,
                             system_prompt=self._static_system_prompt)

    def strengthen_cta(
        self,
        content: str,
        desired_action: str
    ) -> Dict:
        """
        Strengthen the call-to-action.

        Great content with a weak CTA = wasted opportunity.
        """
        prompt = f"""CONTENT:
{content}

DESIRED ACTION: {desired_action}

Create a compelling call-to-action that:
1. Flows naturally from the content
2. Gives a clear, specific next step
3. Creates urgency without being pushy
4. Tells reader what they'll get

Return JSON:
{{
    "primary_cta": "The main call to action",
    "cta_placement": "Where in the content to place it",
    "supporting_text": "Text that leads into the CTA",
    "urgency_element": "What creates appropriate urgency",
    "value_reminder": "What the reader gets by acting"
}}"""

        return self.generate(prompt, expect_json=True,
                             system_prompt=self._static_system_prompt)

    def eliminate_weak_copy(self, content: str) -> Dict:
        """
        Identify and fix weak copy patterns.

        Common problems:
        - Passive voice
        - Hedge words
        - Vague language
        - Unnecessary qualifiers
        - Weak verbs
        """
        prompt = f"""CONTENT:
{content}

Analyze this content for weak copy patterns and fix them.

Look for:
1. Passive voice → Make active
2. Hedge words (might, maybe, perhaps, somewhat) → Be decisive
3. Vague language (things, stuff, aspects) → Be specific
4. Unnecessary qualifiers (very, really, quite) → Remove
5. Weak verbs (is, was, have) → Use strong verbs
6. Long sentences → Break up
7. Jargon without explanation → Clarify or remove
8. Redundancy → Cut

Return JSON:
{{
    "issues_found": [
        {{
            "type": "passive_voice/hedge_word/vague/etc",
            "original": "The problematic text",
            "fixed": "The improved version",
            "explanation": "Why this is better"
        }}
    ],
    "improved_content": "The full content with all fixes applied",
    "readability_improvement": "How much clearer this is now"
}}"""

        return self.generate(prompt, expect_json=True,
                             system_prompt=self._static_system_prompt)

    def craft_headline(
        self,
        content: str,
        style: str = "benefit"
    ) -> Dict:
        """
        Create compelling headlines.

        Headlines determine if content gets read.
        """
        prompt = f"""CONTENT:
{content}

Create 10 headlines in different styles:
1. Benefit-focused: What the reader gains
2. How-to: Promise to teach
3. Question: Make them curious
4. Number: Specific and scannable
5. Contrarian: Challenge assumptions
6. Urgency: Time-sensitive
7. Curiosity gap: Incomplete info
8. Direct: State the main point
9. Story: Narrative hook
10. Authority: Expert perspective

For each, explain why it works.

Return JSON:
{{
    "headlines": [
        {{
            "style": "benefit/how-to/question/etc",
            "headline": "The actual headline",
            "why_it_works": "Brief explanation",
            "best_platform": "Where this headline works best"
        }}
    ],
    "top_pick": 0,
    "top_pick_reason": "Why this is the best choice"
}}"""

        return self.generate(prompt, expect_json=True,
                             system_prompt=self._static_system_prompt)

    def full_copy_transformation(
        self,
        content: str,
        framework: str = "aida",
        voice: str = "conversational",
        desired_action: str = None
    ) -> Dict:
        """
        Complete copy transformation pipeline.

        Takes raw content and outputs polished, persuasive copy.
        """
        # Step 1: Apply framework
        structured = self.apply_framework(content, framework)

        # Step 2: Inject voice
        voiced_content = self.inject_voice(
            structured.get("restructured_content", content),
            voice
        )

        # Step 3: Eliminate weak copy
        strengthened = self.eliminate_weak_copy(voiced_content)

        # Step 4: Craft headlines
        headlines = self.craft_headline(strengthened.get("improved_content", voiced_content))

        # Step 5: Strengthen CTA if provided
        cta = None
        if desired_action:
            cta = self.strengthen_cta(
                strengthened.get("improved_content", voiced_content),
                desired_action
            )

        return {
            "original_content": content,
            "final_content": strengthened.get("improved_content", voiced_content),
            "headlines": headlines,
            "cta": cta,
            "framework_used": framework,
            "voice_applied": voice,
            "transformations_applied": [
                "framework_restructure",
                "voice_injection",
                "weak_copy_elimination",
                "headline_crafting",
                "cta_strengthening" if cta else None
            ]
        }


# Convenience function
def transform_to_compelling_copy(
    content: str,
    voice: str = "conversational"
) -> str:
    """Quick function to improve content copy."""
    agent = CopywriterAgent()
    result = agent.full_copy_transformation(content, voice=voice)
    return result.get("final_content", content)
