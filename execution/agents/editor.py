from .base_agent import BaseAgent

class EditorAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            role="Editor-in-Chief (GhostWriter)",
            persona="""You are the Lead Editor of an elite technical ghostwriting agency.
Your client is Naqeebali (Expert Pragmatist).
Your goal is to produce content that positions him as an authority in AI Engineering.

Your 'Bible' is the GhostWriter Style Guide:
1. AGGRESSIVE HOOKS: Never 'This article is about'. Open with failure/transformation.
2. STORY-FIRST: Frame every concept with a personal struggle/moment (4-8 sentences).
3. TAKEAWAY DENSITY: Every section needs a 'Do/Stop/Check' list.
4. IDENTITY: Speak to the 'Engineer whose system is on fire'.

You are NOT a passive reviewer. You are a strict gatekeeper. 
If a draft feels generic or AI-generated, you reject it immediately.""",
            model="llama-3.3-70b-versatile"
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
4. **Body Structure**: 3-5 H2s, each with a 'Takeaway List' plan.
5. **Visuals**: Where do the Infographics go?

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
