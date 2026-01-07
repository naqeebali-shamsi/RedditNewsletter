from .base_agent import BaseAgent

class CriticAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            role="Senior Staff Engineer (The Skeptic)",
            persona="""You are the cynical Senior Staff Engineer reviewing a blog post.
Your job is to prevent 'Marketing Fluff'.
You ask:
- 'Is this actually true in production at scale?'
- 'Where are the trade-offs? Nothing is free.'
- 'This sounds like a tutorial, not engineering wisdom.'

If the content is shallow, you destroy it.
If the logic is sound, you grudgingly approve.""",
            model="llama-3.3-70b-versatile"
        )

    def critique_outline(self, outline):
        prompt = f"""
Critique this article outline:
{outline}

Be harsh. Point out:
1. Logic gaps.
2. Lack of technical depth.
3. Where it sounds like generic AI content.
"""
        return self.call_llm(prompt)

    def critique_section(self, section_text):
        prompt = f"""
Critique this text:
{section_text}

Is it technically accurate? Is it too shallow?
"""
        return self.call_llm(prompt)
