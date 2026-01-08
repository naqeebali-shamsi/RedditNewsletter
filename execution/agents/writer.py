from .base_agent import BaseAgent

class WriterAgent(BaseAgent):
    """
    Senior Technical Ghostwriter with consensus-driven quality standards.

    Updated with recommendations from adversarial expert panel:
    - Kill generic openers and weak CTAs
    - Demand specificity over vagueness
    - Require memorable moments, not templates
    - Enforce data/metrics presence
    """

    # KILL PHRASES - Never use these
    FORBIDDEN_PHRASES = [
        "What's been your experience?",  # Weak CTA
        "This aligns with what I'm seeing",  # Template phrase
        "In this article",  # Boring opener
        "As AI engineering matures",  # Cliche closer
        "Drop a comment below",  # Weak CTA
        "soul-crushing",  # Melodrama
        "game-changer",  # Buzzword
        "paradigm shift",  # Corporate speak
        "It goes without saying",  # Filler
        "In today's fast-paced world",  # Generic opener
    ]

    # STRONG ALTERNATIVES
    HOOK_PATTERNS = [
        "I burned $X on [mistake]. Here's what I wish someone told me:",
        "Everyone's talking about [trend]. Nobody's talking about [reality].",
        "[Specific number] engineers hit this wall. [Specific number] know how to fix it.",
        "The [tool/approach] that saved us [specific outcome]:",
        "Why I stopped [common practice] and started [better approach]:",
    ]

    CTA_PATTERNS = [
        "Share your worst [specific experience] - I'll go first: [your example]",
        "If you've hit [specific problem], reply with [specific ask]",
        "One question: [specific, answerable question about reader's situation]",
        "The fix that worked for us: [specific action]. What's yours?",
    ]

    def __init__(self):
        super().__init__(
            role="Senior Technical Ghostwriter",
            persona="""You are an elite technical ghostwriter who has studied the best content from:
- Digital Commerce Partners (conversion-focused)
- Apple (elegant simplicity)
- Zomato/Swiggy (personality and wit)

Your writing MUST:
1. **Hook in 10 words**: Reader decides to stay or scroll in the first line. Make it count.
2. **Specificity over vagueness**: "$10,000 bill" beats "significant cost". "3 weeks" beats "a while".
3. **One memorable moment per piece**: Something quotable, screenshot-worthy.
4. **REAL data only**: If you cite a number, it must come from the source material. NO FABRICATED STATS.
5. **CTAs that convert**: Specific, urgent, value-offering. Never "What do you think?"

CRITICAL - TECHNICAL HONESTY:
- NEVER invent statistics ("40% improvement", "3x faster") unless from source material
- NEVER claim causation without mechanism ("X improved accuracy" - how exactly?)
- NEVER use phantom evidence ("studies show", "research indicates") without citation
- If you don't have data, write with conviction without numbers. Strong opinion > fake stat.
- Code examples MUST be syntactically correct and runnable

FORBIDDEN:
- Generic openers ("In this article...")
- Weak CTAs ("What's been your experience?")
- Template phrases that appear in every post
- Placeholder text ("...")
- Melodrama ("soul-crushing", "game-changer")
- Fabricated percentages or metrics
- "Studies show" without specific source

You write for cynical senior engineers who can smell AI-generated content from a mile away.""",
            model="llama-3.1-8b-instant"
        )

    def write_section(self, section_plan, critique=None, source_type: str = "external"):
        """
        Write a section based on the plan.

        Args:
            section_plan: The outline/plan for the section
            critique: Optional critique to address
            source_type: 'external' (Reddit/observer voice) or 'internal' (GitHub/owner voice)
        """
        # Voice instruction based on source type
        if source_type == "internal":
            voice_instruction = """
6. **VOICE** (Practitioner Owner - This is YOUR work):
   - Use "I", "we", "our" naturally for ownership
   - Share YOUR specific moments, failures, wins
   - Conversational but technical
   - Allowed: 1 moment of dry wit (earned, not forced)
"""
        else:
            # Default to external (observer voice) - NEVER claim ownership
            voice_instruction = """
6. **VOICE** (Journalist Observer - You did NOT build this):
   - CRITICAL: You are REPORTING on others' work, not claiming it as yours
   - FORBIDDEN: "I built", "we created", "our team", "my approach"
   - USE INSTEAD: "teams found", "engineers discovered", "this approach", "the implementation"
   - You CAN use "I" for observations: "I noticed", "I've been tracking", "I find this interesting"
   - Conversational but technical
   - Allowed: 1 moment of dry wit (earned, not forced)
"""

        prompt = f"""
Draft the following section based on the plan:
{section_plan}

**QUALITY REQUIREMENTS (Non-negotiable)**:

1. **HOOK** (First 10 words):
   - Create curiosity gap OR promise transformation OR make a bold claim
   - BAD: "In this section, we'll explore..."
   - GOOD: "Teams wasted 3 months on this approach. Here's why."

2. **SPECIFICITY**:
   - Replace vague words with numbers: "fast" → "47ms p99 latency"
   - Replace abstractions with examples: "performance issues" → "3-second page loads on mobile"

3. **ONE MEMORABLE MOMENT**:
   - A line someone would screenshot and share
   - A metaphor that clicks
   - A counterintuitive insight

4. **STORY ELEMENT** (2-4 sentences):
   - A specific moment (time, place, emotion)
   - What went wrong or what was learned
   - NOT generic ("engineers often face challenges...")

5. **ACTIONABLE TAKEAWAY**:
   - What should the reader DO differently after reading this?
   - Be specific: tool names, code patterns, mental models
{voice_instruction}
**FORBIDDEN** (Instant rejection):
- "What's been your experience?"
- "In this article..."
- "As AI engineering matures..."
- "..." (placeholder text)
- Same structure as every other section
"""
        if critique:
            prompt += f"""

**REVISION CONTEXT**:
The previous version was rejected. Address these issues explicitly:
{critique}

Show clear improvement in this revision.
"""

        return self.call_llm(prompt)

    def write_linkedin_post(self, signal_data: dict, source_type: str = "external") -> str:
        """
        Generate a LinkedIn post from signal data.

        Args:
            signal_data: Dict with title, content, url, reasoning
            source_type: 'external' (observer voice) or 'internal' (owner voice)
        """
        title = signal_data.get('title', '')
        content = signal_data.get('content', '')[:500]
        url = signal_data.get('url', '')
        reasoning = signal_data.get('reasoning', '')

        voice_instruction = ""
        if source_type == "external":
            voice_instruction = """
VOICE: Journalist Observer (you didn't build this, you're reporting on it)
- Use: "teams found", "engineers discovered", "one approach"
- Avoid: "we built", "our team", "I created"
"""
        else:
            voice_instruction = """
VOICE: Practitioner Owner (this is your experience)
- Use: "I", "we", "our team"
- Share personal wins and failures
"""

        prompt = f"""Write a LinkedIn post about this signal.

SIGNAL:
Title: {title}
Context: {content}
Why it matters: {reasoning}
Source: {url}

{voice_instruction}

**LINKEDIN POST REQUIREMENTS**:

1. **HOOK** (Line 1):
   - Stop the scroll. Create curiosity or tension.
   - EXAMPLES that work:
     - "I burned $4,000 on a GPU that's now paperweight."
     - "Everyone's shipping RAG. Nobody's shipping RAG that works."
     - "The 3am Slack message that changed how I think about LLMs:"

2. **BODY** (3-5 short paragraphs):
   - One insight per paragraph
   - Specific > vague (numbers, tools, outcomes)
   - Break up text for mobile reading

3. **CTA** (Final line):
   - Specific and engaging
   - GOOD: "What's the most expensive mistake you've made with GPUs? Mine cost $4K and 3 months."
   - BAD: "What's been your experience with this?"

4. **HASHTAGS** (3-4 max):
   - Vary them! Not the same 4 every post.
   - Mix broad (#AI) with specific (#RAGPipelines)

**FORBIDDEN**:
- "🚀 Interesting insight from r/..."
- "This aligns with what I'm seeing..."
- "What's been your experience?"
- Same hashtags as every other post

Output ONLY the LinkedIn post. No meta-commentary."""

        return self.call_llm(prompt)

    def write_medium_article(self, signal_data: dict, source_type: str = "external") -> str:
        """
        Generate a Medium article from signal data.

        Args:
            signal_data: Dict with title, content, url, reasoning
            source_type: 'external' (observer voice) or 'internal' (owner voice)
        """
        title = signal_data.get('title', '')
        content = signal_data.get('content', '')
        url = signal_data.get('url', '')
        reasoning = signal_data.get('reasoning', '')

        voice_instruction = ""
        if source_type == "external":
            voice_instruction = """
VOICE: Technology journalist sharing insights from the community
- You are an OBSERVER, not the builder
- Use: "teams found", "engineers discovered", "the implementation"
- Avoid: "we built", "our approach", "I created"
- You can use "I" for observations: "I noticed", "I've been tracking"
"""
        else:
            voice_instruction = """
VOICE: Practitioner sharing firsthand experience
- Use "I", "we", "our" naturally
- Share specific moments, failures, wins
- This is YOUR story
"""

        prompt = f"""Write a Medium article about this signal.

SIGNAL:
Title: {title}
Full Context: {content}
Why it matters: {reasoning}
Reference: {url}

{voice_instruction}

**MEDIUM ARTICLE STRUCTURE**:

1. **TITLE** (H1):
   - Create curiosity or promise value
   - GOOD: "Why I Stopped Using RAG (And What I Use Instead)"
   - BAD: "An Overview of RAG Systems in Production"

2. **OPENING** (First 3 paragraphs):
   - Hook: Specific moment, failure, or counterintuitive claim
   - Context: Why this matters NOW
   - Promise: What reader will learn

3. **BODY** (3-5 H2 sections):
   - Each section: ONE clear idea
   - Include: Specific examples, numbers, code if relevant
   - End each section with actionable insight

4. **PRACTICAL IMPLICATIONS** (Required section):
   - NOT placeholder "..."
   - Specific actions for: individual engineers, teams, and industry
   - Real tools, approaches, mental models

5. **CONCLUSION**:
   - Summarize the one thing to remember
   - Forward-looking but grounded
   - Specific CTA (not "drop a comment")

**QUALITY GATES**:
- [ ] No placeholder text ("...")
- [ ] No generic phrases ("As AI matures...")
- [ ] At least 3 specific numbers/metrics
- [ ] At least 1 memorable/quotable line
- [ ] Practical Implications fully written (not "...")
- [ ] CTA is specific, not generic

Output ONLY the article in Markdown. No meta-commentary."""

        return self.call_llm(prompt)
