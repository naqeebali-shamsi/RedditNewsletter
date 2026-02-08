# Day 5: Prompt Engineering for Logic — Building Policies, Not Prose

![Prompting for Logic: The Policy Framework](assets/day-5.png)

**If you are writing long, emotional prompts to get your agent to work, you are still vibe coding. In production, we write Policies.**

I spent months "shouting" at my agents in all-caps: "DO NOT DO THIS" or "BE VERY CAREFUL." It worked maybe 60% of the time. Then I shifted to a **Policy-based approach.**

Prompt Engineering for agents isn't about "Creative Writing." It's about **Logical Governance.**

---

## The Blueprint of a Binary Prompt

1. **Constraints as Logical Gates**
   Instead of saying "Don't use external libraries," define it as a rule within a structured `<constraints>` tag. Treat your prompt like a configuration file, not a letter.

2. **The "Definition of Done" (DoD) Pattern**
   Every agent needs a success signal. By explicitly defining the DoD (e.g., "A valid JSON file in the `/output` folder"), you give the agent a deterministic goal to "Anneal" against.

3. **Standard Operating Procedures (SOPs)**
   The most effective prompts are just Markdown-based SOPs. They define the "Persona," the "Process," and the "Safety Rails." This ensures that even if the model varies slightly, the logic remains rigid.

4. **Few-Shot Logical Examples**
   Don't just describe the task; show the reasoning. Providing 2-3 examples of "Input -> Thought -> Correct Action" is the single fastest way to lock in complex behavior.

---

## The Tradeoff

**Rigid Control vs. Creative Problem-Solving.** The stricter your prompt-policy, the more predictable the agent becomes—but the less it will surprise you with "Clever" solutions. For production systems serving customers, predictability is the only thing that matters.

**Day 5 of 30: Moving from Prose to Policies.**

**The Guardrails:** Do you use structured tags (XML/Markdown) in your prompts, or do you stick to plain text? How do you enforce strict logic? 👇

#AgenticAI #PromptEngineering #SystemDesign #SoftwareEngineering #TheWritingStack #30DayChallenge

---
**Hero Image Prompt**:
> A ByteByteGo style diagram comparing 'Prose' vs 'Policy'. Left side: 'Prose Prompt' (messy text, fuzzy edges). Right side: 'Policy Prompt' (Structured XML tags, lock icon, clear boundaries). Arrow travels from left to right labeled 'Structuring Logic'. Professional aesthetic.
