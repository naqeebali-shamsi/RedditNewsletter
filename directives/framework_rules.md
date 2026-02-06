# Directive: Architected Writing Framework

> The 5-Pillar framework for producing high-quality technical content. Apply these pillars to every article and post.

## The 5-Pillar Framework

### 1. The Contrast Hook (Status Quo vs. Pro)

- **Purpose**: Challenge a common industry belief or failure
- **Formula**: [Common Amateur Mistake] vs. [Senior Architectural Reality]
- **Example**: "Prompting is for prototypes. Orchestration is for production."
- **Guidelines**:
  - Challenge status quo only when you have evidence or experience to back it up
  - If status quo is correct, find different angle: implementation details, edge cases, or common misunderstandings
  - Avoid contrarianism for its own sake -- authenticity over controversy
- **Check**: Does the hook challenge a status quo or common misconception? (Only if supported by evidence)

### 2. The "Human Variable" (War Story)

- **Purpose**: Insert authentic struggle and human experience
- **Keywords**: "I built," "I broke," "Pager duty," "Context leak"
- **Target**: 80% Technical, 20% Narrative
  - Narrative: 1-2 sentences war story + technical context
  - Use appropriate voice based on experience level (see `directives/writing_rules.md`)
- **Check**: Does this sound like a battle-scarred engineer or an AI bot?

### 3. Takeaway Density (Signal-to-Noise)

- **Paragraph Rules** (varies by content type):
  - LinkedIn Posts: Max 3 lines (~120-150 words, 2-3 sentences)
  - Articles: Max 4-5 lines (~150-200 words, 3-4 sentences)
  - Long-form: Max 6 lines (~200-250 words), allow longer for flow
  - Max 2 sentences per bullet
  - Format: **[The Mechanic]**: [The Consequence/Value] (for takeaway bullets, not headers)
- **Fluff Removal**: If sentence starts with "In this post," "Furthermore," or "Transitioning to," CUT IT
- **Check**: Is any paragraph longer than the limit for this content type? (If yes, split or delete)

### 4. The "Tradeoff" (Senior Perspective)

- **Purpose**: Acknowledge technical decisions and their costs when meaningful
- **Format**: State the opportunity cost or architectural tradeoff when it exists
- **Note**: Not every decision requires a tradeoff -- some approaches are clearly superior
- **Example**: "Tradeoff: Higher initial design latency vs. zero production hallucinations"
- **Check**: If tradeoffs exist, are they stated? (Avoid forcing tradeoffs when none exist)

### 5. Visual Anchor

- **Purpose**: Generate diagram prompts that are "Self-Explaining"
- **Style**: **ByteByteGo style is MANDATORY** -- must explicitly specify "high-fidelity ByteByteGo style technical diagram"
- **Requirements**:
  - Every article/post MUST include a hero image prompt (no exceptions)
  - Prompts must be extremely specific and ready for direct use
  - See `directives/technical_rules.md` for full ByteByteGo style specifications
- **Aesthetic**: Minimalist, high information density, professional palette (deep blue, slate, emerald green)

---

## Quality Check Framework

### Writing Quality Checklist

Run this checklist against every draft before finalizing:

1. **Contrast Hook**: Does the hook challenge a status quo or common misconception? (Only if you have evidence -- avoid contrarianism for its own sake)
2. **Sentence Compression**: Is any paragraph longer than the limit for this content type? (LinkedIn: 3 lines, Articles: 4-5 lines, Long-form: 6 lines -- if yes, split or delete)
3. **Tradeoff Enforcement**: If tradeoffs exist, are they stated? (Don't force tradeoffs when none exist)
4. **Authentic Voice**: Does this sound like a battle-scarred engineer (Expert Pragmatist) or an AI bot?
5. **No Meta-Labels**: Does the post avoid functional labels like "### The Hook" in the final output?

---

## Workflow Integration

### Session Sync (Living Logic)

Before starting any writing session:

- Review current session context and `.planning/PROJECT.md` for any "Self-Annealing" updates
- Check for recent user feedback that modifies writing rules
- Ensure all recent constraints (like 'No Meta-Labeling') are enforced

### Quality Linter Phase

Before writing, define:

- The "Status Quo" we are challenging
- The "War Story" we are weaving in
- All recent constraints that must be enforced

### Self-Correction

Run the draft against the writing quality check markers before finalizing. This is the final gate before content is considered ready.

---

## References

- Writing voice & style: `directives/writing_rules.md`
- Technical standards: `directives/technical_rules.md`
