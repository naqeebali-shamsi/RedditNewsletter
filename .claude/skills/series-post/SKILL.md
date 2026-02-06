---
name: series-post
description: Generates ultra-high-quality, Byte-sized LinkedIn posts for the 30-day challenge using the "Architected Writing" framework.
allowed-tools:
  - read_file
  - write_to_file
  - generate_image
---

# /series:post [day_number]

<required_reading>
- @.planning/PROJECT.md
- @.planning/SERIES_MASTER.md
- @.planning/STATE.md
- @templates/article_spec.md
- @directives/writing_rules.md
- @directives/framework_rules.md
- @directives/technical_rules.md
</required_reading>

<writing_framework_architected>
1. **The Contrast Hook (Status Quo vs. Pro)**:
   - Challenge a common industry belief or failure.
   - Formula: [Common Amateur Mistake] vs. [Senior Architectural Reality].
   - Example: "Prompting is for prototypes. Orchestration is for production."

2. **The "Human Variable" (War Story)**:
   - Insert 1 sentence of authentic struggle. Use keywords like "I built," "I broke," "Pager duty," "Context leak."
   - Target: 80% Technical, 20% Narrative.

3. **Takeaway Density (Signal-to-Noise)**:
   - Max 3 lines per paragraph.
   - Max 2 sentences per bullet.
   - Format: **[The Mechanic]**: [The Consequence/Value].
   - No fluff. If the sentence starts with "In this post," "Furthermore," or "Transitioning to," CUT IT.

4. **The "Tradeoff" (Senior Perspective)**:
   - Every technical decision has a cost. Stat it.
   - Example: "Tradeoff: Higher initial design latency vs. zero production hallucinations."

5. **Visual Anchor**:
   - Generate a diagram prompt that is "Self-Explaining."
   - Style: Minimalist, ByteByteGo-inspired, high information density, professional palette.
</writing_framework_architected>

<workflow_steps>
0. **Session Sync (Living Logic)**: Review the current session context and the `.planning/PROJECT.md` for any "Self-Annealing" updates or recent user feedback that modifies writing rules.
1. **Context Extraction**: Pull the day's topic from `SERIES_MASTER.md`.
2. **Quality Linter Phase**: Before writing, define the "Status Quo" we are challenging and the "War Story" we are weaving in, ensuring all recent constraints (like 'No Meta-Labeling') are enforced.
3. **Drafting**: Execute the Architected Writing framework.
4. **Self-Correction**: Run the draft against the <writing_quality_check> markers.
5. **Asset Generation**:
   - **REQUIRED**: Generate Hero Image prompt using `/generate-image-prompt` skill
   - Prompt must follow ByteByteGo style requirements (see directives/technical_rules.md)
   - Prompt must be extremely specific and ready for direct use
   - Save prompt in article file and save final MD file in `articles/day-[day_number].md`
</workflow_steps>

<writing_quality_check>
  <checker tag="contrast_hook">Does the hook challenge a status quo or common misconception?</checker>
  <checker tag="sentence_compression">Is any paragraph longer than 3 lines? (If yes, split or delete).</checker>
  <checker tag="tradeoff_enforcement">Is the opportunity cost or architectural tradeoff stated?</checker>
  <checker tag="authentic_voice">Does this sound like a battle-scarred engineer (Expert Pragmatist) or an AI bot?</checker>
  <checker tag="no_meta_labels">Does the post avoid functional labels like "### The Hook" in the final output?</checker>
</writing_quality_check>

<constraints>
- **Length**: Max 300 words (Byte-sized).
- **Tone**: Authentic, technical, no academic fluff.
- **Style**: ByteByteGo/System Design heavy.
- **No Meta-Labeling**: DO NOT use functional labels in headers (e.g., "### The Hook", "### The Pivot"). Headers should be narrative or informative, and contributes to the knowledge.
</constraints>
