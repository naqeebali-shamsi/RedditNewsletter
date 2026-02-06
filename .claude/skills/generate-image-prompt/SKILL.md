---
name: generate-image-prompt
description: Generates extremely specific ByteByteGo-style image prompts for articles and posts. Essential for creating high-quality visual content.
allowed-tools:
  - read_file
  - write_to_file
---

# /generate-image-prompt [article-slug]

<required_reading>

- @.planning/PROJECT.md
- @.planning/articles/[article-slug].spec.md (or current article context)
- @directives/technical_rules.md
- @directives/framework_rules.md
</required_reading>

<workflow_steps>

1. **Load Article Context**: Read the article/post content or spec to understand the topic, key concepts, and visual narrative
2. **Identify Visual Narrative**: Determine what the diagram should communicate:
   - Main concept or progression
   - Key stages, components, or relationships
   - Contrast or comparison (if applicable)
   - Technical architecture or flow
3. **Generate ByteByteGo Prompt**: Create an extremely specific prompt following the ByteByteGo style requirements
4. **Save Prompt**: Append or save the prompt to the article file or spec
5. **Verify Completeness**: Ensure prompt includes all required elements (style, structure, colors, labels, etc.)
</workflow_steps>

<bytebytego_style_requirements>
The ByteByteGo style is **ESSENTIAL** and must be explicitly specified in every prompt. Key requirements:

**Base Style**:

- High-fidelity ByteByteGo style technical diagram
- Clean white background
- Minimalist, high-signal engineering aesthetic
- High information density
- Professional palette
- No surrounding device frames

**Structure Elements**:

- Title: Clear, descriptive title for the diagram
- Layout: Specify layout type (horizontal progression, vertical flow, node-based, comparison, etc.)
- Stages/Components: Each stage/component must have:
  - Clear label
  - Icon description (simple, clean, technical)
  - Specific visual characteristics
  - Color assignment (if using color coding)

**Color Palette**:

- Professional deep blue (#1E3A8A or similar)
- Slate gray (#475569 or similar)
- Vibrant emerald green (#10B981 or similar) for emphasis/highlights
- Use color strategically to show progression, importance, or contrast
- Maintain professional, consistent palette

**Visual Hierarchy**:

- High information density without clutter
- Clear visual flow (left-to-right, top-to-bottom, or circular)
- Balanced composition
- Self-explaining (diagram should communicate without additional text)

**Prohibited Elements**:

- Device frames (no phone/tablet/computer mockups)
- Decorative elements that don't add information
- Low information density designs
- Generic or vague descriptions
</bytebytego_style_requirements>

<prompt_structure_template>

```
A high-fidelity ByteByteGo style technical diagram on a clean white background. Title: '[Article Title or Key Concept]'. The diagram shows [layout type: horizontal progression/vertical flow/node-based architecture/comparison/etc.].

[Stage/Component 1]: '[Label]' ([Icon description], labeled '[Key Detail]').
[Stage/Component 2]: '[Label]' ([Icon description], labeled '[Key Detail]').
[Stage/Component 3]: '[Label]' ([Icon description], labeled '[Key Detail]').

Use [color scheme description] for [purpose: progression/emphasis/contrast]. Minimalist, high-signal engineering aesthetic, high information density. No surrounding device frames.
```

</prompt_structure_template>

<prompt_generation_guidelines>

1. **Analyze Article Content**:
   - Extract the core technical concept
   - Identify progression, stages, or components
   - Note any contrasts or comparisons
   - Understand the visual narrative

2. **Determine Layout**:
   - **Horizontal Progression**: For evolution, stages, or time-based concepts
   - **Vertical Flow**: For hierarchical structures or top-down processes
   - **Node-Based**: For architectures, systems, or relationships
   - **Comparison**: For contrasting approaches or before/after
   - **Circular/Flow**: For cycles or iterative processes

3. **Specify Each Element**:
   - Give each stage/component a clear label
   - Describe the icon (simple, technical, clean)
   - Include key details as labels
   - Assign colors purposefully

4. **Create Visual Narrative**:
   - The diagram should tell a story or explain a concept
   - Use color to show progression, importance, or contrast
   - Ensure visual flow guides the eye logically

5. **Be Extremely Specific**:
   - No vague descriptions like "some icons" or "various elements"
   - Specify exact number of stages/components
   - Describe icons clearly (e.g., "Simple icon", "Chaos/Sketchy cloud icon", "Sharp, structured node-based icon")
   - Include exact labels and text
   - Specify color assignments for each element
</prompt_generation_guidelines>

<example_prompt>

```
A high-fidelity ByteByteGo style technical diagram on a clean white background. Title: 'The Evolution of the AI Developer'. The diagram shows a three-stage horizontal progression.

Stage 1: 'Prompt Engineer' (Simple icon, labeled 'Input -> Output'). Stage 2: 'Vibe Coder' (Chaos/Sketchy cloud icon, labeled 'Iterative Chat'). Stage 3: 'Agentic Architect' (Sharp, structured node-based icon, labeled 'Systems, Memory, Orchestration').

Use professional deep blue and slate colors for the first two stages, and a vibrant emerald green for the final 'Agentic Architect' stage. Minimalist, high-signal engineering aesthetic, high information density. No surrounding device frames.
```

</example_prompt>

<integration_with_workflow>
This skill should be called:

- **After drafting**: Once article content is complete, generate the image prompt
- **As part of asset generation**: In the final workflow step alongside article completion
- **Before publishing**: Ensure every article has a hero image prompt

The generated prompt should be:

- Saved in the article file (markdown frontmatter or dedicated section)
- Included in the article spec if using specs
- Referenced in the final article output
</integration_with_workflow>

<output_format>
The prompt should be:

1. Saved to the article file in a dedicated section or frontmatter
2. Formatted as a code block for easy copying
3. Include a note: "Use this prompt with DALL-E, Midjourney, or other AI image generators"
4. Be ready for direct use—no additional editing needed

Example article integration:

```markdown
---
title: "Article Title"
image_prompt: |
  A high-fidelity ByteByteGo style technical diagram...
---
```

</output_format>

<constraints>
- **ByteByteGo style is mandatory**—every prompt must explicitly specify this
- Prompts must be extremely specific—no generic descriptions
- Every article/post must have an image prompt (no exceptions)
- Prompts must be self-contained and ready for direct use
- Visual narrative must align with article content and key takeaways
</constraints>
