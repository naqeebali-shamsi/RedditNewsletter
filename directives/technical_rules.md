# Directive: Technical Rules

> Standards for technical accuracy, research methodology, asset generation, SEO, and content maintenance.

## Technical Accuracy Standards

### Research Requirements

- Find 3-5 specific technical facts per article
- Identify 1-2 real-world "failure modes" related to the topic
- Include relevant performance benchmarks when available
- Use `browser_subagent` for complex technical documentation

### Code Snippet Standards

- Code must be accurate and runnable
- Include context and explanation
- Show real-world patterns, not toy examples
- Use proper syntax highlighting
- **For LinkedIn posts (300 words)**: Use minimal code (1-3 lines) or link to full examples. Prioritize concepts over code.
- **For articles**: Full code examples with explanation are appropriate

### Technical Depth

- Target audience: Mid-Senior Engineers and Tech Leads
- Balance between accessibility and depth
- Include practical "gotchas," not just theory
- Maintain "Expert Pragmatist" lens

---

## Research Methodology

### Search Strategy

1. **Observer Voice Sources**: Reddit/HN/Twitter discussions
2. **Official Documentation**: GitHub issues, official docs for technical edge cases
3. **Real-World Patterns**: Production systems, war stories, failure modes

### Handling Topics Without First-Hand Experience

- **First-hand experience** (preferred): Use "I built," "I broke" authentically
- **Second-hand (team/company)**: Use "We built," "We encountered," "Our team faced"
- **Research-based**: Use "In production systems," "Common failure mode," acknowledge sources
- **Never**: Fake first-person experience -- authenticity over forced personalization
- If writing about technology you haven't used, be transparent about research sources

### Synthesis Requirements

Extract and document:

- 3-5 specific technical facts
- 1-2 real-world "failure modes"
- Relevant performance benchmarks
- Practical "gotchas" and edge cases

### Research Tools

- Use `browser_subagent` for complex technical documentation
- Use `search_web` for finding discussions and examples
- Maintain "Expert Pragmatist" lens -- look for practical insights, not just theory

---

## Asset Generation

### Image Prompts (ESSENTIAL)

- **Style**: **ByteByteGo style is MANDATORY** -- every prompt must explicitly specify "high-fidelity ByteByteGo style technical diagram"
- **Base Requirements**:
  - Clean white background
  - Minimalist, high-signal engineering aesthetic
  - High information density
  - Professional palette (deep blue, slate, emerald green for emphasis)
  - No surrounding device frames
- **Structure**: Must include:
  - Title: Clear, descriptive title for the diagram
  - Layout type: Horizontal progression, vertical flow, node-based, comparison, etc.
  - Stages/Components: Each with label, icon description, and key details
  - Color scheme: Purposeful color assignment (progression, emphasis, contrast)
- **Purpose**: Self-explaining diagrams that enhance understanding
- **Format**: Extremely detailed prompts for AI image generation (DALL-E, Midjourney, etc.)

### Diagram Requirements

- **Every article/post MUST include a hero image prompt** (no exceptions)
- Diagrams should be "Self-Explaining" (communicate without additional text)
- High information density without clutter
- Professional aesthetic consistent with ByteByteGo style
- Visual narrative must align with article content and key takeaways

### Prompt Quality Standards

- **Extremely specific**: No vague descriptions
- **Complete**: Includes all required elements (style, structure, colors, labels)
- **Ready for use**: Can be copied directly to image generators without editing
- **Aligned with content**: Visual narrative matches article's core concept

---

## SEO & Metadata

### Article Metadata

- **Target Audience**: Mid-Senior Engineers (default, can be customized)
- **Status Tracking**: Draft-In-Progress, Review, Published
- **SEO Keywords**: Relevant technical terms and concepts
- **Format**: Markdown with frontmatter or structured metadata

### Content Structure

- Article in Markdown format
- Includes context-specific image prompts
- Contains at least 3 high-value technical takeaways (see `directives/writing_rules.md` for criteria)
- Passes "Voice Check" (minimize passive voice, no academic fluff)
- Includes SEO-optimized metadata

---

## Content Maintenance

- **Update articles when technical information changes**: Keep content current with latest best practices, tool versions, and patterns
- **Acknowledge corrections transparently**: If you discover an error, add a correction note with date
- **Date articles and note last updated**: Include publication date and last updated date in metadata
- **Version awareness**: When writing about tools/libraries, specify versions if relevant
- **Link rot prevention**: Periodically check and update external links
- **Deprecation notices**: If content becomes outdated, add a note or consider archiving

---

## Integration Standards

### Tech Stack

- **Structure**: 3-layer orchestration (directives / orchestration / execution)
- **Research**: Browser Subagent, search_web
- **Formatting**: Markdown
- **Generation**: Claude Opus / Sonnet

### File Organization

- Article specs: `.planning/articles/[topic-slug].spec.md`
- Final articles: `articles/[slug].md` or `articles/day-[number].md`
- Templates: `templates/article_spec.md`, `templates/plan_template.xml`
- Directives: `directives/`
- Execution scripts: `execution/`

---

## References

- Writing voice & style: `directives/writing_rules.md`
- 5-Pillar framework: `directives/framework_rules.md`
