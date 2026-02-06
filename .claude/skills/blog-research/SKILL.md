---
name: blog-research
description: Performs deep technical research on a topic to find metrics, war stories, and edge cases.
allowed-tools:
  - search_web
  - browser_subagent
  - read_url_content
---

# /blog:research

<required_reading>
- @.planning/PROJECT.md
- @.planning/STATE.md
- @.planning/articles/[current-article].spec.md
</required_reading>

<workflow_steps>
1. **Identify Gaps**: Analyze the current article spec to see what technical details or metrics are missing.
2. **Search Strategy**:
    - Search for Reddit/HN/Twitter discussions (Observer voice source).
    - Search for official docs or GitHub issues for technical edge cases.
3. **Synthesize**: extract:
    - 3-5 specific technical facts.
    - 1-2 real-world "failure modes" related to the topic.
    - Any relevant performance benchmarks.
4. **Update Spec**: Append the findings to the article spec file under a `<research_data>` tag.
</workflow_steps>

<constraints>
- Use `browser_subagent` for complex technical documentation.
- Maintain the "Expert Pragmatist" lens—look for practical "gotchas," not just theory.
</constraints>
