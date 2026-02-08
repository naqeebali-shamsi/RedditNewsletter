---
name: tws-plan-article
description: Generates XML task plan for an article, creating structured plan in PLAN.md.
allowed-tools:
  - read_file
  - write_to_file
  - list_dir
---

# /tws:plan-article [slug]

<required_reading>
- @.planning/PROJECT.md
- @.planning/ROADMAP.md
- @.planning/STATE.md
- @templates/plan_template.xml
- @.claude/rules/WRITING_RULES.md
- @.claude/rules/FRAMEWORK_RULES.md
</required_reading>

<workflow_steps>
1. **Gather Context**: 
   - If slug provided, check if article spec exists in `.planning/articles/[slug].spec.md`
   - If no slug, ask user for article topic
   - Read existing spec or gather requirements from user

2. **Generate XML Plan**: 
   - Use `templates/plan_template.xml` as base
   - Fill in: article_name, article_slug, research_tasks, draft_tasks, review_tasks, asset_tasks
   - Define verification criteria based on writing rules
   - Set completion criteria

3. **Save Plan**: Write XML plan to `.planning/PLAN.md`

4. **Update State**: Log plan creation in STATE.md
</workflow_steps>

<xml_plan_structure>
```xml
<article type="technical">
  <name>Article Title</name>
  <files>articles/article-slug.md</files>
  <action>
    Research: [Specific research tasks - find 3-5 technical facts, 1-2 failure modes]
    Draft: [Drafting tasks - write with contrast hook, war story, takeaways]
    Review: [Review tasks - verify quality gates, voice check, takeaway density]
    Assets: [Asset tasks - generate hero image prompt, diagram prompts]
  </action>
  <verify>
    - Hook challenges status quo
    - Contains war story ("I built, I broke")
    - Max 3 lines per paragraph
    - No meta-labels in headers
    - Minimum 3 high-value takeaways
    - Passes voice check (no passive voice, no academic fluff)
  </verify>
  <done>Article published with hero image prompt, all quality gates passed</done>
</article>
```
</xml_plan_structure>

<constraints>
- Always use the "Expert Pragmatist" voice in the plan
- Ensure the slug is URL-friendly (lowercase, hyphens)
- Include all required verification criteria from writing rules
- Plan must be actionable and specific
</constraints>
