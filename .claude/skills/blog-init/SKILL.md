---
name: blog-init
description: Initializes a new technical article by creating a specification file and setting the state.
allowed-tools:
  - list_dir
  - write_to_file
  - read_file
---

# /blog:init

<required_reading>
- @.planning/PROJECT.md
- @.planning/ROADMAP.md
- @.planning/STATE.md
</required_reading>

<workflow_steps>
1. **Gather Intel**: Ask the user for the primary topic, target audience (if different from default), and any specific "war stories" or anecdotes they want to include.
2. **Context Sync**: Check the `.planning/STATE.md` to ensure we are not duplicating work.
3. **Spec Creation**: Create a new file in `.planning/articles/[topic-slug].spec.md` using the Article Specification template.
4. **Update State**: Log the start of this new article in `.planning/STATE.md`.
</workflow_steps>

<output_format>
A confirmation message stating that the article spec has been created, with a link to the file.
</output_format>

<constraints>
- Always use the "Expert Pragmatist" voice in the spec.
- Ensure the slug is URL-friendly (lowercase, hyphens).
</constraints>
