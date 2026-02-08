---
name: tws-execute-phase
description: Executes all plans in a phase with parallel subagent execution for research, drafting, and review workflows.
allowed-tools:
  - read_file
  - write_to_file
  - subagent
---

# /tws:execute-phase <N>

<required_reading>
- @.planning/PROJECT.md
- @.planning/ROADMAP.md
- @.planning/STATE.md
- @.claude/rules/WRITING_RULES.md
- @.claude/rules/FRAMEWORK_RULES.md
- @.claude/rules/TECHNICAL_RULES.md
</required_reading>

<workflow_steps>
1. **Load Phase**: Read ROADMAP.md to identify all articles/plans in phase N.
2. **Create Plans**: For each article in the phase, ensure PLAN.md exists (create if needed).
3. **Parallel Execution**: Spawn parallel subagents for:
   - **Research Subagent**: Web search, browser navigation, data extraction
   - **Drafting Subagent**: Content generation following architected writing framework
   - **Review Subagent**: Quality checks, voice validation, takeaway density verification
4. **Status Tracking**: Monitor subagent progress and log in STATE.md.
5. **Completion**: When all subagents complete, update ROADMAP.md and create atomic commits.
</workflow_steps>

<parallel_execution>
For each article in the phase:

1. **Research Subagent**:
   - Task: Gather technical facts, war stories, edge cases
   - Tools: browser_subagent, search_web
   - Output: Research data appended to article spec

2. **Drafting Subagent** (runs after research):
   - Task: Generate article following Architected Writing Framework
   - Tools: read_file, write_to_file
   - Output: Draft article in articles/ directory

3. **Review Subagent** (runs after draft):
   - Task: Quality checks and verification
   - Tools: read_file, write_to_file
   - Output: Finalized article with all quality gates passed
</parallel_execution>

<constraints>
- Each subagent runs in fresh context (200k tokens) to prevent quality degradation
- Subagents execute in sequence per article (research → draft → review)
- Multiple articles can be processed in parallel
- All verification criteria must be met before phase completion
- Atomic commits: one commit per article task
</constraints>
