---
name: tws-execute-plan
description: Executes a single XML task plan using subagent for fresh context execution.
allowed-tools:
  - read_file
  - write_to_file
  - subagent
---

# /tws:execute-plan

<required_reading>
- @.planning/PROJECT.md
- @.planning/PLAN.md
- @.planning/STATE.md
- @.claude/rules/WRITING_RULES.md
- @.claude/rules/FRAMEWORK_RULES.md
- @.claude/rules/TECHNICAL_RULES.md
</required_reading>

<workflow_steps>
1. **Load Plan**: Read `.planning/PLAN.md` to get the current XML task plan.
2. **Parse XML**: Extract article name, files, action steps, verification criteria, and done conditions.
3. **Subagent Execution**: Use `subagent` tool to execute the plan in a fresh context (200k tokens).
   - The subagent receives the full PLAN.md, PROJECT.md, and relevant rules
   - Subagent executes: Research → Draft → Review → Assets
4. **Verify**: Check that all verification criteria from the plan are met.
5. **Update State**: Log completion in STATE.md and SUMMARY.md.
6. **Atomic Commit**: Create git commit for this task.
</workflow_steps>

<subagent_instructions>
You are executing a writing task plan. Follow these steps:

1. **Research Phase**: 
   - Use browser_subagent and search_web to gather technical facts, war stories, and edge cases
   - Document findings in structured format

2. **Draft Phase**:
   - Follow Architected Writing Framework (see FRAMEWORK_RULES.md)
   - Apply writing rules (see WRITING_RULES.md)
   - Generate content following the action steps in the plan

3. **Review Phase**:
   - Verify all criteria in the <verify> section are met
   - Run quality checks: contrast hook, sentence compression, tradeoff enforcement, authentic voice, no meta-labels
   - Make corrections as needed

4. **Assets Phase**:
   - **REQUIRED**: Generate hero image prompt using ByteByteGo style (use `/tws:generate-image-prompt` or follow TECHNICAL_RULES.md specifications)
   - Prompt must be extremely specific and ready for direct use
   - Save prompt in article file (frontmatter or dedicated section)
   - Create any additional visual assets if needed

5. **Completion**:
   - Save final article to the specified file path
   - Ensure all verification criteria are met
   - **Verify**: Article includes hero image prompt (mandatory)
   - Report completion status
</subagent_instructions>

<constraints>
- Each subagent execution runs in fresh context to prevent quality degradation
- All verification criteria must be met before marking as done
- Follow atomic commit pattern: one commit per task
- Update STATE.md and SUMMARY.md after completion
</constraints>
