---
name: series-init
description: Initializes a multi-part series (like the 30-day LinkedIn challenge) with a master plan and story arc.
allowed-tools:
  - write_to_file
  - read_file
---

# /series:init

<required_reading>
- @.planning/PROJECT.md
- @.planning/STATE.md
</required_reading>

<workflow_steps>
1. **Define the Arc**: Ask the user for the "Start Point" (Current Skill Level) and "End Point" (Desired Mastery) of the 30-day journey.
2. **Master List Creation**: Create `.planning/SERIES_MASTER.md` to track all 30 days, their topics, and their status.
3. **Weekly Buckets**: Group the 30 days into 4 logical "Difficulty/Complexity" buckets to show a progression curve.
4. **Update State**: Mark the series as established in `STATE.md`.
</workflow_steps>

<constraints>
- Each day must have a "Learning Tag" (e.g., #Day1, #Day2).
- Ensure the topics flow logically (e.g., don't teach Multi-Agent Orchestration before Basic Directives).
</constraints>
