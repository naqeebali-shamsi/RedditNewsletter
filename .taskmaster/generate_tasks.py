import json

tasks_data = {
    "meta": {
        "project": "GhostWriter 3.0: WSJ-Tier AI Writing Agency",
        "prd": ".taskmaster/docs/prd.md",
        "created": "2026-01-08",
        "total_tasks": 64,
        "phases": 4,
        "version": "3.0.0"
    },
    "tasks": []
}

# Phase 1: Critical Infrastructure (Tasks 1-15)
phase1 = [
    {"id": "TASK-001", "title": "Create centralized config module", "phase": 1, "priority": "critical", "status": "completed", "requirements": ["REQ-101"], "dependencies": [], "estimated_minutes": 60, "notes": "Already implemented"},
    {"id": "TASK-002", "title": "Migrate all hardcoded paths to config", "phase": 1, "priority": "critical", "status": "pending", "requirements": ["REQ-102"], "dependencies": ["TASK-001"], "estimated_minutes": 120},
    {"id": "TASK-003", "title": "Add environment variable support", "phase": 1, "priority": "critical", "status": "completed", "requirements": ["REQ-103"], "dependencies": ["TASK-001"], "estimated_minutes": 45},
    {"id": "TASK-004", "title": "Validate paths on startup", "phase": 1, "priority": "high", "status": "completed", "requirements": ["REQ-104"], "dependencies": ["TASK-001"], "estimated_minutes": 30},
    {"id": "TASK-005", "title": "Create FactVerificationAgent", "phase": 1, "priority": "critical", "status": "pending", "requirements": ["REQ-105"], "dependencies": ["TASK-001"], "estimated_minutes": 180},
    {"id": "TASK-006", "title": "Implement claim extraction", "phase": 1, "priority": "high", "status": "pending", "requirements": ["REQ-108"], "dependencies": ["TASK-005"], "estimated_minutes": 120},
    {"id": "TASK-007", "title": "Build verification API wrapper", "phase": 1, "priority": "high", "status": "pending", "requirements": ["REQ-109"], "dependencies": ["TASK-005"], "estimated_minutes": 90},
    {"id": "TASK-008", "title": "Add verification status to schema", "phase": 1, "priority": "high", "status": "pending", "requirements": ["REQ-110"], "dependencies": ["TASK-005", "TASK-006"], "estimated_minutes": 45},
    {"id": "TASK-009", "title": "Create quality gate blocking rule", "phase": 1, "priority": "critical", "status": "pending", "requirements": ["REQ-107"], "dependencies": ["TASK-005", "TASK-008"], "estimated_minutes": 60},
    {"id": "TASK-010", "title": "Update pipeline to require verification", "phase": 1, "priority": "critical", "status": "pending", "requirements": ["REQ-106"], "dependencies": ["TASK-005", "TASK-009"], "estimated_minutes": 90},
    {"id": "TASK-011", "title": "Test cross-platform path handling", "phase": 1, "priority": "high", "status": "pending", "requirements": ["REQ-102"], "dependencies": ["TASK-002"], "estimated_minutes": 60},
    {"id": "TASK-012", "title": "Document configuration options", "phase": 1, "priority": "medium", "status": "pending", "requirements": ["REQ-103"], "dependencies": ["TASK-003"], "estimated_minutes": 45},
    {"id": "TASK-013", "title": "Add config validation tests", "phase": 1, "priority": "medium", "status": "pending", "requirements": ["REQ-104"], "dependencies": ["TASK-004"], "estimated_minutes": 45},
    {"id": "TASK-014", "title": "Migration script for existing code", "phase": 1, "priority": "medium", "status": "pending", "requirements": ["REQ-102"], "dependencies": ["TASK-002"], "estimated_minutes": 60},
    {"id": "TASK-015", "title": "USER-TEST-1: Verify fact verification", "phase": 1, "priority": "critical", "status": "pending", "requirements": ["REQ-105", "REQ-106", "REQ-107"], "dependencies": ["TASK-010"], "estimated_minutes": 30, "is_checkpoint": True},
]

# Phase 2: Architecture Modernization (Tasks 16-32)
phase2 = [
    {"id": "TASK-016", "title": "Design LangGraph state schema", "phase": 2, "priority": "critical", "status": "pending", "requirements": ["REQ-202"], "dependencies": ["TASK-015"], "estimated_minutes": 90},
    {"id": "TASK-017", "title": "Create StateGraph builder", "phase": 2, "priority": "critical", "status": "pending", "requirements": ["REQ-201"], "dependencies": ["TASK-016"], "estimated_minutes": 120},
    {"id": "TASK-018", "title": "Migrate research phase to LangGraph", "phase": 2, "priority": "high", "status": "pending", "requirements": ["REQ-201"], "dependencies": ["TASK-017"], "estimated_minutes": 90},
    {"id": "TASK-019", "title": "Migrate generation phase to LangGraph", "phase": 2, "priority": "high", "status": "pending", "requirements": ["REQ-201"], "dependencies": ["TASK-018"], "estimated_minutes": 90},
    {"id": "TASK-020", "title": "Migrate verification phase to LangGraph", "phase": 2, "priority": "high", "status": "pending", "requirements": ["REQ-201"], "dependencies": ["TASK-019"], "estimated_minutes": 90},
    {"id": "TASK-021", "title": "Add SQLite checkpointer", "phase": 2, "priority": "medium", "status": "pending", "requirements": ["REQ-203"], "dependencies": ["TASK-017"], "estimated_minutes": 60},
    {"id": "TASK-022", "title": "Implement multi-model adversarial panel", "phase": 2, "priority": "critical", "status": "pending", "requirements": ["REQ-205"], "dependencies": ["TASK-020"], "estimated_minutes": 120},
    {"id": "TASK-023", "title": "Create ethics reviewer (Claude)", "phase": 2, "priority": "high", "status": "pending", "requirements": ["REQ-205"], "dependencies": ["TASK-022"], "estimated_minutes": 60},
    {"id": "TASK-024", "title": "Create accuracy reviewer (Gemini)", "phase": 2, "priority": "high", "status": "pending", "requirements": ["REQ-205"], "dependencies": ["TASK-022"], "estimated_minutes": 60},
    {"id": "TASK-025", "title": "Create structure reviewer (GPT-4o)", "phase": 2, "priority": "high", "status": "pending", "requirements": ["REQ-205"], "dependencies": ["TASK-022"], "estimated_minutes": 60},
    {"id": "TASK-026", "title": "Implement weighted voting aggregation", "phase": 2, "priority": "critical", "status": "pending", "requirements": ["REQ-206"], "dependencies": ["TASK-023", "TASK-024", "TASK-025"], "estimated_minutes": 60},
    {"id": "TASK-027", "title": "Add WSJ Four Showstoppers checklist", "phase": 2, "priority": "critical", "status": "pending", "requirements": ["REQ-207"], "dependencies": ["TASK-022"], "estimated_minutes": 90},
    {"id": "TASK-028", "title": "Implement kill phrase detection", "phase": 2, "priority": "high", "status": "pending", "requirements": ["REQ-208"], "dependencies": ["TASK-022"], "estimated_minutes": 90},
    {"id": "TASK-029", "title": "Add source-aware voice selection", "phase": 2, "priority": "high", "status": "pending", "requirements": ["REQ-211"], "dependencies": ["TASK-019"], "estimated_minutes": 60},
    {"id": "TASK-030", "title": "Create voice validation post-processing", "phase": 2, "priority": "high", "status": "pending", "requirements": ["REQ-212"], "dependencies": ["TASK-029"], "estimated_minutes": 60},
    {"id": "TASK-031", "title": "Add escalation logic", "phase": 2, "priority": "high", "status": "pending", "requirements": ["REQ-210"], "dependencies": ["TASK-026"], "estimated_minutes": 60},
    {"id": "TASK-032", "title": "USER-TEST-2: Verify adversarial review", "phase": 2, "priority": "critical", "status": "pending", "requirements": ["REQ-205", "REQ-206", "REQ-207"], "dependencies": ["TASK-031"], "estimated_minutes": 45, "is_checkpoint": True},
]

# Phase 3: WSJ-Tier Quality (Tasks 33-48)
phase3 = [
    {"id": "TASK-033", "title": "Implement C2PA manifest generation", "phase": 3, "priority": "critical", "status": "pending", "requirements": ["REQ-301"], "dependencies": ["TASK-032"], "estimated_minutes": 120},
    {"id": "TASK-034", "title": "Add Schema.org JSON-LD output", "phase": 3, "priority": "high", "status": "pending", "requirements": ["REQ-302"], "dependencies": ["TASK-033"], "estimated_minutes": 60},
    {"id": "TASK-035", "title": "Create inline disclosure generator", "phase": 3, "priority": "critical", "status": "pending", "requirements": ["REQ-304"], "dependencies": ["TASK-033"], "estimated_minutes": 45},
    {"id": "TASK-036", "title": "Build provenance tracking module", "phase": 3, "priority": "high", "status": "pending", "requirements": ["REQ-303"], "dependencies": ["TASK-033"], "estimated_minutes": 90},
    {"id": "TASK-037", "title": "Create Streamlit dashboard shell", "phase": 3, "priority": "critical", "status": "pending", "requirements": ["REQ-305"], "dependencies": ["TASK-032"], "estimated_minutes": 120},
    {"id": "TASK-038", "title": "Implement review queue view", "phase": 3, "priority": "critical", "status": "pending", "requirements": ["REQ-305"], "dependencies": ["TASK-037"], "estimated_minutes": 90},
    {"id": "TASK-039", "title": "Add side-by-side comparison view", "phase": 3, "priority": "high", "status": "pending", "requirements": ["REQ-306"], "dependencies": ["TASK-038"], "estimated_minutes": 90},
    {"id": "TASK-040", "title": "Create fact-check highlighting panel", "phase": 3, "priority": "critical", "status": "pending", "requirements": ["REQ-307"], "dependencies": ["TASK-038"], "estimated_minutes": 90},
    {"id": "TASK-041", "title": "Implement approval workflow", "phase": 3, "priority": "critical", "status": "pending", "requirements": ["REQ-305"], "dependencies": ["TASK-039", "TASK-040"], "estimated_minutes": 90},
    {"id": "TASK-042", "title": "Add audit trail display", "phase": 3, "priority": "high", "status": "pending", "requirements": ["REQ-308"], "dependencies": ["TASK-041"], "estimated_minutes": 60},
    {"id": "TASK-043", "title": "Implement escalation UI", "phase": 3, "priority": "high", "status": "pending", "requirements": ["REQ-309"], "dependencies": ["TASK-041"], "estimated_minutes": 60},
    {"id": "TASK-044", "title": "Add LangGraph interrupt integration", "phase": 3, "priority": "critical", "status": "pending", "requirements": ["REQ-204"], "dependencies": ["TASK-041"], "estimated_minutes": 120},
    {"id": "TASK-045", "title": "Research Agent-Lightning integration", "phase": 3, "priority": "low", "status": "pending", "requirements": ["REQ-310"], "dependencies": ["TASK-044"], "estimated_minutes": 120},
    {"id": "TASK-046", "title": "Create optimization feedback loop", "phase": 3, "priority": "low", "status": "pending", "requirements": ["REQ-310"], "dependencies": ["TASK-045"], "estimated_minutes": 180},
    {"id": "TASK-047", "title": "End-to-end integration testing", "phase": 3, "priority": "high", "status": "pending", "requirements": ["REQ-305"], "dependencies": ["TASK-044"], "estimated_minutes": 120},
    {"id": "TASK-048", "title": "USER-TEST-3: Verify HITL dashboard", "phase": 3, "priority": "critical", "status": "pending", "requirements": ["REQ-305", "REQ-307", "REQ-308"], "dependencies": ["TASK-047"], "estimated_minutes": 45, "is_checkpoint": True},
]

# Phase 4: Production Hardening (Tasks 49-64)
phase4 = [
    {"id": "TASK-049", "title": "Create test infrastructure", "phase": 4, "priority": "critical", "status": "pending", "requirements": ["REQ-401"], "dependencies": ["TASK-048"], "estimated_minutes": 90},
    {"id": "TASK-050", "title": "Add unit tests for config module", "phase": 4, "priority": "high", "status": "pending", "requirements": ["REQ-401"], "dependencies": ["TASK-049"], "estimated_minutes": 60},
    {"id": "TASK-051", "title": "Add unit tests for agents", "phase": 4, "priority": "critical", "status": "pending", "requirements": ["REQ-402"], "dependencies": ["TASK-049"], "estimated_minutes": 180},
    {"id": "TASK-052", "title": "Add unit tests for quality gate", "phase": 4, "priority": "critical", "status": "pending", "requirements": ["REQ-402"], "dependencies": ["TASK-049"], "estimated_minutes": 90},
    {"id": "TASK-053", "title": "Create integration test suite", "phase": 4, "priority": "critical", "status": "pending", "requirements": ["REQ-403"], "dependencies": ["TASK-051", "TASK-052"], "estimated_minutes": 120},
    {"id": "TASK-054", "title": "Add end-to-end pipeline test", "phase": 4, "priority": "high", "status": "pending", "requirements": ["REQ-403"], "dependencies": ["TASK-053"], "estimated_minutes": 90},
    {"id": "TASK-055", "title": "Implement cost tracking", "phase": 4, "priority": "high", "status": "pending", "requirements": ["REQ-404"], "dependencies": ["TASK-049"], "estimated_minutes": 90},
    {"id": "TASK-056", "title": "Add structured logging", "phase": 4, "priority": "critical", "status": "pending", "requirements": ["REQ-405"], "dependencies": ["TASK-049"], "estimated_minutes": 60},
    {"id": "TASK-057", "title": "Create health check endpoints", "phase": 4, "priority": "high", "status": "pending", "requirements": ["REQ-408"], "dependencies": ["TASK-049"], "estimated_minutes": 60},
    {"id": "TASK-058", "title": "Implement rate limiting", "phase": 4, "priority": "critical", "status": "pending", "requirements": ["REQ-407"], "dependencies": ["TASK-049"], "estimated_minutes": 60},
    {"id": "TASK-059", "title": "Add error recovery logic", "phase": 4, "priority": "critical", "status": "pending", "requirements": ["REQ-407"], "dependencies": ["TASK-058"], "estimated_minutes": 90},
    {"id": "TASK-060", "title": "Create Docker configuration", "phase": 4, "priority": "high", "status": "pending", "requirements": ["REQ-409"], "dependencies": ["TASK-056"], "estimated_minutes": 90},
    {"id": "TASK-061", "title": "Write deployment documentation", "phase": 4, "priority": "high", "status": "pending", "requirements": ["REQ-409"], "dependencies": ["TASK-060"], "estimated_minutes": 60},
    {"id": "TASK-062", "title": "Performance optimization", "phase": 4, "priority": "medium", "status": "pending", "requirements": [], "dependencies": ["TASK-054"], "estimated_minutes": 120},
    {"id": "TASK-063", "title": "Security audit", "phase": 4, "priority": "critical", "status": "pending", "requirements": [], "dependencies": ["TASK-060"], "estimated_minutes": 90},
    {"id": "TASK-064", "title": "USER-TEST-4: Full system verification", "phase": 4, "priority": "critical", "status": "pending", "requirements": ["REQ-401", "REQ-403"], "dependencies": ["TASK-063"], "estimated_minutes": 60, "is_checkpoint": True},
]

tasks_data["tasks"] = phase1 + phase2 + phase3 + phase4

with open("N:/RedditNews/.taskmaster/tasks/tasks.json", "w") as f:
    json.dump(tasks_data, f, indent=2)

print(f"Successfully created tasks.json with {len(tasks_data['tasks'])} tasks")
print(f"Phase 1: {len(phase1)} tasks (Critical Infrastructure)")
print(f"Phase 2: {len(phase2)} tasks (Architecture Modernization)")
print(f"Phase 3: {len(phase3)} tasks (WSJ-Tier Quality)")
print(f"Phase 4: {len(phase4)} tasks (Production Hardening)")
