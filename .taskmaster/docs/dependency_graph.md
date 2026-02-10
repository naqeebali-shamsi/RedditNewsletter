# Task Dependency Graph

## Visual Dependency Map

```
PHASE 1: Source Detection
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  TASK-001: Add source_type field                                │
│      │                                                          │
│      ├──────────────┬───────────────┐                          │
│      ▼              ▼               │                          │
│  TASK-002       TASK-003            │                          │
│  fetch_reddit   fetch_github        │                          │
│  (external)     (internal)          │                          │
│                                     │                          │
└─────────────────────────────────────│──────────────────────────┘
                                      │
PHASE 2: Voice Templates              │
┌─────────────────────────────────────│──────────────────────────┐
│                                     │                          │
│  TASK-004: voice_rules.md ──────────┼───────────────┐          │
│      │                              │               │          │
│      ├──────────────┐               │               │          │
│      ▼              ▼               │               │          │
│  TASK-005       TASK-006            │               │          │
│  External       Internal            │               │          │
│  Voice Prompt   Voice Prompt        │               │          │
│      │              │               │               │          │
└──────│──────────────│───────────────│───────────────│──────────┘
       │              │               │               │
       └──────────────┼───────────────┘               │
                      │                               │
PHASE 3: Voice Selection                              │
┌─────────────────────│───────────────────────────────│──────────┐
│                     ▼                               │          │
│  TASK-007: Voice selector (generate_drafts.py) ◄───┘          │
│      │                                                         │
│      ▼                                                         │
│  TASK-008: Voice selector (generate_medium_full.py)           │
│      │                                                         │
│      ▼                                                         │
│  TASK-009: Test with sample signals                            │
│      │                                                         │
└──────│─────────────────────────────────────────────────────────┘
       │
PHASE 4: Validation
┌──────│─────────────────────────────────────────────────────────┐
│      │                                                         │
│  TASK-010: validate_voice.py ◄──── (also depends on TASK-004) │
│      │                                                         │
│      ▼                                                         │
│  TASK-011: Integrate validation ◄──── (also depends on TASK-008)
│      │                                                         │
│      ▼                                                         │
│  TASK-012: Logging and reporting                               │
│      │                                                         │
└──────│─────────────────────────────────────────────────────────┘
       │
PHASE 5: Testing & Polish
┌──────│─────────────────────────────────────────────────────────┐
│      ▼                                                         │
│  TASK-013: End-to-end testing                                  │
│      │                                                         │
│      ▼                                                         │
│  TASK-014: Quality comparison                                  │
│      │                                                         │
│      ▼                                                         │
│  TASK-015: Documentation updates                               │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

## Dependency Matrix

| Task | Depends On | Blocking |
|------|------------|----------|
| TASK-001 | - | TASK-002, TASK-003, TASK-007 |
| TASK-002 | TASK-001 | - |
| TASK-003 | TASK-001 | - |
| TASK-004 | - | TASK-005, TASK-006, TASK-010 |
| TASK-005 | TASK-004 | TASK-007 |
| TASK-006 | TASK-004 | TASK-007 |
| TASK-007 | TASK-001, TASK-005, TASK-006 | TASK-008 |
| TASK-008 | TASK-007 | TASK-009, TASK-011 |
| TASK-009 | TASK-008 | - |
| TASK-010 | TASK-004 | TASK-011 |
| TASK-011 | TASK-010, TASK-008 | TASK-012 |
| TASK-012 | TASK-011 | TASK-013 |
| TASK-013 | TASK-012 | TASK-014 |
| TASK-014 | TASK-013 | TASK-015 |
| TASK-015 | TASK-014 | - |

## Parallelization Opportunities

### Can Run in Parallel:
- **TASK-002 + TASK-003** (both depend only on TASK-001)
- **TASK-005 + TASK-006** (both depend only on TASK-004)
- **TASK-001 + TASK-004** (no shared dependencies - can start Phase 1 and Phase 2 together)
- **TASK-010** can start as soon as TASK-004 is done (doesn't need to wait for Phase 3)

### Critical Path:
```
TASK-001 → TASK-007 → TASK-008 → TASK-011 → TASK-012 → TASK-013 → TASK-014 → TASK-015
            ↑
    TASK-005 (must wait for TASK-004)
```

**Estimated Critical Path Duration:** ~6.5 hours

## Execution Order Recommendation

### Optimal Order (with parallelization):

**Batch 1** (start immediately, parallel):
- TASK-001: Add source_type field (30 min)
- TASK-004: Create voice_rules.md (45 min)

**Batch 2** (after Batch 1, parallel):
- TASK-002: Update fetch_reddit.py (20 min)
- TASK-003: Update fetch_github.py (20 min)
- TASK-005: External voice prompt (60 min)
- TASK-006: Internal voice prompt (30 min)
- TASK-010: Create validate_voice.py (60 min)

**Batch 3** (after Batch 2):
- TASK-007: Voice selector in generate_drafts.py (45 min)

**Batch 4** (after Batch 3):
- TASK-008: Voice selector in generate_medium_full.py (30 min)

**Batch 5** (after Batch 4, parallel):
- TASK-009: Test voice selection (40 min)
- TASK-011: Integrate validation (30 min)

**Batch 6** (after Batch 5):
- TASK-012: Validation logging (25 min)

**Batch 7** (after Batch 6):
- TASK-013: End-to-end testing (60 min)

**Batch 8** (after Batch 7):
- TASK-014: Quality comparison (45 min)

**Batch 9** (after Batch 8):
- TASK-015: Documentation (30 min)

**Total Estimated Time:** ~8.5 hours (sequential) or ~5 hours (with parallelization)
