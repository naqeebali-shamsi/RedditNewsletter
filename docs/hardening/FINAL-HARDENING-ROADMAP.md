# FINAL HARDENING ROADMAP

**Generated**: 2026-02-08
**Source**: Product Hardening Dossier + 5-Feature Deep Analysis
**Documents**: 15 feature files in `docs/hardening/`

---

## Executive Summary

The GhostWriter codebase was analyzed across 5 priority features using a simulated multi-agent protocol (Deconstruction → Research + Red Team → Upgrade Planning). Analysis covered 15 source files totaling ~6,500 lines of production code.

**Total upgrades identified**: 39 actionable upgrades
- CRITICAL: 6 (must fix before ship)
- HIGH: 20 (should fix before ship)
- MEDIUM: 12 (fix in next sprint)
- LOW: 1 (nice to have)

**Estimated total implementation**: 14-19 days for a single engineer

**Key architectural findings not in original dossier:**
1. THREE pipelines exist (not two) — dashboard has its own 447-line inline pipeline
2. THREE fail-open verification paths (not one) — fact agent, quality gate, pipeline
3. Review dashboard decisions don't persist — lost on browser refresh
4. Reddit data invisible to pulse aggregator by default — wrong DB table
5. XSS vulnerabilities in both dashboards via `unsafe_allow_html`

---

## Ship-Blocking Upgrades (Week 1)

These MUST be completed before any external demo or production use.

### Day 1-2: Foundation Fixes (Zero-Risk, High-Impact)

| # | Upgrade | Feature | Effort | Risk |
|---|---------|---------|--------|------|
| 1 | Fix verify_node error routing → ESCALATE | Pipeline #6 | S | Low |
| 2 | Fix "no claims = pass" verification bypass | Fact Verify #1 | S | Low |
| 3 | Fix Reddit dual-DB default (pulse data completeness) | Sources #2 | S | Low |
| 4 | Fix style check fail-open default | Pipeline #7 | S | Low |
| 5 | Add LLM output validation in pipeline nodes | Pipeline #5 | S | Low |
| 6 | Persist review decisions to disk | Dashboard #2 | S | Low |

**All one-line or small-scope changes. Can be done in parallel. Zero regression risk.**

### Day 2-3: Error Boundaries & Resilience

| # | Upgrade | Feature | Effort | Risk |
|---|---------|---------|--------|------|
| 7 | Add per-phase error boundaries in dashboard | Dashboard #1 | S | Low |
| 8 | Replace bare except blocks in dashboard | Dashboard #6 | S | Low |
| 9 | Sanitize HTML output (fix XSS) | Dashboard #3 | S | None |
| 10 | Fix copy_to_clipboard JS injection | Dashboard #3 | S | None |
| 11 | Add basic authentication | Dashboard #4 | S | Low |
| 12 | Add generation rate limiting | Dashboard #5 | S | None |
| 13 | Rename "C2PA manifest" → "Content Metadata" | Dashboard #7 | S | None |

### Day 3-4: BaseAgent Exceptions (Foundation)

| # | Upgrade | Feature | Effort | Risk |
|---|---------|---------|--------|------|
| 14 | Raise exceptions instead of error strings | BaseAgent #1 | M | Medium |

**This is the highest-risk change.** It modifies the foundation class used by 12 agents with 25+ call sites. Requires updating all callers that currently handle error strings. Test thoroughly. However, it unblocks multiple downstream upgrades.

---

## High-Priority Upgrades (Week 2)

### Day 5-6: Agent Reliability

| # | Upgrade | Feature | Effort | Risk |
|---|---------|---------|--------|------|
| 15 | Add retry with exponential backoff (BaseAgent) | BaseAgent #2 | S | Low |
| 16 | Add response validation (BaseAgent) | BaseAgent #3 | S | Low |
| 17 | Add per-agent provider override | BaseAgent #4 | S | Low |
| 18 | Add source health check + registration warnings | Sources #1 | S | None |
| 19 | Add retry with backoff (sources) | Sources #3 | S | Low |

### Day 7-8: Pipeline Robustness

| # | Upgrade | Feature | Effort | Risk |
|---|---------|---------|--------|------|
| 20 | Use typed state in StateGraph | Pipeline #1 | S | Medium |
| 21 | Fix state merge in run_pipeline | Pipeline #2 | S | Low |
| 22 | Add per-node timeouts | Pipeline #3 | S | Low |

### Day 9-10: Verification Hardening

| # | Upgrade | Feature | Effort | Risk |
|---|---------|---------|--------|------|
| 23 | Add minimum claim count validation | Fact Verify #5 | S | Low |
| 24 | Add verification provider health check | Fact Verify #3 | S | None |
| 25 | Remove content truncation (6000→12000 chars) | Fact Verify #4 | S | Low |
| 26 | Add source failure circuit breaker | Sources #4 | M | Low |

---

## Medium-Priority Upgrades (Week 3)

| # | Upgrade | Feature | Effort | Risk |
|---|---------|---------|--------|------|
| 27 | Add token counting (BaseAgent) | BaseAgent #5 | M | None |
| 28 | Add call timeouts (BaseAgent) | BaseAgent #6 | S | Low |
| 29 | Use structured output for claims | Fact Verify #2 | M | Low |
| 30 | Add back-reference validation | Fact Verify #7 | S | None |
| 31 | Add verification audit logging | Fact Verify #6 | M | None |
| 32 | Add per-run cost tracking | Pipeline #4 | M | None |
| 33 | Enable checkpointing by default | Pipeline #8 | S | Low |
| 34 | Fix Reddit engagement data | Sources #5 | M | Low |
| 35 | Add concurrent source fetching | Sources #6 | S | Low |
| 36 | Add semantic deduplication | Sources #7 | M | Low |
| 37 | Wire status_callback to UI | Dashboard #8 | S | None |
| 38 | Save pipeline metadata for crash recovery | Dashboard #9 | S | None |

---

## Deferred (Post-Ship)

| # | Upgrade | Feature | Effort | Risk |
|---|---------|---------|--------|------|
| 39 | Dashboard calls LangGraph pipeline (consolidation) | Pipeline #9 | L | Medium |
| 40 | Move trust tiers to configuration | Sources #8 | S | None |

Pipeline consolidation (Upgrade 39) is the architecturally correct long-term fix but requires reconciling two different pipeline implementations with different agent ordering, specialist chains, and progress callback patterns. This should be planned as a dedicated project after all ship-blocking fixes are stable.

---

## Dependency Graph

```
                    BaseAgent #1 (exceptions)
                   /           |            \
                  /            |             \
    Pipeline #5          Fact Verify #2       Dashboard #1
  (output validation)   (structured output)   (error boundaries)
         |                                         |
    Pipeline #4                              Pipeline #9
   (cost tracking)                        (consolidation)
         |
    BaseAgent #5
   (token counting)

    Sources #4 (circuit breaker)
         |
    Pipeline #6 (error routing)  ← INDEPENDENT

    Fact Verify #1 (fail-closed)  ← INDEPENDENT, DO FIRST
    Sources #2 (unified DB)       ← INDEPENDENT, DO FIRST
    Dashboard #2 (persist reviews) ← INDEPENDENT, DO FIRST
```

**Critical path**: BaseAgent #1 → Pipeline #5 → Pipeline #4 → BaseAgent #5

**Parallel tracks** (can proceed independently):
- Fact Verification fixes (#1, #3, #4, #5)
- Source fixes (#1, #2, #3)
- Dashboard fixes (#2, #3, #4, #5, #6)
- Pipeline safety fixes (#6, #7)

---

## Files Modified (Summary)

| File | Upgrade Count | Features Affected |
|------|--------------|-------------------|
| `execution/agents/base_agent.py` | 6 | BaseAgent, Pipeline |
| `execution/pipeline.py` | 8 | Pipeline, Fact Verify |
| `app.py` | 9 | Dashboard |
| `execution/agents/fact_verification_agent.py` | 6 | Fact Verify |
| `execution/quality_gate.py` | 2 | Fact Verify, Dashboard |
| `execution/sources/__init__.py` | 3 | Sources |
| `execution/sources/reddit_source.py` | 3 | Sources |
| `execution/dashboard/app.py` | 4 | Dashboard |
| `execution/pulse_aggregator.py` | 2 | Sources |
| `requirements.txt` | 4 | BaseAgent, Fact Verify, Sources |

**New files created:**
- `execution/cost_tracker.py` (Pipeline #4)
- `execution/sources/circuit_breaker.py` (Sources #4)

---

## Product Trustworthiness Scorecard

### Current State (Pre-Hardening)

| Dimension | Score | Critical Issues |
|-----------|-------|----------------|
| Error Handling | 2/10 | Error strings as return values; bare excepts; fail-open verification |
| Data Integrity | 3/10 | State merge loses errors; review decisions ephemeral; dual DB |
| Security | 2/10 | No auth; XSS via unsafe_allow_html; JS injection in clipboard |
| Reliability | 3/10 | No timeouts; no retries; no circuit breakers; crash = lost work |
| Consistency | 2/10 | Three pipelines; different quality gates; different agent ordering |
| Observability | 1/10 | No cost tracking; no structured logging; silent source failures |
| Cost Control | 1/10 | Zero budget caps; zero cost visibility; unbounded revision loops |
| **Overall** | **2.0/10** | |

### Projected State (After Ship-Blocking Fixes, Week 1)

| Dimension | Score | Improvement |
|-----------|-------|-------------|
| Error Handling | 5/10 | Exceptions replace error strings; error boundaries in dashboard |
| Data Integrity | 6/10 | Fail-closed verification; persisted reviews; unified DB default |
| Security | 5/10 | Auth gate; HTML sanitization; JS escaping fixed |
| Reliability | 3/10 | (Timeouts and retries come in Week 2) |
| Consistency | 3/10 | (Pipeline consolidation is deferred) |
| Observability | 2/10 | Source health warnings; status callback wired |
| Cost Control | 1/10 | (Cost tracking comes in Week 2-3) |
| **Overall** | **3.6/10** | +80% improvement |

### Projected State (After All HIGH Fixes, Week 2)

| Dimension | Score | Improvement |
|-----------|-------|-------------|
| Error Handling | 7/10 | Per-phase boundaries; validated outputs; typed state |
| Data Integrity | 7/10 | Claim count validation; content truncation fixed; proper merge |
| Security | 6/10 | Auth + sanitization + rate limiting |
| Reliability | 6/10 | Timeouts; retries; circuit breaker |
| Consistency | 3/10 | (Still three pipelines) |
| Observability | 4/10 | Provider health; source health; status display |
| Cost Control | 2/10 | (Cost tracking comes in Week 3) |
| **Overall** | **5.0/10** | +150% from baseline |

### Projected State (After All MEDIUM Fixes, Week 3)

| Dimension | Score | Improvement |
|-----------|-------|-------------|
| Error Handling | 8/10 | Structured output; back-reference validation |
| Data Integrity | 8/10 | Audit logging; semantic dedup; checkpointing |
| Security | 6/10 | (Unchanged from Week 2) |
| Reliability | 7/10 | Call timeouts; concurrent fetching; crash recovery |
| Consistency | 3/10 | (Pipeline consolidation is post-ship) |
| Observability | 6/10 | Cost tracking; verification audit logs; token counting |
| Cost Control | 6/10 | Budget caps; per-run cost display; cost-per-article tracking |
| **Overall** | **6.3/10** | +215% from baseline |

### Target State (After Pipeline Consolidation, Post-Ship)

| Dimension | Score |
|-----------|-------|
| Error Handling | 8/10 |
| Data Integrity | 8/10 |
| Security | 7/10 |
| Reliability | 8/10 |
| Consistency | 8/10 |
| Observability | 7/10 |
| Cost Control | 7/10 |
| **Overall** | **7.6/10** |

---

## Risk Matrix

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| BaseAgent exception change breaks callers | Medium | High | Run full test suite; update callers incrementally |
| Typed state surfaces hidden bugs | Medium | Medium | Run pipeline E2E after change; add missing keys to PipelineState |
| Pipeline consolidation breaks dashboard | Medium | High | Defer to post-ship; maintain adapter pattern |
| Reddit rate limiting during backoff testing | Low | Low | Test with `--dry-run`; mock HTTP calls |
| Structured output breaks claim extraction | Low | Medium | Roll out per-provider; keep fallback parser |

---

## Source Documents

All analysis files live in `docs/hardening/`:

| Feature | Current State | Critique | Upgrade Plan |
|---------|--------------|----------|-------------|
| BaseAgent | `feature-baseagent-01-current-state.md` | `feature-baseagent-02-critique.md` | `feature-baseagent-03-upgrade-plan.md` |
| Fact Verification | `feature-factverify-01-current-state.md` | `feature-factverify-02-critique.md` | `feature-factverify-03-upgrade-plan.md` |
| Pipeline | `feature-pipeline-01-current-state.md` | `feature-pipeline-02-critique.md` | `feature-pipeline-03-upgrade-plan.md` |
| Sources | `feature-sources-01-current-state.md` | `feature-sources-02-critique.md` | `feature-sources-03-upgrade-plan.md` |
| Dashboard | `feature-dashboard-01-current-state.md` | `feature-dashboard-02-critique.md` | `feature-dashboard-03-upgrade-plan.md` |

Master tracking: `00-master-plan.md`
