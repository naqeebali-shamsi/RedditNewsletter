# Product Hardening Master Plan

**Created**: 2026-02-08
**Source**: PRODUCT_HARDENING_DOSSIER.md
**Status**: COMPLETE

---

## Priority Features (Processing Order)

| # | Feature | Trust Score | Status | Files |
|---|---------|-------------|--------|-------|
| 1 | BaseAgent & Multi-Provider | Foundation | COMPLETE | `feature-baseagent-*.md` |
| 2 | Fact Verification | 3/10 | COMPLETE | `feature-factverify-*.md` |
| 3 | Pipeline Orchestration | 5/10 | COMPLETE | `feature-pipeline-*.md` |
| 4 | Source Ingestion | 4/10 | COMPLETE | `feature-sources-*.md` |
| 5 | Dashboard | 3/10 | COMPLETE | `feature-dashboard-*.md` |

## Phase 1 Ship-Blocking Items (from Dossier)

1. BaseAgent: Raise exceptions, not error strings (M)
2. Fix "no claims = pass" verification bypass (S)
3. Add error boundaries in Streamlit dashboard (S)
4. Add source failure circuit breaker (S)
5. Add per-node timeouts in pipeline (S)
6. Rename "C2PA manifest" to "Content Metadata" (S)

## Progress Log

- [x] Feature 1: BaseAgent — Deconstruction / Critique / Upgrade Plan (6 upgrades identified)
- [x] Feature 2: Fact Verification — Deconstruction / Critique / Upgrade Plan (7 upgrades identified)
- [x] Feature 3: Pipeline Orchestration — Deconstruction / Critique / Upgrade Plan (9 upgrades identified; discovered 3-pipeline problem, not 2)
- [x] Feature 4: Source Ingestion — Deconstruction / Critique / Upgrade Plan (8 upgrades identified; discovered dashboard bypasses ContentSource entirely, Reddit data invisible to pulse aggregator by default)
- [x] Feature 5: Dashboard — Deconstruction / Critique / Upgrade Plan (9 upgrades identified; discovered review decisions don't persist, XSS via unsafe_allow_html, no authentication)
- [x] FINAL: Synthesize FINAL-HARDENING-ROADMAP.md (39 upgrades, 3-week roadmap, trustworthiness scorecard 2.0→7.6/10)
