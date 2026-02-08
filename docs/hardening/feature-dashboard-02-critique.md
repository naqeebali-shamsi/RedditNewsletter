# Feature: Dashboard — Research & Adversarial Critique

**Agents**: Technique Research + Red Team
**Date**: 2026-02-08

---

## Research: Industry Standard Patterns for Streamlit Dashboards

### R1: Error Boundaries in Streamlit (Streamlit Best Practice)

**The problem**: A single uncaught exception in the pipeline crashes the entire page. The user sees a stack trace and loses all UI state.

**Industry solutions:**
- **Per-phase try/except**: Wrap each pipeline phase (outline, draft, review, etc.) in individual try/except blocks
- **`st.error()` + graceful degradation**: Show user-friendly error, offer retry, preserve completed phases
- **Streamlit `@st.cache_data`**: Cache completed intermediate results so a retry doesn't re-run successful phases
- **Background worker pattern**: Run pipeline in a separate thread/process; poll for results from UI

**Pattern**: Each pipeline phase returns a typed result or raises a specific exception. The UI layer catches per-phase exceptions and shows recovery options ("Skip this step", "Retry", "Save partial results").

### R2: Persistent State for Streamlit Apps (Production Standard)

**The problem**: All review decisions, audit logs, and pipeline state live in `st.session_state` and are lost on page refresh.

**Industry solutions:**
- **SQLite backend**: Store review queue, decisions, audit log in a local SQLite database
- **Streamlit-Supabase/Firebase**: Cloud-backed persistent state
- **File-based state**: Write JSON state files per article (decisions, review notes)
- **Redis/memcached**: Shared state across multiple Streamlit instances

**Pattern**: Minimum viable persistence: write review decisions to a JSON sidecar file next to the article (e.g., `article_xxx_review.json`). This survives page refreshes and is discoverable by the review dashboard's file scanner.

### R3: Input Sanitization for unsafe_allow_html (OWASP)

**The problem**: Dynamic content (LLM output, article titles) injected into `unsafe_allow_html=True` markdown creates XSS vectors.

**Industry solutions:**
- **`markupsafe.escape()`**: Escape HTML entities in all dynamic content before injection
- **Bleach library**: Allow-list specific HTML tags while sanitizing others
- **Content Security Policy (CSP)**: HTTP headers to restrict script execution (limited in Streamlit)
- **Avoid `unsafe_allow_html`**: Use native Streamlit components where possible

**Pattern**: Create a safe helper:
```python
from markupsafe import escape

def safe_html(template: str, **kwargs) -> str:
    """Render HTML template with escaped dynamic values."""
    escaped = {k: escape(str(v)) for k, v in kwargs.items()}
    return template.format(**escaped)
```

### R4: Authentication for Streamlit (Community Pattern)

**The problem**: No authentication — anyone with network access can generate articles, spending API credits, or approve articles for publication.

**Industry solutions:**
- **`streamlit-authenticator`**: Library for username/password auth with session cookies
- **Streamlit Community Cloud**: Built-in Google/GitHub OAuth
- **HTTP proxy auth**: Nginx/Caddy with basic auth in front of Streamlit
- **API key gating**: Require an API key in the sidebar before any action

**Pattern**: For a single-user tool, simplest approach is an environment-variable password check:
```python
def require_auth():
    password = st.sidebar.text_input("Password", type="password")
    if password != os.getenv("DASHBOARD_PASSWORD"):
        st.stop()
```

### R5: Consolidating Dashboard with Pipeline (Architecture)

**The problem**: The dashboard reimplements the full pipeline inline instead of calling `pipeline.py`. Bug fixes and improvements must be applied twice.

**Industry solutions:**
- **Single pipeline, multiple frontends**: Dashboard calls `run_pipeline()` from `pipeline.py` via a thin adapter
- **Event-driven architecture**: Pipeline emits progress events; dashboard subscribes and renders
- **Dagster/Prefect UI**: Pipeline definition is separate from execution UI

**Pattern**: Refactor `run_full_pipeline()` in `app.py` to:
1. Map dashboard inputs to `pipeline.py`'s `run_pipeline()` parameters
2. Subscribe to LangGraph stream events for progress updates
3. Map pipeline output dict to dashboard's display format

### R6: JavaScript Safety in Streamlit Components (Security)

**The problem**: `copy_to_clipboard()` injects article content into a JavaScript template literal with incomplete escaping.

**Industry solutions:**
- **`json.dumps()` for JS strings**: Properly escapes all special characters for JavaScript embedding
- **Streamlit clipboard component**: Use a community component like `streamlit-clipboard`
- **postMessage API**: Communicate between iframe and parent safely
- **Data URI approach**: Encode content as base64 data attribute, read from JS

**Pattern**: Replace manual escaping with `json.dumps()`:
```python
import json
safe_text = json.dumps(text)  # Produces valid JS string with all escapes
# In template: const text = {safe_text};  (already quoted by json.dumps)
```

---

## Red Team: Trust-Breaking Attack Scenarios

### Attack 1: "The Phantom Approval" (SEVERITY: CRITICAL)

**Vector**: Review decisions are in-memory only
**Steps**:
1. Reviewer opens review dashboard, reviews 5 articles over 30 minutes
2. Approves 3 articles, rejects 1, escalates 1
3. Reviewer's browser tab crashes (memory leak, Streamlit auto-refresh, sleep)
4. All 5 decisions are lost — `st.session_state.review_queue` reverts to pending
5. Another reviewer opens dashboard — sees all 5 articles as "pending_review"
6. Re-reviews same articles, potentially making different decisions
7. No audit trail of the first reviewer's work (audit log also in-memory)
8. No conflict detection — contradictory decisions possible

**Likelihood**: HIGH (Streamlit session state is inherently volatile)
**Impact**: Complete loss of editorial decisions. No provenance of review actions.

### Attack 2: "The Script Injection Article" (SEVERITY: HIGH)

**Vector**: LLM output + unsafe_allow_html
**Steps**:
1. User provides custom topic: `<img src=x onerror="alert(document.cookie)">`
2. `st.session_state.selected_topic` stores this string
3. After generation, line 2031 renders: `st.markdown(f'<div class="topic-display">{topic}</div>', unsafe_allow_html=True)`
4. Browser executes the injected script
5. Could steal session cookies, redirect to malicious page, or modify page content

**Alternative vector via LLM:**
1. If an LLM is compromised or jailbroken, it could return a title containing `<script>`
2. The title propagates through `selected_topic` to the HTML rendering
3. Every viewer of the results page gets the injected script

**Likelihood**: MEDIUM (requires either malicious input or LLM prompt injection)
**Impact**: XSS execution in reviewer's browser. Potential session hijacking.

### Attack 3: "The Credit Drain" (SEVERITY: HIGH)

**Vector**: No authentication + no rate limiting + no cost tracking
**Steps**:
1. Dashboard is exposed to network (common for Streamlit apps on servers)
2. Attacker discovers the URL and clicks "Generate Article" repeatedly
3. Each click triggers the full pipeline: TopicResearch → Perplexity → Writer → 4 Specialists → QualityGate → Visuals
4. Each run costs $2-$10 in API credits (per Feature 3 analysis)
5. 50 clicks in 30 minutes = $100-$500 in API costs
6. No rate limit, no CAPTCHA, no auth, no budget cap, no alert

**Likelihood**: MEDIUM (requires network exposure, which is common for demo/staging)
**Impact**: Unbounded API cost. No alerting or automatic shutdown.

### Attack 4: "The Refresh Catastrophe" (SEVERITY: HIGH)

**Vector**: Session state lost on browser refresh
**Steps**:
1. User clicks "Generate Article" — 5-minute pipeline starts
2. Pipeline reaches 92% (Quality Gate passed, visuals generating)
3. User accidentally hits F5 (refresh), or browser auto-refreshes
4. `st.session_state.is_running` resets to False
5. Pipeline is still running in the Python process (no cancellation mechanism)
6. Result is written to disk but `st.session_state.generation_complete` is False
7. User sees the hero section again, not the results
8. User clicks "Generate Article" again — runs a SECOND pipeline simultaneously
9. Both pipelines consume API credits. Second pipeline starts from scratch.

**Likelihood**: HIGH (accidental refresh during 5-minute wait is common)
**Impact**: Double API cost. Lost time. User frustration.

### Attack 5: "The Clipboard Injection" (SEVERITY: MEDIUM)

**Vector**: Incomplete JavaScript escaping in copy_to_clipboard
**Steps**:
1. Article contains the text: `</script><script>alert('XSS')</script>`
2. `escaped_text` only handles `\`, backtick, and `${` — NOT `</script>`
3. The template literal breaks: `const text = \`...content...</script><script>alert('XSS')</script>...\``
4. Browser parses `</script>` as end of script block
5. Injected `<script>alert('XSS')</script>` executes in the iframe context

**Likelihood**: LOW (requires specific content in article, but LLMs can produce it)
**Impact**: Script execution in Streamlit component iframe.

### Attack 6: "The .env Corruptor" (SEVERITY: MEDIUM)

**Vector**: save_to_env naive .env parsing
**Steps**:
1. Existing .env contains: `API_KEY="sk-abc123"` (quoted value)
2. User saves a GitHub token via the dashboard
3. `save_to_env()` reads the file, splits on `=`: key=`API_KEY`, value=`"sk-abc123"`
4. Writes back: `API_KEY="sk-abc123"` — preserves quotes in this case
5. But if .env has: `PROMPT="Hello = World"` (value contains `=`)
6. Split on first `=` gets: key=`PROMPT`, value=`"Hello = World"`
7. Works for simple cases, BUT:
8. Multiline values, comments, or export prefixes are destroyed
9. `export API_KEY=abc` becomes `export API_KEY`=`abc` (broken)

**Likelihood**: LOW (depends on .env file complexity)
**Impact**: Potential .env file corruption breaking API keys for all pipelines.

### Attack 7: "The Pipeline Roulette" (SEVERITY: HIGH — Cross-Feature)

**Vector**: Dashboard pipeline ≠ LangGraph pipeline
**Steps**:
1. Developer fixes "no claims = pass" bug in `pipeline.py` (Feature 2 Upgrade 1)
2. Developer tests via CLI: `python pipeline.py --topic "AI Safety"` — verification works correctly
3. User generates article via Dashboard — uses `app.py:run_full_pipeline()`
4. Dashboard pipeline uses `QualityGate.process()` (line 1535), NOT the LangGraph pipeline
5. The QualityGate STILL has the old fail-open bug (if not also fixed there)
6. Dashboard produces articles that pass verification incorrectly
7. CLI produces correct results. Dashboard produces buggy results. Same codebase.

**Likelihood**: HIGH (this is the current state — ANY fix to pipeline.py doesn't propagate to dashboard)
**Impact**: Inconsistent quality between CLI and dashboard. Bugs "fixed" but still present in production.

---

## Summary Verdict

| Dimension | Rating | Notes |
|-----------|--------|-------|
| Error handling | F | Single try/except for entire pipeline, bare excepts elsewhere |
| State persistence | F | All review decisions in-memory, lost on refresh |
| Security | D | No auth, XSS via unsafe_allow_html, JS injection in clipboard |
| Consistency | F | Inline pipeline diverges from LangGraph pipeline |
| User experience | C | Nice UI/UX design, but catastrophic failure modes |
| Architecture | D | Two disconnected dashboards, inline pipeline, no shared state |
| Cost protection | F | No rate limit, no auth, no budget cap on generation |
