# Feature: Dashboard — Current State

**Agent**: Feature Deconstruction Agent
**Date**: 2026-02-08
**Files Analyzed**:
- `app.py` (2076 lines) — Main generation dashboard (Streamlit)
- `execution/dashboard/app.py` (685 lines) — HITL review dashboard (Streamlit)
- `execution/dashboard/__init__.py` (11 lines) — Package stub
- `execution/quality_gate.py` (461 lines) — Used by main dashboard's pipeline

---

## Architecture Summary

### TWO Separate Dashboard Apps

| Dashboard | File | Lines | Purpose | Run Command |
|-----------|------|-------|---------|-------------|
| Generation | `app.py` | 2076 | One-click article generation | `streamlit run app.py` |
| Review | `execution/dashboard/app.py` | 685 | HITL review & approval | `streamlit run execution/dashboard/app.py` |

**These two dashboards share NO state.** The generation dashboard writes articles to disk. The review dashboard reads articles from disk. There is no real-time connection, no shared database, no event system.

### Main Dashboard Structure (app.py)

```
Lines 1-35:     Imports and page config
Lines 36-968:   CSS (~933 lines — 45% of the file)
Lines 970-1155: Session state, history, utilities
Lines 1157-1603: run_full_pipeline() — INLINE PIPELINE (447 lines)
Lines 1606-2076: main() — UI rendering, event handling
```

### Generation Pipeline (Inline in app.py)

```
TOPIC RESEARCH (TopicResearchAgent or CommitAnalysisAgent)
    → FACT RESEARCH (Perplexity, optional)
    → AGENT INIT (Editor, Critic, Writer, Visuals)
    → OUTLINE → CRITIQUE → REFINE
    → DRAFT (writer.write_section)
    → 4 SPECIALISTS (Hook, Storytelling, Voice, Value Density)
    → FINAL POLISH (SpecialistAgent)
    → DRAFT VERIFICATION (Perplexity verify_draft, optional)
    → QUALITY GATE (QualityGate.process)
    → VISUALS (VisualsAgent)
    → SAVE (write to disk)
```

### Review Dashboard Structure

```
Sidebar: Navigation, reviewer name, queue stats
Pages:
  - Review Queue:    List pending articles, filter/sort
  - Article Review:  Content tab, Fact Check tab, Provenance tab, Decision tab
  - Escalations:     List escalated articles
  - Audit Trail:     In-memory action log
  - Settings:        Quality thresholds, API status (read-only)
```

---

## Critical Findings

### F1: No Error Boundaries — One Exception Crashes Everything (CRITICAL)

**Location**: `app.py:1970-1987`

```python
try:
    result = run_full_pipeline(update_progress, update_status, provided_topic=...)
    ...
except Exception as e:
    st.session_state.is_running = False
    st.error(f"Error: {str(e)}")
    st.exception(e)
```

This is the ONLY error boundary for the entire pipeline. Inside `run_full_pipeline()`, individual agent failures are NOT caught per-phase. A single agent crash (e.g., WriterAgent fails) kills the entire pipeline with no partial recovery.

**Specific uncaught failure points inside `run_full_pipeline()`:**
- `editor.create_outline(topic)` (line 1327) — no try/except
- `critic.critique_outline(outline)` (line 1331) — no try/except
- `editor.call_llm(...)` (line 1336) — no try/except, returns error string on failure
- `writer.write_section(...)` (line 1356) — no try/except, returns error string on failure
- Each specialist `.refine(draft)` call (line 1441) — no try/except
- `quality_gate.process(...)` (line 1535) — no try/except

Only fact research (line 1305) and draft verification (line 1521) have individual try/except blocks.

### F2: Review Dashboard Decisions Don't Persist (CRITICAL)

**Location**: `execution/dashboard/app.py:502-506`

```python
# Update in queue
for item in st.session_state.review_queue:
    if item["id"] == article["id"]:
        item["status"] = new_status
        break
```

Approval/rejection decisions only update `st.session_state`. They are NOT written to disk, not written to a database, and not written to the provenance file. A browser refresh loses ALL decisions.

Similarly, the audit trail (line 148):
```python
st.session_state.audit_log.append(entry)
```

Audit log is in-memory only. All audit records are lost on page refresh.

### F3: Inline Pipeline Differs from LangGraph Pipeline (HIGH)

Cross-referencing with Feature 3 analysis:

| Capability | LangGraph (`pipeline.py`) | Dashboard (`app.py`) |
|-----------|--------------------------|---------------------|
| Research | Gemini → Perplexity fallback | TopicResearchAgent (separate) |
| Generation | WriterAgent.call_llm | WriterAgent.write_section |
| Specialists | None | 4 specialists + final polish |
| Verification | FactVerificationAgent (full) | Perplexity verify_draft (optional) |
| Review | AdversarialPanelAgent | QualityGate.process (separate code) |
| Revision loop | LangGraph conditional edges | QualityGate internal loop |
| State management | StateGraph dict + stream | st.session_state |
| Provenance | Full ContentProvenance | None |
| Cost tracking | None | None |
| Error routing | Escalate on failure | Crash on failure |

The dashboard pipeline is 447 lines of inline code that reimplements what `pipeline.py` (1083 lines) does with different agents, different ordering, and different error handling.

### F4: Bare except Blocks Swallow Errors Silently (HIGH)

**Location**: `app.py:1013, 1027, 1049`

```python
# Line 1013 (get_draft_history):
except:
    date_formatted = "Unknown"

# Line 1027 (get_draft_history):
except:
    pass

# Line 1049 (load_draft):
except:
    return None
```

Three bare `except:` blocks catch ALL exceptions including `KeyboardInterrupt`, `SystemExit`, and `MemoryError`. File corruption, permission errors, and encoding errors are all silently swallowed.

### F5: XSS Risk via unsafe_allow_html (HIGH)

**Location**: Multiple locations throughout `app.py`

The dashboard uses `st.markdown(..., unsafe_allow_html=True)` extensively (40+ occurrences). Most inject CSS or static HTML, but some include dynamic data:

```python
# Line 1741 - article title from file system:
st.markdown(f'<div class="topic-display">{st.session_state.viewing_history["title"]}</div>', unsafe_allow_html=True)

# Line 2031 - topic from LLM output:
st.markdown(f'<div class="topic-display">{st.session_state.selected_topic}</div>', unsafe_allow_html=True)
```

If an article title or LLM output contains `<script>` tags or other HTML, it gets rendered directly. The LLM can inject arbitrary HTML into the page.

### F6: copy_to_clipboard Uses Unescaped JavaScript (HIGH)

**Location**: `app.py:1086-1154`

```python
escaped_text = text.replace('\\', '\\\\').replace('`', '\\`').replace('${', '\\${')
...
components.html(f'''
    <script>
        function copyToClipboard(btn) {{
            const text = `{escaped_text}`;
            ...
    </script>
''', height=50)
```

The escaping is incomplete. Text containing `</script>` would break out of the script tag. Text with `'` or `"` in specific positions could enable injection. Article content (which comes from LLM output) is passed directly into a JavaScript template literal.

### F7: No Authentication or Authorization (HIGH)

Neither dashboard has any authentication. Anyone with network access can:
- Generate articles (consuming API credits)
- Approve/reject articles in the review dashboard
- View all historical articles
- Access API key status information

### F8: Session State Lost on Refresh (MEDIUM)

All state is in `st.session_state`:
- `generation_complete`, `article_content`, `image_paths` — generation results
- `is_running` — pipeline execution state
- `review_queue`, `audit_log` — review decisions

Browser refresh, tab close, or Streamlit rerun cycle resets ALL of this. A 5-minute pipeline run that completes successfully can be lost if the user accidentally refreshes before copying/downloading.

### F9: status_callback Is a No-Op (MEDIUM)

**Location**: `app.py:1967-1968`

```python
def update_status(message):
    pass  # Status is now integrated into the progress card
```

The status callback passed to `run_full_pipeline()` does nothing. Pipeline progress messages like "Writing first draft..." and "Quality gate review..." are generated but never displayed. The progress UI only shows phase names from the progress value, not the actual status messages.

### F10: No Input Validation on Custom Topic (MEDIUM)

**Location**: `app.py:1799-1801`

```python
topic_input = st.text_input(
    "What do you want to write about?",
    value=st.session_state.custom_topic,
    ...
)
```

No length limit, no content filtering, no sanitization. A user could enter:
- An extremely long string (causing LLM token limit issues)
- Prompt injection attacks that modify agent behavior
- HTML/JavaScript that gets rendered via `unsafe_allow_html`

### F11: Two Dashboard Apps, Zero Shared State (MEDIUM)

The generation dashboard (`app.py`) and review dashboard (`execution/dashboard/app.py`) are completely independent Streamlit apps:
- Run on different ports
- No shared database
- No event bus
- Review dashboard discovers articles by scanning the file system
- No way for review decisions to trigger pipeline re-runs

### F12: save_to_env Writes Tokens to .env (LOW)

**Location**: `app.py:1053-1076`

```python
def save_to_env(key: str, value: str):
    ...
    with open(env_path, 'w') as f:
        for k, v in existing.items():
            f.write(f"{k}={v}\n")
```

The GitHub token save feature reads the entire `.env` file, parses it naively (no handling for comments, multiline values, or quoted strings), modifies it, and writes it back. This could corrupt other env vars with special characters.

---

## State Management Diagram

```
app.py (Generation Dashboard):
  st.session_state ─── generation_complete, article_content, image_paths
         │              is_running, selected_topic, topic_reasoning
         │              data_source, custom_topic, viewing_history
         │
         └─── Persisted: Only the final .md file (written to OUTPUT_DIR)
              Everything else is lost on refresh.

execution/dashboard/app.py (Review Dashboard):
  st.session_state ─── review_queue (loaded from file scan)
         │              current_article, audit_log, reviewer_name
         │
         └─── Persisted: NOTHING. Decisions are in-memory only.
              Refresh = all review state lost.
```

## Error Handling Summary

| Location | Error Type | Handling |
|----------|-----------|----------|
| `run_full_pipeline()` outer | Exception | `st.error()` + `st.exception()` |
| Fact research (line 1305) | Exception | Print + "skipped (continuing)" |
| Draft verification (line 1521) | Exception | Print + "skipped (continuing)" |
| `get_draft_history()` (line 1013) | bare except | Swallow, use "Unknown" |
| `get_draft_history()` (line 1027) | bare except | Swallow, pass |
| `load_draft()` (line 1049) | bare except | Return None |
| All agent calls inside pipeline | None | Uncaught → bubbles to outer try |
| Review dashboard decisions | None | In-memory only, no error handling |
