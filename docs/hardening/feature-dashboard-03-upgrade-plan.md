# Feature: Dashboard — Upgrade Plan

**Agent**: Engineering Upgrade Agent
**Date**: 2026-02-08
**Constraint**: No rewrites. Incremental fixes only.

---

## Upgrade 1: Add Per-Phase Error Boundaries in run_full_pipeline

**Priority**: CRITICAL (ship-blocking)
**Effort**: S
**Regression Risk**: Low

### What to Change

Wrap each major phase in `run_full_pipeline()` with individual try/except blocks:

**In `app.py:1324-1340` (outline phase):**

```python
# BEFORE (no error handling):
outline = editor.create_outline(topic)
critique = critic.critique_outline(outline)
refined_outline = editor.call_llm(f"Refine this outline...")

# AFTER (per-phase error boundary):
try:
    outline = editor.create_outline(topic)
except Exception as e:
    raise PipelinePhaseError("outline", f"Outline creation failed: {e}")

try:
    critique = critic.critique_outline(outline)
except Exception as e:
    # Outline succeeded, critique failed — continue with uncritiqued outline
    critique = ""
    status_callback(f"Critique skipped ({e})")

try:
    refined_outline = editor.call_llm(f"Refine this outline...")
    # Validate output is not an error string
    if not refined_outline or len(refined_outline.strip()) < 50:
        refined_outline = outline  # Fall back to original outline
except Exception as e:
    refined_outline = outline
    status_callback(f"Refinement skipped ({e})")
```

**Same pattern for draft generation (line 1356):**

```python
try:
    draft = writer.write_section(refined_outline, critique=draft_instruction, source_type=source_type)

    # Validate: catch error string returns from BaseAgent
    error_prefixes = ["Error:", "Groq Error:", "Gemini Error:", "OpenAI Error:"]
    if any(draft.strip().startswith(p) for p in error_prefixes):
        raise ValueError(f"Writer returned error: {draft[:100]}")
    if len(draft.strip()) < 200:
        raise ValueError(f"Draft too short ({len(draft)} chars)")
except Exception as e:
    raise PipelinePhaseError("draft", f"Draft generation failed: {e}")
```

**Same pattern for each specialist (line 1438-1443):**

```python
for name, status_msg, instruction in specialists:
    status_callback(status_msg)
    try:
        specialist = SpecialistAgent(constraint_name=name, constraint_instruction=instruction)
        result = specialist.refine(draft)
        # Validate: specialist didn't return error string or empty
        if result and len(result.strip()) > len(draft.strip()) * 0.3:
            draft = result
        else:
            status_callback(f"{name} output invalid, keeping previous draft")
    except Exception as e:
        status_callback(f"{name} failed ({e}), skipping")
    current += progress_per
    progress_callback(current)
```

### Define PipelinePhaseError

```python
class PipelinePhaseError(Exception):
    """Raised when a critical pipeline phase fails."""
    def __init__(self, phase: str, message: str):
        self.phase = phase
        super().__init__(f"[{phase}] {message}")
```

---

## Upgrade 2: Persist Review Decisions to Disk

**Priority**: CRITICAL (ship-blocking)
**Effort**: S
**Regression Risk**: Low

### What to Change

**In `execution/dashboard/app.py`, add persistence functions:**

```python
REVIEW_STATE_DIR = OUTPUT_DIR / "review_state"

def save_review_decision(article_id: str, status: str, reviewer: str, notes: str):
    """Persist review decision to JSON sidecar file."""
    REVIEW_STATE_DIR.mkdir(parents=True, exist_ok=True)
    filepath = REVIEW_STATE_DIR / f"{article_id}_review.json"

    decision = {
        "article_id": article_id,
        "status": status,
        "reviewer": reviewer,
        "notes": notes,
        "timestamp": datetime.now().isoformat(),
    }

    # Append to history (don't overwrite previous decisions)
    history = []
    if filepath.exists():
        with open(filepath, 'r') as f:
            existing = json.load(f)
            history = existing.get("history", [])

    history.append(decision)

    with open(filepath, 'w') as f:
        json.dump({"current": decision, "history": history}, f, indent=2)

def save_audit_log(entries: list):
    """Persist audit log to JSON file."""
    REVIEW_STATE_DIR.mkdir(parents=True, exist_ok=True)
    filepath = REVIEW_STATE_DIR / "audit_log.json"
    with open(filepath, 'w') as f:
        json.dump(entries, f, indent=2, default=str)
```

**Update decision submission (line 502-517):**

```python
# After updating session state, also persist:
save_review_decision(article["id"], new_status, st.session_state.reviewer_name, notes)
save_audit_log(st.session_state.audit_log)
```

**Update `load_review_queue()` to read persisted decisions:**

```python
# After building queue item, check for persisted decision:
review_file = REVIEW_STATE_DIR / f"{article_id}_review.json"
if review_file.exists():
    with open(review_file, 'r') as f:
        review_data = json.load(f)
        item["status"] = review_data["current"]["status"]
```

---

## Upgrade 3: Sanitize HTML Output (Fix XSS)

**Priority**: HIGH
**Effort**: S
**Regression Risk**: None

### What to Change

Add a sanitization helper and apply it to ALL dynamic content rendered with `unsafe_allow_html=True`:

```python
import html

def safe_html(text: str) -> str:
    """Escape HTML entities in dynamic text before injection into unsafe_allow_html."""
    return html.escape(str(text)) if text else ""
```

**Apply throughout `app.py`:**

```python
# BEFORE (line 1741):
st.markdown(f'<div class="topic-display">{st.session_state.viewing_history["title"]}</div>', unsafe_allow_html=True)

# AFTER:
st.markdown(f'<div class="topic-display">{safe_html(st.session_state.viewing_history["title"])}</div>', unsafe_allow_html=True)

# BEFORE (line 2031):
st.markdown(f'<div class="topic-display">{st.session_state.selected_topic}</div>', unsafe_allow_html=True)

# AFTER:
st.markdown(f'<div class="topic-display">{safe_html(st.session_state.selected_topic)}</div>', unsafe_allow_html=True)

# Same for topic_reasoning (line 2033), and ALL other dynamic injections
```

### Also fix copy_to_clipboard (line 1086)

```python
# BEFORE (incomplete escaping):
escaped_text = text.replace('\\', '\\\\').replace('`', '\\`').replace('${', '\\${')

# AFTER (proper JS string escaping):
import json
safe_js_text = json.dumps(text)  # json.dumps produces valid JS string with all escapes
# In template: const text = {safe_js_text};  (already includes quotes)
```

---

## Upgrade 4: Add Basic Authentication

**Priority**: HIGH
**Effort**: S
**Regression Risk**: Low

### What to Change

Add a simple password gate at the top of both dashboard apps:

**In `app.py`, at the start of `main()`:**

```python
def check_auth():
    """Simple password gate for dashboard access."""
    dashboard_password = os.getenv("DASHBOARD_PASSWORD")
    if not dashboard_password:
        return True  # No password configured, allow access (dev mode)

    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if not st.session_state.authenticated:
        st.markdown('<p class="main-header">GhostWriter</p>', unsafe_allow_html=True)
        password = st.text_input("Dashboard Password", type="password")
        if st.button("Login"):
            if password == dashboard_password:
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("Incorrect password")
        st.stop()

def main():
    check_auth()
    init_session_state()
    ...
```

Add `DASHBOARD_PASSWORD` to `.env.example`.

---

## Upgrade 5: Add Generation Rate Limiting

**Priority**: HIGH
**Effort**: S
**Regression Risk**: None

### What to Change

Add a cooldown mechanism to prevent rapid-fire generation:

**In `app.py`, before the Generate button handler:**

```python
GENERATION_COOLDOWN_SECONDS = 30  # Minimum time between generations

def can_generate() -> tuple[bool, str]:
    """Check if generation is allowed (cooldown, concurrent limit)."""
    last_gen = st.session_state.get("last_generation_time", 0)
    elapsed = time.time() - last_gen

    if elapsed < GENERATION_COOLDOWN_SECONDS:
        remaining = int(GENERATION_COOLDOWN_SECONDS - elapsed)
        return False, f"Please wait {remaining}s before generating again"

    if st.session_state.is_running:
        return False, "A generation is already in progress"

    return True, ""
```

**Update the generate button (line 1830-1838):**

```python
can_gen, gen_message = can_generate()
if st.button(
    "Generate Article",
    type="primary",
    use_container_width=True,
    disabled=not can_gen or st.session_state.is_running,
):
    st.session_state.is_running = True
    st.session_state.last_generation_time = time.time()
    st.rerun()

if gen_message:
    st.caption(gen_message)
```

---

## Upgrade 6: Replace Bare except Blocks

**Priority**: HIGH
**Effort**: S
**Regression Risk**: Low

### What to Change

**In `app.py:1013`:**
```python
# BEFORE:
except:
    date_formatted = "Unknown"

# AFTER:
except (ValueError, IndexError, OSError) as e:
    date_formatted = "Unknown"
```

**In `app.py:1027`:**
```python
# BEFORE:
except:
    pass

# AFTER:
except (IOError, UnicodeDecodeError) as e:
    pass  # File read error, skip word count
```

**In `app.py:1049`:**
```python
# BEFORE:
except:
    return None

# AFTER:
except (IOError, UnicodeDecodeError, PermissionError) as e:
    print(f"Failed to load draft {filepath}: {e}")
    return None
```

---

## Upgrade 7: Rename "C2PA manifest" to "Content Metadata"

**Priority**: MEDIUM (ship-blocking per dossier)
**Effort**: S
**Regression Risk**: None

### What to Change

**In `execution/dashboard/app.py:31`:**

```python
# BEFORE:
from execution.provenance import (
    ContentProvenance, ProvenanceTracker,
    generate_inline_disclosure, generate_c2pa_manifest
)

# AFTER:
from execution.provenance import (
    ContentProvenance, ProvenanceTracker,
    generate_inline_disclosure, generate_content_metadata
)
```

Also rename the function in `execution/provenance.py` (keeping an alias for backward compatibility):

```python
def generate_content_metadata(provenance: ContentProvenance) -> dict:
    """Generate content metadata manifest."""
    ...

# Backward compatibility alias
generate_c2pa_manifest = generate_content_metadata
```

Update ALL references to "C2PA manifest" in UI text and variable names to "Content Metadata."

---

## Upgrade 8: Wire status_callback to UI

**Priority**: MEDIUM
**Effort**: S
**Regression Risk**: None

### What to Change

**In `app.py:1967-1968`, replace no-op with actual status display:**

```python
# BEFORE:
def update_status(message):
    pass  # Status is now integrated into the progress card

# AFTER:
status_display = st.empty()

def update_status(message):
    status_display.markdown(
        f'<div class="status-text">{safe_html(message)}</div>',
        unsafe_allow_html=True
    )
```

This gives users real-time visibility into what the pipeline is doing, not just a progress percentage.

---

## Upgrade 9: Save Pipeline Results Before Displaying (Crash Resilience)

**Priority**: MEDIUM
**Effort**: S
**Regression Risk**: None

### What to Change

Move the file save to BEFORE the `st.rerun()` call, and save a result metadata file alongside the article:

**In `app.py:1970-1982`:**

```python
try:
    result = run_full_pipeline(update_progress, update_status, provided_topic=...)

    # Save results to session state FIRST (before any rerun)
    st.session_state.generation_complete = True
    st.session_state.article_content = result['content']
    st.session_state.filepath = result['filepath']
    ...

    # Also save a result metadata file for crash recovery
    meta_filepath = Path(result['filepath']).with_suffix('.meta.json')
    with open(meta_filepath, 'w') as f:
        json.dump({
            'topic': result['topic'],
            'quality_score': result.get('quality_score'),
            'quality_passed': result.get('quality_passed'),
            'topic_reasoning': result.get('topic_reasoning', ''),
            'perplexity_used': result.get('perplexity_used', False),
            'timestamp': datetime.now().isoformat(),
        }, f, indent=2)

    st.rerun()
except Exception as e:
    ...
```

The `.md` file is already saved inside `run_full_pipeline()` at line 1583. This upgrade adds metadata for the results display to be recoverable from disk if session state is lost.

---

## Implementation Order

```
1. [CRITICAL] Upgrade 1: Per-phase error boundaries (prevents full-crash)
2. [CRITICAL] Upgrade 2: Persist review decisions (prevents lost approvals)
3. [HIGH]     Upgrade 3: Sanitize HTML output (fix XSS)
4. [HIGH]     Upgrade 4: Add basic authentication (prevent unauthorized access)
5. [HIGH]     Upgrade 5: Add generation rate limiting (prevent credit drain)
6. [HIGH]     Upgrade 6: Replace bare except blocks (proper error handling)
7. [MEDIUM]   Upgrade 7: Rename C2PA to Content Metadata (ship-blocking naming fix)
8. [MEDIUM]   Upgrade 8: Wire status_callback to UI (user visibility)
9. [MEDIUM]   Upgrade 9: Save pipeline metadata for crash recovery (resilience)
```

## Estimated Total Effort: 2-3 days for a single engineer

## Files Modified

| File | Changes |
|------|---------|
| `app.py` | Upgrades 1, 3, 4, 5, 6, 8, 9: error boundaries, HTML sanitization, auth, rate limit, bare excepts, status display, metadata save |
| `execution/dashboard/app.py` | Upgrades 2, 3, 4, 7: decision persistence, HTML sanitization, auth, C2PA rename |
| `execution/provenance.py` | Upgrade 7: Rename `generate_c2pa_manifest` → `generate_content_metadata` |
| `.env.example` | Upgrade 4: Add `DASHBOARD_PASSWORD` |

## Cross-Feature Dependencies

| Upgrade | Depends On |
|---------|------------|
| Upgrade 1 (error boundaries) | Benefits from Feature 1 Upgrade 1 (BaseAgent exceptions) — without it, error strings still pass through |
| Upgrade 3 (HTML sanitization) | Independent, but LLM output validation (Feature 3 Upgrade 5) reduces injection surface |
| Upgrade 7 (C2PA rename) | Independent naming change, but should be done when touching provenance.py |
| Pipeline consolidation | Feature 3 Upgrade 9 — replacing inline pipeline with `run_pipeline()` call is the long-term fix for Attack 7 |
