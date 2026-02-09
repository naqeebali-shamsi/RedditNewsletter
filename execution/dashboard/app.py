"""
Human-in-the-Loop Review Dashboard - Streamlit App.

WSJ-tier content review interface for:
- Review queue management
- Side-by-side source/draft comparison
- Fact-check highlighting
- Approval workflow
- Audit trail display
- Escalation handling

Run with: streamlit run execution/dashboard/app.py
"""

import streamlit as st
import json
import os
import sys
from pathlib import Path
from datetime import datetime
from execution.utils.datetime_utils import utc_iso
from typing import Dict, List, Optional
import pandas as pd

# Add parent paths
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from execution.config import config, OUTPUT_DIR
from execution.provenance import (
    ContentProvenance, ProvenanceTracker,
    generate_inline_disclosure, generate_content_metadata
)
from execution.sources.database import (
    save_review_decision as db_save_review_decision,
    get_review_decision_for_article,
    get_review_history,
    get_decision_stats,
)

REVIEW_STATE_DIR = OUTPUT_DIR / "review_state"

# Page config
st.set_page_config(
    page_title="GhostWriter Review Dashboard",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .review-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        border-left: 4px solid #007bff;
    }
    .escalation-card {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
    }
    .approved-card {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
    }
    .rejected-card {
        background-color: #f8d7da;
        border-left: 4px solid #dc3545;
    }
    .fact-verified {
        background-color: #d4edda;
        padding: 2px 6px;
        border-radius: 3px;
    }
    .fact-unverified {
        background-color: #fff3cd;
        padding: 2px 6px;
        border-radius: 3px;
    }
    .fact-false {
        background-color: #f8d7da;
        padding: 2px 6px;
        border-radius: 3px;
    }
    .score-high { color: #28a745; font-weight: bold; }
    .score-medium { color: #ffc107; font-weight: bold; }
    .score-low { color: #dc3545; font-weight: bold; }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# State Management
# ============================================================================

def init_session_state():
    """Initialize session state variables."""
    if 'review_queue' not in st.session_state:
        st.session_state.review_queue = load_review_queue()
    if 'current_article' not in st.session_state:
        st.session_state.current_article = None
    if 'audit_log' not in st.session_state:
        st.session_state.audit_log = []
    if 'reviewer_name' not in st.session_state:
        st.session_state.reviewer_name = ""


def load_review_queue() -> List[Dict]:
    """Load pending reviews from output directory.

    Checks the SQLAlchemy database first for persisted review decisions,
    falling back to legacy JSON sidecar files if no DB record exists.
    """
    queue = []
    output_dir = OUTPUT_DIR

    if not output_dir.exists():
        return queue

    # Look for article files and their provenance
    for article_file in output_dir.glob("article_*.md"):
        article_id = article_file.stem.replace("article_", "")

        # Check for provenance file
        provenance_file = output_dir / "provenance" / f"{article_id}_provenance.json"

        item = {
            "id": article_id,
            "file": str(article_file),
            "title": f"Article {article_id[:8]}...",
            "status": "pending_review",
            "quality_score": 0,
            "created_at": datetime.fromtimestamp(article_file.stat().st_mtime).isoformat(),
            "provenance": None
        }

        if provenance_file.exists():
            with open(provenance_file, 'r') as f:
                item["provenance"] = json.load(f)
                item["quality_score"] = item["provenance"].get("quality_score", 0)
                item["title"] = item["provenance"].get("topic", item["title"])

        # Check DB first for persisted review decision
        db_decision = get_review_decision_for_article(article_id)
        if db_decision:
            item["status"] = db_decision["decision"]
        else:
            # Fallback: legacy JSON sidecar files
            review_file = REVIEW_STATE_DIR / f"{article_id}_review.json"
            if review_file.exists():
                with open(review_file, 'r') as f:
                    review_data = json.load(f)
                    item["status"] = review_data["current"]["status"]
                # Migrate legacy JSON decision to DB
                _migrate_json_decision(article_id, review_file, item)

        queue.append(item)

    # Sort by creation date (newest first)
    queue.sort(key=lambda x: x["created_at"], reverse=True)
    return queue


def _migrate_json_decision(article_id: str, json_path: Path, item: Dict):
    """Migrate a legacy JSON review decision to the database."""
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        current = data.get("current", {})
        db_save_review_decision(
            article_id=article_id,
            decision=current.get("status", "pending_review"),
            notes=current.get("notes", ""),
            topic=item.get("title", ""),
            quality_score=item.get("quality_score", 0.0),
        )
    except Exception:
        pass  # Non-critical: migration is best-effort


def log_audit_action(action: str, article_id: str, details: Dict = None):
    """Log an audit action."""
    entry = {
        "timestamp": utc_iso(),
        "action": action,
        "article_id": article_id,
        "reviewer": st.session_state.reviewer_name or "Anonymous",
        "details": details or {}
    }
    st.session_state.audit_log.append(entry)


def save_review_decision(article_id: str, status: str, reviewer: str, notes: str):
    """Persist review decision to the SQLAlchemy database.

    Each call inserts a new row so full decision history is preserved.
    The most-recent row for a given article_id is treated as current.
    """
    # Resolve topic / quality from current session queue
    topic = ""
    quality_score = 0.0
    for item in st.session_state.get("review_queue", []):
        if item["id"] == article_id:
            topic = item.get("title", "")
            quality_score = item.get("quality_score", 0.0)
            break

    db_save_review_decision(
        article_id=article_id,
        decision=status,
        notes=f"[{reviewer}] {notes}" if reviewer else notes,
        topic=topic,
        quality_score=quality_score,
    )


def save_audit_log(entries: list):
    """Persist audit log to JSON file."""
    REVIEW_STATE_DIR.mkdir(parents=True, exist_ok=True)
    filepath = REVIEW_STATE_DIR / "audit_log.json"
    with open(filepath, 'w') as f:
        json.dump(entries, f, indent=2, default=str)


# ============================================================================
# Sidebar - Navigation & Settings
# ============================================================================

def render_sidebar():
    """Render sidebar navigation."""
    with st.sidebar:
        st.title("📝 GhostWriter")
        st.caption("Human-in-the-Loop Review")

        st.divider()

        # Reviewer identification
        st.session_state.reviewer_name = st.text_input(
            "Reviewer Name",
            value=st.session_state.reviewer_name,
            placeholder="Enter your name"
        )

        st.divider()

        # Navigation
        page = st.radio(
            "Navigation",
            ["Review Queue", "Article Review", "Escalations", "Audit Trail", "Settings"],
            label_visibility="collapsed"
        )

        st.divider()

        # Queue stats
        queue = st.session_state.review_queue
        pending = len([q for q in queue if q["status"] == "pending_review"])
        approved = len([q for q in queue if q["status"] == "approved"])
        escalated = len([q for q in queue if q["status"] == "escalated"])

        st.metric("Pending Review", pending)
        col1, col2 = st.columns(2)
        col1.metric("Approved", approved)
        col2.metric("Escalated", escalated)

        st.divider()

        # Refresh button
        if st.button("🔄 Refresh Queue", use_container_width=True):
            st.session_state.review_queue = load_review_queue()
            st.rerun()

        return page


# ============================================================================
# Page: Review Queue
# ============================================================================

def render_review_queue():
    """Render the review queue page."""
    st.header("📋 Review Queue")

    queue = st.session_state.review_queue

    if not queue:
        st.info("No articles in the review queue. Run the pipeline to generate content.")
        return

    # Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        status_filter = st.selectbox(
            "Status Filter",
            ["All", "Pending Review", "Approved", "Rejected", "Escalated"]
        )
    with col2:
        score_filter = st.slider("Min Quality Score", 0.0, 10.0, 0.0)
    with col3:
        sort_by = st.selectbox("Sort By", ["Newest", "Oldest", "Score (High)", "Score (Low)"])

    # Apply filters
    filtered = queue
    if status_filter != "All":
        status_map = {
            "Pending Review": "pending_review",
            "Approved": "approved",
            "Rejected": "rejected",
            "Escalated": "escalated"
        }
        filtered = [q for q in filtered if q["status"] == status_map.get(status_filter, "")]

    filtered = [q for q in filtered if q["quality_score"] >= score_filter]

    # Sort
    if sort_by == "Newest":
        filtered.sort(key=lambda x: x["created_at"], reverse=True)
    elif sort_by == "Oldest":
        filtered.sort(key=lambda x: x["created_at"])
    elif sort_by == "Score (High)":
        filtered.sort(key=lambda x: x["quality_score"], reverse=True)
    elif sort_by == "Score (Low)":
        filtered.sort(key=lambda x: x["quality_score"])

    st.divider()

    # Display queue items
    for item in filtered:
        score = item["quality_score"]
        score_class = "score-high" if score >= 7 else ("score-medium" if score >= 5 else "score-low")

        status_emoji = {
            "pending_review": "⏳",
            "approved": "✅",
            "rejected": "❌",
            "escalated": "⚠️"
        }.get(item["status"], "❓")

        with st.container():
            col1, col2, col3, col4 = st.columns([3, 1, 1, 1])

            with col1:
                st.markdown(f"**{item['title'][:50]}...**")
                st.caption(f"ID: {item['id'][:12]}... | Created: {item['created_at'][:16]}")

            with col2:
                st.markdown(f"<span class='{score_class}'>{score}/10</span>", unsafe_allow_html=True)

            with col3:
                st.write(f"{status_emoji} {item['status'].replace('_', ' ').title()}")

            with col4:
                if st.button("Review", key=f"review_{item['id']}", use_container_width=True):
                    st.session_state.current_article = item
                    st.rerun()

        st.divider()


# ============================================================================
# Page: Article Review
# ============================================================================

def render_article_review():
    """Render the article review page."""
    st.header("📖 Article Review")

    article = st.session_state.current_article

    if not article:
        st.info("Select an article from the Review Queue to begin.")
        return

    # Article header
    col1, col2 = st.columns([3, 1])
    with col1:
        st.subheader(article["title"])
        st.caption(f"Content ID: {article['id']}")
    with col2:
        score = article["quality_score"]
        score_color = "green" if score >= 7 else ("orange" if score >= 5 else "red")
        st.metric("Quality Score", f"{score}/10", delta=None)

    st.divider()

    # Tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(["📄 Content", "✓ Fact Check", "📊 Provenance", "⚖️ Decision"])

    # Tab 1: Content View
    with tab1:
        render_content_tab(article)

    # Tab 2: Fact Check
    with tab2:
        render_fact_check_tab(article)

    # Tab 3: Provenance
    with tab3:
        render_provenance_tab(article)

    # Tab 4: Decision
    with tab4:
        render_decision_tab(article)


def render_content_tab(article: Dict):
    """Render content comparison view."""
    st.subheader("Content Review")

    # Load article content
    if os.path.exists(article["file"]):
        with open(article["file"], 'r', encoding='utf-8') as f:
            content = f.read()

        # Side-by-side view option
        view_mode = st.radio("View Mode", ["Full Article", "Side-by-Side"], horizontal=True)

        if view_mode == "Full Article":
            st.markdown(content)
        else:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Source Material**")
                if article.get("provenance", {}).get("source_url"):
                    st.info(f"Source: {article['provenance']['source_url']}")
                else:
                    st.info("Source material not available")
            with col2:
                st.markdown("**Generated Draft**")
                st.markdown(content)
    else:
        st.error("Article file not found")


def render_fact_check_tab(article: Dict):
    """Render fact check highlighting panel."""
    st.subheader("Fact Verification Status")

    provenance = article.get("provenance", {})

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Verified Claims", provenance.get("verified_claims_count", 0))
    col2.metric("Verification Passed", "Yes" if provenance.get("fact_verification_passed") else "No")
    col3.metric("WSJ Checklist", "Passed" if provenance.get("wsj_checklist_passed") else "Review Needed")
    col4.metric("Human Reviewed", "Yes" if provenance.get("human_reviewed") else "No")

    st.divider()

    # Fact verification details (if available in provenance)
    actions = provenance.get("actions", [])
    verification_actions = [a for a in actions if a.get("action_type") == "verified"]

    if verification_actions:
        st.markdown("**Verification Details**")
        for action in verification_actions:
            details = action.get("details", {})
            st.write(f"- Claims Verified: {details.get('claims_verified', 0)}")
            st.write(f"- False Claims: {details.get('false_claims', 0)}")
            st.write(f"- Timestamp: {action.get('timestamp', 'N/A')}")
    else:
        st.info("Detailed verification data not available")


def render_provenance_tab(article: Dict):
    """Render provenance and audit trail."""
    st.subheader("Content Provenance")

    provenance = article.get("provenance", {})

    if not provenance:
        st.warning("No provenance data available for this article")
        return

    # Provenance overview
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Generation Details**")
        st.write(f"- Content ID: `{provenance.get('content_id', 'N/A')}`")
        st.write(f"- Content Hash: `{provenance.get('content_hash', 'N/A')[:24]}...`")
        st.write(f"- Created: {provenance.get('created_at', 'N/A')}")
        st.write(f"- Platform: {provenance.get('platform', 'N/A')}")
        st.write(f"- Source Type: {provenance.get('source_type', 'N/A')}")

    with col2:
        st.markdown("**Models Used**")
        models = provenance.get("models_used", [])
        if models:
            for model in models:
                st.write(f"- {model}")
        else:
            st.write("- No model data available")

    st.divider()

    # Action timeline
    st.markdown("**Pipeline Timeline**")
    actions = provenance.get("actions", [])

    if actions:
        for action in actions:
            action_type = action.get("action_type", "unknown")
            agent = action.get("agent", "unknown")
            timestamp = action.get("timestamp", "")[:19]

            emoji = {
                "created": "🆕",
                "research": "🔍",
                "generated": "✍️",
                "verified": "✓",
                "reviewed": "📋",
                "revised": "📝",
                "human_reviewed": "👤",
                "finalized": "✅"
            }.get(action_type, "•")

            st.write(f"{emoji} **{action_type.title()}** by {agent} at {timestamp}")
    else:
        st.info("No action timeline available")

    # Raw JSON expander
    with st.expander("View Raw Provenance JSON"):
        st.json(provenance)


def render_decision_tab(article: Dict):
    """Render decision/approval workflow."""
    st.subheader("Editorial Decision")

    if not st.session_state.reviewer_name:
        st.warning("Please enter your name in the sidebar before making decisions")
        return

    provenance = article.get("provenance", {})
    score = article.get("quality_score", 0)

    # Quality summary
    st.markdown("**Quality Assessment**")
    col1, col2, col3 = st.columns(3)
    col1.metric("Quality Score", f"{score}/10")
    col2.metric("Fact Verified", "Yes" if provenance.get("fact_verification_passed") else "No")
    col3.metric("WSJ Standards", "Met" if score >= 7 else "Review")

    st.divider()

    # Decision form
    st.markdown("**Make Decision**")

    decision = st.radio(
        "Decision",
        ["Approve for Publication", "Request Revisions", "Reject", "Escalate"],
        horizontal=True
    )

    notes = st.text_area(
        "Review Notes",
        placeholder="Add any notes about your decision...",
        height=100
    )

    col1, col2 = st.columns([1, 3])

    with col1:
        if st.button("Submit Decision", type="primary", use_container_width=True):
            # Update article status
            status_map = {
                "Approve for Publication": "approved",
                "Request Revisions": "pending_review",
                "Reject": "rejected",
                "Escalate": "escalated"
            }

            new_status = status_map.get(decision, "pending_review")

            # Update in queue
            for item in st.session_state.review_queue:
                if item["id"] == article["id"]:
                    item["status"] = new_status
                    break

            # Log audit
            log_audit_action(
                f"decision_{new_status}",
                article["id"],
                {"decision": decision, "notes": notes, "score": score}
            )

            # Persist to disk
            save_review_decision(article["id"], new_status, st.session_state.reviewer_name, notes)
            save_audit_log(st.session_state.audit_log)

            st.success(f"Decision recorded: {decision}")
            st.session_state.current_article = None
            st.rerun()


# ============================================================================
# Page: Escalations
# ============================================================================

def render_escalations():
    """Render escalations page."""
    st.header("⚠️ Escalations")

    escalated = [q for q in st.session_state.review_queue if q["status"] == "escalated"]

    if not escalated:
        st.success("No escalations pending!")
        return

    st.warning(f"{len(escalated)} articles require attention")

    for item in escalated:
        with st.container():
            st.markdown(f"### {item['title']}")
            st.caption(f"ID: {item['id']}")

            provenance = item.get("provenance", {})

            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Quality Score:** {item['quality_score']}/10")
                st.write(f"**Created:** {item['created_at'][:16]}")

            with col2:
                if st.button("Review Now", key=f"esc_{item['id']}", type="primary"):
                    st.session_state.current_article = item
                    st.rerun()

        st.divider()


# ============================================================================
# Page: Audit Trail
# ============================================================================

def render_audit_trail():
    """Render audit trail page."""
    st.header("📜 Audit Trail")

    if not st.session_state.audit_log:
        st.info("No audit entries yet. Actions will be logged as you review articles.")
        return

    # Convert to dataframe for display
    df = pd.DataFrame(st.session_state.audit_log)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp", ascending=False)

    # Filters
    col1, col2 = st.columns(2)
    with col1:
        action_filter = st.multiselect(
            "Filter by Action",
            df["action"].unique().tolist()
        )
    with col2:
        reviewer_filter = st.multiselect(
            "Filter by Reviewer",
            df["reviewer"].unique().tolist()
        )

    if action_filter:
        df = df[df["action"].isin(action_filter)]
    if reviewer_filter:
        df = df[df["reviewer"].isin(reviewer_filter)]

    st.divider()

    # Display
    for _, row in df.iterrows():
        with st.container():
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                st.write(f"**{row['action'].replace('_', ' ').title()}**")
                st.caption(f"Article: {row['article_id'][:12]}...")
            with col2:
                st.write(row["reviewer"])
            with col3:
                st.write(row["timestamp"].strftime("%Y-%m-%d %H:%M"))

            if row["details"]:
                with st.expander("Details"):
                    st.json(row["details"])

        st.divider()

    # Export option
    if st.button("Export Audit Log"):
        st.download_button(
            "Download JSON",
            json.dumps(st.session_state.audit_log, indent=2),
            "audit_log.json",
            "application/json"
        )


# ============================================================================
# Page: Settings
# ============================================================================

def render_settings():
    """Render settings page."""
    st.header("⚙️ Settings")

    st.subheader("Quality Thresholds")
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**Pass Threshold:** {config.quality.PASS_THRESHOLD}")
        st.write(f"**Escalation Threshold:** {config.quality.ESCALATION_THRESHOLD}")
        st.write(f"**Auto-Approve Threshold:** {config.quality.MIN_SCORE_FOR_AUTO_APPROVE}")
    with col2:
        st.write(f"**Max Iterations:** {config.quality.MAX_ITERATIONS}")
        st.write(f"**Fact Verification Required:** {config.quality.FACT_VERIFICATION_REQUIRED}")
        st.write(f"**Human Review Required:** {config.quality.HUMAN_REVIEW_REQUIRED_FOR_PUBLISH}")

    st.divider()

    st.subheader("Output Directory")
    st.code(str(OUTPUT_DIR))

    st.divider()

    st.subheader("API Status")
    api_status = {
        "Google (Gemini)": config.api.has_api_key("gemini"),
        "Perplexity": config.api.has_api_key("perplexity"),
        "Anthropic (Claude)": config.api.has_api_key("anthropic"),
        "OpenAI (GPT)": config.api.has_api_key("openai"),
        "Groq": config.api.has_api_key("groq")
    }

    for api, status in api_status.items():
        icon = "✅" if status else "❌"
        st.write(f"{icon} {api}")


# ============================================================================
# Main App
# ============================================================================

def main():
    """Main app entry point."""
    init_session_state()

    page = render_sidebar()

    if page == "Review Queue":
        render_review_queue()
    elif page == "Article Review":
        render_article_review()
    elif page == "Escalations":
        render_escalations()
    elif page == "Audit Trail":
        render_audit_trail()
    elif page == "Settings":
        render_settings()


if __name__ == "__main__":
    main()
