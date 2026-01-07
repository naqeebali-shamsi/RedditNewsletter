"""
GhostWriter - AI Ghostwriting Pipeline
One-Click Content Generation for AI Engineers

Target: Build thought leadership, attract recruiters passively
Run with: streamlit run app.py
"""

import streamlit as st
import sys
import os
from pathlib import Path
from datetime import datetime
import json
import time

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Page config
st.set_page_config(
    page_title="GhostWriter",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Premium modern CSS
st.markdown("""
<style>
    /* Import Inter font with full weight range */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* CSS Custom Properties - Dark Mode First (Streamlit default) */
    :root {
        --color-primary: #818cf8;
        --color-primary-hover: #a5b4fc;
        --color-primary-light: #c7d2fe;
        --color-success: #34d399;

        /* Dark theme (default for Streamlit) */
        --color-bg-primary: #0e1117;
        --color-bg-secondary: #1a1f2e;
        --color-bg-tertiary: #262c3a;
        --color-text-primary: #fafafa;
        --color-text-secondary: #a1a1aa;
        --color-text-muted: #71717a;
        --color-border: #3f3f46;
        --color-border-hover: #52525b;

        --shadow-sm: 0 1px 2px 0 rgb(0 0 0 / 0.3);
        --shadow-md: 0 4px 6px -1px rgb(0 0 0 / 0.4), 0 2px 4px -2px rgb(0 0 0 / 0.3);
        --shadow-lg: 0 10px 15px -3px rgb(0 0 0 / 0.5), 0 4px 6px -4px rgb(0 0 0 / 0.4);

        --radius-md: 8px;
        --radius-lg: 12px;
    }

    /* Light theme override */
    @media (prefers-color-scheme: light) {
        :root {
            --color-primary: #6366f1;
            --color-primary-hover: #4f46e5;
            --color-primary-light: #818cf8;
            --color-success: #10b981;

            --color-bg-primary: #ffffff;
            --color-bg-secondary: #f8fafc;
            --color-bg-tertiary: #f1f5f9;
            --color-text-primary: #0f172a;
            --color-text-secondary: #475569;
            --color-text-muted: #64748b;
            --color-border: #e2e8f0;
            --color-border-hover: #cbd5e1;

            --shadow-sm: 0 1px 2px 0 rgb(0 0 0 / 0.05);
            --shadow-md: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1);
            --shadow-lg: 0 10px 15px -3px rgb(0 0 0 / 0.1), 0 4px 6px -4px rgb(0 0 0 / 0.1);
        }
    }

    /* Global styles */
    .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        font-size: 16px;
        -webkit-font-smoothing: antialiased;
    }

    /* Headers - High Contrast */
    .main-header {
        font-size: 2.25rem;
        font-weight: 700;
        color: var(--color-text-primary);
        margin-bottom: 0.5rem;
        letter-spacing: -0.025em;
        line-height: 1.2;
    }
    .sub-header {
        font-size: 1.0625rem;
        font-weight: 400;
        color: var(--color-text-secondary);
        margin-bottom: 2rem;
        line-height: 1.5;
    }

    /* Phase indicator - Pill style with accent */
    .phase-indicator {
        display: inline-block;
        font-size: 0.8125rem;
        font-weight: 600;
        color: var(--color-primary);
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.1) 0%, rgba(139, 92, 246, 0.1) 100%);
        padding: 0.5rem 1rem;
        border-radius: 9999px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 0.75rem;
        border: 1px solid rgba(99, 102, 241, 0.2);
        animation: fadeInUp 0.3s ease-out;
    }

    /* Status text - Modern card */
    .status-text {
        font-size: 0.9375rem;
        font-weight: 500;
        color: var(--color-text-primary);
        padding: 1rem 1.25rem;
        background: linear-gradient(135deg, var(--color-bg-secondary) 0%, var(--color-bg-tertiary) 100%);
        border-left: 3px solid var(--color-primary);
        border-radius: 0 var(--radius-md) var(--radius-md) 0;
        margin: 0.75rem 0;
        box-shadow: var(--shadow-sm);
        animation: slideInLeft 0.25s ease-out;
    }

    /* Info box - Elevated card */
    .info-box {
        background: var(--color-bg-secondary);
        border: 1px solid var(--color-border);
        border-radius: var(--radius-lg);
        padding: 1.5rem;
        margin: 1.25rem 0;
        box-shadow: var(--shadow-md);
        transition: box-shadow 0.2s ease, transform 0.2s ease;
    }
    .info-box:hover {
        box-shadow: var(--shadow-lg);
        transform: translateY(-2px);
        border-color: var(--color-border-hover);
    }
    .info-box h4 {
        font-size: 0.8125rem;
        font-weight: 600;
        color: var(--color-text-secondary);
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 1rem;
        padding-bottom: 0.75rem;
        border-bottom: 1px solid var(--color-border);
    }
    .info-box ul {
        margin: 0;
        padding-left: 0;
        list-style: none;
        color: var(--color-text-primary);
        font-size: 0.9375rem;
        line-height: 1.8;
    }
    .info-box ul li {
        position: relative;
        padding-left: 1.5rem;
        margin-bottom: 0.5rem;
    }
    .info-box ul li::before {
        content: "";
        position: absolute;
        left: 0;
        top: 0.6rem;
        width: 6px;
        height: 6px;
        border-radius: 50%;
        background: var(--color-primary);
    }

    /* Progress bar - Gradient accent */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, var(--color-primary) 0%, var(--color-primary-light) 50%, #a78bfa 100%) !important;
        height: 6px !important;
        border-radius: 9999px !important;
        transition: width 0.3s ease-out !important;
    }
    .stProgress > div > div > div {
        background: var(--color-bg-tertiary) !important;
        height: 6px !important;
        border-radius: 9999px !important;
    }

    /* Buttons - Premium design */
    .stButton > button {
        border-radius: var(--radius-md) !important;
        border: 1px solid var(--color-border) !important;
        background: var(--color-bg-primary) !important;
        color: var(--color-text-primary) !important;
        font-weight: 500 !important;
        font-size: 0.9375rem !important;
        min-height: 48px !important;
        padding: 0.75rem 1.5rem !important;
        transition: all 0.15s ease !important;
        box-shadow: var(--shadow-sm) !important;
    }
    .stButton > button:hover {
        border-color: var(--color-border-hover) !important;
        background: var(--color-bg-secondary) !important;
        box-shadow: var(--shadow-md) !important;
        transform: translateY(-1px) !important;
    }
    .stButton > button:active {
        transform: translateY(0) !important;
    }
    .stButton > button:focus {
        outline: 2px solid var(--color-primary) !important;
        outline-offset: 2px !important;
    }

    /* Primary button - Gradient accent */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, var(--color-primary) 0%, #8b5cf6 100%) !important;
        border-color: transparent !important;
        color: white !important;
        font-weight: 600 !important;
        box-shadow: 0 4px 14px 0 rgba(99, 102, 241, 0.4) !important;
    }
    .stButton > button[kind="primary"]:hover {
        background: linear-gradient(135deg, var(--color-primary-hover) 0%, #7c3aed 100%) !important;
        box-shadow: 0 6px 20px 0 rgba(99, 102, 241, 0.5) !important;
        transform: translateY(-2px) !important;
    }

    /* Download button */
    .stDownloadButton > button {
        border-radius: var(--radius-md) !important;
        border: 1px solid var(--color-border) !important;
        font-weight: 500 !important;
        font-size: 0.9375rem !important;
        min-height: 48px !important;
        padding: 0.75rem 1.5rem !important;
        transition: all 0.15s ease !important;
        box-shadow: var(--shadow-sm) !important;
    }
    .stDownloadButton > button:hover {
        border-color: var(--color-success) !important;
        background: rgba(16, 185, 129, 0.05) !important;
        color: var(--color-success) !important;
        box-shadow: var(--shadow-md) !important;
    }

    /* Expander - Clean card */
    .streamlit-expanderHeader {
        font-size: 0.9375rem !important;
        font-weight: 600 !important;
        color: var(--color-text-primary) !important;
        background: var(--color-bg-secondary) !important;
        border-radius: var(--radius-md) !important;
        padding: 1rem 1.25rem !important;
        border: 1px solid var(--color-border) !important;
        transition: all 0.15s ease !important;
    }
    .streamlit-expanderHeader:hover {
        background: var(--color-bg-tertiary) !important;
        border-color: var(--color-border-hover) !important;
    }
    .streamlit-expanderContent {
        border: 1px solid var(--color-border) !important;
        border-top: none !important;
        border-radius: 0 0 var(--radius-md) var(--radius-md) !important;
        background: var(--color-bg-primary) !important;
    }

    /* Divider */
    hr {
        border: none;
        border-top: 1px solid var(--color-border);
        margin: 2rem 0;
    }

    /* Success/info messages */
    .stSuccess, .stInfo {
        background: var(--color-bg-secondary) !important;
        border: 1px solid var(--color-border) !important;
        border-radius: var(--radius-md) !important;
        color: var(--color-text-primary) !important;
    }
    .stSuccess {
        border-left: 3px solid var(--color-success) !important;
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.05) 0%, var(--color-bg-secondary) 100%) !important;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: var(--color-bg-secondary) !important;
        border-right: 1px solid var(--color-border) !important;
    }
    [data-testid="stSidebar"] .section-header {
        color: var(--color-text-secondary);
    }

    /* Topic display - Premium card */
    .topic-display {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        color: #f8fafc;
        padding: 1.25rem 1.5rem;
        border-radius: var(--radius-lg);
        font-size: 1rem;
        font-weight: 600;
        margin: 1.25rem 0;
        box-shadow: 0 10px 40px -10px rgba(15, 23, 42, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.1);
        position: relative;
        overflow: hidden;
    }
    .topic-display::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, var(--color-primary) 0%, #8b5cf6 50%, #a78bfa 100%);
    }
    .topic-reasoning {
        font-size: 0.9375rem;
        color: var(--color-text-muted);
        margin-top: 0.75rem;
        font-style: italic;
        line-height: 1.6;
    }

    /* Animations */
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(8px); }
        to { opacity: 1; transform: translateY(0); }
    }
    @keyframes slideInLeft {
        from { opacity: 0; transform: translateX(-12px); }
        to { opacity: 1; transform: translateX(0); }
    }

    /* Code blocks */
    .stCodeBlock {
        border-radius: var(--radius-md) !important;
        border: 1px solid var(--color-border) !important;
    }

    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Section header */
    .section-header {
        font-size: 0.8125rem;
        font-weight: 700;
        color: var(--color-text-muted);
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin: 2rem 0 1rem 0;
    }

    /* History item */
    .history-item {
        background: var(--color-bg-secondary);
        border: 1px solid var(--color-border);
        border-radius: var(--radius-md);
        padding: 1rem 1.25rem;
        margin-bottom: 0.75rem;
        cursor: pointer;
        transition: all 0.15s ease;
    }
    .history-item:hover {
        border-color: var(--color-primary);
        background: var(--color-bg-tertiary);
        transform: translateX(4px);
    }
    .history-item-title {
        font-size: 0.9375rem;
        font-weight: 600;
        color: var(--color-text-primary);
        margin-bottom: 0.25rem;
        line-height: 1.4;
    }
    .history-item-meta {
        font-size: 0.8125rem;
        color: var(--color-text-muted);
    }
    .history-empty {
        text-align: center;
        padding: 2rem;
        color: var(--color-text-muted);
        font-size: 0.9375rem;
    }

    /* Scrollbar styling */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    ::-webkit-scrollbar-track {
        background: var(--color-bg-secondary);
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb {
        background: var(--color-border-hover);
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: var(--color-text-muted);
    }

    /* Mobile responsive */
    @media (max-width: 768px) {
        .main-header {
            font-size: 1.75rem;
        }
        .stButton > button,
        .stDownloadButton > button {
            width: 100% !important;
        }
    }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """Initialize session state variables."""
    defaults = {
        'generation_complete': False,
        'article_content': None,
        'image_paths': [],
        'visual_plan': None,
        'selected_topic': None,
        'topic_reasoning': None,
        'is_running': False,
        'viewing_history': None,  # Currently viewed history file
        'data_source': 'reddit',  # 'reddit' or 'github'
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def get_draft_history():
    """
    Scan drafts directory and return list of saved articles.

    Returns:
        List of dicts with filename, title, date, filepath
    """
    drafts_dir = Path("n:/RedditNews/drafts")
    if not drafts_dir.exists():
        return []

    history = []
    for md_file in sorted(drafts_dir.glob("*.md"), reverse=True):
        # Extract date from filename (medium_full_YYYYMMDD_HHMMSS.md)
        try:
            parts = md_file.stem.split("_")
            if len(parts) >= 3:
                date_str = parts[-2]
                time_str = parts[-1]
                date_formatted = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]} {time_str[:2]}:{time_str[2:4]}"
            else:
                date_formatted = md_file.stat().st_mtime
                date_formatted = datetime.fromtimestamp(date_formatted).strftime("%Y-%m-%d %H:%M")
        except:
            date_formatted = "Unknown"

        # Extract title from first line
        try:
            with open(md_file, 'r', encoding='utf-8') as f:
                first_line = f.readline().strip()
                title = first_line.lstrip('#').strip()[:80]
                if not title:
                    title = md_file.stem
        except:
            title = md_file.stem

        history.append({
            'filename': md_file.name,
            'title': title,
            'date': date_formatted,
            'filepath': str(md_file),
        })

    return history


def load_draft(filepath):
    """Load a draft from file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except:
        return None


def run_full_pipeline(progress_callback, status_callback):
    """
    Run the COMPLETE automated pipeline:
    1. Topic Research Agent selects best topic
    2. Full article generation with all agents
    3. Visual generation

    Returns:
        dict with content, filepath, topic info, images
    """
    from execution.agents.topic_researcher import TopicResearchAgent
    from execution.agents.editor import EditorAgent
    from execution.agents.critic import CriticAgent
    from execution.agents.writer import WriterAgent
    from execution.agents.specialist import SpecialistAgent
    from execution.agents.visuals import VisualsAgent

    output_dir = Path("n:/RedditNews/drafts")
    output_dir.mkdir(parents=True, exist_ok=True)

    # PHASE 1: TOPIC RESEARCH (0-10%)
    data_source = st.session_state.get('data_source', 'reddit')

    if data_source == 'github':
        # GitHub commit analysis
        from execution.agents.commit_analyzer import CommitAnalysisAgent

        status_callback("Analyzing GitHub commits for topics...")
        progress_callback(0.02)

        commit_agent = CommitAnalysisAgent()

        status_callback("Fetching commits from database...")
        progress_callback(0.05)

        status_callback("Extracting themes from commit activity...")
        progress_callback(0.08)
        selected = commit_agent.research_github_topics_from_db()

        topic = selected['title']
        topic_reasoning = selected.get('reasoning', '')
        topic_angle = selected.get('angle', '')

    else:
        # Reddit topic research (default)
        status_callback("Scanning Reddit for trending topics...")
        progress_callback(0.02)

        topic_agent = TopicResearchAgent()

        status_callback("Fetching from AI/ML subreddits...")
        progress_callback(0.04)
        topics = topic_agent.fetch_trending_topics()

        status_callback("Analyzing topics for positioning fit...")
        progress_callback(0.08)
        selected = topic_agent.analyze_and_select(topics)

        topic = selected['title']
        topic_reasoning = selected.get('reasoning', '')
        topic_angle = selected.get('angle', '')

    status_callback(f"Topic selected")
    progress_callback(0.10)

    # PHASE 2: AGENT INITIALIZATION (10-15%)
    status_callback("Initializing agents...")
    progress_callback(0.12)

    editor = EditorAgent()
    critic = CriticAgent()
    writer = WriterAgent()
    visuals = VisualsAgent()
    progress_callback(0.15)

    # PHASE 3: OUTLINE (15-25%)
    status_callback("Creating outline...")
    progress_callback(0.18)
    outline = editor.create_outline(topic)
    progress_callback(0.22)

    status_callback("Reviewing outline...")
    critique = critic.critique_outline(outline)
    progress_callback(0.25)

    # PHASE 4: REFINE OUTLINE (25-30%)
    status_callback("Refining outline...")
    refined_outline = editor.call_llm(
        f"Refine this outline based on critique. Topic angle: {topic_angle}\n\nOutline:\n{outline}\n\nCritique:\n{critique}"
    )
    progress_callback(0.30)

    # PHASE 5: DRAFT (30-50%)
    status_callback("Writing first draft...")
    progress_callback(0.35)
    draft = writer.write_section(refined_outline, critique="Write the full article draft.")
    progress_callback(0.50)

    # PHASE 6: SPECIALISTS (50-75%)
    specialists = [
        ("Hook Specialist", "Optimizing title and hook...",
         """Your job is to make readers STOP scrolling.

1. Rewrite the title to create curiosity or promise transformation.
   Examples of good patterns: "Why I stopped X and started Y", "The X that nobody talks about", "What 3 years of X taught me about Y"
2. The opening line should feel like the start of a conversation, not a thesis statement.
3. The hook should make the reader think "this person gets it" within 10 seconds.

Keep the author's voice - don't make it sound like marketing copy."""),

        ("Storytelling Architect", "Adding authentic narrative...",
         """Your job is to make this feel like it's written by a REAL engineer, not an AI.

1. Add 1-2 brief personal moments: a frustration, a realization, a late-night debugging session.
   Keep these SHORT (2-3 sentences max) and weave them naturally into transitions.
2. Use "I" and "you" to create conversation - "I've seen this pattern fail" or "You've probably hit this wall"
3. The personality should feel like a smart colleague explaining something over coffee, not a textbook.

Do NOT add fake-sounding stories. If it feels forced, cut it."""),

        ("Voice & Tone Specialist", "Refining voice...",
         """Your job is to make this sound like ONE consistent, authentic person.

1. Remove anything that sounds like corporate speak, marketing jargon, or AI-generated filler.
2. Add subtle wit where natural - a wry observation, a knowing aside. NOT forced jokes.
3. Vary sentence rhythm: mix punchy short sentences with longer explanations.
4. The tone should be: confident but not arrogant, technical but accessible, opinionated but fair.

Read it aloud - if it sounds like a robot wrote it, rewrite those parts."""),

        ("Value Density Specialist", "Ensuring takeaways...",
         """Your job is to make every paragraph EARN its place.

1. Every section should leave the reader with something actionable or a new perspective.
2. Cut fluff, throat-clearing, and obvious statements. Get to the point faster.
3. Where appropriate, add concrete specifics: tools, numbers, code snippets, real examples.
4. The reader should finish thinking "I learned something useful" not "I read a lot of words"

Do NOT add bullet lists for the sake of it. Natural prose with clear takeaways > forced structure."""),
    ]

    progress_per = 0.0625
    current = 0.50

    for name, status_msg, instruction in specialists:
        status_callback(status_msg)
        specialist = SpecialistAgent(constraint_name=name, constraint_instruction=instruction)
        draft = specialist.refine(draft)
        current += progress_per
        progress_callback(current)

    # PHASE 7: POLISH (75-80%)
    status_callback("Final polish...")
    polisher = SpecialistAgent(
        constraint_name="Final Editor",
        constraint_instruction="""Final pass before publication. Your job is COHESION and CLEAN OUTPUT.

1. Strip ALL internal labels, metadata markers, section tags, and any "Value-Bait:", "Hook:", etc. prefixes.
2. Format first line as # H1 Title (clean, no labels).
3. Ensure smooth transitions between sections - no jarring jumps.
4. Remove any repetitive phrases or ideas that got duplicated across specialist passes.
5. If there are forced bullet lists that interrupt flow, convert them to natural prose.
6. The final piece should read like ONE person wrote it in ONE sitting - not a committee.

Output ONLY the polished markdown. No explanations, no meta-commentary."""
    )
    draft = polisher.refine(draft)
    progress_callback(0.80)

    # PHASE 8: FINAL REVIEW (80-85%)
    status_callback("Quality check...")
    review = editor.review_draft(draft, "Full Article Post-Specialists")
    if "REVISE" in review:
        draft = writer.call_llm(f"Apply corrections:\n{review}\n\nDraft:\n{draft}")
    progress_callback(0.85)

    # PHASE 9: VISUALS (85-95%)
    status_callback("Generating visuals...")
    visual_plan = visuals.suggest_visuals(draft)
    image_paths = []

    if visual_plan and isinstance(visual_plan, list):
        status_callback(f"Creating {len(visual_plan)} infographics...")
        images_dir = output_dir / "images"
        image_paths = visuals.generate_all_visuals(visual_plan, output_dir=str(images_dir))

    progress_callback(0.95)

    # PHASE 10: SAVE (95-100%)
    status_callback("Saving...")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"medium_full_{timestamp}.md"
    filepath = output_dir / filename

    images_section = ""
    if image_paths:
        images_section = "\n## Generated Images\n"
        for i, path in enumerate(image_paths, 1):
            rel_path = Path(path).name
            images_section += f"![Visual {i}](./images/{rel_path})\n\n"

    final_content = f"""{draft}

---
{images_section}
"""

    with open(filepath, "w", encoding='utf-8') as f:
        f.write(final_content)

    progress_callback(1.0)
    status_callback("Complete")

    return {
        'content': draft,
        'filepath': str(filepath),
        'image_paths': image_paths,
        'visual_plan': visual_plan,
        'topic': topic,
        'topic_reasoning': topic_reasoning,
        'topic_angle': topic_angle,
    }


def main():
    init_session_state()

    # Header
    st.markdown('<p class="main-header">GhostWriter</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">One-click AI ghostwriting for technical thought leadership</p>', unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.markdown('<p class="section-header">Configuration</p>', unsafe_allow_html=True)

        google_key = os.getenv("GOOGLE_API_KEY")
        groq_key = os.getenv("GROQ_API_KEY")

        st.markdown(f"Groq API: {'Connected' if groq_key else 'Not configured'}")
        st.markdown(f"Google API: {'Connected' if google_key else 'Not configured'}")

        st.divider()

        st.markdown('<p class="section-header">Data Source</p>', unsafe_allow_html=True)
        source = st.radio(
            "Topic Source",
            ["reddit", "github"],
            index=0 if st.session_state.get('data_source', 'reddit') == 'reddit' else 1,
            format_func=lambda x: "Reddit (Trending)" if x == "reddit" else "GitHub (Commits)",
            key="source_selector",
            label_visibility="collapsed"
        )
        st.session_state.data_source = source

        st.divider()

        st.markdown('<p class="section-header">Pipeline</p>', unsafe_allow_html=True)
        st.markdown("""
        1. Topic Researcher
        2. Editor
        3. Critic
        4. Writer
        5. Hook Specialist
        6. Storytelling Architect
        7. Voice & Tone
        8. Value Density
        9. Final Editor
        10. Visuals
        """)

    # Main content
    if st.session_state.viewing_history:
        # Viewing a history item
        draft_content = load_draft(st.session_state.viewing_history['filepath'])

        if draft_content:
            st.markdown(f'<div class="topic-display">{st.session_state.viewing_history["title"]}</div>', unsafe_allow_html=True)
            st.markdown(f'<p class="topic-reasoning">Saved: {st.session_state.viewing_history["date"]}</p>', unsafe_allow_html=True)

            st.divider()

            col1, col2, col3 = st.columns(3)
            with col1:
                st.download_button(
                    label="Download",
                    data=draft_content,
                    file_name=st.session_state.viewing_history['filename'],
                    mime="text/markdown",
                    use_container_width=True
                )
            with col2:
                if st.button("Copy", use_container_width=True, key="history_copy"):
                    st.code(draft_content, language="markdown")
                    st.caption("Select all and copy (Ctrl+A, Ctrl+C)")
            with col3:
                if st.button("Back", use_container_width=True, key="history_back"):
                    st.session_state.viewing_history = None
                    st.rerun()

            st.markdown('<p class="section-header">Article Content</p>', unsafe_allow_html=True)
            with st.expander("View Full Article", expanded=True):
                st.markdown(draft_content)
        else:
            st.error("Could not load draft file")
            if st.button("Back"):
                st.session_state.viewing_history = None
                st.rerun()

    elif not st.session_state.generation_complete:

        # Info box
        st.markdown("""
        <div class="info-box">
            <h4>Pipeline Overview</h4>
            <ul>
                <li>Scans trending topics from AI/ML subreddits</li>
                <li>Selects optimal topic for recruiter visibility</li>
                <li>Generates article with 10 specialized agents</li>
                <li>Creates supporting infographics</li>
                <li>Outputs ready-to-publish content</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        st.divider()

        # Generate button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button(
                "Generate Article",
                type="primary",
                use_container_width=True,
                disabled=st.session_state.is_running,
                key="main_generate"
            ):
                st.session_state.is_running = True
                st.rerun()

        # History section
        st.divider()
        st.markdown('<p class="section-header">History</p>', unsafe_allow_html=True)

        history = get_draft_history()
        if history:
            for i, item in enumerate(history):
                col1, col2 = st.columns([5, 1])
                with col1:
                    st.markdown(f'''
                    <div class="history-item">
                        <div class="history-item-title">{item["title"]}</div>
                        <div class="history-item-meta">{item["date"]}</div>
                    </div>
                    ''', unsafe_allow_html=True)
                with col2:
                    if st.button("View", key=f"view_{i}", use_container_width=True):
                        st.session_state.viewing_history = item
                        st.rerun()
        else:
            st.markdown('<div class="history-empty">No articles generated yet</div>', unsafe_allow_html=True)

        # Pipeline execution
        if st.session_state.is_running:
            st.divider()

            progress_bar = st.progress(0)
            status_container = st.empty()
            phase_container = st.empty()

            def update_progress(value):
                progress_bar.progress(value)
                phases = [
                    (0.10, "Topic Research"),
                    (0.15, "Initialization"),
                    (0.25, "Outline"),
                    (0.30, "Refinement"),
                    (0.50, "Draft"),
                    (0.75, "Specialists"),
                    (0.80, "Polish"),
                    (0.85, "Review"),
                    (0.95, "Visuals"),
                    (1.00, "Complete"),
                ]
                for threshold, name in phases:
                    if value < threshold:
                        phase_container.markdown(f'<p class="phase-indicator">{name}</p>', unsafe_allow_html=True)
                        break

            def update_status(message):
                status_container.markdown(f'<div class="status-text">{message}</div>', unsafe_allow_html=True)

            try:
                result = run_full_pipeline(update_progress, update_status)

                st.session_state.generation_complete = True
                st.session_state.article_content = result['content']
                st.session_state.image_paths = result['image_paths']
                st.session_state.visual_plan = result['visual_plan']
                st.session_state.filepath = result['filepath']
                st.session_state.selected_topic = result['topic']
                st.session_state.topic_reasoning = result.get('topic_reasoning', '')
                st.session_state.is_running = False

                st.rerun()

            except Exception as e:
                st.session_state.is_running = False
                st.error(f"Error: {str(e)}")
                st.exception(e)

    else:
        # Results display
        st.markdown(f'<div class="topic-display">{st.session_state.selected_topic}</div>', unsafe_allow_html=True)
        if st.session_state.topic_reasoning:
            st.markdown(f'<p class="topic-reasoning">{st.session_state.topic_reasoning}</p>', unsafe_allow_html=True)

        st.divider()

        # Action buttons
        col1, col2, col3 = st.columns(3)

        with col1:
            st.download_button(
                label="Download",
                data=st.session_state.article_content,
                file_name="article.md",
                mime="text/markdown",
                use_container_width=True
            )

        with col2:
            if st.button("Copy", use_container_width=True):
                st.code(st.session_state.article_content, language="markdown")
                st.caption("Select all and copy (Ctrl+A, Ctrl+C)")

        with col3:
            if st.button("New Article", use_container_width=True):
                st.session_state.generation_complete = False
                st.session_state.article_content = None
                st.session_state.image_paths = []
                st.session_state.selected_topic = None
                st.rerun()

        # Article preview
        st.markdown('<p class="section-header">Article Preview</p>', unsafe_allow_html=True)
        with st.expander("View Full Article", expanded=True):
            st.markdown(st.session_state.article_content)

        # Generated images
        if st.session_state.image_paths:
            st.markdown('<p class="section-header">Infographics</p>', unsafe_allow_html=True)
            img_cols = st.columns(min(len(st.session_state.image_paths), 3))
            for i, img_path in enumerate(st.session_state.image_paths):
                if Path(img_path).exists():
                    img_cols[i % 3].image(img_path, caption=f"Visual {i+1}")

        # Visual plan
        if st.session_state.visual_plan:
            with st.expander("Visual Plan"):
                st.json(st.session_state.visual_plan)

        # File location
        if hasattr(st.session_state, 'filepath'):
            st.caption(f"Saved: {st.session_state.filepath}")


if __name__ == "__main__":
    main()
