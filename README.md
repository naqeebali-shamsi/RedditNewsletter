# ✍️ GhostWriter: AI Ghostwriting Pipeline
### *One-Click Content Generation for Elite AI Engineering Thought Leadership*

GhostWriter is a premium, agentic content production pipeline designed specifically for AI Engineers. It automates the entire lifecycle of high-signal technical content—from scanning subreddits for trending technical shifts to producing polished, multi-agent refined Medium articles and LinkedIn posts with custom infographics.

---

## 💎 The 10-Agent Editorial Team

GhostWriter doesn't just "generate text." It runs your ideas through a specialized editorial board of 10 autonomous agents, each mimicking a high-level publishing professional:

1.  **🔍 Topic Research Strategist**: Scans Reddit/GitHub for trending high-signal technical shifts.
2.  **🏗️ Editor-in-Chief**: Sets the vision and approves the initial skeletal outline.
3.  **⚖️ Critic Agent**: Challenges the outline for technical depth and uniqueness.
4.  **✍️ Senior technical Ghostwriter**: Drafts the initial practitioner-level prose.
5.  **🪝 Hook Specialist**: Optimizes the first 10 words for maximal scroll-stopping tension.
6.  **🎭 Storytelling Architect**: Weaves authentic narrative tension into technical explanations.
7.  **🎙️ Voice & Tone Specialist**: Ensures consistency and removes "AI-generated" artifacts.
8.  **📈 Value Density Specialist**: Strips fluff and ensures every paragraph offers a practical takeaway.
9.  **🛡️ Quality Gate (Expert Panel)**: An adversarial review loop that iterates content until it passes elite standards.
10. **🎨 Visuals Agent**: Automatically designs infographics and visual plans to accompany the text.

---

## 🔥 Key Features

### 🛡️ Adversarial Quality Gate
The pipeline includes a specialized review loop where an "Expert Panel" subjects the draft to rigorous technical and editorial criticism. The content is iteratively refined until it meets non-negotiable standards for specificity, hook strength, and memorability.

### 🎭 Voice-Aware Intelligence
GhostWriter understands the difference between an **Observer** (reporting on trends) and a **Practitioner** (sharing firsthand experience). It automatically adjusts the narrative voice based on the data source:
- **GitHub Source**: Practitioner voice ("We built...", "I learned...")
- **Reddit Source**: Observer voice ("Teams found...", "The community discovered...")

### 📊 Visual Content Generation
Integrated DALL-E/Image generation for infographics and visual layouts, ensuring your articles are not just readable, but shareable.

### 🚀 Custom Topic Injection
Beyond autonomous research, users can inject custom topics or choose from curated "Typing Suggestions" to steer the agents toward specific expertise or system design patterns.

---

## 🏗️ 3-Layer Architecture

This project follows the **Advanced Agentic Architecture**:

1.  **Layer 1: Directives** (`directives/`) — Deterministic SOPs and strategy documents that define content standards and market positioning.
2.  **Layer 2: Orchestration** (The Agent) — Intelligent routing and decision-making that manages the multi-agent editorial loop.
3.  **Layer 3: Execution** (`execution/`) — High-performance Python tools for data fetching, LLM interaction, and visual generation.

---

## 🛠️ Tech Stack

- **Frontend**: Streamlit (Premium Glassmorphism UI)
- **Intelligence**: Llama 3.1 & 3.3 (via Groq), Claude-3.5-Sonnet (via Anthropic)
- **Data**: Reddit RSS, GitHub API, SQLite
- **Visuals**: OpenAI DALL-E 3 / Visual Generation Agents

---

## 🚀 Getting Started

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Environment Setup
Configure your API keys in `.env` (see `.env.example` for details):
```bash
GROQ_API_KEY=your_key
GOOGLE_API_KEY=your_key
GITHUB_TOKEN=your_token
```

### 3. Launch the Dashboard
```bash
streamlit run app.py
```

---

## 📁 Project Layout

- `app.py`: The primary interactive command center.
- `execution/agents/`: The brains of the operation (Researcher, Writer, Quality Gate, etc.).
- `execution/`: Individual tools for fetching and processing data.
- `directives/`: Standard Operating Procedures for the agents.
- `drafts/`: Final output storage for Medium and LinkedIn content.

---

**Built for the Silent Recruiting Era.** GhostWriter positions you as a builder in an industry that prizes practitioners over passive observers.
