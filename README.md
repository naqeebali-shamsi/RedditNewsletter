# GhostWriter: Autonomous AI Content Pipeline
## Specialized Multi-Agent System for Technical Thought Leadership

GhostWriter is an enterprise-grade agentic pipeline designed to automate the production of high-signal technical content. It orchestrates a sophisticated editorial board of specialized AI agents to transform raw technical signals from Reddit and GitHub into publication-ready Medium articles and LinkedIn posts.

---

## Agentic Editorial Architecture

The core of GhostWriter is a multi-agent workflow that mirrors a professional publishing house. Each stage of the content lifecycle is managed by an autonomous agent with distinct quality objectives:

1.  **Topic Research Strategist**: Analyzes trending technical shifts and developer sentiment to prioritize high-value content themes.
2.  **Editor-in-Chief**: Establishes the strategic vision and approves structured outlines for production.
3.  **Critic Agent**: Performs adversarial review of outlines to ensure technical depth and practitioners' perspective.
4.  **Senior Technical Ghostwriter**: Generates the primary narrative draft, focusing on engineering precision.
5.  **Hook Specialist**: Optimizes the lead-in to maximize reader retention and engagement.
6.  **Storytelling Architect**: Weaves authentic practitioner narratives and technical challenges into the prose.
7.  **Voice & Tone Specialist**: Harmonizes the output to maintain personality consistency while removing LLM artifacts.
8.  **Value Density Specialist**: Prunes redundant content to maximize technical takeaway for senior engineers.
9.  **Quality Gate (Expert Panel)**: Executes an adversarial review-and-fix loop, iterating until the content meets standardized benchmarks.
10. **Visuals Agent**: Synthesizes infographic plans and generates assets to support the technical narrative.

---

## Core Capabilities

### Adversarial Quality Assurance
The pipeline utilizes a dedicated Quality Gate where an expert panel subjects drafts to rigorous technical and editorial scrutiny. This iterative loop ensures compliance with non-negotiable standards for specificity, technical accuracy, and readability.

### Context-Aware Voice Modulation
GhostWriter dynamically adjusts its narrative voice based on the provenance of the input data:
- **Internal (GitHub/Direct)**: Employs a practitioner voice centered on firsthand ownership and specific implementation realizations.
- **External (Reddit/Community)**: Employs an observer voice focused on community trends, emerging patterns, and technical consensus.

### Automated Visual Reasoning
The Visuals Agent analyzes technical drafts to propose and generate complex infographics, ensuring that conceptual density is balanced with visual clarity.

### Adaptive Topic Control
The system supports both autonomous "best-fit" research and user-guided customization, allowing for targeted thought leadership in specific engineering domains such as system design, MLOps, or distributed systems.

---

## Technical Architecture

GhostWriter is built on a Three-Layer Architecture designed for reliability and deterministic execution:

1.  **Layer 1: Directives** (`directives/`): A repository of Standard Operating Procedures (SOPs) and market strategy documents that govern agent behavior and quality thresholds.
2.  **Layer 2: Orchestration** (The Agency Agent): A decision-making layer that manages state, routes tasks between specialist agents, and handles error recovery.
3.  **Layer 3: Execution** (`execution/`): A suite of deterministic Python modules that handle data ingestion, LLM interface protocols, and asset generation.

---

## Technical Stack

- **Dashboard**: Streamlit-based interface for pipeline visualization and state management.
- **Large Language Models**: Llama 3.1 & 3.3 (optimized for speed/inference) and Claude 3.5 Sonnet (optimized for editorial reasoning).
- **Data Persistence**: SQLite for tracking signals, evaluations, and draft history.
- **Integration Protocols**: Reddit RSS feeds, GitHub REST API, and Groq/Anthropic/OpenAI inference endpoints.

---

## Installation and Deployment

### 1. Dependency Installation
Initialize your environment and install the required packages:
```bash
pip install -r requirements.txt
```

### 2. Configuration
Define your environment variables in a `.env` file:
```bash
GROQ_API_KEY=your_key
GOOGLE_API_KEY=your_key
GITHUB_TOKEN=your_token
```

### 3. Execution
Launch the orchestration dashboard:
```bash
streamlit run app.py
```

---

## Project Structure

- `app.py`: Central orchestration dashboard and UI.
- `execution/agents/`: Modular implementation of the specialist agent editorial board.
- `execution/`: Deterministic execution scripts for data fetching and processing.
- `directives/`: Standard Operating Procedures and strategy specifications.
- `drafts/`: Persistent storage for generated technical artifacts.

---

**Positioning for the Practitioner.** GhostWriter is engineered to establish technical credibility by prioritizing engineering depth over generic coverage.
