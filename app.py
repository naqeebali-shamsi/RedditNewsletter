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
from execution.utils.datetime_utils import utc_now
import html
import json
import math
import time

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Load environment variables
from dotenv import load_dotenv, set_key
load_dotenv()

# Centralized configuration
from execution.config import config, OUTPUT_DIR, PROJECT_ROOT

# Page config
st.set_page_config(
    page_title="GhostWriter",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium modern CSS
st.markdown("""
<style>
    /* Import Inter font with full weight range */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Flaticon UIcons - Regular Rounded */
    @import url('https://cdn-uicons.flaticon.com/3.0.0/uicons-regular-rounded/css/uicons-regular-rounded.css');

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

        /* Dark theme glass/gradient defaults */
        --glass-bg: rgba(26, 31, 46, 0.6);
        --glass-border: rgba(255, 255, 255, 0.08);
        --glass-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
        --hero-bg: linear-gradient(135deg, rgba(26, 31, 46, 0.8) 0%, rgba(38, 44, 58, 0.6) 100%);
        --progress-bg: linear-gradient(135deg, rgba(26, 31, 46, 0.9) 0%, rgba(38, 44, 58, 0.7) 100%);
        --success-bg: linear-gradient(135deg, rgba(16, 185, 129, 0.15) 0%, rgba(26, 31, 46, 0.8) 100%);
        --mesh-color-1: rgba(99, 102, 241, 0.12);
        --mesh-color-2: rgba(139, 92, 246, 0.08);
        --mesh-color-3: rgba(99, 102, 241, 0.06);
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

            /* Light theme glass/gradient overrides */
            --glass-bg: rgba(255, 255, 255, 0.8);
            --glass-border: rgba(0, 0, 0, 0.1);
            --glass-shadow: 0 8px 32px rgba(0, 0, 0, 0.08);
            --hero-bg: linear-gradient(135deg, rgba(248, 250, 252, 0.95) 0%, rgba(241, 245, 249, 0.9) 100%);
            --progress-bg: linear-gradient(135deg, rgba(255, 255, 255, 0.95) 0%, rgba(248, 250, 252, 0.9) 100%);
            --success-bg: linear-gradient(135deg, rgba(16, 185, 129, 0.1) 0%, rgba(255, 255, 255, 0.95) 100%);
            --mesh-color-1: rgba(99, 102, 241, 0.06);
            --mesh-color-2: rgba(139, 92, 246, 0.04);
            --mesh-color-3: rgba(99, 102, 241, 0.03);
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
            font-size: 1.5rem;
        }
        .sub-header {
            font-size: 0.9375rem;
        }
        .stButton > button,
        .stDownloadButton > button {
            width: 100% !important;
        }
        .hero-section {
            padding: 2rem 1rem;
        }
        .hero-title {
            font-size: 1.375rem;
        }
        .hero-tagline {
            font-size: 1rem;
        }
        .hero-features {
            gap: 1.25rem;
        }
        .hero-feature-text {
            font-size: 0.8125rem;
        }
        .progress-card {
            padding: 1.5rem;
        }
        .success-banner {
            padding: 1.5rem;
        }
        .success-stats {
            gap: 1rem;
        }
        .stat-value {
            font-size: 1.25rem;
        }
        .history-card {
            padding: 0.875rem;
        }
        .tip-container {
            padding: 1rem;
        }
    }

    /* Accessibility - Reduced Motion */
    @media (prefers-reduced-motion: reduce) {
        *, *::before, *::after {
            animation-duration: 0.01ms !important;
            animation-iteration-count: 1 !important;
            transition-duration: 0.01ms !important;
        }
        .confetti { display: none !important; }
    }

    /* ========== GLASSMORPHISM & GRADIENT MESH ========== */
    .glass-card {
        background: var(--glass-bg);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid var(--glass-border);
        box-shadow: var(--glass-shadow);
        border-radius: var(--radius-lg);
    }

    .gradient-mesh-bg {
        position: relative;
    }
    .gradient-mesh-bg::before {
        content: "";
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background:
            radial-gradient(ellipse at 20% 0%, var(--mesh-color-1) 0px, transparent 50%),
            radial-gradient(ellipse at 80% 20%, var(--mesh-color-2) 0px, transparent 50%),
            radial-gradient(ellipse at 0% 60%, var(--mesh-color-3) 0px, transparent 50%);
        pointer-events: none;
        z-index: -1;
    }

    /* ========== HERO SECTION ========== */
    .hero-section {
        text-align: center;
        padding: 3rem 2rem;
        background: var(--hero-bg);
        backdrop-filter: blur(12px);
        border: 1px solid var(--glass-border);
        border-radius: var(--radius-lg);
        margin-bottom: 2rem;
        position: relative;
        overflow: hidden;
    }
    .hero-section::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 2px;
        background: linear-gradient(90deg, transparent, var(--color-primary), #8b5cf6, transparent);
    }
    .hero-icon {
        font-size: 3rem;
        margin-bottom: 1rem;
        display: block;
    }
    .hero-tagline {
        font-size: 1.375rem;
        font-weight: 300;
        color: var(--color-text-secondary);
        margin-bottom: 0.75rem;
        line-height: 1.4;
    }
    .hero-value {
        font-size: 1rem;
        color: var(--color-primary);
        font-weight: 500;
        margin-bottom: 1.5rem;
    }
    .hero-agents {
        display: flex;
        justify-content: center;
        gap: 0.5rem;
        flex-wrap: wrap;
        margin-top: 1rem;
    }
    .hero-agent-badge {
        background: rgba(99, 102, 241, 0.15);
        border: 1px solid rgba(99, 102, 241, 0.3);
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        color: var(--color-primary-light);
        font-weight: 500;
    }
    .hero-title {
        font-size: 1.75rem;
        font-weight: 700;
        color: var(--color-text-primary);
        margin-bottom: 0.5rem;
        background: linear-gradient(135deg, var(--color-text-primary) 0%, var(--color-primary-light) 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .hero-features {
        display: flex;
        justify-content: center;
        gap: 2rem;
        margin: 2rem 0;
        flex-wrap: wrap;
    }
    .hero-feature {
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 0.5rem;
    }
    .hero-feature-icon {
        font-size: 1.75rem;
        display: block;
        color: var(--color-primary);
    }
    .hero-feature-icon i {
        display: inline-block;
    }
    .hero-feature-text {
        font-size: 0.875rem;
        color: var(--color-text-secondary);
        font-weight: 500;
    }
    .hero-social-proof {
        margin-top: 1.5rem;
        padding-top: 1.5rem;
        border-top: 1px solid rgba(255, 255, 255, 0.08);
    }
    .social-proof-text {
        display: block;
        font-size: 0.75rem;
        color: var(--color-text-muted);
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.5rem;
    }
    .social-proof-agents {
        font-size: 0.875rem;
        color: var(--color-primary-light);
        font-weight: 500;
    }

    /* ========== SUCCESS CELEBRATION ========== */
    @keyframes celebrate {
        0% { transform: scale(0.9); opacity: 0; }
        50% { transform: scale(1.02); }
        100% { transform: scale(1); opacity: 1; }
    }
    @keyframes confettiFall {
        0% { transform: translateY(-100%) rotate(0deg); opacity: 1; }
        100% { transform: translateY(100vh) rotate(720deg); opacity: 0; }
    }
    .success-banner {
        animation: celebrate 0.5s ease-out;
        background: var(--success-bg);
        border: 1px solid var(--color-success);
        padding: 1.5rem 2rem;
        border-radius: var(--radius-lg);
        text-align: center;
        margin-bottom: 1.5rem;
        position: relative;
        overflow: hidden;
    }
    .success-banner::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, var(--color-success), #34d399, var(--color-success));
    }
    .success-icon {
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
        display: block;
        color: var(--color-success);
    }
    .success-icon i {
        display: inline-block;
    }
    .success-title {
        font-size: 1.25rem;
        font-weight: 600;
        color: var(--color-success);
        margin-bottom: 0.5rem;
    }
    .success-stats {
        display: flex;
        justify-content: center;
        gap: 2rem;
        margin-top: 1rem;
        flex-wrap: wrap;
    }
    .success-stat {
        text-align: center;
    }
    .success-stat-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: var(--color-text-primary);
    }
    .success-stat-label {
        font-size: 0.75rem;
        color: var(--color-text-muted);
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .confetti {
        position: fixed;
        width: 10px;
        height: 10px;
        top: -10px;
        z-index: 1000;
        animation: confettiFall 3s ease-out forwards;
        pointer-events: none;
    }

    /* ========== ENHANCED PROGRESS/WAIT STATE ========== */
    .progress-card {
        background: var(--progress-bg);
        backdrop-filter: blur(12px);
        border: 1px solid var(--glass-border);
        border-radius: var(--radius-lg);
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
    }
    .progress-ring-container {
        position: relative;
        width: 120px;
        height: 120px;
        margin: 0 auto 1.5rem;
    }
    .progress-ring {
        transform: rotate(-90deg);
    }
    .progress-ring-bg {
        fill: none;
        stroke: var(--color-bg-tertiary);
        stroke-width: 8;
    }
    .progress-ring-fill {
        fill: none;
        stroke: url(#progressGradient);
        stroke-width: 8;
        stroke-linecap: round;
        transition: stroke-dashoffset 0.3s ease;
    }
    .progress-percent {
        position: absolute;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        font-size: 1.5rem;
        font-weight: 700;
        color: var(--color-text-primary);
    }
    .progress-eta {
        font-size: 0.875rem;
        color: var(--color-text-muted);
        margin-bottom: 1rem;
    }
    .progress-agent {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        background: rgba(99, 102, 241, 0.15);
        border: 1px solid rgba(99, 102, 241, 0.3);
        padding: 0.5rem 1rem;
        border-radius: 9999px;
        font-size: 0.875rem;
        color: var(--color-primary-light);
        margin-bottom: 1rem;
    }
    .progress-agent-icon {
        font-size: 1.25rem;
    }
    .progress-tip {
        font-size: 0.8125rem;
        color: var(--color-text-muted);
        font-style: italic;
        margin-top: 1.5rem;
        padding-top: 1rem;
        border-top: 1px solid var(--color-border);
    }
    .progress-tip strong {
        color: var(--color-primary-light);
    }
    .progress-ring-text {
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    .progress-info {
        text-align: center;
    }
    .agent-icon {
        font-size: 1.25rem;
        color: inherit;
    }
    .agent-icon i {
        display: inline-block;
    }
    .agent-name {
        font-weight: 500;
    }

    /* ========== TIP CONTAINER ========== */
    .tip-container {
        margin-top: 1.5rem;
        padding: 1.25rem 1.5rem;
        text-align: center;
    }
    .tip-label {
        display: block;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: var(--color-primary-light);
        margin-bottom: 0.5rem;
    }
    .tip-text {
        font-size: 0.9375rem;
        color: var(--color-text-secondary);
        margin: 0;
        line-height: 1.5;
    }

    /* ========== TOAST NOTIFICATION ========== */
    @keyframes toastSlideIn {
        from { transform: translateY(100%); opacity: 0; }
        to { transform: translateY(0); opacity: 1; }
    }
    @keyframes toastSlideOut {
        from { transform: translateY(0); opacity: 1; }
        to { transform: translateY(100%); opacity: 0; }
    }
    .toast {
        position: fixed;
        bottom: 2rem;
        left: 50%;
        transform: translateX(-50%);
        background: var(--color-success);
        color: white;
        padding: 0.875rem 1.5rem;
        border-radius: var(--radius-md);
        font-weight: 500;
        font-size: 0.9375rem;
        z-index: 9999;
        box-shadow: 0 4px 20px rgba(16, 185, 129, 0.4);
        animation: toastSlideIn 0.3s ease-out;
    }
    .toast.hiding {
        animation: toastSlideOut 0.3s ease-out forwards;
    }

    /* ========== IMPROVED HISTORY CARDS ========== */
    .history-card {
        background: linear-gradient(135deg, var(--color-bg-secondary) 0%, var(--color-bg-tertiary) 100%);
        border: 1px solid var(--color-border);
        border-radius: var(--radius-md);
        padding: 1rem;
        margin-bottom: 0.75rem;
        cursor: pointer;
        transition: all 0.2s ease;
        position: relative;
    }
    .history-card:hover {
        border-color: var(--color-primary);
        background: var(--color-bg-tertiary);
        transform: translateX(4px);
        box-shadow: var(--shadow-md);
    }
    .history-card-title {
        font-size: 0.875rem;
        font-weight: 600;
        color: var(--color-text-primary);
        margin-bottom: 0.375rem;
        line-height: 1.3;
        display: -webkit-box;
        -webkit-line-clamp: 2;
        -webkit-box-orient: vertical;
        overflow: hidden;
    }
    .history-card-meta {
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-size: 0.75rem;
        color: var(--color-text-muted);
    }
    .history-card-date {
        display: flex;
        align-items: center;
        gap: 0.25rem;
    }
    .history-card-words {
        background: rgba(99, 102, 241, 0.1);
        padding: 0.125rem 0.5rem;
        border-radius: 9999px;
        color: var(--color-primary-light);
    }
    .history-card-stats {
        background: rgba(99, 102, 241, 0.1);
        padding: 0.125rem 0.5rem;
        border-radius: 9999px;
        color: var(--color-primary-light);
        font-size: 0.6875rem;
    }
    .history-search {
        margin-bottom: 1rem;
    }

    /* Empty state */
    .empty-state {
        text-align: center;
        padding: 2rem 1rem;
        color: var(--color-text-muted);
    }
    .empty-state-icon {
        font-size: 2.5rem;
        margin-bottom: 0.75rem;
        display: block;
        color: var(--color-text-muted);
    }
    .empty-state-icon i {
        display: inline-block;
    }
    .empty-state-text {
        font-size: 0.9375rem;
        font-weight: 500;
        color: var(--color-text-secondary);
        margin-bottom: 0.25rem;
    }
    .empty-state-hint {
        font-size: 0.8125rem;
        color: var(--color-text-muted);
    }

    .history-empty-state {
        text-align: center;
        padding: 2rem 1rem;
        color: var(--color-text-muted);
    }
    .history-empty-icon {
        font-size: 2rem;
        margin-bottom: 0.5rem;
        opacity: 0.5;
    }
    .history-empty-text {
        font-size: 0.875rem;
    }
    .history-empty-cta {
        font-size: 0.75rem;
        color: var(--color-primary);
        margin-top: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)


def safe_html(text: str) -> str:
    """Escape HTML entities in dynamic text before injection into unsafe_allow_html."""
    return html.escape(str(text)) if text else ""


GENERATION_COOLDOWN_SECONDS = 30  # Minimum time between generations


def can_generate() -> tuple:
    """Check if generation is allowed (cooldown, concurrent limit)."""
    last_gen = st.session_state.get("last_generation_time", 0)
    elapsed = time.time() - last_gen

    if elapsed < GENERATION_COOLDOWN_SECONDS:
        remaining = int(GENERATION_COOLDOWN_SECONDS - elapsed)
        return False, f"Please wait {remaining}s before generating again"

    if st.session_state.is_running:
        return False, "A generation is already in progress"

    return True, ""


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
        'custom_topic': '',  # User-provided custom topic
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def get_draft_history():
    """
    Scan drafts directory and return list of saved articles.

    Returns:
        List of dicts with filename, title, date, filepath, word_count, read_time
    """
    drafts_dir = OUTPUT_DIR
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
        except (ValueError, IndexError, OSError):
            date_formatted = "Unknown"

        # Extract title and word count from content
        title = md_file.stem
        word_count = 0
        try:
            with open(md_file, 'r', encoding='utf-8') as f:
                content = f.read()
                first_line = content.split('\n')[0].strip()
                title = first_line.lstrip('#').strip()[:80]
                if not title:
                    title = md_file.stem
                word_count = len(content.split())
        except (IOError, UnicodeDecodeError):
            pass  # File read error, skip word count

        read_time = max(1, word_count // 200)  # Average reading speed

        history.append({
            'filename': md_file.name,
            'title': title,
            'date': date_formatted,
            'filepath': str(md_file),
            'word_count': word_count,
            'read_time': read_time,
        })

    return history


def load_draft(filepath):
    """Load a draft from file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except (IOError, UnicodeDecodeError, PermissionError) as e:
        print(f"Failed to load draft {filepath}: {e}")
        return None


def save_to_env(key: str, value: str):
    """Save or update a key in .env file."""
    env_path = PROJECT_ROOT / ".env"

    # Ensure the .env file exists before set_key tries to read it
    if not env_path.exists():
        env_path.touch()

    # Use python-dotenv's set_key which preserves comments, blank lines,
    # and handles quoting correctly
    set_key(str(env_path), key, value)

    # Reload into environment
    os.environ[key] = value


def copy_to_clipboard(text: str, button_text: str = "Copy", key: str = "copy_btn"):
    """
    Create a one-click copy button with toast notification.
    Uses JavaScript clipboard API for instant copy.
    """
    import streamlit.components.v1 as components

    # Proper JS string escaping via json.dumps (handles all special chars)
    safe_js_text = json.dumps(text)

    components.html(f'''
        <style>
            .copy-btn {{
                background: linear-gradient(135deg, #818cf8 0%, #6366f1 100%);
                color: white;
                border: none;
                padding: 0.625rem 1.5rem;
                border-radius: 8px;
                font-size: 0.9375rem;
                font-weight: 500;
                cursor: pointer;
                width: 100%;
                transition: all 0.2s ease;
                font-family: 'Inter', -apple-system, sans-serif;
            }}
            .copy-btn:hover {{
                transform: translateY(-1px);
                box-shadow: 0 4px 12px rgba(99, 102, 241, 0.4);
            }}
            .copy-btn.copied {{
                background: linear-gradient(135deg, #34d399 0%, #10b981 100%);
            }}
            .toast {{
                position: fixed;
                bottom: 20px;
                left: 50%;
                transform: translateX(-50%) translateY(100px);
                background: #34d399;
                color: white;
                padding: 12px 24px;
                border-radius: 8px;
                font-size: 14px;
                font-weight: 500;
                opacity: 0;
                transition: all 0.3s ease;
                z-index: 9999;
                font-family: 'Inter', -apple-system, sans-serif;
                box-shadow: 0 4px 12px rgba(0,0,0,0.3);
            }}
            .toast.show {{
                transform: translateX(-50%) translateY(0);
                opacity: 1;
            }}
        </style>
        <button class="copy-btn" onclick="copyToClipboard(this)">{safe_html(button_text)}</button>
        <div class="toast" id="toast-{safe_html(key)}"><i class="fi fi-rr-check" style="margin-right: 6px;"></i>Copied to clipboard!</div>
        <script>
            function copyToClipboard(btn) {{
                const text = {safe_js_text};
                navigator.clipboard.writeText(text).then(() => {{
                    btn.innerHTML = '<i class="fi fi-rr-check" style="margin-right: 4px;"></i>Copied!';
                    btn.classList.add('copied');
                    const toast = document.getElementById('toast-{key}');
                    toast.classList.add('show');
                    setTimeout(() => {{
                        btn.innerHTML = '{button_text}';
                        btn.classList.remove('copied');
                        toast.classList.remove('show');
                    }}, 2000);
                }}).catch(err => {{
                    btn.innerHTML = '<i class="fi fi-rr-cross" style="margin-right: 4px;"></i>Failed';
                    setTimeout(() => btn.innerHTML = '{button_text}', 2000);
                }});
            }}
        </script>
    ''', height=50)


from execution.exceptions import (
    PipelineError,
    ResearchError,
    WriterError,
    VerificationError,
    QualityGateError,
    StyleError,
)


def _phase_research_topic(progress_callback, status_callback, provided_topic=None):
    """
    Phase 1: Select or research a topic. Returns dict with topic info and source metadata.

    Raises ResearchError on failure.
    """
    from execution.agents.topic_researcher import TopicResearchAgent

    data_source = st.session_state.get('data_source', 'reddit')
    source_type = "internal" if data_source == "github" else "external"

    try:
        if provided_topic:
            status_callback(f"Using provided topic: {provided_topic[:30]}...")
            progress_callback(0.05)

            topic_agent = TopicResearchAgent()
            topics = [{'title': provided_topic, 'subreddit': 'custom'}]
            selected = topic_agent.analyze_and_select(topics)

        elif data_source == 'github':
            from execution.agents.commit_analyzer import CommitAnalysisAgent
            from execution.fetch_github import fetch_repo_commits

            status_callback("Analyzing GitHub commits for topics...")
            progress_callback(0.02)

            commit_agent = CommitAnalysisAgent()
            custom_repos = st.session_state.get('custom_github_repos')

            if custom_repos:
                status_callback(f"Fetching from {len(custom_repos)} custom repo(s)...")
                progress_callback(0.04)

                all_commits = []
                for repo_str in custom_repos[:5]:
                    if "/" in repo_str:
                        owner, repo = repo_str.split("/", 1)
                        commits = fetch_repo_commits(owner, repo, max_commits=20, since_hours=168)
                        all_commits.extend(commits)

                status_callback("Extracting themes from commit activity...")
                progress_callback(0.08)

                if all_commits:
                    selected = commit_agent.research_github_topics(all_commits)
                else:
                    selected = commit_agent.research_github_topics_from_db()
            else:
                status_callback("Fetching commits from database...")
                progress_callback(0.05)
                status_callback("Extracting themes from commit activity...")
                progress_callback(0.08)
                selected = commit_agent.research_github_topics_from_db()

        else:
            status_callback("Scanning Reddit for trending topics...")
            progress_callback(0.02)

            topic_agent = TopicResearchAgent()

            status_callback("Fetching from AI/ML subreddits...")
            progress_callback(0.04)
            topics = topic_agent.fetch_trending_topics()

            status_callback("Analyzing topics for positioning fit...")
            progress_callback(0.08)
            selected = topic_agent.analyze_and_select(topics)

    except Exception as e:
        raise ResearchError(f"Topic research failed: {e}") from e

    status_callback("Topic selected")
    progress_callback(0.10)

    return {
        'topic': selected['title'],
        'topic_reasoning': selected.get('reasoning', ''),
        'topic_angle': selected.get('angle', ''),
        'source_type': source_type,
        'selected': selected,
    }


def _phase_research_facts(topic, selected, progress_callback, status_callback):
    """
    Phase 1.5: Fact research via Perplexity. Returns dict with fact_sheet and writer_constraints.

    Non-fatal: returns empty results if Perplexity unavailable.
    """
    perplexity_available = bool(os.getenv("PERPLEXITY_API_KEY"))
    fact_researcher = None
    fact_sheet = None
    writer_constraints = ""

    if perplexity_available:
        try:
            from execution.agents.perplexity_researcher import PerplexityResearchAgent
            fact_researcher = PerplexityResearchAgent()
        except Exception as e:
            print(f"Perplexity init failed: {e}")
            perplexity_available = False

    if perplexity_available and fact_researcher:
        status_callback("Researching facts via Perplexity...")
        progress_callback(0.12)

        try:
            source_content = selected.get('content', '') or selected.get('body', '')
            fact_sheet = fact_researcher.research_topic(
                topic=topic,
                source_content=source_content[:3000] if source_content else ""
            )
            writer_constraints = fact_sheet.get('writer_constraints', '')

            verified_count = len(fact_sheet.get('verified_facts', []))
            unverified_count = len(fact_sheet.get('unverified_claims', []))
            status_callback(f"Found {verified_count} verified facts, {unverified_count} flagged claims")
        except Exception as e:
            print(f"Fact research failed: {e}")
            status_callback("Fact research skipped (continuing)")
    else:
        status_callback("Skipping fact research (no Perplexity API key)")

    progress_callback(0.18)

    return {
        'fact_sheet': fact_sheet,
        'writer_constraints': writer_constraints,
        'fact_researcher': fact_researcher,
        'perplexity_available': perplexity_available,
    }


def _phase_generate_draft(topic, topic_angle, writer_constraints, source_type, progress_callback, status_callback):
    """
    Phases 2-5: Initialize agents, create outline, refine, and generate draft.

    Raises WriterError if outline or draft generation fails fatally.
    """
    from execution.agents.editor import EditorAgent
    from execution.agents.critic import CriticAgent
    from execution.agents.writer import WriterAgent

    # Agent initialization (18-22%)
    status_callback("Initializing agents...")
    progress_callback(0.20)

    editor = EditorAgent()
    critic = CriticAgent()
    writer = WriterAgent()
    progress_callback(0.22)

    # Outline (22-30%)
    status_callback("Creating outline...")
    progress_callback(0.24)
    try:
        outline = editor.create_outline(topic)
    except Exception as e:
        raise WriterError(f"Outline creation failed: {e}") from e
    progress_callback(0.27)

    status_callback("Reviewing outline...")
    try:
        critique = critic.critique_outline(outline)
    except Exception as e:
        critique = ""
        status_callback(f"Critique skipped ({e})")
    progress_callback(0.30)

    # Refine outline (30-35%)
    status_callback("Refining outline...")
    try:
        refined_outline = editor.call_llm(
            f"Refine this outline based on critique. Topic angle: {topic_angle}\n\nOutline:\n{outline}\n\nCritique:\n{critique}"
        )
        if not refined_outline or len(refined_outline.strip()) < 50:
            refined_outline = outline
    except Exception as e:
        refined_outline = outline
        status_callback(f"Refinement skipped ({e})")
    progress_callback(0.35)

    # Draft (35-50%)
    status_callback("Writing first draft...")
    progress_callback(0.38)

    draft_instruction = "Write the full article draft."
    if writer_constraints:
        draft_instruction = f"""Write the full article draft.

CRITICAL - FACT CONSTRAINTS FROM RESEARCH:
{writer_constraints}

You MUST follow these constraints. Only use verified facts. Do NOT invent statistics."""

    try:
        draft = writer.write_section(refined_outline, critique=draft_instruction, source_type=source_type)

        error_prefixes = ["Error:", "Groq Error:", "Gemini Error:", "OpenAI Error:"]
        if any(draft.strip().startswith(p) for p in error_prefixes):
            raise WriterError(f"Writer returned error: {draft[:100]}")
        if len(draft.strip()) < 200:
            raise WriterError(f"Draft too short ({len(draft)} chars)")
    except WriterError:
        raise
    except Exception as e:
        raise WriterError(f"Draft generation failed: {e}") from e

    progress_callback(0.50)
    return draft


def _phase_specialists(draft, source_type, progress_callback, status_callback):
    """
    Phase 6: Run specialist agents (hook, storytelling, voice, value density) + final polish.

    Returns the refined draft string.
    """
    from execution.agents.specialist import SpecialistAgent

    if source_type == "internal":
        storytelling_instruction = """Your job is to make this feel like it's written by a REAL engineer, not an AI.

1. Add 1-2 brief personal moments: a frustration, a realization, a late-night debugging session.
   Keep these SHORT (2-3 sentences max) and weave them naturally into transitions.
2. Use "I" and "we" to create conversation - "I've seen this pattern fail" or "We hit this wall"
3. The personality should feel like a smart colleague explaining something over coffee, not a textbook.

Do NOT add fake-sounding stories. If it feels forced, cut it."""

        voice_instruction = """Your job is to make this sound like ONE consistent, authentic PRACTITIONER.

1. Remove anything that sounds like corporate speak, marketing jargon, or AI-generated filler.
2. You ARE the person who did this work - "I", "we", "our" are encouraged
3. Add subtle wit where natural - a wry observation, a knowing aside. NOT forced jokes.
4. Vary sentence rhythm: mix punchy short sentences with longer explanations.
5. The tone should be: confident but not arrogant, technical but accessible, opinionated but fair.

Read it aloud - if it sounds like a robot wrote it, rewrite those parts."""

    else:
        storytelling_instruction = """Your job is to make this feel like it's written by a REAL tech journalist, not an AI.

CRITICAL: You are an OBSERVER. You did NOT build this. Never claim ownership.

1. Add narrative tension through OTHERS' experiences: "The team hit a wall", "Engineers discovered..."
2. Use "you" to engage readers - "You've probably hit this wall"
3. FORBIDDEN: "I built", "we created", "our team", "my approach"
4. ALLOWED: "I noticed", "I've been tracking", "I find this interesting" (observations only)
5. The personality should feel like a knowledgeable journalist sharing insights.

Do NOT add fake ownership claims. If you catch yourself writing "we built" or "I created", STOP and rewrite."""

        voice_instruction = """Your job is to make this sound like ONE consistent, authentic OBSERVER/JOURNALIST.

CRITICAL VOICE CHECK - You are NOT the builder, you are REPORTING on others' work:
- FORBIDDEN: "I built", "we created", "our team", "my approach", "we discovered"
- USE INSTEAD: "teams found", "engineers discovered", "this approach", "the implementation"
- ALLOWED: "I noticed", "I've observed", "I find this interesting" (observations only)

1. Remove anything that sounds like corporate speak, marketing jargon, or AI-generated filler.
2. Scan for ANY ownership claims ("I built", "we created", "our") and replace with observer language.
3. Add subtle wit where natural - a wry observation, a knowing aside. NOT forced jokes.
4. Vary sentence rhythm: mix punchy short sentences with longer explanations.
5. The tone should be: confident but not arrogant, technical but accessible, authoritative but fair.

FINAL CHECK: Read through and ensure NO "we built", "I created", "our team" phrases remain."""

    specialists = [
        ("Hook Specialist", "Optimizing title and hook...",
         """Your job is to make readers STOP scrolling.

1. Rewrite the title to create curiosity or promise transformation.
   Examples of good patterns: "Why teams are stopping X and starting Y", "The X that nobody talks about", "What engineers learned from X"
2. The opening line should feel like the start of a conversation, not a thesis statement.
3. The hook should make the reader think "this person gets it" within 10 seconds.

Keep the author's voice - don't make it sound like marketing copy."""),

        ("Storytelling Architect", "Adding authentic narrative...", storytelling_instruction),

        ("Voice & Tone Specialist", "Refining voice...", voice_instruction),

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
        try:
            specialist = SpecialistAgent(constraint_name=name, constraint_instruction=instruction)
            result = specialist.refine(draft)
            if result and len(result.strip()) > len(draft.strip()) * 0.3:
                draft = result
            else:
                status_callback(f"{name} output invalid, keeping previous draft")
        except Exception as e:
            status_callback(f"{name} failed ({e}), skipping")
        current += progress_per
        progress_callback(current)

    # Final polish (75%)
    status_callback("Final polish...")

    if source_type == "internal":
        final_polish_voice = """
7. VOICE CHECK: This is a practitioner piece - "I", "we", "our" are appropriate and encouraged."""
    else:
        final_polish_voice = """
7. CRITICAL VOICE CHECK: This is an OBSERVER piece (Reddit source).
   - SCAN AND FIX any "I built", "we created", "our team", "my approach" phrases
   - Replace with: "teams found", "engineers discovered", "this approach", "the implementation"
   - ONLY allowed: "I noticed", "I've observed" (observations, not ownership)"""

    polisher = SpecialistAgent(
        constraint_name="Final Editor",
        constraint_instruction=f"""Final pass before publication. Your job is COHESION and CLEAN OUTPUT.

1. Strip ALL internal labels, metadata markers, section tags, and any "Value-Bait:", "Hook:", etc. prefixes.
2. Format first line as # H1 Title (clean, no labels).
3. Ensure smooth transitions between sections - no jarring jumps.
4. Remove any repetitive phrases or ideas that got duplicated across specialist passes.
5. If there are forced bullet lists that interrupt flow, convert them to natural prose.
6. The final piece should read like ONE person wrote it in ONE sitting - not a committee.
{final_polish_voice}

Output ONLY the polished markdown. No explanations, no meta-commentary."""
    )
    draft = polisher.refine(draft)
    progress_callback(0.75)

    return draft


def _phase_verify_draft(draft, topic, fact_researcher, perplexity_available, progress_callback, status_callback):
    """
    Phase 7.5: Verify claims in draft via Perplexity. Returns verification_result dict (or None).

    Non-fatal: returns None if verification unavailable or fails.
    """
    from execution.agents.specialist import SpecialistAgent

    verification_result = None

    if perplexity_available and fact_researcher:
        status_callback("Verifying claims via Perplexity...")
        progress_callback(0.77)

        try:
            verification_result = fact_researcher.verify_draft(
                draft=draft,
                topic=topic
            )

            recommendation = verification_result.get('recommendation', 'PASS')
            false_claims = len(verification_result.get('false_claims', []))
            unverifiable = len(verification_result.get('unverifiable_claims', []))

            if recommendation != 'PASS' and (false_claims > 0 or unverifiable > 2):
                status_callback(f"Fixing {false_claims} false, {unverifiable} unverifiable claims...")

                revision_instructions = verification_result.get('revision_instructions', '')
                if revision_instructions:
                    fact_fixer = SpecialistAgent(
                        constraint_name="Fact Correction Specialist",
                        constraint_instruction=f"""You are fixing factual issues identified by web search verification.

{revision_instructions}

INSTRUCTIONS:
1. Remove or correct FALSE claims (replace with verified alternatives if available)
2. Remove specific numbers from UNVERIFIABLE claims (write around them)
3. Keep VERIFIED claims exactly as they are
4. Maintain the article's flow and voice while fixing facts
5. Do NOT add new claims or statistics

Output ONLY the corrected article. No explanations."""
                    )
                    draft = fact_fixer.refine(draft)
                    status_callback("Fact corrections applied")
            else:
                status_callback(f"Verification passed (score: {verification_result.get('overall_accuracy_score', 'N/A')}/100)")

        except Exception as e:
            print(f"Draft verification failed: {e}")
            status_callback("Verification skipped (continuing)")

    progress_callback(0.82)

    return draft, verification_result


def _phase_quality_gate(draft, source_type, progress_callback, status_callback):
    """
    Phase 8: Adversarial expert panel review. Returns (updated_draft, quality_result).

    Raises QualityGateError on fatal failure.
    """
    from execution.quality_gate import QualityGate

    status_callback("Quality gate review...")
    progress_callback(0.82)

    try:
        quality_gate = QualityGate(max_iterations=3, verbose=False)
        quality_result = quality_gate.process(
            content=draft,
            platform="medium",
            source_type=source_type
        )
    except Exception as e:
        raise QualityGateError(f"Quality gate failed: {e}") from e

    draft = quality_result.final_content

    if quality_result.passed:
        status_callback(f"Quality gate passed ({quality_result.final_score}/10)")
    else:
        status_callback(f"Escalated for review ({quality_result.final_score}/10)")

    progress_callback(0.92)

    return draft, quality_result


def _phase_generate_visuals(draft, progress_callback, status_callback):
    """
    Phase 9: Generate visual suggestions and infographic images.

    Returns (visual_plan, image_paths).
    """
    from execution.agents.visuals import VisualsAgent

    status_callback("Generating visuals...")
    visuals = VisualsAgent()
    visual_plan = visuals.suggest_visuals(draft)
    image_paths = []

    output_dir = OUTPUT_DIR
    if visual_plan and isinstance(visual_plan, list):
        status_callback(f"Creating {len(visual_plan)} infographics...")
        images_dir = output_dir / "images"
        image_paths = visuals.generate_all_visuals(visual_plan, output_dir=str(images_dir))

    progress_callback(0.98)

    return visual_plan, image_paths


def _phase_save_article(draft, image_paths, progress_callback, status_callback):
    """
    Phase 10: Save final article to disk. Returns filepath string.
    """
    status_callback("Saving...")

    output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = utc_now().strftime("%Y%m%d_%H%M%S")
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

    return str(filepath)


def run_full_pipeline(progress_callback, status_callback, provided_topic=None):
    """
    Run the COMPLETE automated pipeline by orchestrating phase functions:
    1. Topic Research Agent selects best topic
    2. Fact Research via Perplexity (grounded web search)
    3. Full article generation with all agents
    4. Draft Verification via Perplexity
    5. Quality Gate (adversarial review loop)
    6. Visual generation

    Raises PipelineError subclasses (ResearchError, WriterError, etc.) on fatal failures.
    Non-fatal phases (fact research, verification, specialists) degrade gracefully.

    Returns:
        dict with content, filepath, topic info, images, quality_result, fact_sheet
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Phase 1: Topic research (0-10%)
    research = _phase_research_topic(progress_callback, status_callback, provided_topic)
    topic = research['topic']
    topic_angle = research['topic_angle']
    source_type = research['source_type']
    selected = research['selected']

    # Phase 1.5: Fact research (10-18%)
    facts = _phase_research_facts(topic, selected, progress_callback, status_callback)

    # Phases 2-5: Outline + Draft (18-50%)
    draft = _phase_generate_draft(
        topic, topic_angle, facts['writer_constraints'], source_type,
        progress_callback, status_callback,
    )

    # Phase 6: Specialists + Polish (50-75%)
    draft = _phase_specialists(draft, source_type, progress_callback, status_callback)

    # Phase 7.5: Draft verification (75-82%)
    draft, verification_result = _phase_verify_draft(
        draft, topic, facts['fact_researcher'], facts['perplexity_available'],
        progress_callback, status_callback,
    )

    # Phase 8: Quality gate (82-92%)
    draft, quality_result = _phase_quality_gate(draft, source_type, progress_callback, status_callback)

    # Phase 9: Visuals (92-98%)
    visual_plan, image_paths = _phase_generate_visuals(draft, progress_callback, status_callback)

    # Phase 10: Save (98-100%)
    filepath = _phase_save_article(draft, image_paths, progress_callback, status_callback)

    return {
        'content': draft,
        'filepath': filepath,
        'image_paths': image_paths,
        'visual_plan': visual_plan,
        'topic': topic,
        'topic_reasoning': research['topic_reasoning'],
        'topic_angle': topic_angle,
        'quality_score': quality_result.final_score,
        'quality_passed': quality_result.passed,
        'quality_iterations': quality_result.iterations_used,
        'fact_sheet': facts['fact_sheet'],
        'verification_result': verification_result,
        'perplexity_used': facts['perplexity_available'],
    }


def main():
    check_auth()
    init_session_state()

    # Header
    st.markdown('<p class="main-header">GhostWriter</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">One-click AI ghostwriting for technical thought leadership</p>', unsafe_allow_html=True)

    # Sidebar - History & Config
    with st.sidebar:
        # New Article button at top
        if st.button("New Article", type="primary", use_container_width=True, key="sidebar_new"):
            st.session_state.viewing_history = None
            st.session_state.generation_complete = False
            st.session_state.article_content = None
            st.session_state.custom_topic = ""
            st.rerun()

        st.divider()

        # History section
        st.markdown('<p class="section-header">Recent Articles</p>', unsafe_allow_html=True)

        history = get_draft_history()
        if history:
            for i, item in enumerate(history[:10]):  # Limit to 10 most recent
                # Rich card display with metadata
                display_title = item["title"][:45] + "..." if len(item["title"]) > 45 else item["title"]
                word_count = item.get('word_count', 0)
                read_time = item.get('read_time', 0)

                # Create clickable card using HTML + button
                st.markdown(f'''
                <div class="history-card" data-index="{i}">
                    <div class="history-card-title">{safe_html(display_title)}</div>
                    <div class="history-card-meta">
                        <span class="history-card-date">{safe_html(item['date'])}</span>
                        <span class="history-card-stats">{word_count:,} words · {read_time} min</span>
                    </div>
                </div>
                ''', unsafe_allow_html=True)

                # Hidden button for click handling
                if st.button(
                    "Open",
                    key=f"hist_{i}",
                    use_container_width=True,
                ):
                    st.session_state.viewing_history = item
                    st.session_state.generation_complete = False
                    st.rerun()
        else:
            st.markdown('''
            <div class="empty-state">
                <div class="empty-state-icon"><i class="fi fi-rr-document" style="font-size: 40px;"></i></div>
                <p class="empty-state-text">No articles yet</p>
                <p class="empty-state-hint">Your first article is one click away</p>
            </div>
            ''', unsafe_allow_html=True)

        st.divider()

        # Config section (collapsed by default)
        with st.expander("Settings", expanded=False):
            google_key = os.getenv("GOOGLE_API_KEY")
            groq_key = os.getenv("GROQ_API_KEY")
            github_token = os.getenv("GITHUB_TOKEN") or os.getenv("GITHUB_PAT")
            perplexity_key = os.getenv("PERPLEXITY_API_KEY")

            st.caption(f"Groq: {'OK' if groq_key else 'Missing'}")
            st.caption(f"Google: {'OK' if google_key else 'Missing'}")
            st.caption(f"Perplexity: {'OK (fact verification)' if perplexity_key else 'Missing (optional)'}")
            st.caption(f"GitHub: {'OK (5000 req/hr)' if github_token else 'Anonymous (60 req/hr)'}")

            st.markdown('<p class="section-header">Data Source</p>', unsafe_allow_html=True)
            source = st.radio(
                "Topic Source",
                ["reddit", "github"],
                index=0 if st.session_state.get('data_source', 'reddit') == 'reddit' else 1,
                format_func=lambda x: "Reddit" if x == "reddit" else "GitHub",
                key="source_selector",
                label_visibility="collapsed"
            )
            st.session_state.data_source = source

            # GitHub Configuration (show when GitHub is selected)
            if source == "github":
                st.markdown('<p class="section-header">GitHub Config</p>', unsafe_allow_html=True)

                # Token input (masked)
                token_input = st.text_input(
                    "GitHub Token",
                    value="" if not github_token else "ghp_" + "*" * 36,
                    type="password",
                    placeholder="ghp_xxxxxxxxxxxxxxxxxxxx",
                    help="Create at github.com/settings/tokens (select 'repo' scope for private repos)"
                )

                # Save token button
                if token_input and not token_input.startswith("ghp_*"):
                    if st.button("Save Token", key="save_github_token", use_container_width=True):
                        save_to_env("GITHUB_TOKEN", token_input)
                        st.success("Token saved!")
                        st.rerun()

                if github_token:
                    st.caption("Token configured — 5000 req/hr")
                else:
                    st.caption("No token — using anonymous access (60 req/hr)")

                st.markdown("---")

                # Custom repos input
                default_repos = os.getenv("GITHUB_REPOS", "")
                custom_repos = st.text_area(
                    "Custom Repos (one per line)",
                    value=default_repos.replace(",", "\n") if default_repos else "",
                    height=80,
                    placeholder="owner/repo\nmicrosoft/semantic-kernel\nyour-username/your-repo",
                    help="Leave empty to use default AI/ML repos"
                )

                if custom_repos.strip():
                    # Store in session for pipeline to use
                    repos_list = [r.strip() for r in custom_repos.strip().split("\n") if r.strip() and "/" in r]
                    st.session_state.custom_github_repos = repos_list
                    st.caption(f"{len(repos_list)} repo(s) configured")
                else:
                    st.session_state.custom_github_repos = None

            st.markdown("---")
            if st.button("Run Health Check", key="health_check_btn", use_container_width=True):
                try:
                    from execution.utils.health import check_health
                    result = check_health()
                    status = result["status"]
                    if status == "healthy":
                        st.success(f"System: {status}")
                    elif status == "degraded":
                        st.warning(f"System: {status}")
                    else:
                        st.error(f"System: {status}")
                    for name, info in result["checks"].items():
                        st.caption(f"{name}: {info.get('status', '?')} - {info.get('detail', '')}")
                except Exception as e:
                    st.error(f"Health check failed: {e}")

    # Main content
    if st.session_state.viewing_history:
        # Viewing a history item
        draft_content = load_draft(st.session_state.viewing_history['filepath'])

        if draft_content:
            st.markdown(f'<div class="topic-display">{safe_html(st.session_state.viewing_history["title"])}</div>', unsafe_allow_html=True)
            st.markdown(f'<p class="topic-reasoning">Saved: {safe_html(st.session_state.viewing_history["date"])}</p>', unsafe_allow_html=True)

            st.divider()

            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    label="Download",
                    data=draft_content,
                    file_name=st.session_state.viewing_history['filename'],
                    mime="text/markdown",
                    use_container_width=True
                )
            with col2:
                copy_to_clipboard(draft_content, "Copy Article", "history_copy")

            st.markdown('<p class="section-header">Article Content</p>', unsafe_allow_html=True)
            with st.expander("View Full Article", expanded=True):
                st.markdown(draft_content)
        else:
            st.error("Could not load draft file")
            if st.button("Back"):
                st.session_state.viewing_history = None
                st.rerun()

    elif not st.session_state.generation_complete:

        # Hero section - compelling value proposition
        hero_html = '''<div class="hero-section glass-card gradient-mesh">
<h2 class="hero-title">Generate Recruiter-Magnet Articles</h2>
<p class="hero-tagline">One click. 10 AI agents. Professional thought leadership.</p>
<div class="hero-features">
<div class="hero-feature">
<span class="hero-feature-icon"><i class="fi fi-rr-search"></i></span>
<span class="hero-feature-text">Scans trending AI/ML topics</span>
</div>
<div class="hero-feature">
<span class="hero-feature-icon"><i class="fi fi-rr-pencil"></i></span>
<span class="hero-feature-text">10-agent writing pipeline</span>
</div>
<div class="hero-feature">
<span class="hero-feature-icon"><i class="fi fi-rr-palette"></i></span>
<span class="hero-feature-text">Auto-generated visuals</span>
</div>
</div>
<div class="hero-social-proof">
<span class="social-proof-text">Powered by specialized AI agents:</span>
<span class="social-proof-agents">Hook • Storytelling • Voice • Value • SEO</span>
</div>
</div>'''
        st.markdown(hero_html, unsafe_allow_html=True)

        # Topic customization - ONLY visible when not running
        if not st.session_state.is_running:
            st.markdown('<p class="section-header">Customize Your Content</p>', unsafe_allow_html=True)
            
            # Simple text input with manual sync to avoid rerun loops
            topic_input = st.text_input(
                "What do you want to write about?",
                value=st.session_state.custom_topic,
                placeholder="e.g. Scaling Postgres to 10M users, The future of Web3, etc.",
                help="Leave empty to let the AI research trending topics for you."
            )
            if topic_input != st.session_state.custom_topic:
                st.session_state.custom_topic = topic_input
                st.rerun()

            # Suggested topics organized as small chips
            suggestions = [
                "Scaling Postgres to 10M",
                "Why Rust for Infrastructure",
                "System Design Mastery",
                "The 2026 Developer Market",
                "Clean Architecture in 2026"
            ]
            
            st.markdown('<p style="font-size: 0.75rem; color: var(--color-text-muted); margin-top: -0.5rem; margin-bottom: 0.5rem;">Or try a suggestion:</p>', unsafe_allow_html=True)
            cols = st.columns(len(suggestions))
            for i, suggestion in enumerate(suggestions):
                if cols[i].button(suggestion, key=f"sug_{i}", use_container_width=True):
                    st.session_state.custom_topic = suggestion
                    st.rerun()

            st.divider()

        # Generate button - centered and prominent with rate limiting
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            can_gen, gen_message = can_generate()
            if st.button(
                "Generate Article",
                type="primary",
                use_container_width=True,
                disabled=not can_gen or st.session_state.is_running,
                key="main_generate"
            ):
                st.session_state.is_running = True
                st.session_state.last_generation_time = time.time()
                st.rerun()

            if gen_message:
                st.caption(gen_message)

        # Pipeline execution
        if st.session_state.is_running:
            st.divider()

            # Enhanced progress display with ring, ETA, agents, and tips
            progress_container = st.empty()
            tips_container = st.empty()

            # Tips to show during wait
            wait_tips = [
                "The Hook Specialist ensures your opening grabs attention in the first 3 seconds.",
                "Voice & Tone Agent removes AI-sounding phrases for authentic human voice.",
                "Value Density Specialist ensures every paragraph earns its place.",
                "Articles generated here have 3x higher engagement than generic AI content.",
                "The multi-agent pipeline mimics how top content teams operate.",
                "Storytelling Architect weaves personal narrative for authenticity.",
                "Each agent specializes in one aspect, just like a real editorial team.",
                "Perplexity Sonar Pro searches the web to verify every fact before writing.",
                "The fact verification step catches fabricated statistics and phantom evidence.",
                "Claims are checked against official documentation and real sources.",
            ]

            # Agent icons (Flaticon UIcons)
            icon_search = '<i class="fi fi-rr-search"></i>'
            icon_gear = '<i class="fi fi-rr-settings"></i>'
            icon_clipboard = '<i class="fi fi-rr-clipboard"></i>'
            icon_pencil = '<i class="fi fi-rr-pencil"></i>'
            icon_note = '<i class="fi fi-rr-document"></i>'
            icon_target = '<i class="fi fi-rr-bullseye"></i>'
            icon_sparkle = '<i class="fi fi-rr-sparkles"></i>'
            icon_magnify = '<i class="fi fi-rr-zoom-in"></i>'
            icon_palette = '<i class="fi fi-rr-palette"></i>'
            icon_check = '<i class="fi fi-rr-check"></i>'
            icon_refresh = '<i class="fi fi-rr-refresh"></i>'
            icon_globe = '<i class="fi fi-rr-globe"></i>'
            icon_shield = '<i class="fi fi-rr-shield-check"></i>'

            # Agent info for display
            agents_info = {
                "Topic Research": {"icon": icon_search, "name": "Research Agent"},
                "Fact Research": {"icon": icon_globe, "name": "Perplexity Sonar"},
                "Initialization": {"icon": icon_gear, "name": "System Init"},
                "Outline": {"icon": icon_clipboard, "name": "Editor Agent"},
                "Refinement": {"icon": icon_pencil, "name": "Critic Agent"},
                "Draft": {"icon": icon_note, "name": "Writer Agent"},
                "Specialists": {"icon": icon_target, "name": "Specialist Team"},
                "Polish": {"icon": icon_sparkle, "name": "Polish Agent"},
                "Verification": {"icon": icon_shield, "name": "Perplexity Verify"},
                "Quality Gate": {"icon": icon_magnify, "name": "Expert Panel"},
                "Visuals": {"icon": icon_palette, "name": "Visuals Agent"},
                "Complete": {"icon": icon_check, "name": "Complete"},
            }

            # Track start time for ETA
            if 'pipeline_start_time' not in st.session_state:
                st.session_state.pipeline_start_time = time.time()

            def update_progress(value):
                # Calculate ETA (average 2.5 minutes total)
                elapsed = time.time() - st.session_state.pipeline_start_time
                if value > 0.05:
                    estimated_total = elapsed / value
                    remaining = max(0, estimated_total - elapsed)
                    eta_text = f"{int(remaining // 60)}:{int(remaining % 60):02d}" if remaining > 60 else f"{int(remaining)}s"
                else:
                    eta_text = "~2:30"

                phases = [
                    (0.10, "Topic Research"),
                    (0.18, "Fact Research"),
                    (0.22, "Initialization"),
                    (0.30, "Outline"),
                    (0.35, "Refinement"),
                    (0.50, "Draft"),
                    (0.70, "Specialists"),
                    (0.75, "Polish"),
                    (0.82, "Verification"),
                    (0.92, "Quality Gate"),
                    (0.98, "Visuals"),
                    (1.00, "Complete"),
                ]

                current_phase = "Starting"
                for threshold, name in phases:
                    if value < threshold:
                        current_phase = name
                        break

                agent_info = agents_info.get(current_phase, {"icon": icon_refresh, "name": current_phase})
                percent = int(value * 100)

                # SVG progress ring
                circumference = 2 * math.pi * 45
                offset = circumference - (value * circumference)

                progress_container.markdown(f'''
                <div class="progress-card glass-card">
                    <div class="progress-ring-container">
                        <svg class="progress-ring" width="120" height="120">
                            <circle class="progress-ring-bg" stroke="var(--color-border)" stroke-width="8" fill="transparent" r="45" cx="60" cy="60"/>
                            <circle class="progress-ring-fill" stroke="var(--color-primary)" stroke-width="8" fill="transparent" r="45" cx="60" cy="60"
                                stroke-dasharray="{circumference}" stroke-dashoffset="{offset}" stroke-linecap="round" transform="rotate(-90 60 60)"/>
                        </svg>
                        <div class="progress-ring-text">
                            <span class="progress-percent">{percent}%</span>
                        </div>
                    </div>
                    <div class="progress-info">
                        <div class="progress-eta">~{eta_text} remaining</div>
                        <div class="progress-agent">
                            <span class="agent-icon">{agent_info["icon"]}</span>
                            <span class="agent-name">{agent_info["name"]}</span>
                        </div>
                    </div>
                </div>
                ''', unsafe_allow_html=True)

                # Rotate tips
                tip_index = int(time.time()) % len(wait_tips)
                icon_lightbulb = '<i class="fi fi-rr-lightbulb" style="margin-right: 6px;"></i>'
                tips_container.markdown(f'''
                <div class="tip-container glass-card">
                    <span class="tip-label">{icon_lightbulb}Did you know?</span>
                    <p class="tip-text">{wait_tips[tip_index]}</p>
                </div>
                ''', unsafe_allow_html=True)

            status_display = st.empty()

            def update_status(message):
                status_display.markdown(
                    f'<div class="status-text">{safe_html(message)}</div>',
                    unsafe_allow_html=True
                )

            try:
                result = run_full_pipeline(update_progress, update_status, provided_topic=st.session_state.get('custom_topic'))

                st.session_state.generation_complete = True
                st.session_state.article_content = result['content']
                st.session_state.image_paths = result['image_paths']
                st.session_state.visual_plan = result['visual_plan']
                st.session_state.filepath = result['filepath']
                st.session_state.selected_topic = result['topic']
                st.session_state.topic_reasoning = result.get('topic_reasoning', '')
                st.session_state.is_running = False

                st.rerun()

            except PipelineError as e:
                st.session_state.is_running = False
                st.error(f"Pipeline failed: {e}")
            except Exception as e:
                st.session_state.is_running = False
                st.error(f"Unexpected error: {str(e)}")
                st.exception(e)

    else:
        # Results display with success celebration

        # Calculate article stats
        article_content = st.session_state.article_content or ""
        word_count = len(article_content.split())
        read_time = max(1, word_count // 200)  # Average reading speed
        visual_count = len(st.session_state.image_paths) if st.session_state.image_paths else 0

        # Success celebration - confetti and banner
        st.markdown(f'''
        <div class="confetti-container">
            <div class="confetti"></div>
            <div class="confetti"></div>
            <div class="confetti"></div>
            <div class="confetti"></div>
            <div class="confetti"></div>
            <div class="confetti"></div>
            <div class="confetti"></div>
            <div class="confetti"></div>
        </div>
        <div class="success-banner">
            <div class="success-icon"><i class="fi fi-rr-check" style="font-size: 32px;"></i></div>
            <h3 class="success-title">Your Article is Ready!</h3>
            <p class="success-subtitle">Generated with 10 specialized AI agents</p>
            <div class="success-stats">
                <div class="stat-item">
                    <span class="stat-value">{word_count:,}</span>
                    <span class="stat-label">words</span>
                </div>
                <div class="stat-item">
                    <span class="stat-value">{read_time}</span>
                    <span class="stat-label">min read</span>
                </div>
                <div class="stat-item">
                    <span class="stat-value">{visual_count}</span>
                    <span class="stat-label">visuals</span>
                </div>
            </div>
        </div>
        ''', unsafe_allow_html=True)

        st.markdown(f'<div class="topic-display">{safe_html(st.session_state.selected_topic)}</div>', unsafe_allow_html=True)
        if st.session_state.topic_reasoning:
            st.markdown(f'<p class="topic-reasoning">{safe_html(st.session_state.topic_reasoning)}</p>', unsafe_allow_html=True)

        st.divider()

        # Action buttons
        col1, col2 = st.columns(2)

        with col1:
            st.download_button(
                label="Download",
                data=st.session_state.article_content,
                file_name="article.md",
                mime="text/markdown",
                use_container_width=True
            )

        with col2:
            copy_to_clipboard(st.session_state.article_content, "Copy Article", "result_copy")

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
