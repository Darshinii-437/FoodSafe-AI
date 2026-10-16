import streamlit as st
import base64
import html
import re
from datetime import datetime
from pathlib import Path
import numpy as np


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="FoodSafe AI - Food Safety Knowledge Assistant",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# PATHS & ASSETS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DOCUMENTS_DIR = BASE_DIR / "documents"
DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
ASSETS_DIR = BASE_DIR / "assets"
ASSETS_DIR.mkdir(parents=True, exist_ok=True)
HERO_IMG_PATH = ASSETS_DIR / "hero_veggies_opt.jpg"
if not HERO_IMG_PATH.exists():
    HERO_IMG_PATH = ASSETS_DIR / "hero_veggies.jpg"


def get_hero_image_base64():
    """Load optimized hero image as base64 string."""
    if HERO_IMG_PATH.exists():
        try:
            with open(HERO_IMG_PATH, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")
        except Exception:
            return ""
    return ""


# ============================================================
# SAFE HTML RENDER HELPER (GUARANTEES ZERO RAW HTML STRINGS)
# ============================================================

def render_html(html_str):
    """
    Safely renders HTML in Streamlit.
    Strips all leading whitespace and line breaks so Streamlit's CommonMark
    parser NEVER treats indented tags as <pre><code> code blocks.
    Guarantees ZERO visible HTML tags in the user UI.
    """
    clean = "".join([line.strip() for line in html_str.splitlines() if line.strip()])
    st.markdown(clean, unsafe_allow_html=True)


# ============================================================
# LOAD RAG SYSTEM (BACKEND UNTOUCHED)
# ============================================================

@st.cache_resource(show_spinner=False)
def load_rag_system():
    import main
    return main


with st.spinner("Initializing FoodSafe AI Knowledge Base..."):
    rag = load_rag_system()


# ============================================================
# HELPER FUNCTIONS (PRESERVED)
# ============================================================

def get_sources(selected_chunks):
    sources = []
    for index in selected_chunks:
        source = rag.clean_source_name(
            rag.all_chunks[index]["document"]
        )
        if source not in sources:
            sources.append(source)
    return sources


def save_uploaded_files(uploaded_files):
    saved = []
    for uploaded_file in uploaded_files:
        filename = Path(uploaded_file.name).name
        if not filename.lower().endswith(".pdf"):
            continue

        destination = DOCUMENTS_DIR / filename
        with open(destination, "wb") as file:
            file.write(uploaded_file.getbuffer())
        saved.append(filename)
    return saved


def format_chat_text(text):
    """Cleanly escape and format response text into HTML paragraphs and lists."""
    if not text:
        return ""
    safe_text = html.escape(text.strip())
    safe_text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", safe_text)

    lines = safe_text.splitlines()
    out_lines = []
    in_list = False

    for line in lines:
        stripped = line.strip()
        if not stripped:
            if in_list:
                out_lines.append("</ul>")
                in_list = False
            continue

        if stripped.startswith("* ") or stripped.startswith("- ") or stripped.startswith("• "):
            if not in_list:
                out_lines.append("<ul style='margin: 6px 0; padding-left: 18px; color: #dbe4ee;'>")
                in_list = True
            bullet_content = stripped[2:].strip()
            out_lines.append(f"<li style='margin-bottom: 4px;'>{bullet_content}</li>")
        else:
            if in_list:
                out_lines.append("</ul>")
                in_list = False
            out_lines.append(f"<p style='margin: 0 0 8px 0;'>{stripped}</p>")

    if in_list:
        out_lines.append("</ul>")

    return "".join(out_lines)


def build_sources_html(sources):
    """Constructs the styled Sources card inside the assistant bubble."""
    if not sources:
        return ""
    items_html = "".join([f"<div class='source-item'>• {html.escape(str(s).strip())}</div>" for s in sources])
    return (
        f"<div class='sources-box-inner'>"
        f"<div class='sources-box-header'>"
        f"<span class='sources-icon'>📁</span>"
        f"<span class='sources-title'>Sources</span>"
        f"</div>"
        f"<div class='sources-box-list'>{items_html}</div>"
        f"</div>"
    )


# ============================================================
# SESSION STATE INITIALIZATION
# ============================================================

if "active_tab" not in st.session_state:
    st.session_state.active_tab = "Chat"

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "user",
            "content": "What is the mercury limit in fish according to FSSAI?",
            "timestamp": "10:24 AM",
        },
        {
            "role": "assistant",
            "content": (
                "According to the FSSAI Contaminants Regulations, the mercury limit for fish is 0.5. "
                "However, the correct unit cannot be established from the same table and column, "
                "so the unit is not specified in the retrieved context."
            ),
            "sources": [
                "FSSAI Contaminants Regulations",
                "FSSAI Licensing and Registration Regulations"
            ],
            "timestamp": "10:24 AM",
        }
    ]

if "previous_question" not in st.session_state:
    st.session_state.previous_question = "What is the mercury limit in fish according to FSSAI?"

if "recent_sources" not in st.session_state:
    st.session_state.recent_sources = [
        {"name": "FSSAI Contaminants Regulations", "score": 0.82},
        {"name": "FSSAI Food Products Standards", "score": 0.76},
        {"name": "FSSAI Licensing and Registration", "score": 0.62},
    ]

if "pending_question" not in st.session_state:
    st.session_state.pending_question = None


# ============================================================
# CUSTOM CSS STYLES (MATCHING REFERENCE EXACTLY)
# ============================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [data-testid="stApp"] {
    background-color: #060c18 !important;
    color: #e2e8f0 !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    overflow-x: hidden;
}

#MainMenu, header, footer, [data-testid="stDecoration"], [data-testid="stToolbar"] {
    display: none !important;
    visibility: hidden !important;
}

.block-container {
    padding-top: 0.8rem !important;
    padding-bottom: 1.5rem !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
    max-width: 98% !important;
}

::-webkit-scrollbar {
    width: 6px;
    height: 6px;
}
::-webkit-scrollbar-track {
    background: #070e1c;
}
::-webkit-scrollbar-thumb {
    background: #1e293b;
    border-radius: 4px;
}
::-webkit-scrollbar-thumb:hover {
    background: #00e5a3;
}

/* TOP NAVBAR */
.top-header-wrapper {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 10px 0 20px 0;
    border-bottom: 1px solid rgba(255, 255, 255, 0.05);
    margin-bottom: 22px;
    position: relative;
}

.brand-section {
    display: flex;
    align-items: center;
    gap: 12px;
}

.shield-logo-box {
    width: 44px;
    height: 48px;
    display: flex;
    align-items: center;
    justify-content: center;
    filter: drop-shadow(0 0 10px rgba(0, 229, 163, 0.35));
}

.brand-text-container {
    display: flex;
    flex-direction: column;
}

.brand-title {
    font-size: 24px;
    font-weight: 700;
    color: #ffffff;
    letter-spacing: -0.4px;
    line-height: 1.1;
}

.brand-title span {
    color: #00e5a3;
}

.brand-subtitle {
    font-size: 11.5px;
    color: #7b91a7;
    margin-top: 3px;
    letter-spacing: 0.1px;
}

.header-tagline {
    color: #64748b;
    font-size: 13px;
    font-weight: 500;
    letter-spacing: 0.6px;
    display: flex;
    align-items: center;
    gap: 12px;
}

.tagline-dot {
    color: #334155;
    font-size: 11px;
}

.header-right-meta {
    display: flex;
    align-items: center;
    gap: 16px;
}

.status-badge {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 13px;
    color: #94a3b8;
    font-weight: 500;
}

.pulse-indicator {
    width: 9px;
    height: 9px;
    background-color: #00e5a3;
    border-radius: 50%;
    box-shadow: 0 0 10px #00e5a3;
    animation: pulseGlow 2s infinite ease-in-out;
}

@keyframes pulseGlow {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.55; transform: scale(1.15); }
}

.settings-btn-icon {
    color: #64748b;
    font-size: 18px;
    cursor: pointer;
}

.header-botanical-art {
    position: absolute;
    right: 220px;
    top: -5px;
    opacity: 0.22;
    pointer-events: none;
}

/* SIDEBAR / CARDS */
.section-label {
    font-size: 11.5px;
    font-weight: 600;
    text-transform: uppercase;
    color: #64748b;
    letter-spacing: 0.6px;
    margin-bottom: 10px;
    margin-top: 14px;
}

.card-container {
    background: #091322;
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-radius: 16px;
    padding: 16px;
    margin-bottom: 18px;
}

.stat-item {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 7px 0;
    border-bottom: 1px solid rgba(255, 255, 255, 0.025);
    font-size: 13px;
}

.stat-item:last-child {
    border-bottom: none;
}

.stat-left {
    display: flex;
    align-items: center;
    gap: 10px;
    color: #8da2ba;
}

.stat-right {
    font-weight: 600;
    color: #ffffff;
}

/* Add PDF Card */
.upload-card-wrapper {
    background: #0a1526;
    border: 1px solid rgba(0, 229, 163, 0.22);
    border-radius: 16px;
    padding: 20px 16px;
    text-align: center;
    margin-bottom: 22px;
    box-shadow: 0 4px 20px rgba(0, 229, 163, 0.04);
}

.upload-icon-cloud {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 44px;
    height: 44px;
    margin-bottom: 10px;
}

.upload-card-title {
    font-size: 14.5px;
    font-weight: 700;
    color: #ffffff;
    margin-bottom: 4px;
}

.upload-card-desc {
    font-size: 11.5px;
    color: #7b91a7;
    line-height: 1.4;
    margin-bottom: 14px;
}

.upload-card-footer {
    font-size: 11px;
    color: #64748b;
    margin-top: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 6px;
}

.sidebar-quote {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 8px 4px;
    color: #10b981;
    font-size: 12.5px;
    font-style: italic;
    opacity: 0.85;
}

/* File Uploader styling */
[data-testid="stFileUploader"] {
    padding: 0 !important;
}

[data-testid="stFileUploaderDropzone"] {
    background: rgba(15, 23, 42, 0.4) !important;
    border: 1px dashed rgba(0, 229, 163, 0.3) !important;
    border-radius: 12px !important;
    padding: 10px !important;
}

[data-testid="stFileUploaderDropzone"] button {
    background: #00e5a3 !important;
    color: #06101e !important;
    font-weight: 600 !important;
    border-radius: 9999px !important;
    border: none !important;
    padding: 7px 22px !important;
    font-size: 13px !important;
    box-shadow: 0 2px 10px rgba(0, 229, 163, 0.3) !important;
}

/* HERO BANNER */
.hero-card {
    background: linear-gradient(135deg, #072a24 0%, #0d3d35 42%, #08212e 100%);
    border: 1px solid rgba(0, 229, 163, 0.3);
    border-radius: 20px;
    padding: 22px 24px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 24px;
    overflow: hidden;
    position: relative;
    box-shadow: 0 6px 24px rgba(0, 229, 163, 0.07);
}

.hero-left-box {
    display: flex;
    align-items: center;
    gap: 16px;
    max-width: 60%;
    z-index: 2;
}

.hero-plant-icon {
    width: 46px;
    height: 46px;
    min-width: 46px;
    background: #00e5a3;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    box-shadow: 0 0 16px rgba(0, 229, 163, 0.4);
}

.hero-title {
    font-size: 21px;
    font-weight: 700;
    color: #ffffff;
    margin-bottom: 4px;
}

.hero-desc {
    font-size: 13px;
    color: #bbf7d0;
    line-height: 1.45;
    opacity: 0.9;
}

.hero-right-img {
    width: 230px;
    height: 110px;
    object-fit: cover;
    border-radius: 16px;
    box-shadow: 0 4px 18px rgba(0, 0, 0, 0.4);
    border: 1px solid rgba(0, 229, 163, 0.25);
    z-index: 2;
}

/* ============================================================
   PIXEL-PERFECT CHAT BUBBLE STYLING
   ============================================================ */
.chat-scroll-area {
    display: flex;
    flex-direction: column;
    gap: 18px;
    margin-bottom: 20px;
    width: 100%;
}

/* User Message Row */
.chat-message-row.user-side {
    display: flex !important;
    justify-content: flex-end !important;
    align-items: flex-start !important;
    gap: 12px !important;
    width: 100% !important;
}

.avatar-circle.user-avatar {
    width: 36px !important;
    height: 36px !important;
    min-width: 36px !important;
    min-height: 36px !important;
    flex-shrink: 0 !important;
    background: #1e3a8a !important;
    border-radius: 50% !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    color: #ffffff !important;
    box-shadow: 0 0 10px rgba(30, 58, 138, 0.4) !important;
    order: 2 !important;
}

.bubble.user-bubble {
    background: #101c33 !important;
    border: 1px solid rgba(59, 130, 246, 0.28) !important;
    border-radius: 18px 4px 18px 18px !important;
    padding: 14px 18px !important;
    max-width: 78% !important;
    color: #ffffff !important;
    font-size: 13.5px !important;
    line-height: 1.55 !important;
    order: 1 !important;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.25) !important;
    word-break: break-word !important;
}

.meta-time.user-meta {
    font-size: 11px !important;
    color: #64748b !important;
    display: flex !important;
    align-items: center !important;
    justify-content: flex-end !important;
    gap: 5px !important;
    margin-top: 6px !important;
}

.check-teal {
    color: #00e5a3 !important;
    font-weight: 700 !important;
}

/* Assistant Message Row */
.chat-message-row.assistant-side {
    display: flex !important;
    justify-content: flex-start !important;
    align-items: flex-start !important;
    gap: 12px !important;
    width: 100% !important;
}

.avatar-circle.assistant-avatar {
    width: 36px !important;
    height: 36px !important;
    min-width: 36px !important;
    min-height: 36px !important;
    flex-shrink: 0 !important;
    background: #059669 !important;
    border-radius: 50% !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    color: #ffffff !important;
    box-shadow: 0 0 12px rgba(5, 150, 105, 0.4) !important;
}

.bubble.assistant-bubble {
    background: #0b1526 !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 4px 18px 18px 18px !important;
    padding: 18px 22px !important;
    max-width: 82% !important;
    color: #dbe4ee !important;
    font-size: 13.5px !important;
    line-height: 1.65 !important;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3) !important;
    word-break: break-word !important;
}

/* STYLED SOURCES CARD INSIDE ASSISTANT BUBBLE */
.sources-box-inner {
    background: rgba(0, 229, 163, 0.08) !important;
    border: 1px solid rgba(0, 229, 163, 0.28) !important;
    border-radius: 10px !important;
    padding: 12px 16px !important;
    margin-top: 14px !important;
    margin-bottom: 6px !important;
    box-shadow: 0 2px 12px rgba(0, 229, 163, 0.04) !important;
}

.sources-box-header {
    display: flex !important;
    align-items: center !important;
    gap: 8px !important;
    color: #00e5a3 !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    margin-bottom: 8px !important;
}

.sources-box-header .sources-icon {
    font-size: 14px !important;
}

.sources-box-header .sources-title {
    color: #00e5a3 !important;
    font-weight: 700 !important;
    letter-spacing: 0.2px !important;
}

.sources-box-list {
    display: flex !important;
    flex-direction: column !important;
    gap: 5px !important;
}

.source-item {
    color: #94a3b8 !important;
    font-size: 12.5px !important;
    line-height: 1.4 !important;
}

.meta-time.assistant-meta {
    font-size: 11px !important;
    color: #64748b !important;
    display: flex !important;
    align-items: center !important;
    justify-content: flex-end !important;
    margin-top: 8px !important;
}

/* FOLLOW-UP SECTION */
.followup-box-wrapper {
    margin: 14px 0 10px 48px !important;
}

.followup-title {
    display: flex !important;
    align-items: center !important;
    gap: 6px !important;
    color: #fbbf24 !important;
    font-size: 12.5px !important;
    font-weight: 600 !important;
    margin-bottom: 10px !important;
}

/* Follow-up Question Buttons */
div[data-testid="stHorizontalBlock"] button {
    border-radius: 9999px !important;
    background: rgba(13, 27, 46, 0.9) !important;
    color: #38bdf8 !important;
    border: 1px solid rgba(56, 189, 248, 0.35) !important;
    font-size: 12px !important;
    font-weight: 500 !important;
    padding: 6px 14px !important;
    transition: all 0.2s ease !important;
}

div[data-testid="stHorizontalBlock"] button:hover {
    background: rgba(0, 229, 163, 0.15) !important;
    border-color: #00e5a3 !important;
    color: #00e5a3 !important;
    box-shadow: 0 0 12px rgba(0, 229, 163, 0.25) !important;
}

/* SIDEBAR NAVIGATION BUTTONS */
.nav-buttons-block button {
    display: flex !important;
    justify-content: flex-start !important;
    align-items: center !important;
    text-align: left !important;
    padding: 10px 14px !important;
    border-radius: 10px !important;
    font-size: 13.5px !important;
    font-weight: 500 !important;
    border: 1px solid transparent !important;
    background: transparent !important;
    color: #8da2ba !important;
    margin-bottom: 4px !important;
    transition: all 0.2s ease !important;
}

.nav-buttons-block button:hover {
    background: rgba(255, 255, 255, 0.04) !important;
    color: #ffffff !important;
    border-color: rgba(255, 255, 255, 0.08) !important;
}

.nav-buttons-block button[kind="primary"] {
    background: rgba(0, 229, 163, 0.12) !important;
    border: 1px solid rgba(0, 229, 163, 0.45) !important;
    color: #00e5a3 !important;
    font-weight: 600 !important;
    box-shadow: 0 0 15px rgba(0, 229, 163, 0.1) !important;
}

/* CHAT INPUT */
[data-testid="stChatInput"] {
    background: #0a1424 !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 30px !important;
    padding: 4px 10px !important;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4) !important;
}

[data-testid="stChatInput"]:focus-within {
    border-color: rgba(0, 229, 163, 0.6) !important;
    box-shadow: 0 0 18px rgba(0, 229, 163, 0.25) !important;
}

[data-testid="stChatInput"] textarea {
    color: #ffffff !important;
    background: transparent !important;
    font-size: 13.5px !important;
}

[data-testid="stChatInput"] textarea::placeholder {
    color: #64748b !important;
}

[data-testid="stChatInput"] button {
    background: #00e5a3 !important;
    color: #06101e !important;
    border-radius: 50% !important;
    width: 36px !important;
    height: 36px !important;
    border: none !important;
    box-shadow: 0 2px 10px rgba(0, 229, 163, 0.4) !important;
}

/* RIGHT PANEL */
.right-panel-card {
    background: #091322;
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-radius: 16px;
    padding: 16px 18px;
    margin-bottom: 18px;
}

.panel-card-header {
    display: flex;
    align-items: center;
    gap: 9px;
    margin-bottom: 14px;
    font-size: 14.5px;
    font-weight: 700;
    color: #ffffff;
}

.panel-header-icon {
    font-size: 17px;
    color: #00e5a3;
}

.kb-file-list {
    display: flex;
    flex-direction: column;
    gap: 8px;
}

.kb-file-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 7px 10px;
    border-radius: 8px;
    background: rgba(255, 255, 255, 0.02);
    border: 1px solid rgba(255, 255, 255, 0.025);
    font-size: 12px;
    color: #cbd5e1;
}

.kb-file-left {
    display: flex;
    align-items: center;
    gap: 8px;
    overflow: hidden;
    white-space: nowrap;
    text-overflow: ellipsis;
}

.pdf-tag-icon {
    background: #ef4444;
    color: #ffffff;
    font-size: 9px;
    font-weight: 700;
    padding: 2px 5px;
    border-radius: 4px;
    letter-spacing: 0.3px;
}

.kb-chevron {
    color: #64748b;
    font-size: 13px;
}

.recent-source-row {
    padding: 8px 0;
    border-bottom: 1px solid rgba(255, 255, 255, 0.03);
}

.recent-source-row:last-child {
    border-bottom: none;
}

.rs-top {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 12.5px;
    color: #cbd5e1;
    font-weight: 500;
}

.rs-dot {
    width: 6.5px;
    height: 6.5px;
    background: #00e5a3;
    border-radius: 50%;
    box-shadow: 0 0 6px #00e5a3;
}

.rs-score {
    font-size: 11px;
    color: #64748b;
    margin-left: 15px;
    margin-top: 2px;
}

.side-footer-quote {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 11.5px;
    color: #64748b;
    font-style: italic;
    border-top: 1px solid rgba(255, 255, 255, 0.03);
    padding-top: 12px;
    margin-top: 12px;
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# TOP NAVBAR
# ============================================================

hero_img_b64 = get_hero_image_base64()

top_nav_html = """
<div class="top-header-wrapper">
    <div class="brand-section">
        <div class="shield-logo-box">
            <svg width="38" height="42" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                <path d="M12 2L3 6V11.5C3 17.5 6.8 21.7 12 23C17.2 21.7 21 17.5 21 11.5V6L12 2Z" fill="rgba(0, 229, 163, 0.16)" stroke="#00e5a3" stroke-width="2" stroke-linejoin="round"/>
                <path d="M12 7C9.8 8.5 9 11.2 9 13.8C9 15.6 10.3 17 12 17C13.7 17 15 15.6 15 13.8C15 10.5 12 7 12 7Z" fill="#00e5a3"/>
                <path d="M12 10.5V15.5" stroke="#061c16" stroke-width="1.5" stroke-linecap="round"/>
            </svg>
        </div>
        <div class="brand-text-container">
            <div class="brand-title">FoodSafe <span>AI</span></div>
            <div class="brand-subtitle">Food Safety Knowledge Assistant</div>
        </div>
    </div>
    
    <div class="header-tagline">
        <span>Safe Food</span>
        <span class="tagline-dot">•</span>
        <span>Healthy People</span>
        <span class="tagline-dot">•</span>
        <span>A Better Tomorrow</span>
    </div>

    <div class="header-botanical-art">
        <svg width="120" height="60" viewBox="0 0 100 50" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M20 40C25 20 45 10 70 12C60 25 45 35 20 40Z" fill="#00e5a3" fill-opacity="0.35"/>
            <path d="M40 45C50 30 70 20 95 20C85 35 70 42 40 45Z" fill="#10b981" fill-opacity="0.25"/>
        </svg>
    </div>

    <div class="header-right-meta">
        <div class="status-badge">
            <div class="pulse-indicator"></div>
            <span>Knowledge Base Online</span>
        </div>
        <div class="settings-btn-icon" title="Settings">⚙</div>
    </div>
</div>
"""
render_html(top_nav_html)


# ============================================================
# THREE-COLUMN DASHBOARD LAYOUT
# ============================================================

col_left, col_center, col_right = st.columns([1.15, 2.6, 1.25], gap="medium")


# ============================================================
# LEFT COLUMN: INTERACTIVE NAVIGATION & STATS
# ============================================================

with col_left:
    st.markdown('<div class="nav-buttons-block">', unsafe_allow_html=True)
    
    current_tab = st.session_state.active_tab

    if st.button(
        "💬   Chat",
        key="btn_tab_chat",
        use_container_width=True,
        type="primary" if current_tab == "Chat" else "secondary"
    ):
        st.session_state.active_tab = "Chat"
        st.rerun()

    if st.button(
        "📦   Knowledge Base",
        key="btn_tab_kb",
        use_container_width=True,
        type="primary" if current_tab == "Knowledge Base" else "secondary"
    ):
        st.session_state.active_tab = "Knowledge Base"
        st.rerun()

    if st.button(
        "📤   Upload Documents",
        key="btn_tab_upload",
        use_container_width=True,
        type="primary" if current_tab == "Upload Documents" else "secondary"
    ):
        st.session_state.active_tab = "Upload Documents"
        st.rerun()

    if st.button(
        "🖥️   System Info",
        key="btn_tab_sys",
        use_container_width=True,
        type="primary" if current_tab == "System Info" else "secondary"
    ):
        st.session_state.active_tab = "System Info"
        st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

    # Quick Stats Card
    quick_stats_html = f"""
    <div class="section-label">Quick Stats</div>
    <div class="card-container">
        <div class="stat-item">
            <div class="stat-left">
                <span class="stat-icon">📄</span>
                <span>Documents</span>
            </div>
            <div class="stat-right">{len(rag.documents)}</div>
        </div>
        <div class="stat-item">
            <div class="stat-left">
                <span class="stat-icon">📦</span>
                <span>Chunks</span>
            </div>
            <div class="stat-right">{len(rag.all_chunks):,}</div>
        </div>
        <div class="stat-item">
            <div class="stat-left">
                <span class="stat-icon">🔍</span>
                <span>Retrieval</span>
            </div>
            <div class="stat-right">Hybrid Search</div>
        </div>
        <div class="stat-item">
            <div class="stat-left">
                <span class="stat-icon">👥</span>
                <span>Reranker</span>
            </div>
            <div class="stat-right">Cross-Encoder</div>
        </div>
        <div class="stat-item">
            <div class="stat-left">
                <span class="stat-icon">🎯</span>
                <span>Final Context</span>
            </div>
            <div class="stat-right">Top {rag.FINAL_TOP_K}</div>
        </div>
        <div class="stat-item">
            <div class="stat-left">
                <span class="stat-icon">🛡️</span>
                <span>Threshold</span>
            </div>
            <div class="stat-right">{rag.RELEVANCE_THRESHOLD}</div>
        </div>
    </div>
    """
    render_html(quick_stats_html)

    # Sidebar Quote
    sidebar_quote_html = """
    <div class="sidebar-quote">
        <span>🌱</span>
        <span>Food Safety is Everyone's Business</span>
    </div>
    """
    render_html(sidebar_quote_html)


# ============================================================
# CENTER COLUMN: TAB-BASED VIEW
# ============================================================

with col_center:
    # --------------------------------------------------------
    # TAB 1: MAIN CHAT VIEW
    # --------------------------------------------------------
    if st.session_state.active_tab == "Chat":
        hero_img_markup = f'<img src="data:image/jpeg;base64,{hero_img_b64}" class="hero-right-img" alt="Fresh Food"/>' if hero_img_b64 else ''
        hero_banner_html = f"""
        <div class="hero-card">
            <div class="hero-left-box">
                <div class="hero-plant-icon">
                    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#042017" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
                        <path d="M12 2a10 10 0 0 1 10 10c0 5.523-4.477 10-10 10S2 17.523 2 12A10 10 0 0 1 12 2z" fill="#00e5a3"></path>
                        <path d="M12 7c-2 2-3 4-3 6a3 3 0 0 0 6 0c0-2-1-4-3-6z" fill="#042017"></path>
                    </svg>
                </div>
                <div>
                    <div class="hero-title">Hello! 👋</div>
                    <div class="hero-desc">Ask me anything about food safety, regulations, contaminants, standards and more.</div>
                </div>
            </div>
            {hero_img_markup}
        </div>
        """
        render_html(hero_banner_html)

        # Chat Messages Rendering
        render_html('<div class="chat-scroll-area">')

        for msg in st.session_state.messages:
            role = msg["role"]
            content = msg["content"]
            timestamp = msg.get("timestamp", datetime.now().strftime("%I:%M %p"))

            if role == "user":
                user_msg_html = f"""
                <div class="chat-message-row user-side">
                    <div class="avatar-circle user-avatar">
                        <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor">
                            <path d="M12 12c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm0 2c-2.67 0-8 1.34-8 4v2h16v-2c0-2.66-5.33-4-8-4z"/>
                        </svg>
                    </div>
                    <div class="bubble user-bubble">
                        <div>{html.escape(content)}</div>
                        <div class="meta-time user-meta">{timestamp} <span class="check-teal">✓✓</span></div>
                    </div>
                </div>
                """
                render_html(user_msg_html)
            else:
                formatted_content = format_chat_text(content)
                sources_html = build_sources_html(msg.get("sources", []))

                assistant_msg_html = f"""
                <div class="chat-message-row assistant-side">
                    <div class="avatar-circle assistant-avatar">
                        <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
                            <path d="M12 2a2 2 0 0 1 2 2v1h2a3 3 0 0 1 3 3v9a3 3 0 0 1-3 3H8a3 3 0 0 1-3-3V8a3 3 0 0 1 3-3h2V4a2 2 0 0 1 2-2zm-3 9a1.5 1.5 0 1 0 0-3 1.5 1.5 0 0 0 0 3zm6 0a1.5 1.5 0 1 0 0-3 1.5 1.5 0 0 0 0 3zm-6 4a1 1 0 0 0 0 2h6a1 1 0 1 0 0-2H9z"/>
                        </svg>
                    </div>
                    <div class="bubble assistant-bubble">
                        {formatted_content}
                        {sources_html}
                        <div class="meta-time assistant-meta">{timestamp}</div>
                    </div>
                </div>
                """
                render_html(assistant_msg_html)

        render_html('</div>')

        # Follow-up Section
        followup_header_html = """
        <div class="followup-box-wrapper">
            <div class="followup-title">
                <span>✨</span>
                <span>Follow-up Questions</span>
            </div>
        </div>
        """
        render_html(followup_header_html)

        fcol1, fcol2, fcol3 = st.columns(3)
        with fcol1:
            if st.button("What about cadmium in turmeric?", key="btn_fu1", use_container_width=True):
                st.session_state.pending_question = "What about cadmium in turmeric?"
                st.rerun()
        with fcol2:
            if st.button("What is the aflatoxin M1 limit in milk?", key="btn_fu2", use_container_width=True):
                st.session_state.pending_question = "What is the aflatoxin M1 limit in milk?"
                st.rerun()
        with fcol3:
            if st.button("Show me the source document", key="btn_fu3", use_container_width=True):
                st.session_state.pending_question = "Show me the source document"
                st.rerun()

        # Chat Input Box
        chat_query = st.chat_input("Ask a food safety question...")

    # --------------------------------------------------------
    # TAB 2: UPLOAD DOCUMENTS VIEW (MATCHING IMAGE 2)
    # --------------------------------------------------------
    elif st.session_state.active_tab == "Upload Documents":
        st.markdown("""
        <div class="card-container" style="padding: 24px; border: 1px solid rgba(0, 229, 163, 0.25);">
            <div style="font-size: 20px; font-weight: 700; color: #ffffff; margin-bottom: 6px;">
                📤 Upload Food-Safety Documents
            </div>
            <div style="font-size: 13px; color: #94a3b8; line-height: 1.5; margin-bottom: 20px;">
                Upload official regulatory PDF documents (e.g., FSSAI, Codex Alimentarius, WHO) to expand the assistant's knowledge base.
            </div>
        """, unsafe_allow_html=True)

        upload_card_header_html = """
        <div class="upload-card-wrapper" style="border: 1px dashed rgba(0, 229, 163, 0.4); padding: 32px 20px;">
            <div class="upload-icon-cloud">
                <svg width="42" height="42" viewBox="0 0 24 24" fill="none" stroke="#00e5a3" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
                    <polyline points="17 8 12 3 7 8"></polyline>
                    <line x1="12" y1="3" x2="12" y2="15"></line>
                </svg>
            </div>
            <div class="upload-card-title" style="font-size: 17px; margin-bottom: 6px;">Add New PDF</div>
            <div class="upload-card-desc" style="font-size: 13px; margin-bottom: 18px;">Upload food-safety PDF documents to expand the knowledge base.</div>
        """
        render_html(upload_card_header_html)

        uploaded_files = st.file_uploader(
            "Upload food-safety PDF documents",
            type=["pdf"],
            accept_multiple_files=True,
            label_visibility="collapsed",
            help="Upload a PDF to add it to the food-safety knowledge base."
        )

        upload_card_footer_html = """
            <div class="upload-card-footer" style="font-size: 12px; margin-top: 10px;">
                <span>200MB per file</span>
                <span>•</span>
                <span>PDF only</span>
            </div>
        </div>
        """
        render_html(upload_card_footer_html)

        if uploaded_files:
            st.info(f"Selected {len(uploaded_files)} file(s). Click below to process and rebuild the vector database.")
            if st.button("🔄 Update & Rebuild Knowledge Base", use_container_width=True, type="primary"):
                saved_files = save_uploaded_files(uploaded_files)
                if saved_files:
                    load_rag_system.clear()
                    st.session_state.messages = []
                    st.session_state.previous_question = None
                    st.success(f"Successfully added {len(saved_files)} document(s)! Knowledge base rebuilt.")
                    st.session_state.active_tab = "Chat"
                    st.rerun()
                else:
                    st.error("Please upload valid PDF files.")

        st.markdown("</div>", unsafe_allow_html=True)

    # --------------------------------------------------------
    # TAB 3: KNOWLEDGE BASE EXPLORER
    # --------------------------------------------------------
    elif st.session_state.active_tab == "Knowledge Base":
        st.markdown(f"""
        <div class="card-container" style="padding: 24px;">
            <div style="font-size: 20px; font-weight: 700; color: #ffffff; margin-bottom: 6px;">
                📦 Knowledge Base Documents ({len(rag.documents)})
            </div>
            <div style="font-size: 13px; color: #94a3b8; line-height: 1.5; margin-bottom: 18px;">
                All indexed food-safety regulatory documents loaded in the vector database and ready for hybrid search.
            </div>
        """, unsafe_allow_html=True)

        for filename in sorted(rag.documents.keys()):
            clean_name = rag.clean_source_name(filename)
            doc_chunks = sum(1 for c in rag.all_chunks if c["document"] == filename)
            st.markdown(f"""
            <div class="kb-file-row" style="padding: 12px 14px; margin-bottom: 8px;">
                <div class="kb-file-left">
                    <span class="pdf-tag-icon" style="font-size: 11px; padding: 3px 6px;">PDF</span>
                    <div>
                        <div style="color: #ffffff; font-weight: 600; font-size: 13.5px;">{html.escape(clean_name)}</div>
                        <div style="color: #64748b; font-size: 11.5px;">File: {html.escape(filename)} • Indexed Chunks: {doc_chunks:,}</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    # --------------------------------------------------------
    # TAB 4: SYSTEM INFO
    # --------------------------------------------------------
    elif st.session_state.active_tab == "System Info":
        st.markdown(f"""
        <div class="card-container" style="padding: 24px;">
            <div style="font-size: 20px; font-weight: 700; color: #ffffff; margin-bottom: 6px;">
                🖥️ System Architecture & Specifications
            </div>
            <div style="font-size: 13px; color: #94a3b8; line-height: 1.5; margin-bottom: 20px;">
                Technical configuration of the FoodSafe AI Retrieval-Augmented Generation (RAG) engine.
            </div>

            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 14px;">
                <div class="card-container" style="margin: 0; background: rgba(255,255,255,0.02);">
                    <div style="color: #00e5a3; font-weight: 600; font-size: 13px; margin-bottom: 6px;">Dense Embedding Model</div>
                    <div style="color: #ffffff; font-size: 14px; font-weight: 600;">{rag.EMBEDDING_MODEL_NAME}</div>
                    <div style="color: #64748b; font-size: 11.5px; margin-top: 3px;">384-dimensional cosine similarity vectors</div>
                </div>

                <div class="card-container" style="margin: 0; background: rgba(255,255,255,0.02);">
                    <div style="color: #00e5a3; font-weight: 600; font-size: 13px; margin-bottom: 6px;">Cross-Encoder Reranker</div>
                    <div style="color: #ffffff; font-size: 14px; font-weight: 600;">{rag.RERANKER_MODEL_NAME}</div>
                    <div style="color: #64748b; font-size: 11.5px; margin-top: 3px;">High-precision transformer neural reranker</div>
                </div>

                <div class="card-container" style="margin: 0; background: rgba(255,255,255,0.02);">
                    <div style="color: #00e5a3; font-weight: 600; font-size: 13px; margin-bottom: 6px;">Generative LLM Model</div>
                    <div style="color: #ffffff; font-size: 14px; font-weight: 600;">{rag.GEMINI_MODEL_NAME}</div>
                    <div style="color: #64748b; font-size: 11.5px; margin-top: 3px;">Google Gemini API with grounded prompting</div>
                </div>

                <div class="card-container" style="margin: 0; background: rgba(255,255,255,0.02);">
                    <div style="color: #00e5a3; font-weight: 600; font-size: 13px; margin-bottom: 6px;">Hybrid Search Parameters</div>
                    <div style="color: #ffffff; font-size: 14px; font-weight: 600;">Top-K: {rag.FINAL_TOP_K} • Threshold: {rag.RELEVANCE_THRESHOLD}</div>
                    <div style="color: #64748b; font-size: 11.5px; margin-top: 3px;">Candidates: {rag.HYBRID_CANDIDATES} • BM25 + Semantic</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ============================================================
# RIGHT COLUMN: KNOWLEDGE BASE & RECENT SOURCES (NO RAG PIPELINE)
# ============================================================

with col_right:
    # 1. Knowledge Base List
    kb_header_html = """
    <div class="right-panel-card">
        <div class="panel-card-header">
            <span class="panel-header-icon">📖</span>
            <span>Knowledge Base</span>
        </div>
        <div class="kb-file-list">
    """
    kb_rows = []
    for filename in sorted(rag.documents.keys()):
        clean_name = rag.clean_source_name(filename)
        display_name = clean_name if clean_name.endswith(".pdf") else f"{clean_name}.pdf"
        kb_rows.append(
            f'<div class="kb-file-row">'
            f'<div class="kb-file-left">'
            f'<span class="pdf-tag-icon">PDF</span>'
            f'<span title="{html.escape(display_name)}">{html.escape(display_name)}</span>'
            f'</div>'
            f'<span class="kb-chevron">›</span>'
            f'</div>'
        )
    kb_footer_html = """
        </div>
    </div>
    """
    render_html(kb_header_html + "".join(kb_rows) + kb_footer_html)

    # 2. Recent Sources (with Relevance Scores)
    rs_header_html = """
    <div class="right-panel-card">
        <div class="panel-card-header">
            <span class="panel-header-icon">📑</span>
            <span>Recent Sources</span>
        </div>
    """
    rs_rows = []
    for rs in st.session_state.recent_sources:
        rs_rows.append(
            f'<div class="recent-source-row">'
            f'<div class="rs-top">'
            f'<div class="rs-dot"></div>'
            f'<div>{html.escape(rs["name"])}</div>'
            f'</div>'
            f'<div class="rs-score">Relevance: {rs["score"]:.2f}</div>'
            f'</div>'
        )
    rs_footer_html = """
        <div class="side-footer-quote">
            <span>🌱</span>
            <span>"Better information. Safer food."</span>
        </div>
    </div>
    """
    render_html(rs_header_html + "".join(rs_rows) + rs_footer_html)


# ============================================================
# QUERY PROCESSING (FULL RAG LOGIC FROM MAIN.PY PRESERVED)
# ============================================================

active_question = None

if st.session_state.pending_question:
    active_question = st.session_state.pending_question
    st.session_state.pending_question = None
elif "chat_query" in locals() and chat_query and chat_query.strip():
    active_question = chat_query.strip()


if active_question:
    current_time_str = datetime.now().strftime("%I:%M %p")

    # 1. Add User Message
    st.session_state.messages.append({
        "role": "user",
        "content": active_question,
        "timestamp": current_time_str
    })

    # 2. Follow-Up Query Expansion
    search_query = rag.build_search_query(
        active_question,
        st.session_state.previous_question
    )

    # 3. Retrieval via Hybrid Search & Cross-Encoder
    with st.spinner("Searching knowledge base..."):
        selected_chunks, reranked, best_score = rag.retrieve_context(search_query)

    if selected_chunks is None:
        answer = (
            "I couldn't find sufficient information to answer "
            "this question in the available food-safety documents."
        )
        sources = []
    else:
        # 4. Generate Grounded Answer via Gemini
        with st.spinner("Generating grounded answer..."):
            answer, error_type = rag.generate_answer(
                active_question,
                selected_chunks
            )
        sources = get_sources(selected_chunks)

        # 5. Update Recent Sources dynamically with real scores
        new_recent = []
        seen_names = set()
        for item in reranked:
            doc_raw = rag.all_chunks[item["index"]]["document"]
            doc_clean = rag.clean_source_name(doc_raw)
            if doc_clean not in seen_names:
                seen_names.add(doc_clean)
                raw_score = float(item["score"])
                if -15 < raw_score < 15:
                    normalized = 1.0 / (1.0 + np.exp(-raw_score))
                else:
                    normalized = max(0.0, min(1.0, raw_score))
                new_recent.append({"name": doc_clean, "score": normalized})
            if len(new_recent) >= 3:
                break
        if new_recent:
            st.session_state.recent_sources = new_recent

    # 6. Save Assistant Response
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
        "timestamp": datetime.now().strftime("%I:%M %p")
    })

    st.session_state.previous_question = active_question
    st.rerun()
