import re
import sys
import tempfile
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.crop_predictor import predict_top3
from src.predict_disease import predict_disease
from src.smart_advisor import generate_advisory

st.set_page_config(
    page_title="SmartCrop",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;1,9..40,400&family=DM+Serif+Display&display=swap');

/* ── Reset & Base ───────────────────────────────────────────── */
*, *::before, *::after { box-sizing: border-box; }

html, body, [class*="css"], .stApp, .stMarkdown, p, li, span, div {
    font-family: 'DM Sans', system-ui, sans-serif !important;
}

.stApp { background: #f0ede6 !important; }
section[data-testid="stMain"] { background: #f0ede6 !important; }

.main .block-container {
    padding: 1.8rem 2.2rem 3.5rem !important;
    max-width: 1100px !important;
}

/* ── Sidebar ────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: #0f1e11 !important;
    border-right: 1px solid rgba(255,255,255,0.06) !important;
}
[data-testid="stSidebar"] > div:first-child {
    background: #0f1e11 !important;
    padding: 0 !important;
}

.sidebar-logo {
    padding: 1.8rem 1.4rem 1.2rem;
    border-bottom: 1px solid rgba(255,255,255,0.07);
    margin-bottom: 0.5rem;
}
.sidebar-logo-badge {
    display: inline-block;
    background: rgba(74,158,76,0.18);
    border: 1px solid rgba(74,158,76,0.3);
    border-radius: 6px;
    padding: 0.18rem 0.55rem;
    font-size: 0.65rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    color: #6abf6c !important;
    margin-bottom: 0.7rem;
}
.sidebar-logo-title {
    font-family: 'DM Serif Display', Georgia, serif !important;
    font-size: 1.6rem !important;
    color: #e8f0e8 !important;
    margin: 0 !important;
    letter-spacing: -0.02em !important;
    line-height: 1.15 !important;
}
.sidebar-logo-sub {
    font-size: 0.73rem !important;
    color: rgba(160,200,162,0.5) !important;
    margin: 0.35rem 0 0 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
}

/* Sidebar nav buttons */
[data-testid="stSidebar"] .stButton button {
    background: transparent !important;
    border: none !important;
    border-radius: 10px !important;
    color: rgba(180,215,182,0.75) !important;
    font-size: 0.87rem !important;
    font-weight: 500 !important;
    text-align: left !important;
    padding: 0.65rem 1.1rem !important;
    width: 100% !important;
    box-shadow: none !important;
    transform: none !important;
    transition: background 0.18s ease, color 0.18s ease !important;
}
[data-testid="stSidebar"] .stButton button:hover {
    background: rgba(74,158,76,0.14) !important;
    color: #d0f0d0 !important;
    transform: none !important;
    box-shadow: none !important;
    border: none !important;
}

.sidebar-divider {
    border: none;
    border-top: 1px solid rgba(255,255,255,0.06);
    margin: 0.8rem 1.2rem;
}
.sidebar-section-label {
    font-size: 0.67rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    color: rgba(140,185,142,0.38) !important;
    padding: 0.7rem 1.4rem 0.25rem !important;
}
.sbar-metric-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.45rem 1.4rem;
    border-bottom: 1px solid rgba(255,255,255,0.035);
}
.sbar-metric-name { font-size: 0.8rem; color: rgba(175,210,178,0.58); }
.sbar-metric-val  { font-size: 0.82rem; font-weight: 700; color: #7ddd7d; }

/* ── Animations ─────────────────────────────────────────────── */
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(16px); }
    to   { opacity: 1; transform: translateY(0); }
}
.fade-up   { animation: fadeUp 0.38s cubic-bezier(0.22,0.61,0.36,1) both; }
.fade-up-1 { animation-delay: 0.04s; }
.fade-up-2 { animation-delay: 0.10s; }
.fade-up-3 { animation-delay: 0.16s; }

/* ── Pipeline tracker ───────────────────────────────────────── */
.pipeline-wrap {
    background: #fff;
    border: 1px solid #ddd8ce;
    border-radius: 14px;
    padding: 1rem 1.6rem;
    display: flex;
    align-items: center;
    margin-bottom: 1.8rem;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
}
.pip-step { display: flex; align-items: center; gap: 0.6rem; flex-shrink: 0; }
.pip-dot {
    width: 32px; height: 32px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.74rem; font-weight: 700;
    border: 2px solid #d4cec4; color: #b0a898; background: #f5f2eb; flex-shrink: 0;
    transition: all 0.2s ease;
}
.pip-dot.pip-active { background: #254d28; border-color: #254d28; color: #fff; box-shadow: 0 0 0 4px rgba(37,77,40,0.12); }
.pip-dot.pip-done   { background: #e8f7e8; border-color: #4a9e4c; color: #2d6a30; }
.pip-label { font-size: 0.8rem; font-weight: 500; color: #b0a898; white-space: nowrap; }
.pip-label.pip-active { color: #1a2e1c; font-weight: 700; }
.pip-label.pip-done   { color: #3d8040; }
.pip-line  { flex: 1; height: 2px; background: #e4dfd4; margin: 0 0.5rem; min-width: 20px; border-radius: 2px; }
.pip-line.pip-done { background: linear-gradient(90deg, #6ab86c, #4a9e4c); }

/* ── Page header ────────────────────────────────────────────── */
.pg-header { margin-bottom: 1.8rem; padding-bottom: 1.2rem; border-bottom: 1.5px solid #ddd8ce; }
.pg-eyebrow {
    font-size: 0.71rem; font-weight: 700; letter-spacing: 0.13em;
    text-transform: uppercase; color: #4a9e4c; margin-bottom: 0.4rem;
    display: flex; align-items: center; gap: 0.4rem;
}
.pg-eyebrow::before { content:''; display:inline-block; width:18px; height:2px; background:#4a9e4c; border-radius:2px; }
.pg-title {
    font-family: 'DM Serif Display', Georgia, serif !important;
    font-size: 2rem !important; color: #0f1e11 !important;
    margin: 0 0 0.5rem !important; letter-spacing: -0.03em !important; line-height: 1.1 !important;
}
.pg-sub { font-size: 0.92rem !important; color: #4a5e4b !important; margin: 0 !important; max-width: 60ch !important; line-height: 1.6 !important; }

/* ── White card ─────────────────────────────────────────────── */
.wcard {
    background: #fff; border: 1px solid #e4dfd4; border-radius: 16px;
    padding: 1.4rem 1.6rem; margin-bottom: 1.1rem;
    box-shadow: 0 2px 10px rgba(0,0,0,0.035); transition: box-shadow 0.2s ease;
}
.wcard:hover { box-shadow: 0 4px 18px rgba(0,0,0,0.06); }
.wcard-title { font-size: 0.71rem; font-weight: 700; letter-spacing: 0.1em; text-transform: uppercase; color: #6a9e6b; margin: 0 0 1.1rem; }

/* ── Chat-style soil input ──────────────────────────────────── */
.chat-wrap {
    background: #fff; border: 1.5px solid #e4dfd4; border-radius: 20px;
    padding: 1.6rem 1.8rem 1.2rem; margin-bottom: 1.2rem;
    box-shadow: 0 2px 12px rgba(0,0,0,0.04);
}
.chat-title {
    font-size: 0.71rem; font-weight: 700; letter-spacing: 0.1em;
    text-transform: uppercase; color: #6a9e6b; margin: 0 0 1.4rem;
}
.chat-hint {
    background: #f4fbf4; border: 1px solid #c8e6c8; border-radius: 14px;
    padding: 1rem 1.2rem; margin-bottom: 1.2rem; font-size: 0.88rem;
    color: #2a4a2b; line-height: 1.6;
}
.chat-hint code {
    background: #e0f0e0; color: #1a4a1c; border-radius: 5px;
    padding: 0.1rem 0.4rem; font-size: 0.85rem; font-weight: 600;
}
.chat-preview {
    display: grid; grid-template-columns: repeat(7,1fr); gap: 0.6rem;
    margin-top: 1.2rem;
}
.chat-prev-cell {
    background: #f7faf7; border: 1.5px solid #d4e8d4; border-radius: 10px;
    padding: 0.65rem 0.4rem; text-align: center;
    transition: border-color 0.2s ease;
}
.chat-prev-cell.filled { border-color: #4a9e4c; background: #f0faf0; }
.chat-prev-lbl { font-size: 0.62rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.06em; color: #7a9a7b; margin-bottom: 0.2rem; }
.chat-prev-val { font-family: 'DM Serif Display', Georgia, serif; font-size: 1.05rem; color: #0f1e11; line-height: 1; }
.chat-prev-val.pending { color: #b8c8b8; font-size: 0.9rem; font-family: 'DM Sans', sans-serif; }

/* chat text input box */
.chat-input-wrap .stTextInput input {
    background: #f8f6f0 !important;
    border: 2px solid #d8d4c8 !important;
    border-radius: 14px !important;
    color: #0f1e11 !important;
    font-size: 0.96rem !important;
    font-weight: 500 !important;
    padding: 0.7rem 1rem !important;
}
.chat-input-wrap .stTextInput input:focus {
    border-color: #4a9e4c !important;
    box-shadow: 0 0 0 3px rgba(74,158,76,0.14) !important;
    outline: none !important;
    background: #fff !important;
}
.chat-input-wrap .stTextInput input::placeholder { color: #9aaa9b !important; }
.chat-input-wrap label { display: none !important; }

.chat-error { color: #b03820 !important; font-size: 0.84rem !important; margin-top: 0.4rem; }
.chat-success { color: #2d6a30 !important; font-size: 0.84rem !important; margin-top: 0.4rem; }

/* ── Snapshot grid (old, keep for other pages) ───────────────── */
.snap-grid {
    display: grid; grid-template-columns: repeat(7,1fr); gap: 0.7rem;
    margin-top: 1.1rem; margin-bottom: 0.5rem;
}
.snap-cell { background: #fff; border: 1px solid #e4dfd4; border-radius: 12px; padding: 0.75rem 0.5rem; text-align: center; box-shadow: 0 1px 4px rgba(0,0,0,0.04); }
.snap-lbl { font-size: 0.65rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.07em; color: #7a9a7b; margin-bottom: 0.25rem; }
.snap-val { font-family: 'DM Serif Display', Georgia, serif; font-size: 1.1rem; color: #0f1e11; line-height: 1; }

/* ── Crop result cards ──────────────────────────────────────── */
.result-card {
    background: #fff; border: 1.5px solid #e4dfd4; border-radius: 18px;
    padding: 1.6rem 1.4rem 1.4rem; text-align: center;
    transition: transform 0.22s ease, box-shadow 0.22s ease;
    box-shadow: 0 2px 10px rgba(0,0,0,0.04);
}
.result-card:hover { transform: translateY(-4px); box-shadow: 0 16px 36px rgba(26,46,28,0.11); }
.result-card.top-pick {
    border: 2px solid #4a9e4c;
    background: linear-gradient(160deg, #f4fbf4 0%, #fff 100%);
    box-shadow: 0 4px 20px rgba(74,158,76,0.12);
}
.result-rank { font-size: 0.69rem; font-weight: 700; letter-spacing: 0.11em; text-transform: uppercase; margin-bottom: 0.55rem; }
.rk-gold   { color: #8a6820; }
.rk-silver { color: #607070; }
.rk-bronze { color: #7a5a3a; }
.result-crop-name { font-family: 'DM Serif Display', Georgia, serif; font-size: 1.5rem; color: #0f1e11; margin: 0.2rem 0 0.6rem; letter-spacing: -0.02em; }
.result-conf-text { font-size: 0.83rem; color: #4a5e4b; font-weight: 500; margin-bottom: 0.75rem; }
.conf-bar { height: 6px; background: #eaeee8; border-radius: 999px; overflow: hidden; }
.conf-fill { height: 100%; background: linear-gradient(90deg, #2d6a30, #70cc70); border-radius: 999px; }

/* ── Disease detection ──────────────────────────────────────── */
.det-card { background: #fff; border: 1.5px solid #e4dfd4; border-radius: 18px; padding: 1.8rem 2rem; box-shadow: 0 2px 10px rgba(0,0,0,0.04); }
.det-label { font-size: 0.69rem; font-weight: 700; letter-spacing: 0.1em; text-transform: uppercase; color: #6a9e6b; margin-bottom: 0.25rem; }
.det-value { font-family: 'DM Serif Display', Georgia, serif; font-size: 1.3rem; color: #0f1e11; margin: 0 0 1.2rem; letter-spacing: -0.01em; line-height: 1.2; }
.det-disease-val { color: #b03820; }
.det-empty { text-align: center; padding: 3.5rem 1rem; }
.det-empty-icon { font-size: 3rem; margin-bottom: 0.7rem; opacity: 0.7; }
.det-empty-text { color: #8a9a8b; font-size: 0.88rem; line-height: 1.6; }

/* ── File uploader ──────────────────────────────────────────── */
[data-testid="stFileUploader"],
[data-testid="stFileUploader"] > div,
[data-testid="stFileUploader"] > section,
[data-testid="stFileUploaderDropzone"] > div { background: #ffffff !important; }

[data-testid="stFileUploaderDropzone"] {
    background: #f6fbf6 !important;
    border: 2px dashed #b0d0b2 !important;
    border-radius: 16px !important;
    padding: 1.4rem 1.6rem !important;
    transition: border-color 0.22s ease, background 0.22s ease !important;
}
[data-testid="stFileUploaderDropzone"]:hover {
    border-color: #4a9e4c !important;
    background: #f0faf0 !important;
}

/* The "Browse files" / "Upload" button inside the dropzone */
[data-testid="stFileUploaderDropzone"] button,
[data-testid="stFileUploader"] button {
    background: #f0f7f0 !important;
    color: #1a4a1c !important;
    border: 1.5px solid #7abf7c !important;
    border-radius: 9px !important;
    font-weight: 600 !important;
    font-size: 0.86rem !important;
    padding: 0.45rem 1.1rem !important;
    box-shadow: 0 1px 4px rgba(74,158,76,0.10) !important;
    transition: all 0.18s ease !important;
    min-width: 100px !important;
    position: relative !important;
    overflow: hidden !important;
    color: transparent !important;
    font-size: 0 !important;
    line-height: 0 !important;
}
[data-testid="stFileUploaderDropzone"] button::after,
[data-testid="stFileUploader"] button::after {
    content: "Browse files" !important;
    position: absolute !important;
    inset: 0 !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    color: #1a4a1c !important;
    font-weight: 600 !important;
    font-size: 0.86rem !important;
    line-height: 1 !important;
    pointer-events: none !important;
    z-index: 2 !important;
}
[data-testid="stFileUploaderDropzone"] button *,
[data-testid="stFileUploader"] button * {
    display: none !important;
}
[data-testid="stFileUploaderDropzone"] button:hover,
[data-testid="stFileUploader"] button:hover {
    background: #e0f2e0 !important;
    border-color: #4a9e4c !important;
    color: #0f3a12 !important;
    box-shadow: 0 3px 10px rgba(74,158,76,0.18) !important;
    transform: translateY(-1px) !important;
}

/* Text inside uploader */
[data-testid="stFileUploaderDropzone"] *,
[data-testid="stFileUploader"] label,
[data-testid="stFileUploader"] p,
[data-testid="stFileUploader"] span,
[data-testid="stFileUploader"] small { color: #2a4a2b !important; }

/* "Drag and drop" instruction text */
[data-testid="stFileUploaderDropzone"] > div > div > span {
    color: #5a7a5b !important;
    font-size: 0.88rem !important;
}
/* "200MB per file" limit text */
[data-testid="stFileUploaderDropzone"] small,
[data-testid="stFileUploaderDropzone"] [data-testid="stMarkdownContainer"] small {
    color: #8aaa8b !important;
    font-size: 0.78rem !important;
}

/* ── Advisory ───────────────────────────────────────────────── */
.advisory-card {
    background: #fff; border: 1.5px solid #e4dfd4; border-radius: 18px;
    padding: 2rem 2.4rem; font-size: 0.95rem; line-height: 1.85;
    color: #1a2e1a; white-space: pre-wrap; word-break: break-word;
    box-shadow: 0 2px 12px rgba(0,0,0,0.04);
}

/* ── Analytics cards ───────────────────────────────────────── */
.analytics-grid {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 1rem;
    margin-top: 0.2rem;
}
.analytics-card {
    background: #ffffff;
    border: 1.5px solid #e4dfd4;
    border-radius: 16px;
    padding: 1.2rem 1.25rem 1.15rem;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    min-height: 120px;
}
.analytics-label {
    margin: 0 0 0.55rem;
    font-size: 0.76rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #111111;
}
.analytics-value {
    margin: 0;
    font-size: 1.5rem;
    font-weight: 800;
    line-height: 1.05;
    color: #111111;
}

@media (max-width: 900px) {
    .analytics-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
}
@media (max-width: 640px) {
    .analytics-grid { grid-template-columns: 1fr; }
}

/* ── ALL BUTTONS — global light style ──────────────────────── */
/* This ensures no button ever goes black/dark unless it's a sidebar nav */
.stButton > button {
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    padding: 0.62rem 1.2rem !important;
    background: #ffffff !important;
    color: #1a3a1c !important;
    border: 1.5px solid #c8c0b4 !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.07) !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    background: #f0f8f0 !important;
    border-color: #4a9e4c !important;
    color: #0f3a12 !important;
    box-shadow: 0 4px 14px rgba(74,158,76,0.16) !important;
    transform: translateY(-1px) !important;
}
.stButton > button:active {
    transform: translateY(0) !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.08) !important;
}

/* Primary buttons — green fill */
.stButton > button[kind="primary"],
button[kind="primary"] {
    background: linear-gradient(135deg, #2d6a30, #3a8a3e) !important;
    color: #ffffff !important;
    border-color: #2d6a30 !important;
    box-shadow: 0 3px 12px rgba(45,106,48,0.30) !important;
}
.stButton > button[kind="primary"]:hover,
button[kind="primary"]:hover {
    background: linear-gradient(135deg, #254d28, #2d6a30) !important;
    border-color: #254d28 !important;
    color: #ffffff !important;
    box-shadow: 0 7px 20px rgba(37,77,40,0.35) !important;
    transform: translateY(-1px) !important;
}
/* Disabled buttons */
.stButton > button:disabled {
    background: #e0f2e0 !important;
    color: #a0a898 !important;
    border-color: #d8d4c8 !important;
    box-shadow: none !important;
    transform: none !important;
    cursor: not-allowed !important;
    opacity: 0.65 !important;
}

/* Override sidebar buttons back to dark/transparent */
[data-testid="stSidebar"] .stButton > button {
    background: transparent !important;
}
.stSpinner > div { color: #254d28 !important; }
div[data-testid="stAlert"] p { color: inherit !important; }

/* ── Plotly chart container ──────────────────────────────────── */
div[data-testid="stPlotlyChart"] {
    background: #fff; border: 1.5px solid #e4dfd4; border-radius: 18px;
    padding: 0.6rem; overflow: hidden; box-shadow: 0 2px 10px rgba(0,0,0,0.04);
}

/* ── Image ───────────────────────────────────────────────────── */
img[data-testid="stImage"] { border-radius: 14px !important; box-shadow: 0 4px 16px rgba(0,0,0,0.10) !important; }

/* ── Hide keyboard icon artifact ────────────────────────────── */
[data-testid="InputInstructions"],
[data-testid="stTextInput"] + div > span,
span.st-emotion-cache-1629p8f { display: none !important; visibility: hidden !important; }

/* Also hide any "keyboard_do..." text that appears near inputs */
.stTextInput ~ div[class*="instructions"],
div[class*="InputInstructions"] { display: none !important; }

/* ── Responsive ──────────────────────────────────────────────── */
@media (max-width: 860px) {
    .pip-label { display: none; }
    .main .block-container { padding: 1.2rem 1.2rem 3rem !important; }
    .chat-preview { grid-template-columns: repeat(4,1fr); }
}
@media (max-width: 600px) {
    .pg-title { font-size: 1.6rem !important; }
    .chat-preview { grid-template-columns: repeat(3,1fr); }
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────
# State
# ─────────────────────────────────────────────────────────────────────
SOIL_FIELDS = [
    ("N",           "N",           "Nitrogen",    "kg/ha", 0,   140,  int,   90),
    ("P",           "P",           "Phosphorus",  "kg/ha", 0,   145,  int,   42),
    ("K",           "K",           "Potassium",   "kg/ha", 0,   205,  int,   43),
    ("temperature", "Temp",        "Temperature", "°C",    0,   50,   float, 25.0),
    ("humidity",    "Humidity",    "Humidity",    "%",     0,   100,  float, 80.0),
    ("ph",          "pH",          "Soil pH",     "0–14",  0,   14,   float, 6.5),
    ("rainfall",    "Rainfall",    "Rainfall",    "mm",    0,   300,  float, 200.0),
]

def init_state():
    defaults = {
        "ui_page": 0,
        "soil_inputs": {f[0]: f[7] for f in SOIL_FIELDS},
        "soil_saved":  True,
        "crop_results":       None,
        "detected_crop":      None,
        "detected_disease":   None,
        "disease_confidence": None,
        "advisory_text":      None,
        "_last_uploaded_name": None,
        # chat soil state
        "soil_chat_raw":  "",
        "soil_chat_error": "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def set_page(p):
    st.session_state.ui_page = p
    st.rerun()


def run_crop_model():
    s = st.session_state.soil_inputs
    st.session_state.crop_results = predict_top3(
        s["N"], s["P"], s["K"],
        s["temperature"], s["humidity"], s["ph"], s["rainfall"],
    )


def run_disease_model(f):
    suffix = Path(f.name).suffix if f.name else ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(f.getbuffer())
        tmp_path = Path(tmp.name)
    crop, disease, conf = predict_disease(tmp_path)
    st.session_state.detected_crop      = crop
    st.session_state.detected_disease   = disease
    st.session_state.disease_confidence = conf
    st.session_state._last_uploaded_name = f.name


def run_advisory():
    if st.session_state.crop_results:
        crop_name = st.session_state.crop_results[0]["crop"]
        crop_conf = st.session_state.crop_results[0]["confidence"]
    else:
        crop_name = st.session_state.detected_crop or "Unknown"
        crop_conf = st.session_state.disease_confidence or 0
    st.session_state.advisory_text = generate_advisory(
        crop_name=crop_name, crop_confidence=crop_conf,
        disease_name=st.session_state.detected_disease,
        disease_confidence=st.session_state.disease_confidence,
    )


# ─────────────────────────────────────────────────────────────────────
# Chat-style soil parser
# ─────────────────────────────────────────────────────────────────────
def parse_soil_chat(raw: str):
    """
    Accepts flexible formats:
      90 42 43 25 80 6.5 200          (positional)
      N=90 P=42 K=43 T=25 H=80 pH=6.5 R=200
      90, 42, 43, 25, 80, 6.5, 200
    Returns (dict, error_string).
    """
    raw = raw.strip()
    if not raw:
        return None, ""

    # Try key=value pairs first
    kv_pattern = re.findall(
        r'(?:N|P|K|temp(?:erature)?|t|hum(?:idity)?|h|ph|rain(?:fall)?|r)\s*=\s*[\d.]+',
        raw, flags=re.IGNORECASE
    )

    keys_map = {
        'n': 'N', 'p': 'P', 'k': 'K',
        'temp': 'temperature', 'temperature': 'temperature', 't': 'temperature',
        'hum': 'humidity', 'humidity': 'humidity', 'h': 'humidity',
        'ph': 'ph',
        'rain': 'rainfall', 'rainfall': 'rainfall', 'r': 'rainfall',
    }

    parsed = {}
    if kv_pattern:
        for pair in re.findall(r'(\w+)\s*=\s*([\d.]+)', raw, flags=re.IGNORECASE):
            k_raw, v_raw = pair
            k = keys_map.get(k_raw.lower())
            if k:
                try:
                    parsed[k] = float(v_raw)
                except ValueError:
                    pass
    else:
        # Positional: split by comma, space, or semicolon
        nums = re.split(r'[\s,;]+', raw)
        nums = [n for n in nums if n]
        field_keys = [f[0] for f in SOIL_FIELDS]
        if len(nums) != 7:
            return None, f"Expected 7 values (N P K Temp Humidity pH Rainfall), got {len(nums)}."
        for key, val_str in zip(field_keys, nums):
            try:
                parsed[key] = float(val_str)
            except ValueError:
                return None, f"'{val_str}' is not a valid number."

    # Validate ranges and cast types
    result = {}
    errors = []
    for (key, short, label, unit, lo, hi, typ, default) in SOIL_FIELDS:
        if key not in parsed:
            # use existing value if not supplied
            result[key] = st.session_state.soil_inputs[key]
            continue
        val = parsed[key]
        if not (lo <= val <= hi):
            errors.append(f"{label} ({val}) must be {lo}–{hi} {unit}.")
        else:
            result[key] = typ(val)

    if errors:
        return None, "  •  " + "\n  •  ".join(errors)

    return result, ""


# ─────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div class="sidebar-logo">
            <div class="sidebar-logo-badge">AI Powered</div>
            <div class="sidebar-logo-title">🌱 SmartCrop</div>
            <div class="sidebar-logo-sub">Intelligent Crop Advisor</div>
        </div>
        """, unsafe_allow_html=True)

        ss = st.session_state
        cur = ss.ui_page
        done_map = {
            0: ss.soil_saved,
            1: bool(ss.crop_results),
            2: bool(ss.detected_disease),
            3: bool(ss.advisory_text),
            4: True,
        }
        steps = [
            ("01", "🌍", "Soil Analysis"),
            ("02", "🌾", "Crop Recommendation"),
            ("03", "🔬", "Disease Detection"),
            ("04", "💡", "AI Advisory"),
            ("05", "📊", "Analytics"),
        ]
        for i, (num, icon, label) in enumerate(steps):
            is_done = done_map.get(i, False) and i != cur
            prefix  = "✓" if is_done else num
            if st.button(f"{icon}  {prefix}  {label}", key=f"nav_{i}", use_container_width=True):
                set_page(i)

        st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-section-label">Model Performance</div>', unsafe_allow_html=True)
        for label, val in [
            ("Crop accuracy",    "99.3%"),
            ("Disease accuracy", "94.4%"),
            ("Crop classes",     "22"),
            ("Disease classes",  "38"),
        ]:
            st.markdown(f"""
            <div class="sbar-metric-row">
                <span class="sbar-metric-name">{label}</span>
                <span class="sbar-metric-val">{val}</span>
            </div>
            """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────
# Pipeline tracker
# ─────────────────────────────────────────────────────────────────────
def render_pipeline():
    ss  = st.session_state
    cur = ss.ui_page
    steps      = ["Soil", "Crops", "Disease", "Advisory", "Analytics"]
    done_flags = [ss.soil_saved, bool(ss.crop_results), bool(ss.detected_disease), bool(ss.advisory_text), False]

    parts = ['<div class="pipeline-wrap">']
    for i, (name, done) in enumerate(zip(steps, done_flags)):
        active    = i == cur
        completed = done and not active
        dot_cls   = "pip-active" if active else ("pip-done" if completed else "")
        lbl_cls   = dot_cls
        dot_inner = "✓" if completed else str(i + 1)
        parts.append(f'<div class="pip-step"><div class="pip-dot {dot_cls}">{dot_inner}</div>'
                     f'<span class="pip-label {lbl_cls}">{name}</span></div>')
        if i < len(steps) - 1:
            line_cls = "pip-done" if (done_flags[i] and i < cur) else ""
            parts.append(f'<div class="pip-line {line_cls}"></div>')
    parts.append("</div>")
    st.markdown("".join(parts), unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────
def page_header(eyebrow, title, subtitle):
    st.markdown(f"""
    <div class="pg-header fade-up">
        <div class="pg-eyebrow">{eyebrow}</div>
        <div class="pg-title">{title}</div>
        <div class="pg-sub">{subtitle}</div>
    </div>
    """, unsafe_allow_html=True)


def render_nav(prev=None, nxt=None, prev_label="← Back",
               nxt_label="Continue →", nxt_disabled=False):
    cols = st.columns([1.2, 1.4, 5])
    if prev is not None:
        with cols[0]:
            if st.button(prev_label, key=f"nav_prev_{prev}", use_container_width=True):
                set_page(prev)
    if nxt is not None:
        with cols[1]:
            if st.button(nxt_label, type="primary", key=f"nav_nxt_{nxt}",
                         use_container_width=True, disabled=nxt_disabled):
                set_page(nxt)


# ─────────────────────────────────────────────────────────────────────
# Page 0 — Soil Analysis  (chat-style input)
# ─────────────────────────────────────────────────────────────────────
def soil_page():
    page_header(
        "Step 01", "Soil Analysis",
        "Type all your soil values in one line — the format is flexible and forgiving.",
    )

    sv = st.session_state.soil_inputs

    # ── hint bubble ──────────────────────────────────────────────
    st.markdown("""
    <div class="chat-wrap fade-up fade-up-1">
    <div class="chat-title">Quick Entry — one line, all values</div>
    <div class="chat-hint">
        Enter all 7 values in order: <code>N</code> <code>P</code> <code>K</code>
        <code>Temp&nbsp;°C</code> <code>Humidity&nbsp;%</code> <code>pH</code> <code>Rainfall&nbsp;mm</code><br><br>
        <strong>Format A — space or comma separated (positional):</strong><br>
        &nbsp;&nbsp;&nbsp;<code>90 42 43 25.0 80.0 6.5 200</code><br><br>
        <strong>Format B — labelled pairs (any order):</strong><br>
        &nbsp;&nbsp;&nbsp;<code>N=90 P=42 K=43 T=25 H=80 pH=6.5 R=200</code>
    </div>
    """, unsafe_allow_html=True)

    # ── text input ───────────────────────────────────────────────
    st.markdown('<div class="chat-input-wrap">', unsafe_allow_html=True)
    raw = st.text_input(
        "soil_chat",
        value=st.session_state.soil_chat_raw,
        placeholder="e.g.  90 42 43 25 80 6.5 200   or   N=90 P=42 K=43 T=25 H=80 pH=6.5 R=200",
        label_visibility="collapsed",
        key="soil_chat_input",
    )
    st.markdown('</div>', unsafe_allow_html=True)

    # parse on every keystroke (Streamlit re-runs on change)
    if raw != st.session_state.soil_chat_raw:
        st.session_state.soil_chat_raw = raw
        parsed, err = parse_soil_chat(raw)
        if parsed:
            st.session_state.soil_inputs   = parsed
            st.session_state.soil_saved    = True
            st.session_state.soil_chat_error = ""
        elif err:
            st.session_state.soil_chat_error = err
        else:
            st.session_state.soil_chat_error = ""

    # feedback
    err = st.session_state.soil_chat_error
    if err:
        st.markdown(f'<div class="chat-error">⚠ {err}</div>', unsafe_allow_html=True)
    elif raw and not err:
        st.markdown('<div class="chat-success">✓ Values parsed successfully</div>', unsafe_allow_html=True)

    # ── live preview grid ────────────────────────────────────────
    sv = st.session_state.soil_inputs
    cells_html = ""
    for (key, short, label, unit, lo, hi, typ, default) in SOIL_FIELDS:
        val = sv[key]
        is_filled = "filled" if st.session_state.soil_chat_raw else ""
        val_display = str(int(val)) if typ == int else f"{val:.1f}"
        cells_html += f"""
        <div class="chat-prev-cell {is_filled}">
            <div class="chat-prev-lbl">{short}<br><span style="font-weight:400;color:#9aaa9b">{unit}</span></div>
            <div class="chat-prev-val">{val_display}</div>
        </div>"""

    st.markdown(f'<div class="chat-preview">{cells_html}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)   # close chat-wrap

    st.markdown("<br>", unsafe_allow_html=True)
    render_nav(nxt=1, nxt_label="Continue to Crops →")


# ─────────────────────────────────────────────────────────────────────
# Page 1 — Crop Recommendation
# ─────────────────────────────────────────────────────────────────────
def crop_page():
    page_header(
        "Step 02", "Crop Recommendation",
        "Run the model to get the top three crop matches for your soil profile, ranked by confidence.",
    )

    col_btn, _ = st.columns([2, 6])
    with col_btn:
        if st.button("🌾  Run Crop Model", type="primary", use_container_width=True):
            with st.spinner("Analysing soil profile…"):
                run_crop_model()

    if st.session_state.crop_results:
        results   = st.session_state.crop_results
        rank_defs = [
            ("🥇  Top Pick",      "rk-gold",   True),
            ("🥈  Runner-up",     "rk-silver", False),
            ("🥉  Third Choice",  "rk-bronze", False),
        ]
        st.markdown("<br>", unsafe_allow_html=True)
        cols = st.columns(3)
        for col, item, (rank_text, rk_cls, is_top) in zip(cols, results, rank_defs):
            conf    = item["confidence"]
            top_cls = "top-pick" if is_top else ""
            with col:
                st.markdown(f"""
                <div class="result-card {top_cls} fade-up">
                    <div class="result-rank {rk_cls}">{rank_text}</div>
                    <div class="result-crop-name">{item['crop'].title()}</div>
                    <div class="result-conf-text">{conf}% confidence</div>
                    <div class="conf-bar"><div class="conf-fill" style="width:{min(conf,100)}%"></div></div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        chart_df = pd.DataFrame(results)
        chart_df["crop"] = chart_df["crop"].str.title()
        fig = go.Figure(go.Bar(
            x=chart_df["confidence"], y=chart_df["crop"],
            orientation="h",
            text=[f"{v}%" for v in chart_df["confidence"]],
            textposition="outside",
            textfont=dict(color="#1a2e1c", size=13, family="DM Sans"),
            marker=dict(color=["#254d28","#4a9e4c","#90d090"], line=dict(width=0), cornerradius=6),
        ))
        fig.update_layout(
            plot_bgcolor="#fafaf8", paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=10, r=80, t=28, b=20),
            xaxis=dict(
                range=[0, 118], showgrid=True, gridcolor="#ece8e0",
                title=dict(text="Confidence (%)", font=dict(color="#4a6a4b", size=12, family="DM Sans")),
                tickfont=dict(size=12, color="#3a5a3c", family="DM Sans"),
            ),
            yaxis=dict(showgrid=False, tickfont=dict(size=13, color="#0f1e11", family="DM Sans")),
            font=dict(family="DM Sans", color="#1a2e1a"),
            height=210, bargap=0.4,
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    render_nav(prev=0, prev_label="← Soil", nxt=2,
               nxt_label="Continue to Disease →",
               nxt_disabled=not bool(st.session_state.crop_results))


# ─────────────────────────────────────────────────────────────────────
# Page 2 — Disease Detection
# ─────────────────────────────────────────────────────────────────────
def disease_page():
    page_header(
        "Step 03", "Disease Detection",
        "Upload a clear leaf photo — the model identifies the crop and any disease present.",
    )

    st.markdown("<br>", unsafe_allow_html=True)
    left, right = st.columns([1.2, 0.8], gap="large")

    with left:
        uploaded = st.file_uploader(
            "Upload leaf image (JPG or PNG)",
            type=["jpg", "jpeg", "png"],
        )
        if uploaded is not None:
            if uploaded.name != st.session_state.get("_last_uploaded_name"):
                with st.spinner("Running disease detection…"):
                    run_disease_model(uploaded)
                    if st.session_state.crop_results:
                        run_advisory()
            st.image(uploaded, caption="Uploaded leaf image", use_container_width=True)
            st.success("✅  Scan complete — results shown on the right.")

    with right:
        if st.session_state.detected_disease:
            conf = st.session_state.disease_confidence
            st.markdown(f"""
            <div class="det-card fade-up">
                <div class="det-label">Detected Crop</div>
                <div class="det-value">{str(st.session_state.detected_crop).title()}</div>
                <div class="det-label">Detected Disease</div>
                <div class="det-value det-disease-val">{st.session_state.detected_disease}</div>
                <div class="det-label">Model Confidence</div>
                <div class="det-value">{conf:.1f}%</div>
                <div class="conf-bar" style="height:7px;margin-top:0.2rem">
                    <div class="conf-fill" style="width:{min(conf,100)}%"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="det-card">
                <div class="det-empty">
                    <div class="det-empty-icon">🍃</div>
                    <div class="det-empty-text">No scan yet.<br>Upload a leaf photo to begin.</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    render_nav(prev=1, prev_label="← Crops", nxt=3,
               nxt_label="Continue to Advisory →",
               nxt_disabled=not bool(st.session_state.detected_disease))


# ─────────────────────────────────────────────────────────────────────
# Page 3 — Advisory
# ─────────────────────────────────────────────────────────────────────
def advisory_page():
    page_header(
        "Step 04", "AI Advisory",
        "Combined, actionable guidance based on your crop recommendation and disease scan results.",
    )

    can_gen = st.session_state.crop_results and st.session_state.detected_disease
    if can_gen and not st.session_state.advisory_text:
        with st.spinner("Generating AI advisory…"):
            run_advisory()

    if st.session_state.advisory_text:
        text = (st.session_state.advisory_text
                .replace("&","&amp;").replace("<","&lt;").replace(">","&gt;"))
        st.markdown(f'<div class="advisory-card fade-up">{text}</div>', unsafe_allow_html=True)
    elif not st.session_state.crop_results:
        st.info("⚠️  Complete Step 2 (Crop Recommendation) first.")
    elif not st.session_state.detected_disease:
        st.info("⚠️  Complete Step 3 (Disease Detection) first.")

    st.markdown("<br>", unsafe_allow_html=True)
    render_nav(prev=2, prev_label="← Disease", nxt=4,
               nxt_label="View Analytics →",
               nxt_disabled=not bool(st.session_state.advisory_text))


# ─────────────────────────────────────────────────────────────────────
# Page 4 — Analytics
# ─────────────────────────────────────────────────────────────────────
def analytics_page():
    page_header(
        "Step 05", "Analytics",
        "Model performance metrics and dataset scale — a health check on the full pipeline.",
    )

    metrics = [
        ("Crop Accuracy", "99.3%"),
        ("Disease Accuracy", "94.4%"),
        ("Crop Classes", "22"),
        ("Disease Classes", "38"),
    ]

    cols = st.columns(4)
    for col, (label, value) in zip(cols, metrics):
        with col:
            st.markdown(
                f'''
                <div class="analytics-card fade-up">
                    <p class="analytics-label">{label}</p>
                    <p class="analytics-value">{value}</p>
                </div>
                ''',
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)

    perf = pd.DataFrame({
        "Metric": ["Accuracy", "Precision", "Recall", "F1-Score"],
        "Score":  [99.3, 99.1, 98.9, 99.0],
    })
    fig = go.Figure(go.Bar(
        x=perf["Metric"], y=perf["Score"],
        text=[f"{v}%" for v in perf["Score"]],
        textposition="outside",
        textfont=dict(color="#0f1e11", size=13, family="DM Sans"),
        marker=dict(
            color=["#254d28","#2d6a30","#4a9e4c","#6ab86c"],
            line=dict(width=0), cornerradius=6,
        ),
    ))
    fig.update_layout(
        plot_bgcolor="#fafaf8", paper_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(
            range=[97, 101], showgrid=True, gridcolor="#e8e4dc",
            title=dict(text="Score (%)", font=dict(color="#3a6a3c", size=12, family="DM Sans")),
            tickfont=dict(size=12, color="#3a6a3c", family="DM Sans"),
        ),
        xaxis=dict(showgrid=False, tickfont=dict(size=13, color="#0f1e11", family="DM Sans")),
        margin=dict(l=10, r=10, t=40, b=20),
        font=dict(family="DM Sans", color="#0f1e11"),
        height=340,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    render_nav(prev=3, prev_label="← Advisory")

    st.markdown("<br>", unsafe_allow_html=True)
    _, col_rst, _ = st.columns([4, 2, 4])
    with col_rst:
        if st.button("↺  Start New Session", key="new_session", use_container_width=True):
            for k in ["crop_results","detected_crop","detected_disease",
                      "disease_confidence","advisory_text","_last_uploaded_name",
                      "soil_chat_raw","soil_chat_error"]:
                st.session_state.pop(k, None)
            st.session_state.ui_page = 0
            st.rerun()


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────
init_state()
render_sidebar()

with st.container():
    render_pipeline()
    page = st.session_state.ui_page
    if   page == 0: soil_page()
    elif page == 1: crop_page()
    elif page == 2: disease_page()
    elif page == 3: advisory_page()
    else:           analytics_page()