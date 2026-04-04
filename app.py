"""
app.py
------
Streamlit entry point for HuggingFace Spaces deployment.

Sets up sys.path, downloads model artefacts from HF Hub (if needed),
then delegates to src/app/dashboard.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure src/ is importable
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import streamlit as st

st.set_page_config(
    page_title="Proactive Defaulter Flagging System",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Download model artefacts from HF Hub on first run (Spaces deployment)
try:
    from app.hf_loader import download_artifacts
    download_artifacts(ROOT / "outputs" / "models")
except Exception:
    pass  # Local run without HF Hub artefacts is fine

from app.dashboard import (
    page_overview,
    page_risk_explorer,
    page_explainability,
    page_survival_curves,
    page_model_health,
)

PAGES = {
    "🏠 Overview": page_overview,
    "📊 Risk Explorer": page_risk_explorer,
    "🔍 Explainability (SHAP)": page_explainability,
    "📈 Survival Curves": page_survival_curves,
    "📉 Model Health": page_model_health,
}

with st.sidebar:
    st.image("https://img.shields.io/badge/Risk%20Engine-XGBoost-orange", use_column_width=False)
    st.markdown("## Navigation")
    selected = st.radio("Go to", list(PAGES.keys()), label_visibility="collapsed")
    st.markdown("---")
    st.markdown(
        "**Dataset:** [Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk)\n\n"
        "**Code:** [GitHub](https://github.com/biplobgon/proactive-defaulter-flagging-system)"
    )

PAGES[selected]()
