"""
src/app/dashboard.py
---------------------
Streamlit dashboard pages:

1. 🏠 Overview     — Business context, Traffic Light legend
2. 📊 Risk Explorer — Predict risk band for uploaded CSV or demo data
3. 🔍 Explainability — SHAP waterfall plot for a selected applicant
4. 📈 Survival Curves — Kaplan-Meier curves by risk band
5. 📉 Model Health  — PSI drift gauge, evaluation metrics

Used by app.py as the main entry point.
"""
from __future__ import annotations

import io
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# Paths
ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_DIR = ROOT / "outputs" / "models"
PROCESSED_DIR = ROOT / "data" / "processed"
REPORT_DIR = ROOT / "outputs" / "reports"

BAND_COLORS = {"GREEN": "#2ecc71", "YELLOW": "#f39c12", "RED": "#e74c3c"}
BAND_EMOJI = {"GREEN": "🟢", "YELLOW": "🟡", "RED": "🔴"}


# -------------------------------------------------------------------
# Shared state helpers
# -------------------------------------------------------------------

@st.cache_resource
def _load_model():
    """Load DefaultClassifier once and cache for the session."""
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    from models.xgboost_classifier import DefaultClassifier
    from utils.config import load_config
    cfg = load_config()
    return DefaultClassifier.load(MODEL_DIR, cfg), cfg


@st.cache_data
def _load_demo_data() -> pd.DataFrame:
    demo_path = PROCESSED_DIR / "sample_for_demo.csv"
    if demo_path.exists():
        return pd.read_csv(demo_path)
    return pd.DataFrame()


# -------------------------------------------------------------------
# Page 1: Overview
# -------------------------------------------------------------------

def page_overview() -> None:
    st.title("🚦 Proactive Defaulter Flagging System")
    st.markdown("""
    > **Key Business Question:** *Who will likely default — and how early can we flag them?*

    This system uses an XGBoost-based classifier trained on the
    [Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk)
    dataset to assign each loan applicant a **Traffic Light risk band**:
    """)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            "<div style='background:#2ecc71;padding:20px;border-radius:8px;text-align:center'>"
            "<h2>🟢 GREEN</h2><b>Low Risk</b><br>Score < 15% default probability<br>"
            "<i>Auto-approve / standard rate</i></div>", unsafe_allow_html=True
        )
    with col2:
        st.markdown(
            "<div style='background:#f39c12;padding:20px;border-radius:8px;text-align:center'>"
            "<h2>🟡 YELLOW</h2><b>Medium Risk</b><br>15%–40% default probability<br>"
            "<i>Human review / conditional approval</i></div>", unsafe_allow_html=True
        )
    with col3:
        st.markdown(
            "<div style='background:#e74c3c;padding:20px;border-radius:8px;text-align:center'>"
            "<h2>🔴 RED</h2><b>High Risk</b><br>Score ≥ 40% default probability<br>"
            "<i>Decline or premium rate</i></div>", unsafe_allow_html=True
        )

    st.markdown("---")
    st.subheader("Tech Stack")
    st.markdown("""
    | Component | Technology |
    |-----------|------------|
    | Core Model | XGBoost (calibrated probabilities) |
    | Explainability | SHAP TreeExplainer |
    | Survival Analysis | Kaplan-Meier curves |
    | Threshold Optimisation | Cost-benefit matrix (FP=$100, FN=$5,000) |
    | Fairness Audit | Demographic parity (gender, age band) |
    | Drift Monitoring | Population Stability Index (PSI) |
    | Deployment | HuggingFace Spaces + Docker |
    """)


# -------------------------------------------------------------------
# Page 2: Risk Explorer
# -------------------------------------------------------------------

def page_risk_explorer() -> None:
    st.title("📊 Risk Explorer")
    st.markdown("Upload a CSV of applicant features — or use the built-in demo data.")

    demo_df = _load_demo_data()
    use_demo = st.checkbox("Use demo data (100 sample applicants)", value=True)

    if use_demo and not demo_df.empty:
        df = demo_df.copy()
        st.info(f"Loaded demo data: {len(df)} applicants")
    else:
        uploaded = st.file_uploader("Upload applicant CSV", type=["csv"])
        if uploaded is None:
            st.warning("Please upload a CSV file or enable demo data.")
            return
        df = pd.read_csv(uploaded)

    if st.button("🚦 Predict Risk Bands"):
        try:
            clf, cfg = _load_model()
        except Exception as exc:
            st.error(f"Model not loaded: {exc}. Run the training pipeline first.")
            return

        # Align columns
        valid_cols = [c for c in clf.feature_columns if c in df.columns]
        missing = set(clf.feature_columns) - set(df.columns)
        if missing:
            st.warning(f"{len(missing)} feature columns missing — filling with 0.")
        X = df.reindex(columns=clf.feature_columns, fill_value=0)

        with st.spinner("Predicting ..."):
            results = clf.predict_traffic_light(X)
            results["applicant_id"] = range(1, len(results) + 1)

        # Band summary
        band_counts = results["risk_band"].value_counts().reset_index()
        band_counts.columns = ["Risk Band", "Count"]

        col1, col2 = st.columns([1, 2])
        with col1:
            st.subheader("Band Distribution")
            for _, row in band_counts.iterrows():
                band = row["Risk Band"]
                st.markdown(
                    f"**{BAND_EMOJI.get(band, '')} {band}**: {row['Count']} applicants "
                    f"({row['Count']/len(results)*100:.1f}%)"
                )
        with col2:
            fig, ax = plt.subplots(figsize=(5, 4))
            colors = [BAND_COLORS.get(b, "grey") for b in band_counts["Risk Band"]]
            ax.bar(band_counts["Risk Band"], band_counts["Count"], color=colors, edgecolor="white")
            ax.set_title("Risk Band Distribution")
            ax.set_ylabel("Applicants")
            st.pyplot(fig)
            plt.close(fig)

        st.subheader("Applicant Risk Scores")
        st.dataframe(
            results[["applicant_id", "risk_score", "risk_band"]]
            .style.background_gradient(subset=["risk_score"], cmap="RdYlGn_r"),
            use_container_width=True,
        )

        csv_bytes = results.to_csv(index=False).encode()
        st.download_button("⬇️ Download predictions CSV", csv_bytes, "predictions.csv", "text/csv")


# -------------------------------------------------------------------
# Page 3: Explainability
# -------------------------------------------------------------------

def page_explainability() -> None:
    st.title("🔍 SHAP Explainability")
    st.markdown("""
    SHAP (SHapley Additive exPlanations) shows the contribution of each feature
    to an individual applicant's default risk score — providing regulatory-compliant
    *right-to-explanation* for each decision.
    """)

    demo_df = _load_demo_data()
    if demo_df.empty:
        st.warning("Demo data not found. Run `python src/create_sample.py` first.")
        return

    applicant_idx = st.slider("Select applicant index", 0, min(99, len(demo_df) - 1), 0)

    if st.button("Explain this applicant"):
        try:
            clf, cfg = _load_model()
        except Exception as exc:
            st.error(f"Model not loaded: {exc}")
            return

        try:
            import shap
        except ImportError:
            st.error("shap not installed. Add it to requirements.txt and rebuild.")
            return

        X = demo_df.reindex(columns=clf.feature_columns, fill_value=0)
        row = X.iloc[[applicant_idx]]
        proba = clf.predict_proba(row)[0]
        band_label = "RED" if proba >= clf.red_threshold else ("GREEN" if proba <= clf.green_threshold else "YELLOW")

        st.metric("Default Probability", f"{proba:.1%}", delta=f"{BAND_EMOJI[band_label]} {band_label}")

        shap_vals = clf.shap_values(row)
        # SHAP waterfall plot
        explainer = shap.TreeExplainer(clf.model if clf.model else clf.calibrated_model.estimator)
        shap_explanation = explainer(row)
        fig, ax = plt.subplots(figsize=(10, 5))
        shap.plots.waterfall(shap_explanation[0], max_display=15, show=False)
        st.pyplot(plt.gcf())
        plt.close()


# -------------------------------------------------------------------
# Page 4: Survival Curves
# -------------------------------------------------------------------

def page_survival_curves() -> None:
    st.title("📈 Kaplan-Meier Survival Curves")
    st.markdown("""
    Survival curves show *how long* borrowers in each risk band go without defaulting.
    A steeper curve for RED applicants confirms the Traffic Light system captures
    time-to-default risk, not just binary outcome.
    """)

    cache_path = PROCESSED_DIR / "feature_cache" / "master_features.csv"
    if not cache_path.exists():
        st.warning("Master features not found. Run the training pipeline first.")
        return

    import sys
    sys.path.insert(0, str(ROOT / "src"))
    from models.survival_model import KaplanMeierAnalyser, derive_loan_duration

    try:
        clf, cfg = _load_model()
    except Exception as exc:
        st.error(f"Model not loaded: {exc}")
        return

    with st.spinner("Loading data and generating curves ..."):
        df = pd.read_csv(cache_path, nrows=20000)
        df = derive_loan_duration(df)

        feat_cols = [c for c in clf.feature_columns if c in df.columns]
        X = df.reindex(columns=clf.feature_columns, fill_value=0)
        df["risk_band"] = clf.predict_traffic_light(X)["risk_band"].values

        km = KaplanMeierAnalyser(df, duration_col="loan_months", event_col="TARGET")
        km.fit_by_group("risk_band")
        fig = km.plot()

    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Median survival times")
    median_df = km.median_survival()
    st.dataframe(median_df, use_container_width=True)


# -------------------------------------------------------------------
# Page 5: Model Health
# -------------------------------------------------------------------

def page_model_health() -> None:
    st.title("📉 Model Health & Evaluation")

    eval_report_path = REPORT_DIR / "evaluation_report.csv"
    if eval_report_path.exists():
        report = pd.read_csv(eval_report_path)
        st.subheader("Evaluation Metrics")

        key_metrics = ["roc_auc", "gini_coefficient", "ks_statistic", "pr_auc"]
        cols = st.columns(len(key_metrics))
        for col, metric in zip(cols, key_metrics):
            row = report[report["metric"] == metric]
            if not row.empty:
                val = float(row["value"].iloc[0])
                col.metric(metric.replace("_", " ").title(), f"{val:.4f}")

        st.markdown("---")
        st.subheader("Full Metrics Table")
        st.dataframe(report, use_container_width=True)
    else:
        st.warning("Evaluation report not found. Run the training pipeline first.")

    fairness_path = REPORT_DIR / "fairness_report.csv"
    if fairness_path.exists():
        st.markdown("---")
        st.subheader("Fairness Audit")
        fairness = pd.read_csv(fairness_path)
        st.dataframe(fairness.style.applymap(
            lambda v: "background-color: #ffcccc" if v is False else "",
            subset=["passes_parity", "passes_4_5ths_rule"]
        ), use_container_width=True)
