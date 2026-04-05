# Proactive Defaulter Flagging System 🚦

> **Key Business Question:** *Who will likely default — and how early can we proactively flag them?*  
> **Expected Impact:** Reduced credit risk, data-driven loan decisions, regulatory-compliant AI

[![Python](https://img.shields.io/badge/Python-3.14.3-blue)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange)](https://xgboost.readthedocs.io/)
[![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-red)](https://streamlit.io/)
[![HuggingFace](https://img.shields.io/badge/Demo-HuggingFace%20Spaces-yellow)](https://huggingface.co/spaces/biplobgon/proactive-defaulter-flagging-system)

---

## Overview

This project builds a **production-grade credit default early-warning system** that assigns every loan applicant a **Traffic Light risk score** — proactively flagging high-risk borrowers *before* they default.

| 🟢 GREEN | 🟡 YELLOW | 🔴 RED |
|----------|----------|--------|
| Low risk | Medium risk | High risk |
| P(default) < 15% | 15%–40% | > 40% |
| Auto-approve | Human review | Decline / premium rate |

### Business Framing
- **False Positive cost:** $100 (lost revenue — wrongly declined good borrower)
- **False Negative cost:** $5,000 (default loss — wrongly approved bad borrower)
- Thresholds calibrated to **maximise expected profit**, not just accuracy

---

## Dataset

**[Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk)** (Kaggle)

| Table | Rows | Description |
|-------|------|-------------|
| `application_train.csv` | 307,511 | Main applicant features, target |
| `bureau.csv` | 1,716,428 | Credit bureau loan history |
| `bureau_balance.csv` | 27,299,925 | Monthly bureau balance |
| `previous_application.csv` | 1,670,214 | Prior Home Credit applications |
| `installments_payments.csv` | 13,605,401 | Installment payment timing |
| `credit_card_balance.csv` | 3,840,312 | Revolving credit utilisation |
| `POS_CASH_balance.csv` | 10,001,358 | POS / cash loan balance |

**Default rate: ~8.1%** (11:1 class imbalance)

---

## Architecture

```
Raw Data (7 tables)
      ↓
Feature Engineering
  ├── Behavioral:    Payment velocity, DPD30/60/90, on-time rate
  ├── Applicant:     Income ratios, external credit scores, doc count
  ├── Bureau:        Overdue history, active loans, debt ratio
  └── Previous apps: Approval rate, credit history
      ↓
XGBoost Classifier
  ├── Calibrated probabilities (Platt scaling)
  ├── Early stopping (val AUC)
  └── scale_pos_weight=11 (class imbalance)
      ↓
Traffic Light Scorer
  ├── 🟢 GREEN  → Auto-approve
  ├── 🟡 YELLOW → Manual review
  └── 🔴 RED    → Decline
      ↓
Explainability               Fairness Audit
  └── SHAP TreeExplainer         └── Demographic parity
      (GDPR right-to-explain)        (gender + age band)
      ↓
Streamlit Dashboard (HuggingFace Spaces) + Docker
```

---

## Key Results

| Metric | Value |
|--------|-------|
| ROC-AUC | *run pipeline* |
| Gini Coefficient | *2×AUC − 1* |
| KS Statistic | *run pipeline* |
| GREEN band default rate | < 5% ✅ |
| RED band default rate | > 20% ✅ |

---

## Innovative Differentiators

| Feature | Why It Matters |
|---------|---------------|
| **Survival Curves** | Kaplan-Meier shows *when* RED borrowers default vs GREEN — time dimension |
| **SHAP Explainability** | Per-decision waterfall plots → GDPR Article 22 compliance |
| **Cost-Benefit Threshold Optimizer** | Thresholds set by FP/FN cost matrix, not arbitrary 0.5 |
| **PSI Drift Monitoring** | Detects score distribution shift → automated retraining signal |
| **Fairness Audit Module** | Demographic parity scores for gender & age → Fair Lending compliance |
| **Walk-Forward Split** | No data leakage — trains on earlier cohorts, tests on later |

---

## Quickstart

### Prerequisites
```bash
pip install -r requirements.txt
```

### 1. Download Data
```bash
kaggle competitions download -c home-credit-default-risk -p data/raw/
cd data/raw && unzip home-credit-default-risk.zip
```

### 2. Prepare Features
```bash
python src/data_prep.py
```

### 3. Train Model
```bash
python src/training/train.py
```

### 4. Create Sample for Demo
```bash
python src/create_sample.py
```

### 5. Launch Dashboard
```bash
streamlit run app.py
```

---

## Repository Structure

```
proactive-defaulter-flagging-system/
├── app.py                              # Streamlit entry point (HF Spaces)
├── Dockerfile                          # HF Spaces container
├── requirements.txt                    # Full training deps
├── requirements-spaces.txt             # Slim inference deps (Spaces)
├── configs/
│   ├── model_config.yaml               # XGBoost params, thresholds, data paths
│   └── pipeline_config.yaml            # Pipeline orchestration, fairness, PSI
├── data/
│   ├── raw/                            # Kaggle CSVs (gitignored — download separately)
│   └── processed/                      # Engineered features (gitignored)
├── notebooks/
│   ├── 01_eda_default_patterns.ipynb   # EDA: distributions, demographics, missing
│   ├── 02_feature_engineering.ipynb    # Feature engineering walkthrough
│   ├── 03_model_training.ipynb         # XGBoost training, ROC curves, band validation
│   ├── 04_survival_curves.ipynb        # Kaplan-Meier curves by risk band
│   └── 05_shap_explainability.ipynb    # SHAP + fairness audit + PSI monitoring
├── src/
│   ├── data_prep.py                    # Data validation + merge pipeline
│   ├── create_sample.py                # Stratified sample for notebooks/demo
│   ├── app/
│   │   ├── dashboard.py                # Streamlit pages
│   │   └── hf_loader.py               # Download artefacts from HF Hub
│   ├── features/
│   │   ├── applicant_features.py       # Application + bureau + prev-app features
│   │   └── behavioral_features.py      # Installment + CC + POS behavioral features
│   ├── models/
│   │   ├── xgboost_classifier.py       # DefaultClassifier: train, predict, SHAP, save
│   │   └── survival_model.py           # KaplanMeierAnalyser
│   ├── training/
│   │   ├── train.py                    # Full pipeline orchestrator
│   │   ├── evaluate.py                 # AUC, Gini, KS, PSI, band rates
│   │   └── threshold_optimizer.py      # Cost-benefit Traffic Light calibration
│   └── utils/
│       ├── config.py                   # YAML config loader
│       ├── logger.py                   # Logging setup
│       └── fairness_audit.py           # Demographic parity audit
└── outputs/
    ├── models/                         # Trained model PKL + artefacts
    ├── figures/                        # All generated charts
    └── reports/                        # evaluation_report.csv, fairness_report.csv
```

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| Core Model | XGBoost 2.0 (calibrated via Platt scaling) |
| Explainability | SHAP TreeExplainer |
| Survival Analysis | Custom Kaplan-Meier (no lifelines required) |
| Feature Engineering | Pandas / NumPy |
| Deployment | Streamlit + HuggingFace Spaces + Docker |
| Config | YAML + dot-notation Config class |
| Fairness | Custom demographic parity audit |

---

## Regulatory Compliance Notes

This system is designed with the following regulatory frameworks in mind:

- **GDPR Article 22** — Right not to be subject to automated decision-making; SHAP waterfall plots provide per-decision explanations
- **Fair Lending / ECOA** — Fairness audit module tests for demographic parity and disparate impact (4/5ths rule)
- **Basel III / IFRS-9** — Gini coefficient and KS statistic used (industry standard credit risk metrics)

---

## License

MIT License — see [LICENSE](LICENSE)

---

*Built as a portfolio project to demonstrate production-grade fintech ML capabilities.*
