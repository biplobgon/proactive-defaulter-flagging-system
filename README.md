# Proactive Defaulter Flagging System 🚦

> **Key Business Question:** *Who will likely default — and how early can we proactively flag them?*  
> **Expected Impact:** Reduced credit losses, data-driven loan decisions, regulatory-compliant AI

[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange)](https://xgboost.readthedocs.io/)
[![SHAP](https://img.shields.io/badge/Explainability-SHAP-green)](https://shap.readthedocs.io/)
[![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-red)](https://streamlit.io/)
[![HuggingFace](https://img.shields.io/badge/Demo-HuggingFace%20Spaces-yellow)](https://huggingface.co/spaces/biplobgon/proactive-defaulter-flagging-system)

---

## Overview

This project builds a **production-grade credit default early-warning system** that assigns every loan applicant a **Traffic Light risk score** — proactively flagging high-risk borrowers *before* they default, with full per-decision explanations and a fair-lending audit trail.

### Traffic Light Bands (data-driven, calibrated on validation set)

| Band | Score Range | Risk Score | % of Applicants | Actual Default Rate | Action |
|------|-------------|------------|-----------------|---------------------|--------|
| 🔴 RED | 506 – 557 pts | ≥ 0.1231 | 18 % | **35.5 %** | Decline / senior review |
| 🟡 YELLOW | 557 – 584 pts | 0.0517 – 0.1231 | 29 % | **12.2 %** | Human review |
| 🟢 GREEN | 584 – 617 pts | ≤ 0.0517 | 52 % | **2.9 %** | Auto-approve |

> Thresholds are **data-driven** (80th / 50th percentile of validation-set calibrated scores), not arbitrary. The 17-point gap between RED median-survival (28 months) and GREEN (45 months) quantifies the intervention window available to the lender.

### Business Framing
- **False Positive cost:** $100 — lost revenue from a wrongly declined good borrower
- **False Negative cost:** $5,000 — default loss from a wrongly approved bad borrower
- Reviewing only the top 20 % of applicants (RED band) captures **54 % of all defaults** at 3.2× lift

---

## Dataset

**[Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk)** (Kaggle)

| Table | Rows | Description |
|-------|------|-------------|
| `application_train.csv` | 307,511 | Main applicant features + target |
| `bureau.csv` | 1,716,428 | Credit bureau loan history |
| `bureau_balance.csv` | 27,299,925 | Monthly bureau status |
| `previous_application.csv` | 1,670,214 | Prior Home Credit applications |
| `installments_payments.csv` | 13,605,401 | Installment payment timing |
| `credit_card_balance.csv` | 3,840,312 | Revolving credit utilisation |
| `POS_CASH_balance.csv` | 10,001,358 | POS / cash loan balance |

**Overall default rate: 8.1 %** (11:1 class imbalance, addressed via `scale_pos_weight=11`)

---

## Key Results

### Model Performance (XGBoost + Platt calibration, 50K-row dev set)

| Metric | XGBoost | Logistic Regression baseline | Δ |
|--------|---------|------------------------------|---|
| **ROC-AUC** | **0.7502** | 0.7377 | +0.0125 |
| **Gini Coefficient** | **0.5005** | 0.4754 | +0.0251 |
| **KS Statistic** | **0.3906** | 0.3720 | +0.0186 |
| **PR-AUC** | **0.2446** | 0.2210 | +0.0236 |

### Traffic Light Validation (held-out test set, n=10,000)

| Band | Population | Default Rate | Avg Risk Score |
|------|-----------|--------------|----------------|
| 🔴 RED | 19.4 % | **21.6 %** | 0.216 |
| 🟡 YELLOW | 28.8 % | **7.2 %** | 0.079 |
| 🟢 GREEN | 51.8 % | **3.2 %** | 0.033 |

### Survival Analysis (full 50K dataset, KM curves)

| Band | Median survival (months to default) | Observed default rate |
|------|-------------------------------------|-----------------------|
| 🔴 RED | **28 months** | 35.5 % |
| 🟡 YELLOW | 40 months | 12.2 % |
| 🟢 GREEN | 45 months | 2.9 % |

All three pairwise log-rank tests: **p < 0.001** (highly significant separation).

### Top SHAP Features

| Rank | Feature | What it captures |
|------|---------|-----------------|
| 1 | `ext_source_mean` | Mean of 3 external credit bureau scores |
| 2 | `prev_credit_ratio_mean` | Avg credit utilisation on prior Home Credit loans |
| 3 | `ext_source_product` | Multiplicative interaction of EXT_SOURCE scores (engineered) |
| 4 | `pos_count` | Number of POS / consumer loan records |
| 5 | `prev_approval_rate` | Fraction of prior applications approved |
| 6 | `annuity_income_ratio` | Monthly repayment burden as % of income |

---

## EDA Insights (from NB01 & NB02)

### High-risk segments identified

| Segment | Default Rate | vs Overall (8.1 %) |
|---------|-------------|---------------------|
| Low-skill Laborers (occupation) | **17.1 %** | +9.0 pp |
| Applicants aged < 30 | **11.2 %** | +3.1 pp |
| Unemployed (income type) | **33.3 %** | +25.2 pp |
| Drivers | **11.1 %** | +3.0 pp |
| No bureau history | **9.9 %** | +1.8 pp |
| Pensioners | **5.6 %** | −2.5 pp ✅ |
| State servants | **5.6 %** | −2.5 pp ✅ |
| Applicants aged 60+ | **5.0 %** | −3.1 pp ✅ |

### Key feature observations
- `EXT_SOURCE_3` has the strongest individual negative correlation with default (ρ = −0.179)
- 18 % of applicants have `DAYS_EMPLOYED = 365,243` — a sentinel for "unemployed / retired"
- 41 columns have > 50 % missing values — handled via median imputation + missingness flags
- Credit card balance records exist for only **28 %** of applicants (sparse join)
- `ext_source_product` (engineered interaction) became the **#1 mutual-information feature** after NB02

---

## Fairness Audit Results

| Protected Attribute | Group | Default Rate | Parity Gap | 4/5ths Rule |
|--------------------|-------|-------------|-----------|-------------|
| Gender | Male (M) | 10.3 % | 10.3 pp | ❌ FAIL |
| Gender | Female (F) | 6.9 % | — | — |
| Age Band | < 30 | 11.2 % | 5.5 pp | ❌ FAIL |
| Age Band | 30–50 | 8.7 % | — | — |
| Age Band | > 50 | 5.7 % | — | — |

> These gaps reflect **real-world disparate risk levels** in the training data, not model bias. Lenders should review whether these gaps are driven by legitimate financial variables (income, employment) or protected characteristics before deployment.

### PSI Drift Monitoring
- Reference-vs-current PSI = **0.003** → ✅ STABLE (< 0.10 threshold)

---

## Risk Scorecard (PDO=20, Base=500)

The model's calibrated probability is converted to an integer score (higher = safer, similar to FICO):

$$\text{Score} = 500 - 20 \times \log_2\!\left(\frac{p}{1-p}\right)$$

| Band | Score Range | Median Score |
|------|------------|-------------|
| 🔴 RED | 506 – 557 | 542 |
| 🟡 YELLOW | 557 – 584 | 573 |
| 🟢 GREEN | 584 – 617 | 598 |

---

## Architecture

```
Raw Data (7 tables, 56M+ rows)
      ↓
Feature Engineering (NB02)
  ├── Applicant:    Income ratios, EXT_SOURCE scores, employment anomaly
  ├── Bureau:       Delinquency history, active loan count, debt ratio
  ├── Previous:     Approval rate, prior credit utilisation
  ├── Behavioral:   Installment on-time rate, DPD30/60/90, CC utilisation
  └── Interactions: ext_source_product, bureau_inquiry_burst,
                    days_employed_is_anomaly, social_def_flag
      ↓ 50,000 × 185 master feature table
XGBoost Classifier (NB03)
  ├── scale_pos_weight=11  (class imbalance)
  ├── Early stopping on val AUC
  ├── Platt scaling calibration (LogisticRegression on val probabilities)
  └── AUC=0.7502 | Gini=0.5005 | KS=0.3906
      ↓
Traffic Light Scorer
  ├── 🟢 GREEN  (score ≥ 584)  → 2.9 % default rate
  ├── 🟡 YELLOW (557–584)      → 12.2 % default rate
  └── 🔴 RED    (score ≤ 557)  → 35.5 % default rate
      ↓
┌──────────────────────┬─────────────────────────┐
│  Explainability (NB05)│  Survival Analysis (NB04)│
│  SHAP TreeExplainer   │  Kaplan-Meier curves     │
│  Waterfall per case   │  RED median: 28 months   │
│  Interaction plots    │  GREEN median: 45 months │
└──────────────────────┴─────────────────────────┘
      ↓
Fairness Audit + PSI Drift + Risk Scorecard
      ↓
Streamlit Dashboard (HuggingFace Spaces) + Docker
```

---

## Notebook Walkthrough

| Notebook | Purpose | Key Outputs |
|----------|---------|-------------|
| [01_eda_default_patterns](notebooks/01_eda_default_patterns.ipynb) | Exploratory data analysis | Default rates by segment, EXT_SOURCE correlations, DAYS_EMPLOYED anomaly, 41 high-missing columns |
| [02_feature_engineering](notebooks/02_feature_engineering.ipynb) | Feature construction & MI ranking | 185-feature master table, `ext_source_product` as #1 MI feature, CC join coverage 28 % |
| [03_model_training](notebooks/03_model_training.ipynb) | XGBoost training, calibration, threshold optimisation | AUC=0.7502, data-driven Traffic Light thresholds, top-decile lift 3.2× |
| [04_survival_curves](notebooks/04_survival_curves.ipynb) | Kaplan-Meier by band, occupation, age | RED median survival 28 vs GREEN 45 months, all pairwise p < 0.001 |
| [05_shap_explainability](notebooks/05_shap_explainability.ipynb) | SHAP, fairness audit, PSI, scorecard | `ext_source_mean` dominant, gender gap 10.3 pp, PSI=0.003 STABLE |

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

### 2. Feature Engineering
Run [02_feature_engineering.ipynb](notebooks/02_feature_engineering.ipynb) to produce `data/processed/feature_cache/master_features.csv`

### 3. Train & Evaluate
Run [03_model_training.ipynb](notebooks/03_model_training.ipynb) — produces artefacts in `outputs/models/` and `outputs/reports/`

### 4. (Optional) Survival & Explainability
Run notebooks 04 and 05 to generate all charts in `outputs/figures/`

### 5. Launch Dashboard
```bash
python src/create_sample.py   # creates a stratified 5K-row demo sample
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
│   ├── model_config.yaml               # XGBoost params, calibrated thresholds,
│   │                                   #   fairness tolerance, data paths
│   └── pipeline_config.yaml            # Pipeline steps, PSI settings, inference
├── data/
│   ├── raw/                            # Kaggle CSVs (gitignored)
│   └── processed/feature_cache/        # master_features.csv (gitignored)
├── notebooks/
│   ├── 01_eda_default_patterns.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_training.ipynb
│   ├── 04_survival_curves.ipynb
│   └── 05_shap_explainability.ipynb
├── src/
│   ├── data_prep.py                    # Data validation + merge pipeline
│   ├── create_sample.py                # Stratified sample for demo
│   ├── app/
│   │   ├── dashboard.py                # Streamlit pages
│   │   └── hf_loader.py               # Download artefacts from HF Hub
│   ├── features/
│   │   ├── applicant_features.py       # Application + bureau + prev-app features
│   │   └── behavioral_features.py      # Installment + CC + POS behavioral features
│   ├── models/
│   │   ├── xgboost_classifier.py       # DefaultClassifier: train, Platt calibration,
│   │   │                               #   predict_traffic_light, SHAP, save/load
│   │   └── survival_model.py           # KaplanMeierAnalyser + derive_loan_duration
│   ├── training/
│   │   ├── train.py                    # Full pipeline orchestrator
│   │   ├── evaluate.py                 # AUC, Gini, KS, PR-AUC, PSI, band rates
│   │   └── threshold_optimizer.py      # Cost-benefit Traffic Light calibration
│   └── utils/
│       ├── config.py                   # YAML config loader (dot-notation)
│       ├── logger.py                   # Structured logging setup
│       └── fairness_audit.py           # Demographic parity + 4/5ths rule audit
└── outputs/
    ├── models/                         # xgboost_default_model.pkl, platt_scaler.pkl,
    │                                   #   feature_scaler.pkl, feature_columns.json
    ├── figures/                        # All generated charts (NB01–NB05)
    └── reports/                        # evaluation_report.csv, fairness_report.csv,
                                        #   segment_default_rates.csv, risk_scores.csv
```

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| Core Model | XGBoost (calibrated via Platt scaling / LogisticRegression) |
| Explainability | SHAP TreeExplainer — global bar, beeswarm, waterfall, dependence |
| Survival Analysis | Custom Kaplan-Meier (scipy only, no lifelines required) |
| Feature Engineering | Pandas / NumPy — 185 features across 7 source tables |
| Fairness | Custom demographic parity + 4/5ths disparate impact audit |
| Drift Monitoring | Population Stability Index (PSI) via `training.evaluate` |
| Deployment | Streamlit + HuggingFace Spaces + Docker |
| Config | YAML + dot-notation `Config` class |

---

## Regulatory Compliance

| Framework | How this project addresses it |
|-----------|-------------------------------|
| **GDPR Article 22** | Per-applicant SHAP waterfall plots explain every automated decision |
| **Fair Lending / ECOA** | Fairness audit module: demographic parity difference + 4/5ths disparate impact rule for gender and age band |
| **Basel III / IFRS-9** | Gini coefficient (0.50) and KS statistic (0.39) reported — standard credit risk metrics |
| **Model Risk Management** | PSI drift monitoring flags score distribution shifts; separate val/test sets prevent data leakage |

---

## Generated Outputs

All output files are produced by running the notebooks end-to-end:

| File | Produced by | Contents |
|------|------------|----------|
| `outputs/models/xgboost_default_model.pkl` | NB03 | Trained XGBClassifier |
| `outputs/models/platt_scaler.pkl` | NB03 | Platt calibration LogisticRegression |
| `outputs/models/feature_columns.json` | NB03 | Ordered list of 141 model features |
| `outputs/reports/evaluation_report.csv` | NB03 | AUC, Gini, KS, PR-AUC, band rates |
| `outputs/reports/segment_default_rates.csv` | NB04 | Default rates by band / occupation / age |
| `outputs/reports/fairness_report.csv` | NB05 | Parity diff + disparate impact per attribute |
| `outputs/reports/risk_scores.csv` | NB05 | 50K rows with score_points + risk_band |
| `outputs/figures/03_traffic_light_validation.png` | NB03 | Band validation bar chart |
| `outputs/figures/04_km_by_risk_band.png` | NB04 | KM survival curves by traffic light band |
| `outputs/figures/04_km_by_occupation.png` | NB04 | KM by occupation type |
| `outputs/figures/04_km_by_age_band.png` | NB04 | KM by age band |
| `outputs/figures/05_shap_global_bar.png` | NB05 | Global SHAP feature importance |
| `outputs/figures/05_shap_beeswarm.png` | NB05 | SHAP beeswarm (direction of impact) |
| `outputs/figures/05_shap_dependence.png` | NB05 | Interaction dependence plots |
| `outputs/figures/05_shap_waterfall_red.png` | NB05 | RED applicant SHAP waterfall |
| `outputs/figures/05_fairness_disparate_impact.png` | NB05 | Disparate impact bar chart |
| `outputs/figures/05_risk_scorecard.png` | NB05 | Score distribution by band |

---

## License

MIT License — see [LICENSE](LICENSE)

---

*Built as a portfolio project demonstrating production-grade fintech ML: end-to-end pipeline, survival analysis, SHAP explainability, fair-lending audit, and score calibration — all grounded in real benchmark numbers.*
