"""
training/evaluate.py
---------------------
Model evaluation utilities.

Metrics computed:
- ROC-AUC
- PR-AUC (Average Precision)
- Gini Coefficient (2 * AUC - 1)
- KS Statistic (max separation between TPR and FPR curves)
- Confusion matrix at optimal threshold
- Default rates per Traffic Light band

Usage
-----
    from training.evaluate import evaluate_model
    report = evaluate_model(clf, X_test, y_test, cfg)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    roc_auc_score,
)

from utils.logger import get_logger

log = get_logger(__name__)


def evaluate_model(clf, X_test: pd.DataFrame, y_test: pd.Series, cfg) -> pd.DataFrame:
    """Evaluate a trained DefaultClassifier and return a metrics DataFrame.

    Parameters
    ----------
    clf : DefaultClassifier
        Trained classifier with predict_proba and predict_traffic_light.
    X_test : pd.DataFrame
        Test feature matrix.
    y_test : pd.Series
        Ground-truth labels (1 = default).
    cfg : Config
        Pipeline config.

    Returns
    -------
    pd.DataFrame
        One row per metric with columns [metric, value].
    """
    y_proba = clf.predict_proba(X_test)
    tl_results = clf.predict_traffic_light(X_test)
    tl_results["TARGET"] = y_test.values

    # --- Standard metrics ---
    roc_auc = roc_auc_score(y_test, y_proba)
    pr_auc = average_precision_score(y_test, y_proba)
    gini = 2 * roc_auc - 1
    ks = _ks_statistic(y_test.values, y_proba)

    log.info("ROC-AUC: %.4f | Gini: %.4f | KS: %.4f | PR-AUC: %.4f",
             roc_auc, gini, ks, pr_auc)

    # --- Traffic Light band default rates ---
    band_stats = (
        tl_results.groupby("risk_band")["TARGET"]
        .agg(n_applicants="count", default_rate="mean")
        .reset_index()
    )
    log.info("Default rates by band:\n%s", band_stats.to_string(index=False))

    # --- Confusion matrix at 0.5 threshold ---
    y_pred = (y_proba >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    metrics = [
        ("roc_auc", roc_auc),
        ("gini_coefficient", gini),
        ("ks_statistic", ks),
        ("pr_auc", pr_auc),
        ("true_positives", tp),
        ("false_positives", fp),
        ("true_negatives", tn),
        ("false_negatives", fn),
        ("precision", tp / (tp + fp) if (tp + fp) > 0 else 0.0),
        ("recall", tp / (tp + fn) if (tp + fn) > 0 else 0.0),
        ("n_test_samples", len(y_test)),
        ("positive_rate", y_test.mean()),
    ]

    for _, row in band_stats.iterrows():
        metrics.append((f"default_rate_{row['risk_band']}", row["default_rate"]))
        metrics.append((f"n_{row['risk_band']}", row["n_applicants"]))

    report = pd.DataFrame(metrics, columns=["metric", "value"])
    return report


def _ks_statistic(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """Compute the KS (Kolmogorov-Smirnov) statistic between defaulters and non-defaulters."""
    df = pd.DataFrame({"y": y_true, "p": y_proba}).sort_values("p", ascending=False)
    total_pos = y_true.sum()
    total_neg = (1 - y_true).sum()
    if total_pos == 0 or total_neg == 0:
        return 0.0
    df["cum_pos"] = df["y"].cumsum() / total_pos
    df["cum_neg"] = (1 - df["y"]).cumsum() / total_neg
    return float((df["cum_pos"] - df["cum_neg"]).abs().max())


def compute_psi(reference: np.ndarray, current: np.ndarray, n_bins: int = 10) -> float:
    """Compute Population Stability Index between reference and current score distributions.

    PSI < 0.1   → No significant change
    PSI 0.1-0.25 → Moderate shift (warning)
    PSI > 0.25  → Significant shift (retrain required)
    """
    bins = np.linspace(0, 1, n_bins + 1)
    ref_pct = np.histogram(reference, bins=bins)[0] / len(reference)
    cur_pct = np.histogram(current, bins=bins)[0] / len(current)

    # Avoid log(0)
    ref_pct = np.where(ref_pct == 0, 1e-6, ref_pct)
    cur_pct = np.where(cur_pct == 0, 1e-6, cur_pct)

    psi = float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))
    return psi
