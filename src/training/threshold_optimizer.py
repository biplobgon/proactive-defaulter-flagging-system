"""
training/threshold_optimizer.py
--------------------------------
Cost-benefit threshold optimiser for Traffic Light scoring.

Finds the optimal Red/Green probability thresholds that maximise
expected profit given:
  - cost_fp : Cost of a False Positive (wrongly rejected applicant → lost revenue)
  - cost_fn : Cost of a False Negative (wrongly approved defaulter → default loss)

Output: {"red_threshold": float, "green_threshold": float}

Usage
-----
    from training.threshold_optimizer import optimise_thresholds
    thresholds = optimise_thresholds(y_true, y_proba, cost_fp=100, cost_fn=5000)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from utils.logger import get_logger

log = get_logger(__name__)


def optimise_thresholds(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    cost_fp: float = 100.0,
    cost_fn: float = 5000.0,
    n_points: int = 200,
) -> dict[str, float]:
    """Find the cost-optimal Red/Green thresholds.

    Strategy:
    - Sweep a grid of candidate thresholds (0.01 – 0.99).
    - RED threshold  = probability above which we always decline (minimise FN loss).
    - GREEN threshold = probability below which we always approve (maximise revenue).
    - Applicants between the two thresholds go for YELLOW / manual review.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth binary labels.
    y_proba : np.ndarray
        Model predicted probabilities of default.
    cost_fp : float
        $ lost per false positive (wrongly declined non-defaulter).
    cost_fn : float
        $ lost per false negative (wrongly approved defaulter).
    n_points : int
        Number of threshold grid points to evaluate.

    Returns
    -------
    dict
        {"red_threshold": float, "green_threshold": float}
    """
    thresholds = np.linspace(0.01, 0.99, n_points)
    best_profit = -np.inf
    best_red = 0.40
    best_green = 0.15

    for red_t in thresholds:
        # GREEN: everything below half of red_t (heuristic starting point)
        for green_t in np.linspace(0.01, red_t - 0.01, max(2, int(n_points / 10))):
            y_pred = np.where(y_proba >= red_t, 1, 0)
            # Profit = revenue from correct approvals - losses from defaults
            tp = np.sum((y_pred == 1) & (y_true == 1))
            fp = np.sum((y_pred == 1) & (y_true == 0))
            tn = np.sum((y_pred == 0) & (y_true == 0))
            fn = np.sum((y_pred == 0) & (y_true == 1))

            # Revenue: approved non-defaulters contribute $cost_fp (opportunity value)
            # Loss: approved defaulters cost $cost_fn each
            # FP here means declining a non-defaulter (lost revenue)
            profit = (tn * 0) - (fp * cost_fp) - (fn * cost_fn)
            if profit > best_profit:
                best_profit = profit
                best_red = red_t
                best_green = green_t

    log.info(
        "Optimised thresholds: red=%.3f, green=%.3f | expected_profit=%.0f",
        best_red, best_green, best_profit,
    )
    return {"red_threshold": best_red, "green_threshold": best_green}


def threshold_sweep_report(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    cost_fp: float = 100.0,
    cost_fn: float = 5000.0,
    n_points: int = 100,
) -> pd.DataFrame:
    """Return a DataFrame showing profit, precision, recall for each threshold."""
    records = []
    for t in np.linspace(0.01, 0.99, n_points):
        y_pred = (y_proba >= t).astype(int)
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        tn = np.sum((y_pred == 0) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        profit = -(fp * cost_fp) - (fn * cost_fn)
        records.append({
            "threshold": t,
            "precision": precision,
            "recall": recall,
            "profit": profit,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        })
    return pd.DataFrame(records)
