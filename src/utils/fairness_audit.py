"""
utils/fairness_audit.py
-----------------------
Fairness audit utilities: demographic parity checks across protected groups.

Metrics computed:
- Default rate (positive prediction rate) per group
- Demographic parity difference (max group rate - min group rate)
- Disparate Impact Ratio (min rate / max rate; should be >= 0.8 per 4/5ths rule)

Usage
-----
    from utils.fairness_audit import run_fairness_audit
    report = run_fairness_audit(df, label_col="TARGET", protected_cols=["CODE_GENDER", "age_band"])
"""
from __future__ import annotations

import pandas as pd
import numpy as np

from utils.logger import get_logger

log = get_logger(__name__)


def run_fairness_audit(
    df: pd.DataFrame,
    label_col: str,
    protected_cols: list[str],
    parity_tolerance: float = 0.05,
) -> pd.DataFrame:
    """Compute demographic parity metrics for each protected attribute.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing ground-truth labels and protected attributes.
    label_col : str
        Binary target column (1 = default, 0 = non-default).
    protected_cols : list[str]
        List of column names to audit (e.g. ['CODE_GENDER', 'age_band']).
    parity_tolerance : float
        Maximum allowed demographic parity difference before flagging a warning.

    Returns
    -------
    pd.DataFrame
        One row per (attribute, group) with default_rate, parity_diff,
        disparate_impact_ratio, and a pass/fail flag.
    """
    records = []

    for attr in protected_cols:
        if attr not in df.columns:
            log.warning("Protected attribute '%s' not found in DataFrame — skipping.", attr)
            continue

        group_rates = (
            df.groupby(attr)[label_col]
            .agg(default_rate="mean", n_applicants="count")
            .reset_index()
        )
        group_rates.rename(columns={attr: "group_value"}, inplace=True)
        group_rates["attribute"] = attr

        max_rate = group_rates["default_rate"].max()
        min_rate = group_rates["default_rate"].min()
        parity_diff = max_rate - min_rate
        disparate_impact = min_rate / max_rate if max_rate > 0 else np.nan

        group_rates["parity_diff"] = parity_diff
        group_rates["disparate_impact_ratio"] = disparate_impact
        group_rates["passes_parity"] = parity_diff <= parity_tolerance
        group_rates["passes_4_5ths_rule"] = disparate_impact >= 0.8 if not np.isnan(disparate_impact) else False

        records.append(group_rates)

        status = "PASS" if parity_diff <= parity_tolerance else "FAIL"
        log.info(
            "Fairness [%s]: parity_diff=%.4f | disparate_impact=%.4f | %s",
            attr, parity_diff, disparate_impact if not np.isnan(disparate_impact) else -1, status,
        )

    if not records:
        return pd.DataFrame()

    report = pd.concat(records, ignore_index=True)
    report = report[["attribute", "group_value", "n_applicants", "default_rate",
                      "parity_diff", "disparate_impact_ratio",
                      "passes_parity", "passes_4_5ths_rule"]]
    return report


def add_age_band(df: pd.DataFrame, days_birth_col: str = "DAYS_BIRTH") -> pd.DataFrame:
    """Derive an age_band column from DAYS_BIRTH (negative in Home Credit data)."""
    age_years = np.abs(df[days_birth_col]) / 365.25
    bins = [0, 30, 50, 200]
    labels = ["<30", "30-50", ">50"]
    df = df.copy()
    df["age_band"] = pd.cut(age_years, bins=bins, labels=labels, right=False)
    return df
