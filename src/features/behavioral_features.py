"""
features/behavioral_features.py
--------------------------------
Derives behavioral features from installment payment history and
credit card balance tables.

Key feature families:
- Payment delay statistics (mean/max/std days late across windows)
- Delinquency flags (30/60/90 DPD — Days Past Due)
- Payment velocity (% on-time payments over trailing windows)
- Credit utilisation trajectory (credit_card_balance)

Usage
-----
    from features.behavioral_features import build_installment_features, build_cc_features
    inst_feats = build_installment_features(installments_df)
    cc_feats   = build_cc_features(cc_balance_df)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from utils.logger import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Installment payment features
# ---------------------------------------------------------------------------

def build_installment_features(inst: pd.DataFrame) -> pd.DataFrame:
    """Aggregate installment payment history per SK_ID_CURR.

    Parameters
    ----------
    inst : pd.DataFrame
        Raw installments_payments.csv (Home Credit schema).

    Returns
    -------
    pd.DataFrame
        One row per SK_ID_CURR with behavioral payment features.
    """
    log.info("Building installment payment features for %d records ...", len(inst))

    df = inst.copy()
    # Days paid early (negative) or late (positive) relative to the due date
    df["days_late"] = df["DAYS_PAID_INSTALMENT"] - df["DAYS_INSTALMENT"]
    df["payment_ratio"] = df["AMT_PAYMENT"] / df["AMT_INSTALMENT"].replace(0, np.nan)

    # --- Aggregate across all records per applicant ---
    agg = df.groupby("SK_ID_CURR").agg(
        inst_count=("DAYS_INSTALMENT", "count"),
        inst_days_late_mean=("days_late", "mean"),
        inst_days_late_max=("days_late", "max"),
        inst_days_late_std=("days_late", "std"),
        inst_payment_ratio_mean=("payment_ratio", "mean"),
        inst_payment_ratio_min=("payment_ratio", "min"),
    ).reset_index()

    # --- Delinquency flags ---
    for dpd in [30, 60, 90]:
        late_flag = (df["days_late"] >= dpd).astype(int)
        dpd_agg = (
            df.assign(is_late=late_flag)
            .groupby("SK_ID_CURR")["is_late"]
            .agg(**{f"inst_dpd{dpd}_count": "sum", f"inst_dpd{dpd}_rate": "mean"})
            .reset_index()
        )
        agg = agg.merge(dpd_agg, on="SK_ID_CURR", how="left")

    # --- On-time payment velocity ---
    on_time = (df["days_late"] <= 0).astype(int)
    velocity = (
        df.assign(on_time=on_time)
        .groupby("SK_ID_CURR")["on_time"]
        .mean()
        .reset_index()
        .rename(columns={"on_time": "inst_ontime_rate"})
    )
    agg = agg.merge(velocity, on="SK_ID_CURR", how="left")

    log.info("Installment features shape: %s", agg.shape)
    return agg


# ---------------------------------------------------------------------------
# Credit card balance features
# ---------------------------------------------------------------------------

def build_cc_features(cc: pd.DataFrame) -> pd.DataFrame:
    """Aggregate credit card balance history per SK_ID_CURR.

    Parameters
    ----------
    cc : pd.DataFrame
        Raw credit_card_balance.csv (Home Credit schema).

    Returns
    -------
    pd.DataFrame
        One row per SK_ID_CURR with credit utilisation features.
    """
    log.info("Building credit card balance features for %d records ...", len(cc))

    df = cc.copy()
    df["utilisation"] = (
        df["AMT_BALANCE"] / df["AMT_CREDIT_LIMIT_ACTUAL"].replace(0, np.nan)
    ).clip(upper=1.5)

    # Drawing more than the minimum payment is a risk signal
    df["overspend_flag"] = (
        df["AMT_DRAWINGS_CURRENT"] > df["AMT_RECEIVABLE_PRINCIPAL"]
    ).astype(int)

    agg = df.groupby("SK_ID_CURR").agg(
        cc_count=("MONTHS_BALANCE", "count"),
        cc_utilisation_mean=("utilisation", "mean"),
        cc_utilisation_max=("utilisation", "max"),
        cc_utilisation_std=("utilisation", "std"),
        cc_balance_mean=("AMT_BALANCE", "mean"),
        cc_balance_max=("AMT_BALANCE", "max"),
        cc_overspend_rate=("overspend_flag", "mean"),
        cc_drawings_mean=("AMT_DRAWINGS_CURRENT", "mean"),
    ).reset_index()

    log.info("Credit card features shape: %s", agg.shape)
    return agg


# ---------------------------------------------------------------------------
# POS cash balance features
# ---------------------------------------------------------------------------

def build_pos_features(pos: pd.DataFrame) -> pd.DataFrame:
    """Aggregate POS_CASH_balance history per SK_ID_CURR."""
    log.info("Building POS cash balance features for %d records ...", len(pos))

    df = pos.copy()
    df["is_dpd"] = (df["SK_DPD"] > 0).astype(int)
    df["is_dpd_def"] = (df["SK_DPD_DEF"] > 0).astype(int)

    agg = df.groupby("SK_ID_CURR").agg(
        pos_count=("MONTHS_BALANCE", "count"),
        pos_dpd_mean=("SK_DPD", "mean"),
        pos_dpd_max=("SK_DPD", "max"),
        pos_dpd_rate=("is_dpd", "mean"),
        pos_dpd_def_rate=("is_dpd_def", "mean"),
    ).reset_index()

    log.info("POS features shape: %s", agg.shape)
    return agg
