"""
features/applicant_features.py
-------------------------------
Derives applicant-level features from:
- application_train / application_test (main table)
- bureau.csv + bureau_balance.csv (credit bureau history)
- previous_application.csv (prior Home Credit loans)

Feature families:
- Income/loan ratios and stability indicators
- Bureau delinquency history and inquiry recency
- Previous application approval/rejection patterns
- Weight of Evidence (WOE) encoding for key categoricals

Usage
-----
    from features.applicant_features import (
        build_application_features,
        build_bureau_features,
        build_previous_app_features,
    )
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from utils.logger import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Application table features
# ---------------------------------------------------------------------------

def build_application_features(app: pd.DataFrame) -> pd.DataFrame:
    """Engineer features from the main application table.

    Parameters
    ----------
    app : pd.DataFrame
        application_train.csv or application_test.csv.

    Returns
    -------
    pd.DataFrame
        app with additional engineered columns.
    """
    log.info("Building application features for %d rows ...", len(app))
    df = app.copy()

    # --- Ratio features ---
    df["credit_income_ratio"] = df["AMT_CREDIT"] / df["AMT_INCOME_TOTAL"].replace(0, np.nan)
    df["annuity_income_ratio"] = df["AMT_ANNUITY"] / df["AMT_INCOME_TOTAL"].replace(0, np.nan)
    df["credit_goods_ratio"] = df["AMT_CREDIT"] / df["AMT_GOODS_PRICE"].replace(0, np.nan)
    df["income_per_person"] = df["AMT_INCOME_TOTAL"] / (df["CNT_FAM_MEMBERS"].replace(0, np.nan))

    # --- Age (DAYS_BIRTH is negative in this dataset) ---
    df["age_years"] = np.abs(df["DAYS_BIRTH"]) / 365.25
    df["employed_years"] = np.where(
        df["DAYS_EMPLOYED"] > 0, 0, np.abs(df["DAYS_EMPLOYED"]) / 365.25
    )
    df["employment_ratio"] = df["employed_years"] / df["age_years"].replace(0, np.nan)

    # --- Document submission flags (count of provided docs) ---
    doc_cols = [c for c in df.columns if c.startswith("FLAG_DOCUMENT_")]
    df["n_docs_provided"] = df[doc_cols].sum(axis=1)

    # --- External source scores (credit bureau proxies) ---
    df["ext_source_mean"] = df[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]].mean(axis=1)
    df["ext_source_min"] = df[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]].min(axis=1)

    # --- Contact reachability (more contact info = lower risk) ---
    contact_cols = [c for c in df.columns if c.startswith("FLAG_") and "CONTACT" in c]
    if contact_cols:
        df["n_contact_methods"] = df[contact_cols].sum(axis=1)

    log.info("Application features built. Final shape: %s", df.shape)
    return df


# ---------------------------------------------------------------------------
# Bureau features
# ---------------------------------------------------------------------------

def build_bureau_features(bureau: pd.DataFrame, bureau_balance: pd.DataFrame) -> pd.DataFrame:
    """Aggregate credit bureau history per SK_ID_CURR.

    Parameters
    ----------
    bureau : pd.DataFrame
        bureau.csv
    bureau_balance : pd.DataFrame
        bureau_balance.csv

    Returns
    -------
    pd.DataFrame
        One row per SK_ID_CURR with bureau-derived features.
    """
    log.info("Building bureau features (%d bureau rows, %d balance rows) ...",
             len(bureau), len(bureau_balance))

    # --- Bureau balance: worst status per bureau loan ---
    # Status codes: 0=no DPD, 1=1-30 DPD, ... C=closed, X=unknown
    bb = bureau_balance.copy()
    # Encode numeric DPD
    status_map = {"C": 0, "X": 0, "0": 0, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5}
    bb["status_num"] = bb["STATUS"].map(status_map).fillna(0)
    bb_agg = bb.groupby("SK_ID_BUREAU").agg(
        bb_months_count=("MONTHS_BALANCE", "count"),
        bb_dpd_mean=("status_num", "mean"),
        bb_dpd_max=("status_num", "max"),
    ).reset_index()

    bdf = bureau.merge(bb_agg, on="SK_ID_BUREAU", how="left")

    # --- Aggregate per applicant ---
    agg = bdf.groupby("SK_ID_CURR").agg(
        bureau_n_loans=("SK_ID_BUREAU", "count"),
        bureau_active_loans=("CREDIT_ACTIVE", lambda x: (x == "Active").sum()),
        bureau_closed_loans=("CREDIT_ACTIVE", lambda x: (x == "Closed").sum()),
        bureau_overdue_mean=("AMT_CREDIT_MAX_OVERDUE", "mean"),
        bureau_overdue_max=("AMT_CREDIT_MAX_OVERDUE", "max"),
        bureau_dpd_mean=("CREDIT_DAY_OVERDUE", "mean"),
        bureau_dpd_max=("CREDIT_DAY_OVERDUE", "max"),
        bureau_credit_sum=("AMT_CREDIT_SUM", "sum"),
        bureau_debt_sum=("AMT_CREDIT_SUM_DEBT", "sum"),
        bureau_enquiry_count=("AMT_REQ_CREDIT_BUREAU_YEAR", "sum"),
        bureau_bb_dpd_mean=("bb_dpd_mean", "mean"),
        bureau_bb_dpd_max=("bb_dpd_max", "max"),
    ).reset_index()

    agg["bureau_debt_ratio"] = agg["bureau_debt_sum"] / agg["bureau_credit_sum"].replace(0, np.nan)

    log.info("Bureau features shape: %s", agg.shape)
    return agg


# ---------------------------------------------------------------------------
# Previous application features
# ---------------------------------------------------------------------------

def build_previous_app_features(prev: pd.DataFrame) -> pd.DataFrame:
    """Aggregate previous Home Credit application history per SK_ID_CURR.

    Parameters
    ----------
    prev : pd.DataFrame
        previous_application.csv

    Returns
    -------
    pd.DataFrame
        One row per SK_ID_CURR with previous-application features.
    """
    log.info("Building previous application features for %d records ...", len(prev))

    df = prev.copy()
    df["was_approved"] = (df["NAME_CONTRACT_STATUS"] == "Approved").astype(int)
    df["was_refused"] = (df["NAME_CONTRACT_STATUS"] == "Refused").astype(int)
    df["credit_ratio_prev"] = df["AMT_APPLICATION"] / df["AMT_CREDIT"].replace(0, np.nan)

    agg = df.groupby("SK_ID_CURR").agg(
        prev_n_applications=("SK_ID_PREV", "count"),
        prev_approved_count=("was_approved", "sum"),
        prev_refused_count=("was_refused", "sum"),
        prev_approval_rate=("was_approved", "mean"),
        prev_credit_sum=("AMT_CREDIT", "sum"),
        prev_annuity_mean=("AMT_ANNUITY", "mean"),
        prev_credit_ratio_mean=("credit_ratio_prev", "mean"),
        prev_days_last_due_mean=("DAYS_LAST_DUE", "mean"),
    ).reset_index()

    log.info("Previous application features shape: %s", agg.shape)
    return agg
