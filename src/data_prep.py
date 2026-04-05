"""
src/data_prep.py
----------------
Data preparation pipeline:
1. Validates presence of all required raw CSV files
2. Runs basic data quality checks (row counts, missing %, target distribution)
3. Merges all Home Credit tables into a master feature table
4. Saves processed output to data/processed/

Usage
-----
    python src/data_prep.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from features import (
    build_application_features,
    build_bureau_features,
    build_previous_app_features,
    build_installment_features,
    build_cc_features,
    build_pos_features,
)
from utils.config import load_config
from utils.logger import get_logger

log = get_logger(__name__)

REQUIRED_FILES = [
    "application_train.csv",
    "bureau.csv",
    "bureau_balance.csv",
    "previous_application.csv",
    "installments_payments.csv",
    "credit_card_balance.csv",
    "POS_CASH_balance.csv",
]


def validate_raw_data(raw_dir: Path, cfg) -> bool:
    """Check all required files exist and meet minimum row thresholds."""
    all_ok = True
    for fname in REQUIRED_FILES:
        fpath = raw_dir / fname
        if not fpath.exists():
            log.error("Missing required file: %s", fpath)
            all_ok = False

    if not all_ok:
        return False

    app = pd.read_csv(raw_dir / "application_train.csv", nrows=5)
    row_count_proxy = sum(1 for _ in open(raw_dir / "application_train.csv")) - 1
    min_rows = cfg.data_validation.min_rows
    if row_count_proxy < min_rows:
        log.error("application_train.csv has only %d rows (min: %d)", row_count_proxy, min_rows)
        return False

    log.info("Data validation passed.")
    return True


def run_data_prep(config_path: str | None = None) -> None:
    cfg = load_config(config_path)
    raw_dir = Path(cfg.data.raw_dir)
    processed_dir = Path(cfg.data.processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    log.info("=== Data Preparation Pipeline ===")

    if not validate_raw_data(raw_dir, cfg):
        log.error("Data validation failed. Download the Home Credit dataset from Kaggle first.")
        log.error("kaggle competitions download -c home-credit-default-risk -p data/raw/")
        sys.exit(1)

    # --- Load ---
    log.info("Loading raw tables ...")
    app = pd.read_csv(raw_dir / "application_train.csv")
    bureau = pd.read_csv(raw_dir / "bureau.csv")
    bureau_bal = pd.read_csv(raw_dir / "bureau_balance.csv")
    prev = pd.read_csv(raw_dir / "previous_application.csv")
    inst = pd.read_csv(raw_dir / "installments_payments.csv")
    cc = pd.read_csv(raw_dir / "credit_card_balance.csv")
    pos = pd.read_csv(raw_dir / "POS_CASH_balance.csv")

    log.info("application_train: %s | default_rate=%.2f%%", app.shape, app["TARGET"].mean() * 100)

    # --- Feature engineering ---
    app_feats = build_application_features(app)
    bureau_feats = build_bureau_features(bureau, bureau_bal)
    prev_feats = build_previous_app_features(prev)
    inst_feats = build_installment_features(inst)
    cc_feats = build_cc_features(cc)
    pos_feats = build_pos_features(pos)

    # --- Merge ---
    master = (
        app_feats
        .merge(bureau_feats, on="SK_ID_CURR", how="left")
        .merge(prev_feats, on="SK_ID_CURR", how="left")
        .merge(inst_feats, on="SK_ID_CURR", how="left")
        .merge(cc_feats, on="SK_ID_CURR", how="left")
        .merge(pos_feats, on="SK_ID_CURR", how="left")
    )

    log.info("Master table shape: %s", master.shape)

    # --- Save ---
    out_path = processed_dir / "master_features.csv"
    master.to_csv(out_path, index=False)
    log.info("Master features saved to %s", out_path)

    # Also save to feature_cache for notebook / training pipeline consistency
    cache_dir = Path(cfg.training.feature_cache_dir) if hasattr(cfg, "training") and hasattr(cfg.training, "feature_cache_dir") else processed_dir / "feature_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "master_features.csv"
    master.to_csv(cache_path, index=False)
    log.info("Master features also cached to %s", cache_path)

    # --- Summary stats ---
    summary = pd.DataFrame({
        "column": master.columns,
        "dtype": master.dtypes.values,
        "missing_pct": (master.isnull().mean() * 100).values,
        "nunique": master.nunique().values,
    })
    summary.to_csv(processed_dir / "data_summary.csv", index=False)
    log.info("Data summary saved to %s/data_summary.csv", processed_dir)
    log.info("=== Data preparation complete ===")


if __name__ == "__main__":
    run_data_prep()
