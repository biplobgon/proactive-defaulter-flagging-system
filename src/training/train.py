"""
training/train.py
-----------------
Full training pipeline orchestrator.

Steps:
1. Load config
2. Load & merge all Home Credit tables
3. Feature engineering (behavioral + applicant)
4. Time-aware train/validation/test split
5. Train XGBoost classifier
6. Calibrate thresholds (Traffic Light)
7. Evaluate
8. Save artefacts

Usage
-----
    python src/training/train.py
    python src/training/train.py --config configs/model_config.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Ensure src/ is on the path when run as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from features import (
    build_application_features,
    build_bureau_features,
    build_previous_app_features,
    build_installment_features,
    build_cc_features,
    build_pos_features,
)
from models import DefaultClassifier
from training.evaluate import evaluate_model
from training.threshold_optimizer import optimise_thresholds
from utils.config import load_config
from utils.logger import get_logger

log = get_logger(__name__)


def run_pipeline(config_path: str | None = None) -> None:
    cfg = load_config(config_path)
    log.info("=== Proactive Defaulter Flagging — Training Pipeline ===")

    # ------------------------------------------------------------------
    # 1. Load raw tables
    # ------------------------------------------------------------------
    log.info("Loading raw data ...")
    raw = cfg.data
    app = pd.read_csv(raw.application_train)
    bureau = pd.read_csv(raw.bureau)
    bureau_bal = pd.read_csv(raw.bureau_balance)
    prev = pd.read_csv(raw.previous_application)
    inst = pd.read_csv(raw.installments_payments)
    cc = pd.read_csv(raw.credit_card_balance)
    pos = pd.read_csv(raw.pos_cash_balance)

    log.info("Application train shape: %s | Default rate: %.2f%%",
             app.shape, app["TARGET"].mean() * 100)

    # ------------------------------------------------------------------
    # 2. Feature engineering
    # ------------------------------------------------------------------
    log.info("Engineering features ...")
    app_feats = build_application_features(app)
    bureau_feats = build_bureau_features(bureau, bureau_bal)
    prev_feats = build_previous_app_features(prev)
    inst_feats = build_installment_features(inst)
    cc_feats = build_cc_features(cc)
    pos_feats = build_pos_features(pos)

    master = (
        app_feats
        .merge(bureau_feats, on="SK_ID_CURR", how="left")
        .merge(prev_feats, on="SK_ID_CURR", how="left")
        .merge(inst_feats, on="SK_ID_CURR", how="left")
        .merge(cc_feats, on="SK_ID_CURR", how="left")
        .merge(pos_feats, on="SK_ID_CURR", how="left")
    )
    log.info("Master table shape after merges: %s", master.shape)

    # ------------------------------------------------------------------
    # 3. Prepare feature matrix
    # ------------------------------------------------------------------
    target_col = "TARGET"
    drop_cols = [target_col, "SK_ID_CURR", "SK_ID_BUREAU", "SK_ID_PREV"]
    drop_cols = [c for c in drop_cols if c in master.columns]

    # Drop high-cardinality / low-signal object columns
    obj_cols = master.select_dtypes("object").columns.tolist()
    log.info("Dropping %d object columns: %s ...", len(obj_cols), obj_cols[:5])
    master = master.drop(columns=obj_cols)

    # Drop columns with > 50% missing
    max_missing = cfg.data_validation.max_missing_pct / 100
    missing_rate = master.isnull().mean()
    high_missing = missing_rate[missing_rate > max_missing].index.tolist()
    log.info("Dropping %d high-missing columns.", len(high_missing))
    master = master.drop(columns=high_missing)

    X = master.drop(columns=drop_cols, errors="ignore")
    y = master[target_col]

    # Fill remaining NaN with median
    X = X.fillna(X.median(numeric_only=True))

    log.info("Feature matrix: %s | Positive rate: %.2f%%", X.shape, y.mean() * 100)

    # ------------------------------------------------------------------
    # 4. Time-aware split (use index order as a proxy for time)
    # ------------------------------------------------------------------
    test_size = cfg.evaluation.test_size
    split_idx = int(len(X) * (1 - test_size * 2))
    val_idx = int(len(X) * (1 - test_size))

    X_train, y_train = X.iloc[:split_idx], y.iloc[:split_idx]
    X_val, y_val = X.iloc[split_idx:val_idx], y.iloc[split_idx:val_idx]
    X_test, y_test = X.iloc[val_idx:], y.iloc[val_idx:]

    log.info("Train: %d | Val: %d | Test: %d", len(X_train), len(X_val), len(X_test))

    # Save processed feature matrix (re-usable in notebooks)
    processed_dir = Path(cfg.training.feature_cache_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    master.to_csv(processed_dir / "master_features.csv", index=False)
    log.info("Master features cached to %s", processed_dir)

    # ------------------------------------------------------------------
    # 5. Train XGBoost
    # ------------------------------------------------------------------
    clf = DefaultClassifier(cfg)
    clf.fit(X_train, y_train, X_val, y_val)

    # ------------------------------------------------------------------
    # 6. Optimise Traffic Light thresholds
    # ------------------------------------------------------------------
    val_proba = clf.predict_proba(X_val)
    thresholds = optimise_thresholds(
        y_true=y_val.values,
        y_proba=val_proba,
        cost_fp=cfg.traffic_light.cost_false_positive,
        cost_fn=cfg.traffic_light.cost_false_negative,
    )
    clf.red_threshold = thresholds["red_threshold"]
    clf.green_threshold = thresholds["green_threshold"]
    log.info("Optimised thresholds — Red: %.3f | Green: %.3f",
             clf.red_threshold, clf.green_threshold)

    # ------------------------------------------------------------------
    # 7. Evaluate
    # ------------------------------------------------------------------
    report = evaluate_model(clf, X_test, y_test, cfg)
    log.info("Evaluation report:\n%s", report.to_string())

    report_path = Path(cfg.training.report_dir) / "evaluation_report.csv"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(report_path, index=False)

    # ------------------------------------------------------------------
    # 8. Save artefacts
    # ------------------------------------------------------------------
    clf.save(cfg.training.model_dir)
    log.info("=== Training pipeline complete ===")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train proactive defaulter flagging model.")
    parser.add_argument("--config", type=str, default=None, help="Path to a single YAML config.")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_pipeline(args.config)
