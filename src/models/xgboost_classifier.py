"""
models/xgboost_classifier.py
-----------------------------
XGBoost-based binary default classifier.

Features:
- Early stopping with held-out eval set
- SHAP value computation
- Calibrated probability output (Platt scaling)
- Traffic Light scoring (Red / Yellow / Green)
- Model persistence (joblib)

Usage
-----
    from models.xgboost_classifier import DefaultClassifier
    clf = DefaultClassifier(cfg)
    clf.fit(X_train, y_train, X_val, y_val)
    results = clf.predict_traffic_light(X_test)
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from utils.logger import get_logger

log = get_logger(__name__)

# Traffic Light band labels
RED = "RED"
YELLOW = "YELLOW"
GREEN = "GREEN"


class DefaultClassifier:
    """XGBoost default classifier with Traffic Light risk scoring."""

    def __init__(self, cfg) -> None:
        self.cfg = cfg
        xgb_params = cfg.xgboost.to_dict() if hasattr(cfg.xgboost, "to_dict") else {}
        # Remove keys not accepted by XGBClassifier constructor
        for key in ("early_stopping_rounds", "use_label_encoder"):
            xgb_params.pop(key, None)

        self.model = XGBClassifier(
            **xgb_params,
            random_state=cfg.seed,
            n_jobs=-1,
        )
        self.calibrated_model: LogisticRegression | None = None  # Platt scaler
        self.scaler = StandardScaler()
        self.feature_columns: list[str] = []

        tl = cfg.traffic_light
        self.red_threshold: float = tl.red_threshold
        self.green_threshold: float = tl.green_threshold

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        calibrate: bool = True,
    ) -> "DefaultClassifier":
        """Fit the XGBoost model with early stopping and optional calibration."""
        self.feature_columns = list(X_train.columns)

        log.info("Fitting XGBoost on %d train samples, %d val samples ...",
                 len(X_train), len(X_val))

        early_stopping = getattr(self.cfg.xgboost, "early_stopping_rounds", 50)
        # XGBoost ≥2.0 requires early_stopping_rounds on the estimator, not fit()
        if not getattr(self.model, 'early_stopping_rounds', None):
            self.model.set_params(early_stopping_rounds=early_stopping)
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        log.info("Best iteration: %d", self.model.best_iteration)

        if calibrate:
            log.info("Calibrating probabilities with Platt scaling ...")
            # Manual Platt scaling: sigmoid LogisticRegression on val probabilities
            raw_val = self.model.predict_proba(X_val)[:, 1].reshape(-1, 1)
            self.calibrated_model = LogisticRegression()
            self.calibrated_model.fit(raw_val, y_val)

        return self

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return calibrated default probabilities (P(default=1))."""
        raw = self.model.predict_proba(X[self.feature_columns])[:, 1]
        if self.calibrated_model is not None:
            raw = self.calibrated_model.predict_proba(raw.reshape(-1, 1))[:, 1]
        return raw

    def predict_traffic_light(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return a DataFrame with risk_score, risk_band for each row.

        risk_score: probability of default (0-1), higher = more risky
        risk_band : RED | YELLOW | GREEN
        """
        proba = self.predict_proba(X)
        bands = np.where(
            proba >= self.red_threshold,
            RED,
            np.where(proba <= self.green_threshold, GREEN, YELLOW),
        )
        return pd.DataFrame(
            {"risk_score": proba, "risk_band": bands},
            index=X.index,
        )

    # ------------------------------------------------------------------
    # SHAP
    # ------------------------------------------------------------------

    def shap_values(self, X: pd.DataFrame) -> np.ndarray:
        """Compute SHAP values using the underlying XGBoost tree explainer."""
        try:
            import shap
        except ImportError as exc:
            raise ImportError("Install shap: pip install shap") from exc

        explainer = shap.TreeExplainer(self.model)
        return explainer.shap_values(X[self.feature_columns])

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, output_dir: str | Path) -> None:
        """Save model artefacts to output_dir."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        joblib.dump(self.model, output_dir / "xgboost_default_model.pkl")
        if self.calibrated_model is not None:
            joblib.dump(self.calibrated_model, output_dir / "platt_scaler.pkl")
        joblib.dump(self.scaler, output_dir / "feature_scaler.pkl")
        with open(output_dir / "feature_columns.json", "w") as fh:
            json.dump(self.feature_columns, fh)

        log.info("Model artefacts saved to %s", output_dir)

    @classmethod
    def load(cls, output_dir: str | Path, cfg) -> "DefaultClassifier":
        """Load a previously saved DefaultClassifier."""
        output_dir = Path(output_dir)
        instance = cls.__new__(cls)
        instance.cfg = cfg
        instance.model = joblib.load(output_dir / "xgboost_default_model.pkl")
        platt_path = output_dir / "platt_scaler.pkl"
        instance.calibrated_model = joblib.load(platt_path) if platt_path.exists() else None
        instance.scaler = joblib.load(output_dir / "feature_scaler.pkl")
        with open(output_dir / "feature_columns.json") as fh:
            instance.feature_columns = json.load(fh)

        tl = cfg.traffic_light
        instance.red_threshold = tl.red_threshold
        instance.green_threshold = tl.green_threshold

        log.info("Model loaded from %s", output_dir)
        return instance
