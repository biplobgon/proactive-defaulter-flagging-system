"""
models/survival_model.py
------------------------
Kaplan-Meier survival analysis for time-to-default visualisation.

This module provides:
- Kaplan-Meier curves per risk band and borrower segment
- Log-rank test to verify statistical separation between bands
- Median survival time estimation

Note: This module is for visual storytelling / EDA purposes.
      The XGBoost classifier drives the production Traffic Light scoring.

Usage
-----
    from models.survival_model import KaplanMeierAnalyser
    km = KaplanMeierAnalyser(df, duration_col="loan_months", event_col="TARGET")
    km.fit_by_group("risk_band")
    fig = km.plot()
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.logger import get_logger

log = get_logger(__name__)


class KaplanMeierAnalyser:
    """Kaplan-Meier survival estimator with group comparison."""

    def __init__(
        self,
        df: pd.DataFrame,
        duration_col: str,
        event_col: str,
    ) -> None:
        """
        Parameters
        ----------
        df : pd.DataFrame
            One row per borrower.
        duration_col : str
            Column with time-to-event (months). For non-defaulters this should
            be the loan tenure (i.e. they were observed until end of loan).
        event_col : str
            Binary column: 1 = defaulted, 0 = censored (did not default).
        """
        self.df = df.copy()
        self.duration_col = duration_col
        self.event_col = event_col
        self._groups: dict[str, dict] = {}

    def fit_by_group(self, group_col: str) -> "KaplanMeierAnalyser":
        """Fit a KM curve for each unique value in group_col."""
        log.info("Fitting Kaplan-Meier curves by '%s' ...", group_col)
        self._group_col = group_col
        for group_val, grp in self.df.groupby(group_col):
            km = _km_estimate(grp[self.duration_col].values, grp[self.event_col].values)
            self._groups[str(group_val)] = km
        return self

    def median_survival(self) -> pd.DataFrame:
        """Return median survival time (months) per group."""
        records = []
        for name, km in self._groups.items():
            # First time survival drops below 0.5
            idx = np.searchsorted(-km["survival"], -0.5)
            median = km["timeline"][idx] if idx < len(km["timeline"]) else np.nan
            records.append({"group": name, "median_survival_months": median})
        return pd.DataFrame(records)

    def plot(self, figsize: tuple[int, int] = (10, 6)) -> plt.Figure:
        """Plot KM survival curves for all groups."""
        if not self._groups:
            raise RuntimeError("Call fit_by_group() before plot().")

        fig, ax = plt.subplots(figsize=figsize)

        color_map = {"GREEN": "#2ecc71", "YELLOW": "#f39c12", "RED": "#e74c3c"}

        for name, km in self._groups.items():
            color = color_map.get(name, None)
            ax.step(
                km["timeline"],
                km["survival"],
                where="post",
                label=f"{name} (n={km['n_at_start']:,})",
                color=color,
                linewidth=2,
            )

        ax.set_xlabel("Months since loan origination", fontsize=12)
        ax.set_ylabel("Survival probability (no default)", fontsize=12)
        ax.set_title(
            f"Kaplan-Meier Survival Curves by {self._group_col}",
            fontsize=14, fontweight="bold"
        )
        ax.legend(title=self._group_col, fontsize=10)
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.3)
        ax.axhline(0.5, color="grey", linestyle="--", alpha=0.6, label="50% survival")

        fig.tight_layout()
        return fig

    def log_rank_pvalue(self, group_a: str, group_b: str) -> float:
        """Compute log-rank p-value between two groups (Mantel-Haenszel)."""
        try:
            from lifelines.statistics import logrank_test
        except ImportError:
            # Minimal implementation without lifelines
            return _log_rank_simple(
                self.df, self.duration_col, self.event_col, self._group_col, group_a, group_b
            )
        grp_a = self.df[self.df[self._group_col] == group_a]
        grp_b = self.df[self.df[self._group_col] == group_b]
        result = logrank_test(
            grp_a[self.duration_col], grp_b[self.duration_col],
            grp_a[self.event_col], grp_b[self.event_col],
        )
        return result.p_value


# ---------------------------------------------------------------------------
# Internal KM estimation (no lifelines required)
# ---------------------------------------------------------------------------

def _km_estimate(durations: np.ndarray, events: np.ndarray) -> dict:
    """Compute Kaplan-Meier estimate from raw duration/event arrays."""
    order = np.argsort(durations)
    durations = durations[order]
    events = events[order]

    unique_times = np.unique(durations)
    n = len(durations)
    survival = 1.0
    timeline = [0]
    survival_curve = [1.0]
    n_at_start = n

    for t in unique_times:
        at_risk = np.sum(durations >= t)
        events_at_t = np.sum((durations == t) & (events == 1))
        if at_risk > 0 and events_at_t > 0:
            survival *= 1 - events_at_t / at_risk
        timeline.append(t)
        survival_curve.append(survival)

    return {
        "timeline": np.array(timeline),
        "survival": np.array(survival_curve),
        "n_at_start": n_at_start,
    }


def _log_rank_simple(df, duration_col, event_col, group_col, group_a, group_b) -> float:
    """Minimal log-rank test returning approximate p-value."""
    from scipy import stats

    a = df[df[group_col] == group_a]
    b = df[df[group_col] == group_b]
    # Use observed vs expected events (chi-squared approximation)
    all_times = np.unique(
        np.concatenate([a[duration_col].values, b[duration_col].values])
    )
    o_a, o_b, e_a, e_b = 0.0, 0.0, 0.0, 0.0

    for t in all_times:
        n_a = np.sum(a[duration_col] >= t)
        n_b = np.sum(b[duration_col] >= t)
        d_a = np.sum((a[duration_col] == t) & (a[event_col] == 1))
        d_b = np.sum((b[duration_col] == t) & (b[event_col] == 1))
        n_total = n_a + n_b
        d_total = d_a + d_b
        if n_total == 0:
            continue
        o_a += d_a
        o_b += d_b
        e_a += n_a * d_total / n_total
        e_b += n_b * d_total / n_total

    chi2 = (o_a - e_a) ** 2 / e_a + (o_b - e_b) ** 2 / e_b if e_a > 0 and e_b > 0 else 0.0
    return float(stats.chi2.sf(chi2, df=1))


def derive_loan_duration(app: pd.DataFrame) -> pd.DataFrame:
    """Derive a synthetic loan_months column from application data.

    DAYS_CREDIT or DAYS_TERMINATION fields are used when available.
    Falls back to AMT_CREDIT / AMT_ANNUITY * 12 (estimated months).
    """
    df = app.copy()
    if "DAYS_TERMINATION" in df.columns:
        df["loan_months"] = np.abs(df["DAYS_TERMINATION"]) / 30.5
    elif "AMT_CREDIT" in df.columns and "AMT_ANNUITY" in df.columns:
        df["loan_months"] = (
            df["AMT_CREDIT"] / df["AMT_ANNUITY"].replace(0, np.nan)
        ).clip(1, 360)
    else:
        df["loan_months"] = 12  # fallback
    return df
