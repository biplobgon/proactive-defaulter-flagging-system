"""
src/create_sample.py
---------------------
Creates a stratified sample of the master feature table for fast iteration,
notebook demos, and HuggingFace Spaces inference testing.

Outputs
-------
data/processed/sample_features.csv  — stratified 5K row sample
data/processed/sample_for_demo.csv  — 100 rows for live demo (no TARGET column)

Usage
-----
    python src/create_sample.py
    python src/create_sample.py --n 10000
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils.config import load_config
from utils.logger import get_logger

log = get_logger(__name__)


def create_sample(n_rows: int = 5000, config_path: str | None = None) -> None:
    cfg = load_config(config_path)
    processed_dir = Path(cfg.data.processed_dir)
    master_path = processed_dir / "master_features.csv"

    if not master_path.exists():
        log.error("master_features.csv not found. Run data_prep.py first.")
        sys.exit(1)

    log.info("Loading master features ...")
    full = pd.read_csv(master_path)

    if "TARGET" in full.columns:
        # Stratified sample to preserve class balance
        pos = full[full["TARGET"] == 1]
        neg = full[full["TARGET"] == 0]
        pos_rate = len(pos) / len(full)
        n_pos = int(n_rows * pos_rate)
        n_neg = n_rows - n_pos

        sample = pd.concat([
            pos.sample(min(n_pos, len(pos)), random_state=42),
            neg.sample(min(n_neg, len(neg)), random_state=42),
        ]).sample(frac=1, random_state=42).reset_index(drop=True)

        log.info(
            "Sample: %d rows | default_rate=%.2f%%",
            len(sample), sample["TARGET"].mean() * 100,
        )
    else:
        sample = full.sample(min(n_rows, len(full)), random_state=42).reset_index(drop=True)
        log.info("Sample (no target): %d rows", len(sample))

    sample_path = processed_dir / "sample_features.csv"
    sample.to_csv(sample_path, index=False)
    log.info("Sample saved to %s", sample_path)

    # Demo sample (no TARGET, for live scoring UI)
    demo_cols = [c for c in sample.columns if c not in ("TARGET", "SK_ID_CURR")]
    demo = sample[demo_cols].head(100)
    demo_path = processed_dir / "sample_for_demo.csv"
    demo.to_csv(demo_path, index=False)
    log.info("Demo sample (100 rows, no target) saved to %s", demo_path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a stratified sample for notebooks/demo.")
    parser.add_argument("--n", type=int, default=5000, help="Sample size (default: 5000)")
    parser.add_argument("--config", type=str, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    create_sample(n_rows=args.n, config_path=args.config)
