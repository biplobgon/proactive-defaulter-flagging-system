"""
src/app/hf_loader.py
---------------------
HuggingFace Hub model loader.

Downloads trained model artefacts from a HuggingFace Hub repository
into outputs/models/ at startup when running on HuggingFace Spaces.

Usage
-----
Called automatically by app.py when artefacts are not found locally.
"""
from __future__ import annotations

import os
from pathlib import Path

from utils.logger import get_logger

log = get_logger(__name__)

HF_REPO_ID = os.getenv("HF_MODEL_REPO", "biplobgon/proactive-defaulter-flagging-system")
MODEL_FILES = [
    "xgboost_default_model.pkl",
    "feature_scaler.pkl",
    "feature_columns.json",
]


def download_artifacts(output_dir: str | Path = "outputs/models") -> None:
    """Download model artefacts from HuggingFace Hub if not already present."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    missing = [f for f in MODEL_FILES if not (output_dir / f).exists()]
    if not missing:
        log.info("All model artefacts already present — skipping download.")
        return

    log.info("Downloading %d artefacts from HF Hub: %s", len(missing), HF_REPO_ID)
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise ImportError("Install huggingface_hub: pip install huggingface-hub") from exc

    for fname in missing:
        local_path = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=f"models/{fname}",
            local_dir=str(output_dir.parent),
        )
        log.info("Downloaded: %s → %s", fname, local_path)
