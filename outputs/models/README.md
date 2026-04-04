Trained model artefacts are stored here after running the training pipeline.

Files:
- xgboost_default_model.pkl  — Calibrated XGBoost classifier
- feature_scaler.pkl         — StandardScaler fitted on training data
- feature_columns.json       — Ordered list of feature columns

These artefacts are also uploaded to HuggingFace Hub for Spaces deployment.
Download command: python src/app/hf_loader.py
