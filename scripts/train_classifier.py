from __future__ import annotations

import glob
import json
import os
import pickle
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from xgboost import XGBClassifier


def _load_feature_frames(data_dir: Path) -> pd.DataFrame:
    feat_dir = data_dir / "features"
    files = sorted(glob.glob(str(feat_dir / "*.parquet")))
    frames = []
    for fp in files:
        try:
            frames.append(pd.read_parquet(fp))
        except Exception:
            continue
    if frames:
        df = pd.concat(frames, ignore_index=True)
        return df
    return pd.DataFrame()


def _synthesize_dataset(n: int = 1000) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    rsi14 = rng.uniform(10, 90, size=n)
    foreignBuyStreak = rng.integers(-5, 6, size=n)
    volRatio = rng.uniform(0.5, 2.0, size=n)
    # latent score: mean-reversion on RSI + positive streak
    latent = (50 - rsi14) * 0.02 + np.maximum(foreignBuyStreak, 0) * 0.1 + (volRatio - 1.0) * 0.2
    noise = rng.normal(0, 0.5, size=n)
    y = (latent + noise > 0).astype(int)
    df = pd.DataFrame({
        "rsi14": rsi14,
        "foreignBuyStreak": foreignBuyStreak,
        "volRatio": volRatio,
        "ret_h": y,
    })
    return df


def main():
    root = Path(__file__).resolve().parents[1]
    data_dir = root / "data"
    models_dir = root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    df = _load_feature_frames(data_dir)

    if df.empty:
        # Create synthetic dataset if no features found
        df = _synthesize_dataset(2000)
    else:
        # If real features exist but lack label, create a heuristic label
        if "ret_h" not in df.columns:
            # Build a weak label from a few known columns if present
            rsi = pd.to_numeric(df.get("rsi14", 50), errors="coerce").fillna(50).astype(float)
            fbs = pd.to_numeric(df.get("foreignBuyStreak", 0), errors="coerce").fillna(0).astype(float)
            vol = pd.to_numeric(df.get("volRatio", 1.0), errors="coerce").fillna(1.0).astype(float)
            latent = (50 - rsi) * 0.02 + np.maximum(fbs, 0) * 0.1 + (vol - 1.0) * 0.2
            noise = np.random.normal(0, 0.5, size=len(df))
            df["ret_h"] = ((latent + noise) > 0).astype(int)

    # Select feature columns: all numeric except label
    num_df = df.select_dtypes(include=[np.number]).copy()
    if "ret_h" not in num_df.columns:
        raise SystemExit("No label 'ret_h' found or generated.")

    y = num_df["ret_h"].astype(int).values
    X_df = num_df.drop(columns=["ret_h"])  # features

    # Fallback to at least a couple of known features
    if X_df.empty:
        X_df = df[[c for c in ["rsi14", "foreignBuyStreak"] if c in df.columns]].copy()
        X_df = X_df.fillna(0.0)

    feature_names: List[str] = list(X_df.columns)
    X = X_df.values

    # Train a simple XGBoost classifier
    model = XGBClassifier(
        n_estimators=150,
        max_depth=3,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="logloss",
        n_jobs=0,
        random_state=42,
        verbosity=0,
    )
    model.fit(X, y)

    # Save model and feature names
    with open(models_dir / "model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open(models_dir / "feature_names.json", "w", encoding="utf-8") as f:
        json.dump(feature_names, f)

    print(f"Saved model to {models_dir / 'model.pkl'} with features: {feature_names}")


if __name__ == "__main__":
    main()
