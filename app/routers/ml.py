from __future__ import annotations

import json
import pickle
from pathlib import Path
from datetime import date as date_cls
from typing import Dict, List, Optional

import numpy as np
from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.core.config import settings

router = APIRouter(prefix="/ml", tags=["ml"]) 


# --------- Request/Response Schemas ---------
class Sample(BaseModel):
    ticker: str
    features: Dict[str, float] = Field(default_factory=dict)


class ScoreRequest(BaseModel):
    asOf: Optional[date_cls] = None
    horizonDays: int = 5
    samples: List[Sample] = Field(default_factory=list)


class Prediction(BaseModel):
    ticker: str
    probUp: float
    expectedRetPct: float
    modelVersion: str = "stub_v1"
    explanations: Dict[str, float] = Field(default_factory=dict)


class ScoreResponse(BaseModel):
    horizonDays: int
    predictions: List[Prediction]


# --------- Optional real model loading ---------
_MODEL = None
_FEATURE_NAMES: List[str] = []
_MODEL_VERSION = "stub_v1"

models_dir = Path("models")
model_path = models_dir / "model.pkl"
feat_path = models_dir / "feature_names.json"
try:
    if model_path.exists() and feat_path.exists():
        with open(model_path, "rb") as f:
            _MODEL = pickle.load(f)
        with open(feat_path, "r", encoding="utf-8") as f:
            names = json.load(f) or []
            if isinstance(names, list):
                _FEATURE_NAMES = [str(x) for x in names]
        if _MODEL is not None and _FEATURE_NAMES:
            _MODEL_VERSION = "xgb_v1"
except Exception:
    # Fail quietly and keep stub
    _MODEL = None
    _FEATURE_NAMES = []
    _MODEL_VERSION = "stub_v1"


# --------- Stub Scoring Logic ---------
_DEF_BASE = 0.6  # base probability


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _score_one_stub(sample: Sample, horizon_days: int) -> Prediction:
    feats = sample.features or {}

    # Base probability
    prob = _DEF_BASE
    explanations: Dict[str, float] = {}

    # Rule 1: rsi14 (0..100), lower RSI -> higher probUp (mean-reversion)
    rsi = feats.get("rsi14")
    if isinstance(rsi, (int, float)):
        # Map RSI deviation from 50 to contribution in [-0.1, +0.1]
        contrib_rsi = _clamp((50.0 - float(rsi)) / 100.0, -0.1, 0.1)
        prob += contrib_rsi
        explanations["rsi14"] = round(contrib_rsi, 6)

    # Rule 2: foreignBuyStreak (positive streak increases prob)
    fbs = feats.get("foreignBuyStreak")
    if isinstance(fbs, (int, float)):
        # Each positive day adds 0.02 up to 5 days -> [0, 0.1]
        pos_streak = max(0.0, float(fbs))
        contrib_fbs = _clamp(pos_streak, 0.0, 5.0) * 0.02
        prob += contrib_fbs
        explanations["foreignBuyStreak"] = round(contrib_fbs, 6)

    # Ensure final prob in [0.5, 0.8]
    prob = _clamp(prob, 0.5, 0.8)

    # Expected return percentage: simple mapping from (prob-0.5) to [0, 3]%
    expected_ret_pct = (prob - 0.5) * 10.0  # 0..3

    return Prediction(
        ticker=sample.ticker,
        probUp=round(prob, 6),
        expectedRetPct=round(expected_ret_pct, 6),
        modelVersion="stub_v1",
        explanations=explanations,
    )


def _score_one_model(sample: Sample, horizon_days: int) -> Prediction:
    # Vectorize according to _FEATURE_NAMES, fill missing with 0.0
    feats = sample.features or {}
    if not _FEATURE_NAMES or _MODEL is None:
        return _score_one_stub(sample, horizon_days)
    x = np.array([[float(feats.get(name, 0.0)) for name in _FEATURE_NAMES]], dtype=float)
    prob = None
    try:
        if hasattr(_MODEL, "predict_proba"):
            prob = float(_MODEL.predict_proba(x)[0, 1])
        else:
            # Fallback: some models output raw score; map via sigmoid
            raw = float(_MODEL.predict(x)[0])
            prob = 1.0 / (1.0 + np.exp(-raw))
    except Exception:
        # Any failure -> fallback to stub
        return _score_one_stub(sample, horizon_days)

    # Clamp to reasonable bounds
    prob = _clamp(prob, 0.01, 0.99)
    expected_ret_pct = (prob - 0.5) * 10.0

    return Prediction(
        ticker=sample.ticker,
        probUp=round(prob, 6),
        expectedRetPct=round(expected_ret_pct, 6),
        modelVersion=_MODEL_VERSION,
        explanations={},
    )


@router.post("/score", response_model=ScoreResponse)
async def score(req: ScoreRequest) -> ScoreResponse:
    # Use real model if available; otherwise fallback to stub.
    if _MODEL is not None and _FEATURE_NAMES:
        preds = [_score_one_model(s, req.horizonDays) for s in (req.samples or [])]
    else:
        preds = [_score_one_stub(s, req.horizonDays) for s in (req.samples or [])]
    return ScoreResponse(horizonDays=req.horizonDays, predictions=preds)
