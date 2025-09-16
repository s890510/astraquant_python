from __future__ import annotations

import random
import time
from datetime import datetime, timedelta, date as date_cls
from typing import List, Optional

import pandas as pd
from fastapi import APIRouter, HTTPException, Query

from app.core.config import settings
from pathlib import Path

router = APIRouter(tags=["twse"])  # will be mounted with prefix="/twse" in main


# -----------------------------
# Helper: fetch + cache per date
# -----------------------------

def _instflows_cache_path(d: date_cls) -> Path:
    base = Path(settings.DATA_DIR) / "instflows"
    base.mkdir(parents=True, exist_ok=True)
    return base / f"instflows_{d.strftime('%Y%m%d')}.parquet"


def _mock_download_instflows_for_date(d: date_cls) -> pd.DataFrame:
    """
    Stub downloader that mimics TWSE/TPEx institutional net buy/sell for a single day.
    - Returns empty DataFrame on weekends (Sat/Sun) to simulate no trading.
    - Otherwise returns random rows for a small set of tickers.
    """
    # Weekend -> no data
    if d.weekday() >= 5:
        return pd.DataFrame(columns=[
            "date", "ticker", "foreignNet", "investmentTrustNet", "dealerNet",
        ])

    # Generate a random number of tickers
    num = random.randint(5, 15)
    tickers = random.sample([
        "1101", "1216", "1301", "2002", "2303", "2308", "2317", "2330", "2454", "2603",
        "2609", "2615", "2881", "2882", "2884", "2885", "2891", "3008", "3034", "3711",
    ], num)

    rows = []
    for t in tickers:
        rows.append({
            "date": d.strftime("%Y-%m-%d"),
            "ticker": t,
            "foreignNet": int(random.randint(-50000, 50000)),
            "investmentTrustNet": int(random.randint(-20000, 20000)),
            "dealerNet": int(random.randint(-10000, 10000)),
        })

    df = pd.DataFrame(rows)
    return df


def _retry_download(d: date_cls, retries: int = 3) -> pd.DataFrame:
    backoffs = [0.5, 1.0, 2.0]
    last_exc: Optional[Exception] = None
    for attempt in range(retries):
        try:
            # In future, replace with real TWSE/TPEx fetch
            df = _mock_download_instflows_for_date(d)
            return df
        except Exception as e:  # pragma: no cover - defensive
            last_exc = e
            if attempt < retries - 1:
                time.sleep(backoffs[attempt])
            else:
                raise
    # Should not reach here
    if last_exc:
        raise last_exc
    return pd.DataFrame()


def _normalize_instflows_df(df: pd.DataFrame, d: date_cls) -> pd.DataFrame:
    """
    Ensure schema and types: date(YYYY-MM-DD), ticker(str, strip), foreignNet(int), investmentTrustNet(int), dealerNet(int)
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=[
            "date", "ticker", "foreignNet", "investmentTrustNet", "dealerNet",
        ])

    # Coerce/rename if needed
    cols_map = {
        "date": "date",
        "ticker": "ticker",
        "foreignNet": "foreignNet",
        "investmentTrustNet": "investmentTrustNet",
        "dealerNet": "dealerNet",
    }
    df = df.rename(columns=cols_map)

    # Fill missing and enforce types
    df["date"] = d.strftime("%Y-%m-%d")
    df["ticker"] = df.get("ticker", "").astype(str).str.strip()

    for c in ["foreignNet", "investmentTrustNet", "dealerNet"]:
        if c not in df.columns:
            df[c] = 0
        # to numeric then to int
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)

    # Keep only required columns and drop rows with empty ticker
    df = df[["date", "ticker", "foreignNet", "investmentTrustNet", "dealerNet"]]
    df = df[df["ticker"] != ""]

    return df.reset_index(drop=True)


def get_instflows_for_date(d: date_cls) -> pd.DataFrame:
    """
    Read from cache if available; otherwise download (stub), normalize, write parquet cache, and return DataFrame.
    """
    cache_path = _instflows_cache_path(d)

    # Weekend: short-circuit to empty and avoid creating a cache file
    if d.weekday() >= 5:
        return pd.DataFrame(columns=[
            "date", "ticker", "foreignNet", "investmentTrustNet", "dealerNet",
        ])

    if cache_path.exists():
        try:
            return pd.read_parquet(cache_path)
        except Exception:
            # Corrupted cache -> ignore and refetch
            pass

    df = _retry_download(d, retries=3)
    df = _normalize_instflows_df(df, d)

    # Save to cache
    try:
        df.to_parquet(cache_path, index=False)
    except Exception:
        # Ignore cache write failures
        pass

    return df


# -----------------------------
# API Endpoint
# -----------------------------


def _parse_date(s: str) -> date_cls:
    return datetime.strptime(s, "%Y-%m-%d").date()


@router.get("/instflows")
async def get_instflows(
    date: Optional[str] = None,
    from_: Optional[str] = Query(default=None, alias="from"),
    to: Optional[str] = None,
):
    # Validate query parameters: either date OR (from & to)
    if date and (from_ or to):
        raise HTTPException(status_code=400, detail="Use either 'date' or 'from' & 'to', not both.")
    if not date and not (from_ and to):
        raise HTTPException(status_code=400, detail="Provide 'date' or both 'from' and 'to'.")

    if date:
        try:
            d = _parse_date(date)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD.")
        frames: List[pd.DataFrame] = []
        df = get_instflows_for_date(d)
        if not df.empty:
            frames.append(df)
        if frames:
            out = pd.concat(frames, ignore_index=True)
        else:
            out = pd.DataFrame(columns=["date", "ticker", "foreignNet", "investmentTrustNet", "dealerNet"])
        rows = out.to_dict(orient="records")
        return {
            "source": "twse",
            "from": d.strftime("%Y-%m-%d"),
            "to": d.strftime("%Y-%m-%d"),
            "rows": rows,
        }

    # Range
    try:
        start = _parse_date(from_)
        end = _parse_date(to)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid date format in 'from' or 'to'. Use YYYY-MM-DD.")

    if end < start:
        raise HTTPException(status_code=400, detail="'to' must be on or after 'from'.")

    frames: List[pd.DataFrame] = []
    cur = start
    while cur <= end:
        df = get_instflows_for_date(cur)
        if not df.empty:
            frames.append(df)
        cur += timedelta(days=1)

    if frames:
        out = pd.concat(frames, ignore_index=True)
    else:
        out = pd.DataFrame(columns=["date", "ticker", "foreignNet", "investmentTrustNet", "dealerNet"])

    rows = out.to_dict(orient="records")
    return {
        "source": "twse",
        "from": start.strftime("%Y-%m-%d"),
        "to": end.strftime("%Y-%m-%d"),
        "rows": rows,
    }


# -----------------------------
# Symbols endpoint (stub)
# -----------------------------

@router.get("/symbols")
async def get_symbols():
    """
    Return a stub list of TWSE/TPEx symbols with optional industry and flags.
    Will be replaced by real TWSE/TPEx scraping in a future iteration.
    """
    rows = [
        {"ticker": "2330", "name": "TSMC", "industry": "Semiconductor", "is_tw": True, "is_us": False},
        {"ticker": "2317", "name": "Hon Hai", "industry": "Electronics", "is_tw": True, "is_us": False},
        {"ticker": "2609", "name": "Yang Ming", "industry": "Shipping", "is_tw": True, "is_us": False},
        {"ticker": "2303", "name": "UMC", "industry": "Semiconductor", "is_tw": True, "is_us": False},
        {"ticker": "2882", "name": "Cathay FHC", "industry": "Financial", "is_tw": True, "is_us": False},
    ]

    return {
        "source": "twse",
        "count": len(rows),
        "rows": rows,
    }


@router.get("/symbols/details")
async def get_symbol_details(tickers: str = Query(..., description="Comma-separated list of tickers, e.g. 2308,2609")):
    """
    Enrich symbol metadata for the provided tickers.
    Stub implementation backed by an in-module dict. Later we can swap to TWSE/TPEx official source.
    Response:
    { "rows": [ {"ticker":"2308","name":"...","industry":"...","is_tw":true,"is_us":false}, ... ] }
    """
    # Parse and normalize tickers
    raw = tickers or ""
    parts = [p.strip() for p in raw.split(",") if p and p.strip()]
    # Deduplicate while preserving order
    seen = set()
    req_list = []
    for t in parts:
        if t not in seen:
            seen.add(t)
            req_list.append(t)

    # Stub mapping (can be replaced by CSV/official list later)
    stub_map = {
        "2330": {"name": "TSMC", "industry": "Semiconductor"},
        "2317": {"name": "Hon Hai", "industry": "Electronics"},
        "2609": {"name": "Yang Ming", "industry": "Shipping"},
        "2303": {"name": "UMC", "industry": "Semiconductor"},
        "2882": {"name": "Cathay FHC", "industry": "Financial"},
        "2454": {"name": "MediaTek", "industry": "Semiconductor"},
        "2308": {"name": "Delta Electronics", "industry": "Electronics"},
        "2603": {"name": "Evergreen Marine", "industry": "Shipping"},
        "2881": {"name": "Fubon FHC", "industry": "Financial"},
        "3008": {"name": "Largan", "industry": "Optics"},
    }

    rows = []
    for t in req_list:
        meta = stub_map.get(t, {})
        rows.append({
            "ticker": t,
            "name": meta.get("name"),
            "industry": meta.get("industry"),
            "is_tw": True,
            "is_us": False,
        })

    return {"rows": rows}


# -----------------------------
# Prices endpoint (stub with cache)
# -----------------------------
from typing import Tuple


def _prices_cache_path(ticker: str) -> Path:
    base = Path(settings.DATA_DIR) / "prices"
    base.mkdir(parents=True, exist_ok=True)
    # Normalize ticker (strip spaces)
    safe = str(ticker).strip()
    return base / f"{safe}.parquet"


def _read_prices_cache(ticker: str) -> pd.DataFrame:
    path = _prices_cache_path(ticker)
    if path.exists():
        try:
            df = pd.read_parquet(path)
            # Ensure schema
            exp_cols = ["date", "open", "high", "low", "close", "volume"]
            for c in exp_cols:
                if c not in df.columns:
                    return pd.DataFrame(columns=exp_cols)
            # Coerce types and sort
            df["date"] = df["date"].astype(str)
            for c in ["open", "high", "low", "close"]:
                df[c] = pd.to_numeric(df[c], errors="coerce").astype(float)
            df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0).astype(int)
            df = df.sort_values("date").reset_index(drop=True)
            return df
        except Exception:
            return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])
    return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])


def _daterange_business_days(start: date_cls, end: date_cls):
    cur = start
    while cur <= end:
        if cur.weekday() < 5:  # Mon-Fri
            yield cur
        cur = cur + timedelta(days=1)


def _mock_generate_prices(ticker: str, start: date_cls, end: date_cls) -> pd.DataFrame:
    """Generate deterministic mock daily OHLCV for weekdays in [start, end]."""
    if end < start:
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])

    # Deterministic RNG per (ticker, start)
    seed = (abs(hash(str(ticker).strip())) % (2**31 - 1)) ^ start.toordinal()
    rng = random.Random(seed)

    # Base price from ticker hash
    base = 30.0 + (abs(hash(str(ticker))) % 300) * 0.5  # between ~30 and ~180
    price = max(5.0, base)

    rows = []
    for d in _daterange_business_days(start, end):
        # Simulate daily return in percent
        ret_pct = rng.gauss(0.0, 1.2)  # ~N(0, 1.2%)
        open_px = price * (1.0 + rng.gauss(0.0, 0.2) / 100.0)
        close_px = max(1e-6, open_px * (1.0 + ret_pct / 100.0))
        high_px = max(open_px, close_px) * (1.0 + abs(rng.gauss(0.6, 0.3)) / 100.0)
        low_px = min(open_px, close_px) * (1.0 - abs(rng.gauss(0.6, 0.3)) / 100.0)
        vol = int(max(0, rng.gauss(50_000_000, 15_000_000)))

        rows.append({
            "date": d.strftime("%Y-%m-%d"),
            "open": float(round(open_px, 3)),
            "high": float(round(high_px, 3)),
            "low": float(round(low_px, 3)),
            "close": float(round(close_px, 3)),
            "volume": int(vol),
        })
        price = close_px

    return pd.DataFrame(rows, columns=["date", "open", "high", "low", "close", "volume"])


def _upsert_prices_cache(ticker: str, df_new: pd.DataFrame) -> None:
    if df_new is None or df_new.empty:
        return
    cur = _read_prices_cache(ticker)
    all_df = pd.concat([cur, df_new], ignore_index=True)
    # Deduplicate by date keeping last (new)
    all_df = all_df.drop_duplicates(subset=["date"], keep="last")
    # Coerce types
    all_df["date"] = all_df["date"].astype(str)
    for c in ["open", "high", "low", "close"]:
        all_df[c] = pd.to_numeric(all_df[c], errors="coerce").astype(float)
    all_df["volume"] = pd.to_numeric(all_df["volume"], errors="coerce").fillna(0).astype(int)
    all_df = all_df.sort_values("date").reset_index(drop=True)

    # Save
    try:
        _prices_cache_path(ticker).parent.mkdir(parents=True, exist_ok=True)
        all_df.to_parquet(_prices_cache_path(ticker), index=False)
    except Exception:
        # ignore cache write issues
        pass


@router.get("/prices")
async def get_prices(
    ticker: str = Query(..., description="Ticker symbol, e.g., 2330"),
    from_: str = Query(..., alias="from", description="Start date YYYY-MM-DD"),
    to: str = Query(..., description="End date YYYY-MM-DD"),
):
    """Return daily OHLCV prices for the given ticker and date range.
    For now this uses a deterministic mock generator and caches to data/prices/{ticker}.parquet.
    """
    if not ticker or not str(ticker).strip():
        raise HTTPException(status_code=400, detail="'ticker' is required")

    try:
        start = _parse_date(from_)
        end = _parse_date(to)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD.")

    if end < start:
        raise HTTPException(status_code=400, detail="'to' must be on or after 'from'.")

    # Generate mock for requested range and upsert into cache
    gen_df = _mock_generate_prices(ticker, start, end)
    _upsert_prices_cache(ticker, gen_df)

    # Read cache and filter
    df = _read_prices_cache(ticker)
    if df.empty:
        rows: List[dict] = []
    else:
        mask = (df["date"] >= start.strftime("%Y-%m-%d")) & (df["date"] <= end.strftime("%Y-%m-%d"))
        out = df.loc[mask].copy()
        out = out.sort_values("date").reset_index(drop=True)
        # Ensure types
        for c in ["open", "high", "low", "close"]:
            out[c] = pd.to_numeric(out[c], errors="coerce").astype(float)
        out["volume"] = pd.to_numeric(out["volume"], errors="coerce").fillna(0).astype(int)
        rows = out.to_dict(orient="records")

    return {
        "ticker": str(ticker).strip(),
        "from": start.strftime("%Y-%m-%d"),
        "to": end.strftime("%Y-%m-%d"),
        "rows": rows,
    }
