# scheduler/dynamic_weights.py
from __future__ import annotations

import json
import configparser
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import psycopg2.extras as pgx

from utils.db import get_db_connection

TZ = timezone.utc
DEFAULT_INI = "dynamic_weights.ini"

# ---------------- Config ----------------

@dataclass
class DWCfg:
    tfs: List[str]                        # which TFs to produce weights for
    base_weights: Dict[str, float]        # default fallback
    atr_window: int                       # ATR lookback (on 5m spot resample)
    lookback_days: int                    # how much raw data to pull
    pct_low: float                        # ATR% <= pct_low  => low vol
    pct_high: float                       # ATR% >= pct_high => high vol
    weights_low: Dict[str, float]         # weights in low vol regime
    weights_mid: Dict[str, float]         # weights in mid vol regime
    weights_high: Dict[str, float]        # weights in high vol regime
    blend: str                            # "hard" or "soft"
    soft_k: float                         # softness for logistic blending

def _csv(s: str) -> List[str]:
    return [x.strip() for x in str(s).split(",") if x.strip()]

def _wmap(s: str) -> Dict[str, float]:
    # "25m:0.2,65m:0.3,125m:0.5"
    out: Dict[str, float] = {}
    for part in _csv(s):
        if ":" in part:
            k, v = part.split(":", 1)
            try:
                out[k.strip()] = float(v.strip())
            except:
                pass
    return out

def load_cfg(path: str = DEFAULT_INI) -> DWCfg:
    cfg = configparser.ConfigParser(inline_comment_prefixes=(";","#"), interpolation=None, strict=False)
    cfg.read(path)

    tfs = _csv(cfg.get("dynamic", "tfs", fallback="25m,65m,125m"))
    base = _wmap(cfg.get("dynamic", "base_weights", fallback="25m:0.30,65m:0.30,125m:0.40"))

    return DWCfg(
        tfs=tfs,
        base_weights=base,
        atr_window=cfg.getint("dynamic", "atr_window", fallback=14),
        lookback_days=cfg.getint("dynamic", "lookback_days", fallback=15),
        pct_low=cfg.getfloat("dynamic", "atr_pct_low", fallback=0.60),   # 0.60% daily-ish
        pct_high=cfg.getfloat("dynamic", "atr_pct_high", fallback=1.20), # 1.20%
        weights_low=_wmap(cfg.get("dynamic", "weights_low", fallback="25m:0.20,65m:0.30,125m:0.50")),
        weights_mid=_wmap(cfg.get("dynamic", "weights_mid", fallback="25m:0.30,65m:0.30,125m:0.40")),
        weights_high=_wmap(cfg.get("dynamic", "weights_high", fallback="25m:0.45,65m:0.35,125m:0.20")),
        blend=cfg.get("dynamic", "blend", fallback="hard").strip().lower(),
        soft_k=cfg.getfloat("dynamic", "soft_k", fallback=6.0),
    )

# ---------------- Data helpers ----------------

def _load_spot_5m(symbol: str, lookback_days: int) -> pd.DataFrame:
    cutoff = datetime.now(TZ) - timedelta(days=lookback_days)
    with get_db_connection() as conn, conn.cursor() as cur:
        cur.execute("""
            SELECT ts, open, high, low, close, COALESCE(volume,0)::float8
              FROM market.spot_candles
             WHERE symbol=%s AND interval='5m' AND ts >= %s
             ORDER BY ts ASC
        """, (symbol, cutoff))
        rows = cur.fetchall()
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=["ts","open","high","low","close","volume"])
    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    df = df.dropna(subset=["ts"]).set_index("ts").sort_index()
    for c in ("open","high","low","close","volume"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["open","high","low","close"])

def _resample_25m(df5: pd.DataFrame) -> pd.DataFrame:
    if df5.empty:
        return df5
    rule = "25min"
    out = pd.DataFrame({
        "open":   df5["open"].resample(rule, label="right", closed="right").first(),
        "high":   df5["high"].resample(rule, label="right", closed="right").max(),
        "low":    df5["low"].resample(rule, label="right", closed="right").min(),
        "close":  df5["close"].resample(rule, label="right", closed="right").last(),
        "volume": df5["volume"].resample(rule, label="right", closed="right").sum(),
    }).dropna(how="any")
    return out

def _atr(series_high, series_low, series_close, n:int) -> pd.Series:
    prev_close = series_close.shift(1)
    tr = pd.concat([(series_high-series_low),
                    (series_high-prev_close).abs(),
                    (series_low-prev_close).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0/n, adjust=False).mean()

# ---------------- Regime & weights ----------------

def _atr_pct_latest(df25: pd.DataFrame, atr_window: int) -> Optional[float]:
    if df25.empty:
        return None
    atr = _atr(df25["high"], df25["low"], df25["close"], atr_window)
    close = df25["close"]
    last_atr = float(atr.iloc[-1]) if len(atr) else None
    last_close = float(close.iloc[-1]) if len(close) else None
    if last_atr is None or last_close in (None, 0):
        return None
    return 100.0 * (last_atr / last_close)

def _normalize(weights: Dict[str, float], keys: List[str]) -> Dict[str, float]:
    out = {k: float(weights.get(k, 0.0)) for k in keys}
    s = sum(out.values())
    if s <= 0:
        # evenly spread if all zero
        ev = 1.0 / max(1, len(keys))
        return {k: ev for k in keys}
    return {k: v/s for k, v in out.items()}

def _soft_blend(pct: float, lo: float, hi: float,
                w_lo: Dict[str, float], w_mid: Dict[str, float], w_hi: Dict[str, float],
                keys: List[str], k: float) -> Dict[str, float]:
    """
    Smoothly transition between regimes with logistic ramps.
    - ramp1 ~ goes from low→mid around 'lo'
    - ramp2 ~ goes from mid→high around 'hi'
    """
    # two squashes: below lo → 0, above hi → 1, in between → (0..1)
    r1 = 1.0 / (1.0 + np.exp(-k * (pct - lo)))
    r2 = 1.0 / (1.0 + np.exp(-k * (pct - hi)))
    # weights = (1-r1)*low + (r1-r2)*mid + r2*high
    out = {}
    for key in keys:
        wl = w_lo.get(key, 0.0); wm = w_mid.get(key, 0.0); wh = w_hi.get(key, 0.0)
        out[key] = (1-r1)*wl + (r1-r2)*wm + r2*wh
    return _normalize(out, keys)

def get_dynamic_weights(symbol: str, *, cfg: Optional[DWCfg] = None) -> Dict[str, float]:
    """
    Returns per-TF weights for the given symbol, normalized and regime-aware.
    Fallbacks to base_weights if data is insufficient.
    """
    cfg = cfg or load_cfg()
    tfs = cfg.tfs
    # load 5m spot, resample 25m (good balance), estimate ATR%
    df5 = _load_spot_5m(symbol, cfg.lookback_days)
    if df5.empty:
        return _normalize(cfg.base_weights, tfs)
    df25 = _resample_25m(df5)
    atr_pct = _atr_pct_latest(df25, cfg.atr_window)
    if atr_pct is None:
        return _normalize(cfg.base_weights, tfs)

    if cfg.blend == "hard":
        if atr_pct <= cfg.pct_low:
            return _normalize(cfg.weights_low, tfs)
        if atr_pct >= cfg.pct_high:
            return _normalize(cfg.weights_high, tfs)
        # mid regime
        return _normalize(cfg.weights_mid, tfs)

    # soft regime blending
    return _soft_blend(
        atr_pct, cfg.pct_low, cfg.pct_high,
        cfg.weights_low, cfg.weights_mid, cfg.weights_high,
        tfs, cfg.soft_k
    )
