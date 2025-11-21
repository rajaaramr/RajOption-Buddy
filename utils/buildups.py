# File: utils/buildups.py
# Purpose: Compute Futures OI Buildup & Option-Chain Buildup (kept separate from indicators)
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple
from datetime import timezone

import numpy as np
import pandas as pd

from utils.db import get_db_connection  # your existing helper

# ---------- Config knobs (can be read from INI if you prefer) ----------
DEFAULT_THRESHOLDS = {
    "min_move_pct": 0.20,   # min abs price % move for futures buildup
    "min_oi_pct":   0.50,   # min abs OI % change for futures buildup
    "atm_window":   0.02,   # ±2% band for option-chain ATM aggregation
    "iv_smooth_n":  5,      # EMA window for IV smoothing
}
TRADE_TF = "65"  # compute buildups on your trading timeframe by default

# ---------- Futures OI Buildup ----------
def compute_futures_buildup(symbol: str, *, interval: str = TRADE_TF,
                             thresholds: Dict[str, float] = DEFAULT_THRESHOLDS) -> Dict:
    """
    Reads last two futures candles for (symbol, interval) and classifies buildup.
    Returns dict with tag/percent changes/score.
    """
    with get_db_connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT close, oi, ts
            FROM market.futures_candles
            WHERE symbol=%s AND interval=%s
            ORDER BY ts DESC
            LIMIT 2;
            """,
            (symbol, interval),
        )
        rows = cur.fetchall()

    if not rows or len(rows) < 2:
        return {"tag": "Neutral", "price_chg_pct": 0.0, "oi_chg_pct": 0.0, "score": 50}

    c0, oi0, _ = rows[0]
    c1, oi1, _ = rows[1]
    c0 = float(c0 or 0); c1 = float(c1 or 0)
    oi0 = float(oi0 or 0); oi1 = float(oi1 or 0)

    if c1 == 0 or oi1 == 0:
        return {"tag": "Neutral", "price_chg_pct": 0.0, "oi_chg_pct": 0.0, "score": 50}

    price_chg_pct = 100.0 * (c0 - c1) / c1
    oi_chg_pct    = 100.0 * (oi0 - oi1) / max(oi1, 1.0)

    # Threshold gating
    if abs(price_chg_pct) < thresholds["min_move_pct"] or abs(oi_chg_pct) < thresholds["min_oi_pct"]:
        tag = "Neutral"
    elif price_chg_pct > 0 and oi_chg_pct > 0:
        tag = "Long Buildup"
    elif price_chg_pct < 0 and oi_chg_pct > 0:
        tag = "Short Buildup"
    elif price_chg_pct > 0 and oi_chg_pct < 0:
        tag = "Short Covering"
    else:  # price_chg_pct < 0 and oi_chg_pct < 0
        tag = "Long Unwinding"

    # Map to a 0–100 score
    score_map = {
        "Long Buildup": 100,
        "Short Covering": 60,
        "Neutral": 50,
        "Long Unwinding": 40,
        "Short Buildup": 0,
    }
    score = score_map.get(tag, 50)

    return {
        "tag": tag,
        "price_chg_pct": round(price_chg_pct, 3),
        "oi_chg_pct": round(oi_chg_pct, 3),
        "score": score,
    }

# ---------- Option-Chain Buildup (ATM band) ----------
def compute_optionchain_buildup(symbol: str, *, interval: str = TRADE_TF,
                                thresholds: Dict[str, float] = DEFAULT_THRESHOLDS) -> Dict:
    """
    Aggregates CE/PE OI Δ and IV around ATM (±atm_window * spot) at latest ts.
    options.option_chain schema expected: (symbol, ts, expiry, strike, option_type, oi, oi_chg, iv, last_price, spot_price)
    Returns stance + PCR + deltas + score.
    """
    # Grab the latest spot and ts to align snapshots
    with get_db_connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT close, ts
            FROM market.spot_candles
            WHERE symbol=%s AND interval=%s
            ORDER BY ts DESC
            LIMIT 1;
            """,
            (symbol, interval),
        )
        spot_row = cur.fetchone()

    if not spot_row:
        return {"stance": "Neutral", "pcr": None, "pcr_delta": None, "ce_oi_delta": 0, "pe_oi_delta": 0, "score": 50}

    spot, ref_ts = float(spot_row[0] or 0), spot_row[1]
    if spot <= 0:
        return {"stance": "Neutral", "pcr": None, "pcr_delta": None, "ce_oi_delta": 0, "pe_oi_delta": 0, "score": 50}

    band_lo = spot * (1 - thresholds["atm_window"])
    band_hi = spot * (1 + thresholds["atm_window"])

    # Latest OC rows near the same ts (allow slight drift)
    with get_db_connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT option_type, strike, oi, oi_chg, iv, ts
            FROM options.option_chain
            WHERE symbol=%s
              AND ts = (SELECT max(ts) FROM options.option_chain WHERE symbol=%s)
              AND strike BETWEEN %s AND %s;
            """,
            (symbol, symbol, band_lo, band_hi),
        )
        oc = cur.fetchall()

        # prior snapshot for PCR delta (best effort)
        cur.execute(
            """
            SELECT option_type, strike, oi, oi_chg, iv, ts
            FROM options.option_chain
            WHERE symbol=%s
              AND ts = (
                    SELECT max(ts) FROM options.option_chain
                    WHERE symbol=%s AND ts < (SELECT max(ts) FROM options.option_chain WHERE symbol=%s)
              )
              AND strike BETWEEN %s AND %s;
            """,
            (symbol, symbol, symbol, band_lo, band_hi),
        )
        oc_prev = cur.fetchall()

    if not oc:
        return {"stance": "Neutral", "pcr": None, "pcr_delta": None, "ce_oi_delta": 0, "pe_oi_delta": 0, "score": 50}

    # Aggregate
    CE_oi = CE_oi_d = PE_oi = PE_oi_d = 0.0
    CE_iv_vals, PE_iv_vals = [], []
    for typ, strike, oi, oi_chg, iv, _ in oc:
        if str(typ).upper().startswith("C"):
            CE_oi += float(oi or 0); CE_oi_d += float(oi_chg or 0)
            if iv is not None: CE_iv_vals.append(float(iv))
        else:
            PE_oi += float(oi or 0); PE_oi_d += float(oi_chg or 0)
            if iv is not None: PE_iv_vals.append(float(iv))

    PCR = PE_oi / max(CE_oi, 1.0)

    # previous PCR for delta
    PCR_prev = None
    if oc_prev:
        CEp = PEp = 0.0
        for typ, _, oi, _, _, _ in oc_prev:
            if str(typ).upper().startswith("C"):
                CEp += float(oi or 0)
            else:
                PEp += float(oi or 0)
        PCR_prev = PEp / max(CEp, 1.0)
    PCR_delta = (PCR - PCR_prev) if (PCR_prev is not None) else None

    # Smooth IVs (simple mean here; swap to EMA if you want)
    CE_iv = float(np.mean(CE_iv_vals)) if CE_iv_vals else None
    PE_iv = float(np.mean(PE_iv_vals)) if PE_iv_vals else None

    # Stance (simple, robust)
    # Bullish if PE OI rising (puts written/interest up) and CE OI not rising, or CE OI falling.
    # Bearish if CE OI rising and PE not, or PE falling.
    if PE_oi_d > 0 and CE_oi_d <= 0:
        stance = "Bullish"
    elif CE_oi_d > 0 and PE_oi_d <= 0:
        stance = "Bearish"
    elif PE_oi_d > 0 and CE_oi_d > 0:
        # both rising -> mild, direction via PCR drift
        stance = "Mild Bullish" if (PCR_delta or 0) >= 0 else "Mild Bearish"
    else:
        stance = "Neutral"

    score_map = {
        "Bullish": 90,
        "Mild Bullish": 65,
        "Neutral": 50,
        "Mild Bearish": 35,
        "Bearish": 10,
    }
    score = score_map.get(stance, 50)

    return {
        "stance": stance,
        "pcr": round(PCR, 3),
        "pcr_delta": round(PCR_delta, 3) if PCR_delta is not None else None,
        "ce_oi_delta": int(CE_oi_d),
        "pe_oi_delta": int(PE_oi_d),
        "ce_iv_avg": round(CE_iv, 2) if CE_iv is not None else None,
        "pe_iv_avg": round(PE_iv, 2) if PE_iv is not None else None,
        "score": score,
    }
