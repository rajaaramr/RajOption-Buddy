# File: utils/option_utils.py
# Purpose: Option analytics helpers (Timescale/Postgres).
# - Buildup classification (basic + advanced)
# - Latest CE/PE deltas (price/oi) computed from normalized option_chain_data
# - DB access via utils.db.get_db_connection (UTC handled there)
#
# NOTE: Chain fetching/writes now live in scheduler/fetch_and_store_option_chain.py

from __future__ import annotations

from typing import Dict, Tuple, Optional
import os

from utils.db import get_db_connection

# Schema-qualified table (override via env if needed)
OPTION_CHAIN_TABLE = os.environ.get("OPTION_CHAIN_TABLE", "market.option_chain_data")


# ------------------- Buildup Classification -------------------

def determine_buildup(oi_change: float, price_change: float) -> str:
    if oi_change > 0 and price_change > 0:
        return "Long Build Up"
    elif oi_change > 0 and price_change < 0:
        return "Short Build Up"
    elif oi_change < 0 and price_change < 0:
        return "Long Unwinding"
    elif oi_change < 0 and price_change > 0:
        return "Short Covering"
    else:
        return "Neutral"


def advanced_buildup_rules(data: Dict) -> str:
    """
    data keys (tolerant): oi_chg, prev_oi, price_chg, volume, iv_chg
    """
    oi_chg = float(data.get("oi_chg", 0) or 0)
    prev_oi = float(data.get("prev_oi", 1) or 1)
    price_chg = float(data.get("price_chg", 0) or 0)
    volume = float(data.get("volume", 0) or 0)
    iv_chg = float(data.get("iv_chg", 0) or 0)

    oi_pct = (oi_chg / prev_oi * 100.0) if prev_oi > 0 else 0.0

    if oi_pct > 10 and price_chg > 1.5:
        return "Strong Long Buildup"
    elif oi_pct > 10 and price_chg < -1.5:
        return "Strong Short Buildup"
    elif oi_pct < -10 and price_chg < -1:
        return "Aggressive Long Unwinding"
    elif oi_pct < -10 and price_chg > 1:
        return "Aggressive Short Covering"
    elif abs(iv_chg) > 5 and volume > 100_000:
        return "Volatility Spike Play"
    else:
        return "Neutral"


def classify_iv_regime(iv: float) -> str:
    iv = float(iv or 0)
    if iv < 15:
        return "Low IV"
    elif iv < 25:
        return "Medium IV"
    else:
        return "High IV"


# ------------------- Latest CE/PE snapshot deltas -------------------

def _latest_two_rows_for_type(cursor, symbol: str, opt_type: str):
    """
    Pull the latest two rows for given option_type to compute deltas robustly.
    Assumes option_chain_data has at least:
      - symbol, option_type ('CE'/'PE'), close_price, oi, oi_chg (optional), volume, timestamp
    """
    cursor.execute(
        f"""
        SELECT close_price, oi, oi_chg, volume, timestamp
        FROM {OPTION_CHAIN_TABLE}
        WHERE symbol = %s AND option_type = %s
        ORDER BY timestamp DESC
        LIMIT 2
        """,
        (symbol, opt_type),
    )
    return cursor.fetchall()  # [(close, oi, oi_chg, vol, ts), ...]


def _compute_deltas(rows) -> Tuple[float, float, float]:
    """
    Given up to two rows [(close, oi, oi_chg, vol, ts), ...] (most recent first),
    return (price_change, oi_change, volume_latest).
    Falls back gracefully when only one row exists or fields are NULL.
    """
    if not rows:
        return 0.0, 0.0, 0.0

    close0, oi0, oi_chg0, vol0, _ = rows[0]
    close0 = float(close0 or 0.0)
    oi0 = float(oi0 or 0.0)
    vol0 = float(vol0 or 0.0)

    # Prefer stored oi_chg if present; else derive vs previous row
    if oi_chg0 is not None:
        oi_change = float(oi_chg0 or 0.0)
    elif len(rows) > 1:
        _, oi1, _, _, _ = rows[1]
        oi_change = float(oi0 - float(oi1 or 0.0))
    else:
        oi_change = 0.0

    # Price change = close0 - close1 (if available)
    if len(rows) > 1:
        close1, *_ = rows[1]
        price_change = float(close0 - float(close1 or 0.0))
    else:
        price_change = 0.0

    return price_change, oi_change, vol0


def fetch_latest_option_data(symbol: str) -> Tuple[Dict, Dict]:
    """
    Fetch the latest CE/PE deltas for `symbol` from market.option_chain_data.

    Returns:
      (ce_data, pe_data) where each is:
        {
          "oi_change": float,
          "price_change": float,
          "volume": float
        }
      Missing side returns {}.

    This replaces the old MySQL-based function and aligns to our normalized schema.
    """
    ce, pe = {}, {}
    try:
        with get_db_connection() as conn, conn.cursor() as cur:
            rows_ce = _latest_two_rows_for_type(cur, symbol, "CE")
            rows_pe = _latest_two_rows_for_type(cur, symbol, "PE")

            if rows_ce:
                price_chg, oi_chg, vol = _compute_deltas(rows_ce)
                ce = {"oi_change": oi_chg, "price_change": price_chg, "volume": vol}

            if rows_pe:
                price_chg, oi_chg, vol = _compute_deltas(rows_pe)
                pe = {"oi_change": oi_chg, "price_change": price_chg, "volume": vol}

    except Exception as e:
        print(f"❌ fetch_latest_option_data({symbol}) failed: {e}")

    return ce, pe
