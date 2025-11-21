# File: scheduler/track_zone_breaks.py
from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, Optional, List, Literal
import numpy as np

from utils.db import get_db_connection

TZ = timezone.utc
DEFAULT_INTERVAL = "60m"
LOOKBACK_BARS = 20   # for avg volume & momentum checks
FAKEOUT_LOOKAHEAD = 2  # how many bars to check for fakeout retests

SymbolSource = Literal["open_trades", "webhooks", "recent_prices", "manual"]

# ---------------- DB fetch helpers ----------------

def _fetch_latest_candle(symbol: str, interval: str) -> Optional[dict]:
    with get_db_connection() as conn, conn.cursor() as cur:
        cur.execute("""
            SELECT ts, open, high, low, close, volume
              FROM market.futures_candles
             WHERE symbol=%s AND interval=%s
             ORDER BY ts DESC
             LIMIT 1;
        """, (symbol, interval))
        r = cur.fetchone()
    return None if not r else {
        "ts": r[0], "open": float(r[1] or 0), "high": float(r[2] or 0),
        "low": float(r[3] or 0), "close": float(r[4] or 0), "volume": float(r[5] or 0)
    }

def _fetch_recent_candles(symbol: str, interval: str, n: int = LOOKBACK_BARS) -> List[dict]:
    with get_db_connection() as conn, conn.cursor() as cur:
        cur.execute("""
            SELECT ts, open, high, low, close, volume
              FROM market.futures_candles
             WHERE symbol=%s AND interval=%s
             ORDER BY ts DESC
             LIMIT %s;
        """, (symbol, interval, n))
        rows = cur.fetchall()
    return [
        {"ts": r[0], "open": float(r[1] or 0), "high": float(r[2] or 0),
         "low": float(r[3] or 0), "close": float(r[4] or 0), "volume": float(r[5] or 0)}
        for r in rows or []
    ][::-1]  # chronological

def _fetch_latest_zone(symbol: str, interval: str) -> Optional[dict]:
    with get_db_connection() as conn, conn.cursor() as cur:
        cur.execute("""
            SELECT ts, val, vah, poc,
                   COALESCE(support_break_flag,0),
                   COALESCE(resistance_break_flag,0)
              FROM market.zone_levels
             WHERE symbol=%s AND interval=%s
             ORDER BY ts DESC
             LIMIT 1;
        """, (symbol, interval))
        r = cur.fetchone()
    return None if not r else {
        "ts": r[0],
        "val": float(r[1]) if r[1] else None,
        "vah": float(r[2]) if r[2] else None,
        "poc": float(r[3]) if r[3] else None,
        "sb": int(r[4] or 0), "rb": int(r[5] or 0)
    }

# ---------------- Signal helpers ----------------

def _avg_volume(candles: List[dict]) -> float:
    vols = [c["volume"] for c in candles]
    return float(np.mean(vols)) if vols else 0.0

def _momentum_rsi(candles: List[dict]) -> float:
    closes = np.array([c["close"] for c in candles])
    if len(closes) < 3: return 50.0
    deltas = np.diff(closes)
    ups = deltas[deltas > 0].sum() if any(deltas > 0) else 0
    downs = -deltas[deltas < 0].sum() if any(deltas < 0) else 0
    if downs == 0: return 100.0
    rs = ups / downs if downs > 0 else 0
    return 100 - (100 / (1 + rs))

# ---------------- Main Breakout Logic ----------------

def track_zone_break(symbol: str, *, interval: str = DEFAULT_INTERVAL) -> None:
    cndl = _fetch_latest_candle(symbol, interval)
    if not cndl:
        print(f"⚠️ No candle for {symbol}[{interval}]")
        return
    z = _fetch_latest_zone(symbol, interval)
    if not z:
        print(f"⚠️ No zones for {symbol}[{interval}]")
        return

    ts_cndl = cndl["ts"]
    o, h, l, c = cndl["open"], cndl["high"], cndl["low"], cndl["close"]

    # Body & wick metrics
    rng = max(h - l, 1e-9)
    body = abs(c - o)
    body_ratio = body / rng
    upper_wick = h - max(o, c)
    lower_wick = min(o, c) - l

    # Recent averages
    candles = _fetch_recent_candles(symbol, interval, LOOKBACK_BARS)
    avg_vol = _avg_volume(candles[:-1]) if len(candles) > 1 else 0
    rsi_val = _momentum_rsi(candles)

    set_map: Dict[str, object] = {}
    breakout_type = None

    # Support Break
    if z["val"] and c < z["val"] and body_ratio > 0.5:
        if cndl["volume"] > 2 * avg_vol and rsi_val < 45:  # volume + momentum
            if lower_wick < body * 0.3:  # wick filter
                set_map["support_break_flag"] = 1
                breakout_type = "support_break"

    # Resistance Break
    if z["vah"] and c > z["vah"] and body_ratio > 0.5:
        if cndl["volume"] > 2 * avg_vol and rsi_val > 55:
            if upper_wick < body * 0.3:
                set_map["resistance_break_flag"] = 1
                breakout_type = "resistance_break"

    if not set_map:
        print(f"ℹ️ No breakout for {symbol}[{interval}] @ {ts_cndl} (body_ratio={body_ratio:.2f})")
        return

    # Update DB
    with get_db_connection() as conn, conn.cursor() as cur:
        for k, v in set_map.items():
            cur.execute(f"""
                UPDATE market.zone_levels
                   SET {k}=%s
                 WHERE symbol=%s AND interval=%s AND ts=%s;
            """, (v, symbol, interval, z["ts"]))
        cur.execute("""
            INSERT INTO market.zone_breakouts(symbol,interval,ts,breakout,price,reason)
            VALUES (%s,%s,%s,%s,%s,%s)
        """, (
            symbol, interval, ts_cndl, breakout_type, c,
            f"vol={cndl['volume']:.0f}>2×avg({avg_vol:.0f}); rsi={rsi_val:.1f}; body_ratio={body_ratio:.2f}"
        ))
        conn.commit()

    print(f"✅ ZoneBreak {symbol}[{interval}] {breakout_type} @ {ts_cndl}")

    # TODO: Add fakeout/retest check by scheduling FAKEOUT_LOOKAHEAD candles later.

# ---------------- Symbol Sources ----------------

def _symbols_from_open_trades() -> List[str]:
    with get_db_connection() as conn, conn.cursor() as cur:
        cur.execute("""SELECT DISTINCT symbol FROM journal.trading_journal WHERE status='OPEN';""")
        rows = cur.fetchall()
    return [r[0] for r in rows] if rows else []

def _symbols_from_webhooks(lookback_minutes: int = 1440) -> List[str]:
    with get_db_connection() as conn, conn.cursor() as cur:
        cur.execute("""
            SELECT DISTINCT symbol
              FROM webhooks.webhook_alerts
             WHERE received_at >= now() - INTERVAL '%s minutes';
        """, (lookback_minutes,))
        rows = cur.fetchall()
    return [r[0] for r in rows] if rows else []

def _symbols_from_recent_prices(interval: str, lookback_minutes: int = 180) -> List[str]:
    with get_db_connection() as conn, conn.cursor() as cur:
        cur.execute("""
            SELECT DISTINCT symbol
              FROM market.futures_candles
             WHERE interval=%s
               AND ts >= now() - INTERVAL '%s minutes';
        """, (interval, lookback_minutes))
        rows = cur.fetchall()
    return [r[0] for r in rows] if rows else []

def track_all_symbols(symbols: List[str] | None = None, *,
                      interval: str = DEFAULT_INTERVAL,
                      source: SymbolSource = "open_trades",
                      lookback_minutes: int = 1440) -> None:
    if source == "manual":
        syms = symbols or []
    elif source == "open_trades":
        syms = _symbols_from_open_trades()
    elif source == "webhooks":
        syms = _symbols_from_webhooks(lookback_minutes)
    elif source == "recent_prices":
        syms = _symbols_from_recent_prices(interval, lookback_minutes)
    else:
        syms = []

    if not syms:
        print("ℹ️ No symbols to process.")
        return

    for sym in syms:
        try:
            track_zone_break(sym, interval=interval)
        except Exception as e:
            print(f"❌ {sym}[{interval}] error: {e}")

if __name__ == "__main__":
    track_all_symbols(source="open_trades", interval=DEFAULT_INTERVAL)
