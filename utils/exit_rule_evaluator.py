# File: utils/symbol_utils.py

import configparser

CONFIG_PATH = "zerodha.ini"

def _cfg():
    cfg = configparser.ConfigParser()
    cfg.read(CONFIG_PATH)
    return cfg

# ------------------- Derivative metadata (minimal) -------------------

# If you want lot sizes without DB, keep a small override map here.
_LOT_SIZES = {
    # "RELIANCE": 250,  # example
    # "INFY": 300,
}

def get_lot_size(symbol: str) -> int:
    return int(_LOT_SIZES.get(symbol.upper(), 1))

# ------------------- Spot & Futures tradingsymbols -------------------

def get_spot_tradingsymbol(symbol: str) -> str:
    """Return the exchange spot symbol (normalized)."""
    return (symbol or "").upper()

def get_futures_expiry_suffix() -> str:
    """Read a preconfigured monthly futures suffix, e.g. '25AUGFUT'."""
    cfg = _cfg()
    return cfg.get("settings", "fut_expiry_suffix", fallback="")

def get_futures_tradingsymbol(symbol: str) -> str | None:
    """
    Build futures symbol as SYMBOL + <suffix from config>, e.g. TCS25AUGFUT.
    Prefer using instruments API for accuracy; this is a config-driven fallback.
    """
    sym = get_spot_tradingsymbol(symbol)
    suf = get_futures_expiry_suffix()
    return f"{sym}{suf}" if sym and suf else None

# ------------------- Option helpers -------------------

def get_option_symbol_base(symbol: str) -> str:
    """Base used to build option symbols (config-driven fallback)."""
    sym = get_spot_tradingsymbol(symbol)
    suf = get_futures_expiry_suffix()
    return f"{sym}{suf}" if sym and suf else ""

# ------------------- Misc -------------------

def is_index(symbol: str) -> bool:
    return (symbol or "").upper() in {"NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY"}

def is_valid_tradingsymbol(symbol: str) -> bool:
    s = (symbol or "").upper()
    return s.endswith("FUT") and len(s) > 10

# utils/exit_rule_evaluator.py

from dataclasses import dataclass
from typing import Tuple, Optional, List
from datetime import datetime, timezone, timedelta

from utils.db import get_db_connection

TZ = timezone.utc

# ---------------- Tunables (start here) ----------------
INTERVAL          = "5m"     # zone + ATR timeframe
ATR_LEN           = 14       # bars for ATR
ATR_MULT          = 1.0      # SL: multiple of ATR
FALLBACK_SL_PCT   = 0.008    # 0.8% if ATR not available
OI_LOOKBACK_MIN   = 3        # last N option snapshots to assess unwind
ZONE_TRAIL_WEIGHT = 1.0      # contribution to score
OI_WEIGHT         = 1.0
SL_WEIGHT         = 1.0
MIN_EXIT_SCORE    = 0.6      # 0..1 scale to confirm exit

# -------------------------------------------------------

@dataclass
class ExitSignal:
    should_exit: bool
    reason: str
    score: float  # 0..1


# ======== Public API ========
def evaluate_exit(symbol: str) -> Tuple[bool, str, float]:
    """
    Decide exit for the latest OPEN trade in trading_journal for `symbol`.
    Returns: (should_exit, reason, score) with score in 0..100 for UI compatibility.
    """
    trade = _get_open_trade(symbol)
    if not trade:
        return False, "no_open_trade", 0.0

    side, entry_price = trade["side"], float(trade["entry_price_fut"] or 0.0)

    # Current futures price (last close of interval)
    fut_px = _get_last_futures_close(symbol, INTERVAL)
    if fut_px is None or entry_price <= 0:
        return False, "context_missing", 0.0

    # 1) Footprint/zone trailing stop
    trail_hit, trail_score = _footprint_trail_exit(symbol, side, fut_px)

    # 2) Options OI unwind / short-cover detection
    oi_exit, oi_score = _detect_oi_unwind(symbol, side)

    # 3) Stop-loss (ATR or fallback %)
    sl_hit, sl_score, sl_level = _stoploss_exit(side, entry_price, fut_px, symbol, INTERVAL)

    # Combine (simple normalized vote)
    votes = []
    if trail_hit: votes.append(ZONE_TRAIL_WEIGHT)
    if oi_exit:   votes.append(OI_WEIGHT)
    if sl_hit:    votes.append(SL_WEIGHT)

    raw = sum(votes)
    denom = (ZONE_TRAIL_WEIGHT + OI_WEIGHT + SL_WEIGHT) or 1.0
    score01 = raw / denom

    # Pick the strongest reason (priority: SL > OI > trail)
    reason = "hold"
    if sl_hit:
        reason = f"stoploss_hit({sl_level:.2f})"
    elif oi_exit:
        reason = "oi_unwind" if side == "LONG" else "short_cover"
    elif trail_hit:
        reason = "footprint_trail"

    should = score01 >= MIN_EXIT_SCORE
    return should, reason, round(score01 * 100.0, 2)


# ======== Components ========

def _get_open_trade(symbol: str) -> Optional[dict]:
    with get_db_connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT side, entry_price_fut
              FROM journal.trading_journal
             WHERE symbol=%s AND status='OPEN'
             ORDER BY entry_ts DESC
             LIMIT 1;
            """,
            (symbol,)
        )
        row = cur.fetchone()
    if not row:
        return None
    return {"side": (row[0] or "LONG").upper(), "entry_price_fut": row[1]}

def _get_last_futures_close(symbol: str, interval: str) -> Optional[float]:
    with get_db_connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT close
              FROM market.futures_candles
             WHERE symbol=%s AND interval=%s
             ORDER BY ts DESC
             LIMIT 1;
            """,
            (symbol, interval)
        )
        row = cur.fetchone()
    return float(row[0]) if row else None

def _get_atr(symbol: str, interval: str, length: int = ATR_LEN) -> Optional[float]:
    # Basic ATR (no Wilder smoothing) over last N bars
    with get_db_connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT high, low, close
              FROM market.futures_candles
             WHERE symbol=%s AND interval=%s
             ORDER BY ts DESC
             LIMIT %s;
            """,
            (symbol, interval, length + 1)
        )
        rows = cur.fetchall()
    if not rows or len(rows) < length + 1:
        return None
    # rows are newest->oldest; reverse for chronological
    rows = rows[::-1]
    trs: List[float] = []
    prev_close = float(rows[0][2] or 0.0)
    for h, l, c in rows[1:]:
        h, l, c = float(h or 0.0), float(l or 0.0), float(c or 0.0)
        tr = max(h - l, abs(h - prev_close), abs(l - prev_close))
        trs.append(tr)
        prev_close = c
    if not trs:
        return None
    return sum(trs) / len(trs)

def _stoploss_exit(side: str, entry: float, last: float, symbol: str, interval: str) -> Tuple[bool, float, float]:
    """
    Returns: (hit, score01, sl_level)
    Uses ATR*MULT fallback to FIXED % if ATR unavailable.
    """
    atr = _get_atr(symbol, interval, ATR_LEN)
    if atr and atr > 0:
        sl = entry - ATR_MULT * atr if side == "LONG" else entry + ATR_MULT * atr
        hit = last <= sl if side == "LONG" else last >= sl
        # Normalize score by how far beyond SL we are (cap at 1)
        dist = abs((sl - last) / (ATR_MULT * atr)) if ATR_MULT * atr else 1.0
        score = min(1.0, max(0.0, dist))
        return hit, score, sl
    else:
        sl = entry * (1.0 - FALLBACK_SL_PCT) if side == "LONG" else entry * (1.0 + FALLBACK_SL_PCT)
        hit = last <= sl if side == "LONG" else last >= sl
        # Score based on percent penetration
        base = entry * FALLBACK_SL_PCT
        dist = abs(sl - last) / base if base else 1.0
        score = min(1.0, max(0.0, dist))
        return hit, score, sl

def _latest_zone(symbol: str, interval: str) -> Optional[dict]:
    with get_db_connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT val, vah, poc, support_break_flag, resistance_break_flag
              FROM market.zone_levels
             WHERE symbol=%s AND interval=%s
             ORDER BY ts DESC
             LIMIT 1;
            """,
            (symbol, interval)
        )
        row = cur.fetchone()
    if not row:
        return None
    return {
        "val": row[0], "vah": row[1], "poc": row[2],
        "sb": bool(row[3]), "rb": bool(row[4])
    }

def _footprint_trail_exit(symbol: str, side: str, last: float) -> Tuple[bool, float]:
    """
    Simple zone-trail:
      LONG  -> trail to VAL; exit if close < VAL and support_break_flag
      SHORT -> trail to VAH; exit if close > VAH and resistance_break_flag
    """
    z = _latest_zone(symbol, INTERVAL)
    if not z:
        return (False, 0.0)

    if side == "LONG":
        val = float(z["val"] or 0.0)
        if val and last < val and z["sb"]:
            # score scales with breach depth vs (VAH-VAL); fallback to % if missing
            rng = abs(float(z["vah"] or 0.0) - val) or (0.006 * last)
            depth = (val - last) / rng if rng else 0.0
            return True, min(1.0, max(0.0, depth))
        return False, 0.0
    else:  # SHORT
        vah = float(z["vah"] or 0.0)
        if vah and last > vah and z["rb"]:
            rng = abs(vah - float(z["val"] or vah)) or (0.006 * last)
            depth = (last - vah) / rng if rng else 0.0
            return True, min(1.0, max(0.0, depth))
        return False, 0.0

def _detect_oi_unwind(symbol: str, side: str) -> Tuple[bool, float]:
    """
    Use CE snaps for LONG, PE snaps for SHORT.
    Heuristic:
      LONG  -> 'long unwind' if CE ltp↓ and CE oi↓ persistently
      SHORT -> 'short cover' if PE ltp↑ and PE oi↓ persistently
    """
    table = "options.ce_snapshot" if side == "LONG" else "options.pe_snapshot"
    with get_db_connection() as conn, conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT ltp, oi, oi_change, price_change
              FROM {table}
             WHERE symbol=%s
             ORDER BY ts DESC
             LIMIT %s;
            """,
            (symbol, max(3, OI_LOOKBACK_MIN))
        )
        rows = cur.fetchall()

    if not rows or len(rows) < 2:
        return (False, 0.0)

    # newest -> oldest
    ltp_dir = _dir(rows[0][0] - rows[-1][0])  # +1/-1 over window
    oi_dir  = _dir(rows[0][1] - rows[-1][1])

    if side == "LONG":
        # long unwind if price falling AND OI falling (calls being closed)
        if ltp_dir < 0 and oi_dir < 0:
            # confidence by magnitude of recent oi_change negativity
            neg_oi_chg = [abs(r[2]) for r in rows if (r[2] or 0) < 0]
            conf = min(1.0, (sum(neg_oi_chg) / (sum(abs(r[2] or 0) for r in rows) or 1.0)))
            return True, conf
    else:
        # short cover if price rising AND OI falling (puts being closed)
        if ltp_dir > 0 and oi_dir < 0:
            neg_oi_chg = [abs(r[2]) for r in rows if (r[2] or 0) < 0]
            conf = min(1.0, (sum(neg_oi_chg) / (sum(abs(r[2] or 0) for r in rows) or 1.0)))
            return True, conf

    return (False, 0.0)

def _dir(x: float) -> int:
    return 1 if x > 0 else (-1 if x < 0 else 0)
