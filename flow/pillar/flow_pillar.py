from __future__ import annotations

import json, math, configparser, ast, functools, importlib
from typing import Optional, Tuple, Dict, Any, List
from pathlib import Path  # 👈 add this

import numpy as np
import pandas as pd

from utils.db import get_db_connection

from pillars.common import (  # shared utils
    ema, atr, adx, obv_series, resample, last_metric, write_values, clamp,
    TZ, BaseCfg, min_bars_for_tf, maybe_trim_last_bar
)



# Where this file lives: .../flow/pillar/flow_pillar.py
FLOW_DIR   = Path(__file__).resolve().parent
DEFAULT_INI = FLOW_DIR / "flow_scenarios.ini"

print("🔥 FLOW_PILLAR DEFAULT_INI =", DEFAULT_INI)
print("🔥 Exists?", DEFAULT_INI.exists())

# ============================================================
# Config
# ============================================================

def _cfg(path: str = DEFAULT_INI) -> dict:
    """
    Load Flow pillar config from INI file.

    We expect at minimum:
      [flow]
      [flow_ml]                (optional)
      [flow_scenarios]
      [flow_scenario.*]        (one section per rule)
    """
    cp = configparser.ConfigParser(
        inline_comment_prefixes=(";", "#"),
        interpolation=None,
        strict=False
    )
    cp.read(path)

    # spread / liquidity fallbacks
    spread_bps_veto = cp.getfloat(
        "flow", "spread_bps_veto",
        fallback=cp.getfloat("liquidity", "max_spread_bps", fallback=40.0)
    )
    min_turnover = cp.getfloat(
        "flow", "min_turnover",
        fallback=cp.getfloat("liquidity", "min_turnover", fallback=0.0)
    )

    scenarios_list = [
        s.strip()
        for s in cp.get("flow_scenarios", "list", fallback="")
                  .replace("\n", " ")
                  .split(",")
        if s.strip()
    ]
    return {
        # core knobs
        "mfi_len":         cp.getint("flow", "mfi_len", fallback=14),
        "rvol_strong":     cp.getfloat("flow", "rvol_strong", fallback=1.5),
        "rvol_extreme":    cp.getfloat("flow", "rvol_extreme", fallback=2.0),
        "vol_cv_good":     cp.getfloat("flow", "vol_cv_good", fallback=0.50),
        "vol_cv_bad":      cp.getfloat("flow", "vol_cv_bad",  fallback=1.50),
        "voi_scale":       cp.getfloat("flow", "voi_scale",    fallback=10.0),
        "roll_look":       cp.getint("flow",  "roll_look",     fallback=5),
        "roll_drop_pct":   cp.getfloat("flow","roll_drop_pct", fallback=0.35),
        "spread_bps_veto": spread_bps_veto,
        "min_rvol_veto":   cp.getfloat("flow", "min_rvol_veto",  fallback=0.50),
        "vol_rank_floor":  cp.getfloat("flow", "vol_rank_floor", fallback=0.20),
        "min_turnover":    min_turnover,

        # scenario engine
        "rules_mode": cp.get("flow", "rules_mode", fallback="additive").strip().lower(),
        "clamp_low":  cp.getfloat("flow", "clamp_low",  fallback=0.0),
        "clamp_high": cp.getfloat("flow", "clamp_high", fallback=100.0),
        "write_scenarios_debug": cp.getboolean("flow", "write_scenarios_debug", fallback=False),
        "min_bars": cp.getint("flow", "min_bars", fallback=120),

        # 🔹 ML config (generic ml_pillars table)
        "ml_enabled":      cp.getboolean("flow_ml", "enabled", fallback=False),
        "ml_blend_weight": cp.getfloat("flow_ml", "blend_weight", fallback=0.35),
        "ml_table":        cp.get("flow_ml", "table", fallback="indicators.ml_pillars").strip(),
        "ml_pillar":       cp.get("flow_ml", "pillar", fallback="flow").strip(),
        "ml_target":       cp.get("flow_ml", "main_target", fallback="bull_2pct_4h").strip(),
        "ml_version":      cp.get("flow_ml", "version", fallback="v1").strip(),
        "ml_callback":     cp.get("flow_ml", "callback", fallback="").strip(),
        "ml_veto_if_prob_lt": cp.get("flow_ml", "veto_if_prob_lt", fallback="").strip(),

        # 🔹 scenarios list
        "scenarios_list": [
            s.strip()
            for s in cp.get("flow_scenarios", "list", fallback="")
                      .replace("\n", " ")
                      .split(",")
            if s.strip()
        ],

        "cp": cp,
    }


def _json_safe(x: Any):
    """Convert NaN/inf → None so Postgres JSON accepts it."""
    if isinstance(x, float) and not np.isfinite(x):
        return None
    return x


# ============================================================
# Calibration helpers (Flow ML)
# ============================================================

@functools.lru_cache(maxsize=1)
def _load_flow_calibration() -> List[Dict[str, float]]:
    """
    Load Flow ML calibration buckets from DB.

    Schema (from flow_bucket_stats.py):

        CREATE TABLE IF NOT EXISTS indicators.flow_calibration_4h (
            bucket            integer PRIMARY KEY,
            p_min             double precision,
            p_max             double precision,
            avg_p             double precision,
            realized_up_rate  double precision,
            n                 bigint
        );
    """
    rows: List[Dict[str, float]] = []
    sql = """
        SELECT bucket, p_min, p_max, realized_up_rate
          FROM indicators.flow_calibration_4h
         ORDER BY p_min
    """
    try:
        with get_db_connection() as conn, conn.cursor() as cur:
            cur.execute(sql)
            for bucket, p_min, p_max, realized_up_rate in cur.fetchall():
                if realized_up_rate is None:
                    continue
                rows.append(
                    {
                        "bucket": int(bucket),
                        "p_min": float(p_min),
                        "p_max": float(p_max),
                        "realized_up_rate": float(realized_up_rate),
                    }
                )
    except Exception:
        # If table missing or query fails, we fallback to identity calibration
        rows = []

    return rows


def _calibrate_prob(p: Optional[float]) -> float:
    """
    Map raw ML probability -> calibrated probability using bucket stats.

    - If no calibration table, return p unchanged.
    - If p is outside all bucket ranges, clamp to last bucket value.
    """
    if p is None:
        return 0.0

    try:
        if not np.isfinite(p):
            p = 0.0
    except Exception:
        p = 0.0

    p = max(0.0, min(1.0, float(p)))
    buckets = _load_flow_calibration()
    if not buckets:
        # No calibration available -> identity
        return p

    # Find matching bucket
    for row in buckets:
        p_min = row.get("p_min", 0.0)
        p_max = row.get("p_max", 1.0)
        if p_min <= p < p_max:
            cal = row.get("realized_up_rate", p)
            if cal is None or not np.isfinite(cal):
                return p
            return max(0.0, min(1.0, float(cal)))

    # If we didn't hit any bucket (e.g. p == 1.0), use last bucket
    last = buckets[-1]
    cal = last.get("realized_up_rate", p)
    if cal is None or not np.isfinite(cal):
        return p
    return max(0.0, min(1.0, float(cal)))


def _find_prob_bucket(p: Optional[float]) -> Optional[int]:
    """
    Return calibration bucket index for a given prob (after calibration),
    purely for debug / explainability.
    """
    if p is None:
        return None

    try:
        if not np.isfinite(p):
            return None
    except Exception:
        return None

    p = max(0.0, min(1.0, float(p)))
    buckets = _load_flow_calibration()
    if not buckets:
        return None

    for row in buckets:
        p_min = row.get("p_min", 0.0)
        p_max = row.get("p_max", 1.0)
        if p_min <= p < p_max:
            return int(row.get("bucket"))

    return int(buckets[-1].get("bucket"))

# ============================================================
# Flow-specific min bars (rules-only)
# ============================================================

def min_bars_for_tf_rules(tf: str) -> int:
    """
    Flow pillar's own relaxed thresholds for RULES scoring.

    This does NOT affect other pillars – only used inside score_flow().
    """
    tf = str(tf).lower()
    rules_min = {
        "15m": 2,   # 5 bars = 75min
        "30m": 2,   # 2 hours
        "60m": 1,   # 3 hours
        "120m": 1,  # 3 x 2h bars
        "240m": 1,  # 3 x 4h bars
    }
    return rules_min.get(tf, 3)

# ============================================================
# Min-bars for RULES vs ML
# ============================================================

def min_bars_for_tf_rules(tf: str) -> int:
    """
    Lighter requirement for rules-only Flow.
    Enough for RVOL/MFI/OBV to be stable, but not ML.
    """
    tf = str(tf)
    if tf == "15m":
        return 2
    if tf == "30m":
        return 2
    if tf == "60m":
        return 1
    if tf in ("120m", "240m"):
        return 1
    return 1


# ============================================================
# Safe AST-based evaluator (cached)
# ============================================================

_ALLOWED_BOOL_OPS   = {ast.And, ast.Or}
_ALLOWED_UNARY_BOOL = {ast.Not}
_ALLOWED_NUM_OPS    = {
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod,
    ast.FloorDiv, ast.Pow, ast.USub, ast.UAdd
}

@functools.lru_cache(maxsize=2048)
def _compile_rule(expr: str) -> Optional[ast.AST]:
    if not expr or not expr.strip():
        return None
    try:
        return ast.parse(expr, mode="eval")
    except Exception:
        return None

def _safe_eval(expr_or_tree: str | ast.AST, scope: Dict[str, Any]) -> bool:
    if not expr_or_tree:
        return False

    if isinstance(expr_or_tree, str):
        tree = _compile_rule(expr_or_tree)
        if not tree:
            return False
    else:
        tree = expr_or_tree

    def _eval(node):
        if isinstance(node, ast.Expression):
            return _eval(node.body)

        # bool ops: and/or
        if isinstance(node, ast.BoolOp) and type(node.op) in _ALLOWED_BOOL_OPS:
            vals = [_eval(v) for v in node.values]
            vals = [bool(v) for v in vals]
            if isinstance(node.op, ast.And):
                return all(vals)
            else:
                return any(vals)

        # unary: not / +/- num
        if isinstance(node, ast.UnaryOp):
            if type(node.op) in _ALLOWED_UNARY_BOOL:
                return not bool(_eval(node.operand))
            if type(node.op) in _ALLOWED_NUM_OPS:
                v = _eval(node.operand)
                if not isinstance(v, (int, float)):
                    v = float(v)
                if isinstance(node.op, ast.USub):
                    return -v
                else:
                    return +v

        # binary numeric ops
        if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_NUM_OPS:
            a = _eval(node.left)
            b = _eval(node.right)
            if not isinstance(a, (int, float)):
                a = float(a)
            if not isinstance(b, (int, float)):
                b = float(b)
            if isinstance(node.op, ast.Add):      return a + b
            if isinstance(node.op, ast.Sub):      return a - b
            if isinstance(node.op, ast.Mult):     return a * b
            if isinstance(node.op, ast.Div):      return a / b if b != 0 else 0.0
            if isinstance(node.op, ast.FloorDiv): return a // b if b != 0 else 0.0
            if isinstance(node.op, ast.Mod):      return a % b if b != 0 else 0.0
            if isinstance(node.op, ast.Pow):      return a ** b

        # comparisons (single-op only)
        if isinstance(node, ast.Compare) and len(node.ops) == 1:
            a = _eval(node.left)
            b = _eval(node.comparators[0])
            op = node.ops[0]
            try:
                if isinstance(op, ast.Lt):    return a <  b
                if isinstance(op, ast.LtE):   return a <= b
                if isinstance(op, ast.Gt):    return a >  b
                if isinstance(op, ast.GtE):   return a >= b
                if isinstance(op, ast.Eq):    return a == b
                if isinstance(op, ast.NotEq): return a != b
            except Exception:
                return False

        # name lookups
        if isinstance(node, ast.Name):
            return scope.get(node.id, False)

        # constants
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float, bool, str)):
            return node.value

        raise ValueError("Disallowed token in scenario expression")

    try:
        return bool(_eval(tree))
    except Exception:
        return False

# ============================================================
# Helpers
# ============================================================

def _mfi(high: pd.Series, low: pd.Series, close: pd.Series, vol: pd.Series, n: int = 14) -> pd.Series:
    high  = pd.to_numeric(high,  errors="coerce")
    low   = pd.to_numeric(low,   errors="coerce")
    close = pd.to_numeric(close, errors="coerce")
    vol   = pd.to_numeric(vol,   errors="coerce").fillna(0.0)

    tp  = (high + low + close) / 3.0
    rmf = tp * vol

    diff = tp.diff()
    pos = pd.Series(np.where(diff > 0, rmf, 0.0), index=tp.index)
    neg = pd.Series(np.where(diff < 0, rmf, 0.0), index=tp.index)

    win = max(5, n // 2)
    pos_n = pos.rolling(n, min_periods=win).sum()
    neg_n = neg.rolling(n, min_periods=win).sum()

    mr = pos_n / neg_n.replace(0, np.nan)
    mfi = 100 - (100 / (1 + mr))
    mfi = mfi.clip(0, 100).ffill().fillna(50.0)
    return mfi

def _ist(ts: pd.Timestamp) -> pd.Timestamp:
    try:
        return ts.tz_convert("Asia/Kolkata")
    except Exception:
        return ts.tz_localize("UTC").tz_convert("Asia/Kolkata")

def _hh(series: pd.Series, look: int = 20) -> bool:
    if len(series) < look + 1:
        return False
    return bool(float(series.iloc[-1]) >= float(series.rolling(look).max().iloc[-2]))

def _ll(series: pd.Series, look: int = 20) -> bool:
    if len(series) < look + 1:
        return False
    return bool(float(series.iloc[-1]) <= float(series.rolling(look).min().iloc[-2]))

def _clean_json_dict(d: Dict[str, Any]) -> Dict[str, Any]:
    """
    Replace NaN/inf floats with None so Postgres JSON accepts them.
    """
    clean = {}
    for k, v in d.items():
        if isinstance(v, float):
            if not np.isfinite(v):
                clean[k] = None
            else:
                clean[k] = float(v)
        else:
            clean[k] = v
    return clean

def _load_ml_prob_from_db(
    symbol: str,
    kind: str,
    tf: str,
    ts,
    cfg: dict,
) -> Optional[float]:
    """
    Read latest prob_up from indicators.ml_pillars for this symbol/kind/tf/ts.

    Uses config from [flow_ml] section:
      table       = indicators.ml_pillars
      pillar      = flow
      main_target = flow_ml.target.bull_2pct_4h  (or just bull_2pct_4h)
      version     = xgb_v1
    """
    cp = cfg["cp"]

    table = cp.get("flow_ml", "table", fallback="indicators.ml_pillars").strip()
    pillar = cp.get("flow_ml", "pillar", fallback="flow").strip()
    target_raw = cp.get("flow_ml", "main_target", fallback="bull_2pct_4h").strip()
    version = cp.get("flow_ml", "version", fallback="xgb_v1").strip()

    if not target_raw:
        return None

    # Support both “bull_2pct_4h” and “flow_ml.target.bull_2pct_4h”
    if target_raw.startswith("flow_ml.target."):
        target_name = target_raw
    else:
        target_name = f"flow_ml.target.{target_raw}"

    market_type = "futures" if kind.lower() == "futures" else "spot"

    sql = f"""
        SELECT prob_up
          FROM {table}
         WHERE symbol      = %s
           AND pillar      = %s
           AND market_type = %s
           AND tf          = %s
           AND target_name = %s
           AND version     = %s
           AND ts <= %s
         ORDER BY ts DESC
         LIMIT 1
    """

    with get_db_connection() as conn, conn.cursor() as cur:
        cur.execute(
            sql,
            (symbol, pillar, market_type, tf, target_name, version, ts),
        )
        row = cur.fetchone()

    if not row:
        return None
    val = row[0]
    if val is None:
        return None

    try:
        return float(val)
    except Exception:
        return None

def _find_prob_bucket(p: Optional[float]) -> Optional[int]:
    """
    Given a calibrated probability p, return bucket index based on
    indicators.flow_calibration_4h ranges (p_min, p_max).

    If no calibration exists, returns None.
    """
    if p is None:
        return None

    try:
        p = float(p)
    except Exception:
        return None

    p = max(0.0, min(1.0, p))
    buckets = _load_flow_calibration()
    if not buckets:
        return None

    for idx, row in enumerate(buckets):
        p_min = row.get("p_min", 0.0)
        p_max = row.get("p_max", 1.0)
        if p_min <= p < p_max:
            return idx

    # If p == 1.0 or past last range → last bucket
    return len(buckets) - 1

def _load_ml_prob_from_pillars(
    symbol: str,
    kind: str,
    tf: str,
    ts,
    cfg: dict,
) -> Optional[Tuple[Optional[float], Optional[float]]]:
    """
    Read ML probabilities for Flow from indicators.ml_pillars.

    We look for the latest row *at or before* this ts.
    """
    cp = cfg["cp"]
    pillar = cp.get("flow_ml", "pillar", fallback="flow").strip()
    target_name = cp.get("flow_ml", "main_target", fallback="flow_ml.target.bull_2pct_4h").strip()
    version = cp.get("flow_ml", "version", fallback="xgb_v1").strip()
    market_type = kind.lower()
    interval = tf

    sql = """
        SELECT prob_up, prob_down, ts
          FROM indicators.ml_pillars
         WHERE pillar       = %s
           AND market_type  = %s
           AND interval     = %s
           AND target_name  = %s
           AND version      = %s
           AND symbol       = %s
           AND ts <= %s
         ORDER BY ts DESC
         LIMIT 1
    """
    params = (pillar, market_type, interval, target_name, version, symbol, ts)

    try:
        with get_db_connection() as conn, conn.cursor() as cur:
            cur.execute(sql, params)
            row = cur.fetchone()
    except Exception as e:
        print(f"[FLOW_ML] ERROR loading ml_pillars for {symbol} {kind} {tf} ts={ts}: {e}")
        return None

    if not row:
        # TEMP DEBUG – see when we’re not getting anything
        print(f"[FLOW_ML] NO ml_pillars row for {symbol} {kind} tf={tf} ts<={ts}")
        return None

    p_up, p_dn, ts_row = row
    # TEMP DEBUG
    print(f"[FLOW_ML] HIT ml_pillars {symbol} {kind} tf={tf} bar_ts={ts} ml_ts={ts_row} p_up={p_up} p_dn={p_dn}")

    try:
        p_up = float(p_up) if p_up is not None else None
    except Exception:
        p_up = None
    try:
        p_dn = float(p_dn) if p_dn is not None else None
    except Exception:
        p_dn = None

    return (p_up, p_dn)


# ============================================================
# Feature builder
# ============================================================

def _build_features(
    dtf: pd.DataFrame,
    symbol: str,
    kind: str,
    tf: str,
    cfg: dict,
    context: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:

    # --- Session flags ---
    ts_ist = _ist(dtf.index[-1])
    minutes = ts_ist.hour * 60 + ts_ist.minute
    open_min, close_min = 9*60 + 15, 15*60 + 30
    in_session = (open_min <= minutes <= close_min)

    last_vol_zero = (float(dtf["volume"].iloc[-1]) == 0.0)

    # ignore last bar if off-session & vol=0
    use = dtf.iloc[:-1] if ((not in_session) and last_vol_zero and len(dtf) > 1) else dtf
    o = use["open"]
    c = use["close"]
    h = use["high"]
    l = use["low"]
    v = use["volume"].fillna(0.0)

    # --- MFI ---
    mfi = _mfi(h, l, c, v, n=cfg["mfi_len"])
    mfi_val = float(mfi.iloc[-1] if len(mfi) else 50.0)
    mfi_slope = float(mfi.diff().iloc[-1]) if len(mfi) > 1 else 0.0
    mfi_up = (mfi_slope > 0)
    mfi_prev_high = float(mfi.iloc[-10:-1].max()) if len(mfi) >= 10 else float(mfi_val)

    # --- OBV + structure ---
    obv = obv_series(c, v) if len(c) and len(v) else pd.Series([], dtype=float)
    obv_ema20 = obv.ewm(span=20, adjust=False).mean() if len(obv) else obv
    obv_above_ema = bool(len(obv) and len(obv_ema20) and (float(obv.iloc[-1]) > float(obv_ema20.iloc[-1])))

    price_hh = _hh(h, look=20)
    price_higher_high = price_hh
    price_lower_low = _ll(l, look=20)

    if len(obv) >= 12:
        recent_min = float(obv.tail(5).min())
        prior_min  = float(obv.shift(5).tail(5).min())
        obv_higher_low = bool(recent_min > prior_min)
    else:
        obv_higher_low = False

    obv_hh = _hh(obv, look=20) if len(obv) else False

    # --- RVOL & stability ---
    v_avg = float(v.rolling(20).mean().iloc[-1]) if len(v) >= 20 else float(v.mean() or 1.0)
    rvol_now = float(v.iloc[-1] / max(1e-9, v_avg))
    rvol_strong = (rvol_now >= cfg["rvol_strong"])
    rvol_extreme = (rvol_now >= cfg["rvol_extreme"])
    vol_cv20 = float((v.rolling(20).std(ddof=1) / v.rolling(20).mean()).iloc[-1] or 0.0) if len(v) >= 20 else 1.0

    if len(v) >= 60:
        vol_rank60 = float((v.tail(60).rank(pct=True).iloc[-1]))
    else:
        vol_rank60 = 0.5

    # --- VAH/VAL via context → DB fallback ---
    if context and f"VP.VAH|{tf}" in context:
        vah = context[f"VP.VAH|{tf}"]
    else:
        vah = last_metric(symbol, kind, tf, "VP.VAH")

    if context and f"VP.VAL|{tf}" in context:
        val = context[f"VP.VAL|{tf}"]
    else:
        val = last_metric(symbol, kind, tf, "VP.VAL")

    prev_close = float(c.iloc[-2]) if len(c) > 1 else float(c.iloc[-1])
    last_close = float(c.iloc[-1])
    last_high  = float(h.iloc[-1])
    last_low   = float(l.iloc[-1])

    if vah is not None:
        try: vah = float(vah)
        except Exception: vah = None
    if val is not None:
        try: val = float(val)
        except Exception: val = None

    near_vah_break = bool((vah is not None) and (prev_close <= vah) and (last_close > vah))
    close_vah_break_fail = bool((vah is not None) and (last_high > vah) and (last_close <= vah))
    near_val_break = bool((val is not None) and (prev_close >= val) and (last_close < val))
    close_val_break_fail = bool((val is not None) and (last_low < val) and (last_close >= val))

    # --- Candle anatomy ---
    body = abs(last_close - float(o.iloc[-1]))
    rng = max(1e-9, last_high - last_low)
    upper_wick = last_high - max(last_close, float(o.iloc[-1]))
    lower_wick = min(last_close, float(o.iloc[-1])) - last_low
    wick_ratio = float((upper_wick + lower_wick) / (body if body > 0 else 1e-9))

    bullish_rev = (last_close > float(o.iloc[-1])) and (lower_wick >= 2.0 * body)
    bearish_rev = (last_close < float(o.iloc[-1])) and (upper_wick >= 2.0 * body)
    price_reversal_candle = bool(bullish_rev or bearish_rev)

    # --- VOI & roll (futures only) ---
    voi_long_build = voi_short_build = voi_short_cover = voi_long_unwind = False
    voi_mag = 0.0
    roll_trap = False
    if "oi" in use.columns and kind == "futures":
        doi = use["oi"].diff()
        pc  = c.diff()
        last_doi = float(doi.iloc[-1]) if len(doi) else 0.0
        last_pc  = float(pc.iloc[-1])  if len(pc)  else 0.0

        if last_pc > 0 and last_doi > 0:
            voi_long_build = True
        elif last_pc < 0 and last_doi > 0:
            voi_short_build = True
        elif last_pc > 0 and last_doi < 0:
            voi_short_cover = True
        elif last_pc < 0 and last_doi < 0:
            voi_long_unwind = True

        voi_mag = abs(last_doi) / max(1e-9, float(v.iloc[-1]))

        look = cfg["roll_look"]
        if len(use) > look + 1:
            oi0 = float(use["oi"].iloc[-look-1])
            oin = float(use["oi"].iloc[-1])
            if oi0 > 0:
                drop = (oi0 - oin) / oi0
                if drop >= cfg["roll_drop_pct"] and rvol_strong:
                    roll_trap = True

    # --- Liquidity (context → DB) ---
    spread_bps = None
    found_spread = False
    if context:
        for m in ("MKT.spread_bps", "LIQ.spread_bps", "MKT.spread_bps.mean"):
            key = f"{m}|{tf}"
            if key in context and context[key] is not None:
                spread_bps = float(context[key])
                found_spread = True
                break

    if not found_spread:
        for m in ("MKT.spread_bps", "LIQ.spread_bps", "MKT.spread_bps.mean"):
            x = last_metric(symbol, kind, tf, m)
            if x is not None:
                try:
                    spread_bps = float(x)
                    break
                except Exception:
                    spread_bps = None

    close_last = float(use["close"].iloc[-1])
    vol_last   = float(use["volume"].iloc[-1] or 0.0)
    turnover_est = close_last * vol_last
    min_turn = float(cfg["min_turnover"])

    illiquid = False
    if (spread_bps is not None) and np.isfinite(spread_bps) and (spread_bps >= cfg["spread_bps_veto"]):
        illiquid = True
    elif in_session and (not last_vol_zero) and (rvol_now < cfg["min_rvol_veto"]) and (vol_rank60 <= cfg["vol_rank_floor"]):
        illiquid = True
    elif (min_turn > 0.0) and in_session and (not last_vol_zero) and np.isfinite(turnover_est) and (turnover_est < min_turn):
        illiquid = True

    # --- Meta metrics via context / DB (PCR/MWPL/PIVOT etc.) ---
    def get_meta_metric(name: str) -> float:
        if context and name in context:
            try:
                return float(context[name])
            except Exception:
                return 0.0
        val = last_metric(symbol, kind, tf, name)
        return float(val) if val is not None else 0.0

    pcr_vol_chg = get_meta_metric("FNO.pcr.vol.chg_pct")
    mwpl_pct    = get_meta_metric("FNO.mwpl.pct")
    r1_dist     = get_meta_metric("PIVOT.r1.dist_pct")
    pcr_oi_chg  = get_meta_metric("FNO.pcr.oi.chg_pct")

    # --- Daily F&O context (from orchestrator context) ---
    daily_fut_buildup = context.get("daily_fut_buildup", "") if context else ""
    daily_opt_call_oi_chg_pct = context.get("daily_opt_call_oi_chg_pct", 0.0) if context else 0.0
    daily_opt_put_oi_chg_pct  = context.get("daily_opt_put_oi_chg_pct", 0.0) if context else 0.0

    # --- Session mini flags ---
    near_open  = in_session and (minutes - open_min  <= 30)
    near_close = in_session and (close_min - minutes <= 30)
    mid_lunch  = in_session and (12*60 <= minutes <= 13*60 + 30)

    feats: Dict[str, Any] = {
        "open": float(o.iloc[-1]), "close": last_close, "high": last_high, "low": last_low,
        "volume": float(v.iloc[-1]), "price": last_close,

        # MFI
        "mfi_val": float(mfi_val),
        "mfi_slope": float(mfi_slope),
        "mfi_up": bool(mfi_up),
        "mfi_prev_high": float(mfi_prev_high),

        # OBV structure
        "obv_above_ema": bool(obv_above_ema),
        "price_hh": bool(price_hh),
        "price_higher_high": bool(price_higher_high),
        "price_lower_low": bool(price_lower_low),
        "obv_hh": bool(obv_hh),
        "obv_higher_low": bool(obv_higher_low),

        # RVOL & stability
        "rvol_now": float(rvol_now),
        "rvol_strong": bool(rvol_strong),
        "rvol_extreme": bool(rvol_extreme),
        "vol_cv20": float(vol_cv20),
        "vol_rank60": float(vol_rank60),

        # VAH/VAL breakout/fail
        "near_vah_break": bool(near_vah_break),
        "near_val_break": bool(near_val_break),
        "close_vah_break_fail": bool(close_vah_break_fail),
        "close_val_break_fail": bool(close_val_break_fail),

        # candle anatomy / reversal
        "wick_ratio": float(wick_ratio),
        "price_reversal_candle": bool(price_reversal_candle),

        # VOI & roll
        "voi_long_build": bool(voi_long_build),
        "voi_short_build": bool(voi_short_build),
        "voi_short_cover": bool(voi_short_cover),
        "voi_long_unwind": bool(voi_long_unwind),
        "voi_mag": float(voi_mag),
        "roll_trap": bool(roll_trap),

        # Liquidity
        # Liquidity
        "spread_bps": float(spread_bps) if spread_bps is not None and np.isfinite(spread_bps) else None,
        "turnover_est": float(turnover_est),
        "illiquid_flag": bool(illiquid),


        # Meta metrics (pre-fetched)
        "pcr_vol_chg": pcr_vol_chg,
        "mwpl_pct": mwpl_pct,
        "r1_dist": r1_dist,
        "pcr_oi_chg": pcr_oi_chg,

        # Daily F&O features
        "daily_fut_buildup": daily_fut_buildup,
        "daily_opt_call_oi_chg_pct": float(daily_opt_call_oi_chg_pct),
        "daily_opt_put_oi_chg_pct": float(daily_opt_put_oi_chg_pct),

        # Session
        "in_session": bool(in_session),
        "near_open": bool(near_open),
        "near_close": bool(near_close),
        "mid_lunch": bool(mid_lunch),
        "last_vol_zero": bool(last_vol_zero),
    }

    # Shim to support legacy last_metric(...) usage in rules without hitting DB per call
    feats["last_metric"] = lambda k: feats.get(
        k.replace("FNO.pcr.vol.chg_pct", "pcr_vol_chg")
         .replace("FNO.mwpl.pct", "mwpl_pct")
         .replace("PIVOT.r1.dist_pct", "r1_dist")
         .replace("FNO.pcr.oi.chg_pct", "pcr_oi_chg"),
        0.0
    )

    return feats

# ============================================================
# Scenario engine (flow_scenario.*)
# ============================================================

def _run_scenarios(features: Dict[str, Any], cfg: dict) -> Tuple[float, bool, Dict[str, float], List[tuple]]:
    """
    Scenario engine driven by INI:

      [flow_scenarios]
      list = mfi_rising, obv_confirmed, ...

      [flow_scenario.mfi_rising]
      when  = ...
      score = +15
      bonus_when = ...
      bonus      = ...
      set_veto   = true/false
    """
    cp = cfg["cp"]
    total = 0.0
    veto = False
    parts: Dict[str, float] = {}
    fired: List[tuple[str, float]] = []

    for name in cfg["scenarios_list"]:
        sec = f"flow_scenario.{name}"
        if not cp.has_section(sec):
            continue

        # main condition
        when_str = cp.get(sec, "when", fallback="").strip()
        if not when_str:
            continue

        when_tree = _compile_rule(when_str)
        if not when_tree:
            continue

        try:
            ok = _safe_eval(when_tree, features)
        except Exception:
            ok = False

        if not ok:
            continue

        # base score
        sc_raw = cp.get(sec, "score", fallback="0").strip()
        sc = float(sc_raw or 0.0)

        if cfg["rules_mode"] == "override":
            total = sc
        else:
            total += sc

        parts[name] = sc
        fired.append((name, sc))

        # optional bonus
        if cp.has_option(sec, "bonus_when"):
            try:
                bonus_str = cp.get(sec, "bonus_when", fallback="").strip()
                if bonus_str:
                    bonus_tree = _compile_rule(bonus_str)
                    if bonus_tree and _safe_eval(bonus_tree, features):
                        bonus_val_raw = cp.get(sec, "bonus", fallback="0").strip()
                        bonus_val = float(bonus_val_raw or 0.0)
                        if cfg["rules_mode"] == "override":
                            total = bonus_val
                        else:
                            total += bonus_val
                        parts[name + ".bonus"] = bonus_val
                        fired.append((name + ".bonus", bonus_val))
            except Exception:
                # ignore broken bonus rules
                pass

        # veto flag
        if cp.get(sec, "set_veto", fallback="false").strip().lower() in {"1", "true", "yes", "on"}:
            veto = True

    return total, veto, parts, fired

# ============================================================
# Public API
# ============================================================

# ============================================================
# Public API
# ============================================================
def score_flow(
    symbol: str,
    kind: str,
    tf: str,
    df5: pd.DataFrame,
    base: BaseCfg,
    ini_path=DEFAULT_INI,
    context: Optional[Dict[str, Any]] = None,
):
    """
    Scenario-driven Flow pillar.

    NEW LAYOUT:
      - FLOW.score          → rules-only (no ML blend)
      - FLOW.veto_flag      → rules-only veto
      - FLOW.ml_score       → calibrated ML score (0–100), if available
      - FLOW.ml_p_up_cal    → calibrated long prob (0–1), if available
      - FLOW.ml_p_down_cal  → 1 - ml_p_up_cal
      - FLOW.fused_score    → rules + ML fused score (if ML enabled)
      - FLOW.fused_veto     → veto after fusion
      - FLOW.debug_ctx      → includes rules_score, ml_prob_raw/cal, ml_bucket, fused_score

    RETURNS:
      (ts, rules_score, rules_veto, ml_score, fused_score, fused_veto)
    """

    # --- 0. Prepare TF dataframe ---
    dftf = df5
    if tf != "15m":
        dftf = resample(df5, tf)

    dftf = maybe_trim_last_bar(dftf)

    # Flow-specific min bars (rules-only)
    n = len(dftf)
    need = min_bars_for_tf_rules(tf)
    if n < need:
        print(f"[FLOW] only {n} bars for tf={tf}, need {need} for rules → skipping")
        return None

    # For ML layer (we just need a couple of bars so ts is meaningful)
    ml_min = 2

    cfg = _cfg(ini_path)

    # -----------------------------
    # 1. RULES ENGINE (always runs)
    # -----------------------------
    feats = _build_features(dftf, symbol, kind, tf, cfg, context=context)

    total, veto_scen, parts, fired = _run_scenarios(feats, cfg)

    # auto-veto on hard evidence
    auto_veto = bool(feats["illiquid_flag"] or feats["roll_trap"])
    rules_score = float(clamp(total, cfg["clamp_low"], cfg["clamp_high"]))
    rules_veto = bool(veto_scen or auto_veto)

    ts = dftf.index[-1].to_pydatetime().replace(tzinfo=TZ)

    # Defaults (in case ML is off or fails)
    ml_prob_raw = None
    ml_prob_cal = None
    ml_bucket   = None
    ml_score    = None
    fused_score = rules_score
    fused_veto  = rules_veto

    # -----------------------------
    # 2. OPTIONAL ML LAYER
    # -----------------------------
    try:
        cp = cfg["cp"]
        ml_enabled = cp.getboolean("flow_ml", "enabled", fallback=False)

        # We only *use* ML on 15m for now (that's what you've trained).
        # Other TFs still get fused_score == rules_score.
        if ml_enabled and n >= ml_min:
            ml_prob = None

            # --- get prob from ml_pillars ---
            res = _load_ml_prob_from_pillars(symbol, kind, tf, ts, cfg)
            if res is not None:
                prob_up, prob_down = res
                if prob_up is not None:
                    ml_prob = prob_up
                elif prob_down is not None:
                    ml_prob = 1.0 - prob_down

            # optional callback fallback (fine to leave as-is)
            if ml_prob is None:
                cb_path = cp.get("flow_ml", "callback", fallback="").strip()
                if cb_path:
                    mod_name, _, fn_name = cb_path.rpartition(".")
                    if mod_name and fn_name:
                        mod = importlib.import_module(mod_name)
                        fn = getattr(mod, fn_name)
                        ml_prob = float(
                            fn(symbol=symbol, kind=kind, tf=tf, ts=ts, features=feats)
                        )
                        if not np.isfinite(ml_prob):
                            ml_prob = 0.0

            if ml_prob is not None:
                # Safety clamp raw
                try:
                    ml_prob_raw = float(ml_prob)
                    if not np.isfinite(ml_prob_raw):
                        ml_prob_raw = 0.0
                except Exception:
                    ml_prob_raw = 0.0

                ml_prob_raw = max(0.0, min(1.0, ml_prob_raw))

                # 2.2 Calibration hook (bucket → calibrated prob)
                ml_prob_cal = _calibrate_prob(ml_prob_raw)
                ml_bucket   = _find_prob_bucket(ml_prob_cal)

                ml_score = round(100.0 * ml_prob_cal, 2)

                # -----------------------------
                # 3. FUSION (rules + ML)
                # -----------------------------
                w = float(cp.get("flow_ml", "blend_weight", fallback="0.35"))
                w = max(0.0, min(1.0, w))

                base_prob = float(rules_score) / 100.0
                fused_prob = (1.0 - w) * base_prob + w * ml_prob_cal
                fused_prob = max(0.0, min(1.0, fused_prob))

                fused_score = round(100.0 * fused_prob, 2)
                fused_veto  = rules_veto

                # Optional veto rule based on prob threshold
                veto_if_lt_str = cp.get("flow_ml", "veto_if_prob_lt", fallback="").strip()
                if veto_if_lt_str:
                    try:
                        thr = float(veto_if_lt_str)
                        # allow 55 or 0.55 formats
                        if thr > 1.0:
                            thr = thr / 100.0
                        if fused_prob < thr:
                            fused_veto = True
                    except Exception:
                        pass

    except Exception as e:
        # Keep the pillar robust — if ML dies, rules still work.
        print(f"[FLOW_ML] ERROR during ML blend for {symbol} {kind} {tf} @ {ts}: {e}")
        ml_prob_raw = None
        ml_prob_cal = None
        ml_bucket   = None
        ml_score    = None
        fused_score = rules_score
        fused_veto  = rules_veto

    # -----------------------------
    # 4. WRITE METRICS
    # -----------------------------
    rows = [
        # Rules-only
        (
            symbol,
            kind,
            tf,
            ts,
            "FLOW.score",
            float(rules_score),
            json.dumps({}),
            base.run_id,
            base.source,
        ),
        (
            symbol,
            kind,
            tf,
            ts,
            "FLOW.veto_flag",
            1.0 if rules_veto else 0.0,
            "{}",
            base.run_id,
            base.source,
        ),
    ]

    # ML-only metrics (if available)
    if ml_score is not None and ml_prob_cal is not None:
        rows.extend(
            [
                (
                    symbol,
                    kind,
                    tf,
                    ts,
                    "FLOW.ml_score",
                    float(ml_score),
                    "{}",
                    base.run_id,
                    base.source,
                ),
                (
                    symbol,
                    kind,
                    tf,
                    ts,
                    "FLOW.ml_p_up_cal",
                    float(ml_prob_cal),
                    "{}",
                    base.run_id,
                    base.source,
                ),
                (
                    symbol,
                    kind,
                    tf,
                    ts,
                    "FLOW.ml_p_down_cal",
                    float(1.0 - ml_prob_cal),
                    "{}",
                    base.run_id,
                    base.source,
                ),
            ]
        )
        # Optional: bucket index as a separate metric
        if ml_bucket is not None:
            rows.append(
                (
                    symbol,
                    kind,
                    tf,
                    ts,
                    "FLOW.ml_bucket",
                    float(ml_bucket),
                    "{}",
                    base.run_id,
                    base.source,
                )
            )

    # Fused metrics (always written, but == rules if ML not ready)
    rows.extend(
        [
            (
                symbol,
                kind,
                tf,
                ts,
                "FLOW.fused_score",
                float(fused_score),
                "{}",
                base.run_id,
                base.source,
            ),
            (
                symbol,
                kind,
                tf,
                ts,
                "FLOW.fused_veto",
                1.0 if fused_veto else 0.0,
                "{}",
                base.run_id,
                base.source,
            ),
        ]
    )

    # Debug ctx
    debug_ctx = {
        "mfi_val": feats["mfi_val"],
        "mfi_slope": feats["mfi_slope"],
        "obv_above_ema": feats["obv_above_ema"],
        "price_hh": feats["price_hh"],
        "obv_hh": feats["obv_hh"],
        "rvol": feats["rvol_now"],
        "vol_cv20": feats["vol_cv20"],
        "vol_rank60": feats["vol_rank60"],
        "voi_long_build": feats["voi_long_build"],
        "voi_short_build": feats["voi_short_build"],
        "voi_short_cover": feats["voi_short_cover"],
        "voi_long_unwind": feats["voi_long_unwind"],
        "voi_mag": feats["voi_mag"],
        "spread_bps": feats["spread_bps"],
        "turnover_est": feats["turnover_est"],
        "illiquid_flag": feats["illiquid_flag"],
        "roll_trap": feats["roll_trap"],
        "in_session": feats["in_session"],
        "near_open": feats["near_open"],
        "near_close": feats["near_close"],
        "rules_score": float(rules_score),
        "rules_veto": bool(rules_veto),
        "ml_prob_raw": float(ml_prob_raw) if ml_prob_raw is not None else None,
        "ml_prob_cal": float(ml_prob_cal) if ml_prob_cal is not None else None,
        "ml_bucket": int(ml_bucket) if ml_bucket is not None else None,
        "ml_score": float(ml_score) if ml_score is not None else None,
        "fused_score": float(fused_score),
        "fused_veto": bool(fused_veto),
    }
    rows.append(
        (
            symbol,
            kind,
            tf,
            ts,
            "FLOW.debug_ctx",
            0.0,
            json.dumps(_clean_json_dict(debug_ctx)),
            base.run_id,
            base.source,
        )
    )

    if cfg["write_scenarios_debug"]:
        for name, sc in fired:
            rows.append(
                (
                    symbol,
                    kind,
                    tf,
                    ts,
                    f"FLOW.scenario.{name}",
                    float(sc),
                    "{}",
                    base.run_id,
                    base.source,
                )
            )

    write_values(rows)

    return ts, rules_score, rules_veto, ml_score, fused_score, fused_veto
