# pillars/flow_pillar.py
from __future__ import annotations

import json, math, configparser, ast, functools
from typing import Optional, Tuple, Dict, Any, List
import numpy as np
import pandas as pd

import importlib
from utils.db import get_db_connection

from .common import (  # shared utils
    ema, atr, adx, obv_series, resample, last_metric, write_values, clamp,
    TZ, DEFAULT_INI, BaseCfg
)
from pillars.common import min_bars_for_tf, ensure_min_bars, maybe_trim_last_bar

# -----------------------------
# Config
# -----------------------------
def _cfg(path=DEFAULT_INI) -> dict:
    cp = configparser.ConfigParser(inline_comment_prefixes=(";", "#"), interpolation=None, strict=False)
    cp.read(path)

    # fallbacks for old [liquidity] keys if present
    spread_bps_veto = cp.getfloat(
        "flow", "spread_bps_veto",
        fallback=cp.getfloat("liquidity", "max_spread_bps", fallback=40.0)
    )
    min_turnover = cp.getfloat(
        "flow", "min_turnover",
        fallback=cp.getfloat("liquidity", "min_turnover", fallback=0.0)
    )

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

        "ml_enabled":        cp.getboolean("flow_ml", "enabled", fallback=False),
        "ml_blend_weight":   cp.getfloat("flow_ml", "blend_weight", fallback=0.35),  # 0..1
        "ml_callback":       cp.get("flow_ml", "callback", fallback="").strip(),     # e.g. ml.predict_callback.flow_prob_long
        "ml_source_table":   cp.get("flow_ml", "source_table", fallback="").strip(), # e.g. ml.predictions (optional)
        "ml_veto_if_prob_lt":cp.get("flow_ml", "veto_if_prob_lt", fallback="").strip(), # e.g. "0.35"

        # scenario list
        "scenarios_list": [s.strip() for s in cp.get("flow_scenarios", "list", fallback="").replace("\n"," ").split(",") if s.strip()],
        "cp": cp,
    }

# -----------------------------
# Optimized Safe Scenario Evaluator
# -----------------------------
_ALLOWED_BOOL_OPS = {ast.And, ast.Or}
_ALLOWED_UNARY_BOOL = {ast.Not}
_ALLOWED_CMP_OPS  = {ast.Lt, ast.Gt, ast.Le, ast.Ge, ast.Eq, ast.NotEq}
_ALLOWED_NUM_OPS  = {ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod, ast.FloorDiv, ast.Pow, ast.USub, ast.UAdd}

@functools.lru_cache(maxsize=2048)
def _compile_rule(expr: str):
    """
    OPTIMIZATION: Compiles the string expression into an AST object ONCE and caches it.
    This prevents re-parsing the same strings thousands of times per run.
    """
    if not expr or not expr.strip():
        return None
    try:
        return ast.parse(expr, mode="eval")
    except Exception:
        return None

def _safe_eval(expr_or_tree: str | ast.AST, scope: Dict[str, Any]) -> bool:
    """
    Evaluate simple boolean/arith expressions safely against `scope`.
    Accepts either a raw string (legacy) or a compiled AST (optimized).
    """
    # Handle None
    if not expr_or_tree:
        return False

    # If string, compile it (or get from cache)
    if isinstance(expr_or_tree, str):
        tree = _compile_rule(expr_or_tree)
        if not tree: return False
    else:
        tree = expr_or_tree

    def _eval(node):
        if isinstance(node, ast.Expression):
            return _eval(node.body)

        if isinstance(node, ast.BoolOp) and type(node.op) in _ALLOWED_BOOL_OPS:
            vals = [_eval(v) for v in node.values]
            vals = [bool(v) for v in vals]
            return all(vals) if isinstance(node.op, ast.And) else any(vals)

        if isinstance(node, ast.UnaryOp):
            if type(node.op) in _ALLOWED_UNARY_BOOL:
                return not bool(_eval(node.operand))
            if type(node.op) in _ALLOWED_NUM_OPS:
                v = _eval(node.operand)
                v = float(v) if not isinstance(v, (int, float)) else v
                return -v if isinstance(node.op, ast.USub) else +v

        if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_NUM_OPS:
            a, b = _eval(node.left), _eval(node.right)
            if not isinstance(a, (int, float)): a = float(a)
            if not isinstance(b, (int, float)): b = float(b)
            if isinstance(node.op, ast.Add):       return a + b
            if isinstance(node.op, ast.Sub):       return a - b
            if isinstance(node.op, ast.Mult):      return a * b
            if isinstance(node.op, ast.Div):       return a / b if b != 0 else 0.0
            if isinstance(node.op, ast.FloorDiv):  return a // b if b != 0 else 0.0
            if isinstance(node.op, ast.Mod):       return a % b if b != 0 else 0.0
            if isinstance(node.op, ast.Pow):       return a ** b

        if isinstance(node, ast.Compare) and len(node.ops) == 1:
            a = _eval(node.left)
            b = _eval(node.comparators[0])
            op = node.ops[0]
            try:
                if isinstance(op, ast.Lt):   return a <  b
                if isinstance(op, ast.Le):   return a <= b
                if isinstance(op, ast.Gt):   return a >  b
                if isinstance(op, ast.Ge):   return a >= b
                if isinstance(op, ast.Eq):   return a == b
                if isinstance(op, ast.NotEq):return a != b
            except Exception:
                return False

        if isinstance(node, ast.Name):
            return scope.get(node.id, False)

        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float, bool)):
            return node.value

        raise ValueError("Disallowed token in scenario expression")

    try:
        return bool(_eval(tree))
    except Exception:
        return False

# -----------------------------
# Helpers
# -----------------------------
def _mfi(high: pd.Series, low: pd.Series, close: pd.Series, vol: pd.Series, n:int=14)->pd.Series:
    """Vectorized MFI calculation."""
    high = pd.to_numeric(high, errors="coerce")
    low  = pd.to_numeric(low,  errors="coerce")
    close= pd.to_numeric(close,errors="coerce")
    vol  = pd.to_numeric(vol,  errors="coerce").fillna(0.0)

    tp  = (high + low + close) / 3.0
    rmf = tp * vol

    # Vectorized diff
    diff = tp.diff()

    # Create positive and negative flow series without loop
    pos = pd.Series(np.where(diff > 0, rmf, 0.0), index=tp.index)
    neg = pd.Series(np.where(diff < 0, rmf, 0.0), index=tp.index)

    win = max(5, n//2)
    pos_n = pos.rolling(n, min_periods=win).sum()
    neg_n = neg.rolling(n, min_periods=win).sum()

    # Avoid div by zero
    mr = pos_n / neg_n.replace(0, np.nan)
    mfi = 100 - (100 / (1 + mr))
    mfi = mfi.clip(0, 100).fillna(method="ffill").fillna(50.0)
    return mfi

def _ist(ts: pd.Timestamp) -> pd.Timestamp:
    try:
        return ts.tz_convert("Asia/Kolkata")
    except Exception:
        return ts.tz_localize("UTC").tz_convert("Asia/Kolkata")

def _hh(series: pd.Series, look:int=20) -> bool:
    if len(series) < look + 1: return False
    return bool(float(series.iloc[-1]) >= float(series.rolling(look).max().iloc[-2]))

def _ll(series: pd.Series, look:int=20) -> bool:
    if len(series) < look + 1: return False
    return bool(float(series.iloc[-1]) <= float(series.rolling(look).min().iloc[-2]))

def _resolve_callback(path: str):
    if not path:
        return None
    mod, func = path.rsplit(".", 1)
    return getattr(importlib.import_module(mod), func, None)

# -----------------------------
# Feature builder
# -----------------------------
def _build_features(dtf: pd.DataFrame, symbol: str, kind: str, tf: str, cfg: dict, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    # session flags
    ts_ist = _ist(dtf.index[-1])
    minutes = ts_ist.hour * 60 + ts_ist.minute
    open_min, close_min = 9*60 + 15, 15*60 + 30
    in_session = (open_min <= minutes <= close_min)
    last_vol_zero = (float(dtf["volume"].iloc[-1]) == 0.0)

    # ignore last bar if off-session & vol=0 (don’t poison RVOL/OBV/MFI)
    use = dtf.iloc[:-1] if ((not in_session) and last_vol_zero and len(dtf) > 1) else dtf
    o = use["open"]; c = use["close"]; h = use["high"]; l = use["low"]; v = use["volume"].fillna(0.0)

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

    # OBV higher-low proxy: recent 5-bar min > prior 5-bar min
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

    # --- VAH/VAL from Volume Profile (OPTIMIZED CONTEXT LOOKUP) ---
    # Logic: Try context first (fast), else DB (slow fallback)
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
    last_high  = float(h.iloc[-1]); last_low = float(l.iloc[-1])

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

    # --- Candle anatomy for wick_ratio & reversal ---
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

        if last_pc > 0 and last_doi > 0:   voi_long_build = True
        elif last_pc < 0 and last_doi > 0: voi_short_build = True
        elif last_pc > 0 and last_doi < 0: voi_short_cover = True
        elif last_pc < 0 and last_doi < 0: voi_long_unwind = True

        voi_mag = abs(last_doi) / max(1e-9, float(v.iloc[-1]))

        look = cfg["roll_look"]
        if len(use) > look + 1:
            oi0 = float(use["oi"].iloc[-look-1])
            oin = float(use["oi"].iloc[-1])
            if oi0 > 0:
                drop = (oi0 - oin) / oi0
                if drop >= cfg["roll_drop_pct"] and rvol_strong:
                    roll_trap = True

    # --- Liquidity evidence (OPTIMIZED CONTEXT LOOKUP) ---
    spread_bps = None

    # Try context first
    found_spread = False
    if context:
        for m in ("MKT.spread_bps", "LIQ.spread_bps", "MKT.spread_bps.mean"):
            key = f"{m}|{tf}"
            if key in context and context[key] is not None:
                spread_bps = float(context[key])
                found_spread = True
                break

    # Fallback to DB
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

    # --- Helper for extra indicators (MWPL/PCR/PIVOT) to support context ---
    def get_meta_metric(name: str) -> float:
        """Try context first, then DB."""
        # Context key usually doesn't have TF for these globals, but check your worker logic
        if context and name in context:
            return float(context[name])
        # Fallback to Last Metric (WARNING: Slow N+1 if used heavily)
        val = last_metric(symbol, kind, tf, name) # Passing TF just in case, though some are global
        return float(val) if val is not None else 0.0

    # You can wrap last_metric calls in scenarios by injecting them into `feats`
    # But for dynamic last_metric("...") calls in INI strings, we handle that in the feature set below.
    # NOTE: Dynamic `last_metric` calls inside INI strings are very slow.
    # Better to pre-calculate them here:

    pcr_vol_chg = get_meta_metric("FNO.pcr.vol.chg_pct")
    mwpl_pct = get_meta_metric("FNO.mwpl.pct")
    r1_dist = get_meta_metric("PIVOT.r1.dist_pct")
    pcr_oi_chg = get_meta_metric("FNO.pcr.oi.chg_pct")

    # --- Daily Futures & Options Data ---
    daily_fut_buildup = context.get("daily_fut_buildup", "") if context else ""
    daily_opt_call_oi_chg_pct = context.get("daily_opt_call_oi_chg_pct", 0.0) if context else 0.0
    daily_opt_put_oi_chg_pct = context.get("daily_opt_put_oi_chg_pct", 0.0) if context else 0.0

    # --- Session flags ---
    near_open = in_session and (minutes - open_min <= 30)
    near_close = in_session and (close_min - minutes <= 30)
    mid_lunch = in_session and (12*60 <= minutes <= 13*60 + 30)

    # assemble features for scenarios
    feats: Dict[str, Any] = {
        "open": float(o.iloc[-1]), "close": last_close, "high": last_high, "low": last_low,
        "volume": float(v.iloc[-1]), "price": last_close,

        # MFI
        "mfi_val": float(mfi_val), "mfi_slope": float(mfi_slope), "mfi_up": bool(mfi_up),
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
        "spread_bps": float(spread_bps) if spread_bps is not None else float("nan"),
        "turnover_est": float(turnover_est),
        "illiquid_flag": bool(illiquid),

        # Pre-fetched Meta Metrics (Replacing dynamic calls in INI)
        # You should update your INI to use these keys instead of last_metric(...) function calls if possible
        # But to support existing INI with last_metric(), we add a shim in the AST evaluator or just expose the values here:
        "pcr_vol_chg": pcr_vol_chg,
        "mwpl_pct": mwpl_pct,
        "r1_dist": r1_dist,
        "pcr_oi_chg": pcr_oi_chg,

        # Daily F&O data
        "daily_fut_buildup": daily_fut_buildup,
        "daily_opt_call_oi_chg_pct": daily_opt_call_oi_chg_pct,
        "daily_opt_put_oi_chg_pct": daily_opt_put_oi_chg_pct,

        # Session
        "in_session": bool(in_session),
        "near_open": bool(near_open),
        "near_close": bool(near_close),
        "mid_lunch": bool(mid_lunch),
        "last_vol_zero": bool(last_vol_zero),
    }

    # SHIM for last_metric inside AST (if you keep the function call in INI)
    # This is a hack to make the AST evaluator "find" the function in scope
    # Ideally, change INI to use 'pcr_vol_chg' variable directly.
    feats["last_metric"] = lambda k: feats.get(k.replace("FNO.pcr.vol.chg_pct", "pcr_vol_chg")
                                              .replace("FNO.mwpl.pct", "mwpl_pct")
                                              .replace("PIVOT.r1.dist_pct", "r1_dist")
                                              .replace("FNO.pcr.oi.chg_pct", "pcr_oi_chg"), 0.0)

    return feats

# -----------------------------
# Scenario engine
# -----------------------------
def _run_scenarios(features: Dict[str, Any], cfg: dict) -> Tuple[float, bool, Dict[str, float], List[tuple]]:
    cp = cfg["cp"]
    total = 0.0
    veto = False
    parts: Dict[str, float] = {}
    fired: List[tuple[str, float]] = []

    for name in cfg["scenarios_list"]:
        sec = f"flow_scenario.{name}"
        if not cp.has_section(sec):
            continue

        # OPTIMIZATION: Use cached compilation
        when_str = cp.get(sec, "when", fallback="").strip()
        when_tree = _compile_rule(when_str)

        if not when_tree:
            continue

        ok = False
        try:
            ok = _safe_eval(when_tree, features)
        except Exception:
            ok = False

        if not ok:
            continue

        sc = float(cp.get(sec, "score", fallback="0").strip() or 0.0)
        total = (sc if cfg["rules_mode"] == "override" else (total + sc))
        parts[name] = sc
        fired.append((name, sc))

        if cp.has_option(sec, "bonus_when"):
            try:
                bonus_str = cp.get(sec, "bonus_when")
                bonus_tree = _compile_rule(bonus_str)
                if _safe_eval(bonus_tree, features):
                    bonus_val = float(cp.get(sec, "bonus", fallback="0").strip() or 0.0)
                    total = (bonus_val if cfg["rules_mode"] == "override" else (total + bonus_val))
                    parts[name + ".bonus"] = bonus_val
                    fired.append((name + ".bonus", bonus_val))
            except Exception:
                pass

        if cp.get(sec, "set_veto", fallback="false").strip().lower() in {"1","true","yes","on"}:
            veto = True

    return total, veto, parts, fired

# -----------------------------
# Public API
# -----------------------------
def score_flow(symbol: str, kind: str, tf: str, df5: pd.DataFrame, base: BaseCfg,
               ini_path=DEFAULT_INI, context: Optional[Dict[str, Any]] = None):
    """
    Scenario-driven Flow pillar.
    Updated to accept 'context' for pre-fetched metrics (Speed optimization).
    """
    # 1. CHECK: Did the orchestrator already resample?
    # If 'df5' index freq matches 'tf', use it. Else resample.
    # This assumes 'df5' might actually be 'dftf' if passed from the new orchestrator.

    dftf = df5
    # Only resample if strictly necessary (simple heuristic based on rows/time)
    # If you trust your orchestrator always sends the correct TF data, you can skip this check.
    # For safety, we'll assume if it's 15m and we want 60m, we resample.
    if tf != "15m" and pd.infer_freq(dftf.index) != tf:
         dftf = resample(df5, tf)

    dftf = maybe_trim_last_bar(dftf)
    if not ensure_min_bars(dftf, tf):
        return None

    cfg = _cfg(ini_path)

    # Pass context to feature builder
    feats = _build_features(dftf, symbol, kind, tf, cfg, context=context)

    total, veto_scen, parts, fired = _run_scenarios(feats, cfg)

    # auto-veto on hard evidence
    auto_veto = bool(feats["illiquid_flag"] or feats["roll_trap"])
    score = float(clamp(total, cfg["clamp_low"], cfg["clamp_high"]))
    veto = bool(veto_scen or auto_veto)

    ts = dftf.index[-1].to_pydatetime().replace(tzinfo=TZ)

    # ----- Optional ML blend (flag-driven, no-op if disabled) -----
    ml_prob = None
    ml_score = None
    final_score = score
    final_veto = veto

    try:
        cp = cfg["cp"]
        ml_enabled = cp.getboolean("flow_ml", "enabled", fallback=False)
        if ml_enabled:
            w = float(cp.get("flow_ml", "blend_weight", fallback="0.35"))
            w = max(0.0, min(1.0, w))

            # Path A: DB-driven (if you already store per-(symbol,tf,ts) probs)
            ml_table = cp.get("flow_ml", "source_table", fallback="").strip()
            if ml_table:
                with get_db_connection() as conn, conn.cursor() as cur:
                    cur.execute(f"""
                        SELECT prob_long, prob_short
                          FROM {ml_table}
                         WHERE symbol=%s AND tf=%s AND ts=%s
                         ORDER BY ts DESC
                         LIMIT 1
                    """, (symbol, tf, ts))
                    r = cur.fetchone()
                if r:
                    prob_long = float(r[0] if r[0] is not None else 0.5)
                    prob_short= float(r[1] if r[1] is not None else 0.5)
                    ml_prob = max(prob_long, 1.0 - prob_short)  # simple reconciliation
                    ml_score = round(100.0 * ml_prob, 2)

            # Path B: Python callback (if you want to compute on the fly)
            if ml_prob is None:
                cb_path = cp.get("flow_ml", "callback", fallback="").strip()
                if cb_path:
                    mod_name, _, fn_name = cb_path.rpartition(".")
                    if mod_name and fn_name:
                        import importlib
                        mod = importlib.import_module(mod_name)
                        fn  = getattr(mod, fn_name)
                        ml_prob = float(fn(symbol=symbol, kind=kind, tf=tf, ts=ts, features=feats))
                        if not np.isfinite(ml_prob): ml_prob = 0.0
                        ml_prob = max(0.0, min(1.0, ml_prob))
                        ml_score = round(100.0 * ml_prob, 2)

            # Blend (if we have ml_prob)
            if ml_prob is not None:
                base_prob = float(score) / 100.0
                blended = (1.0 - w) * base_prob + w * ml_prob
                final_score = round(100.0 * blended, 2)
                # Optional: soften veto if ML strongly supports continuation
                if veto and blended >= 0.65:
                    final_veto = False

    except Exception:
        # Keep the pillar robust — ignore ML errors
        pass

    rows = [
        (symbol, kind, tf, ts, "FLOW.score", float(score), json.dumps({}), base.run_id, base.source),
        (symbol, kind, tf, ts, "FLOW.veto_flag", 1.0 if veto else 0.0, "{}", base.run_id, base.source),
        (symbol, kind, tf, ts, "FLOW.score_final", float(final_score), "{}", base.run_id, base.source),
        (symbol, kind, tf, ts, "FLOW.veto_final", 1.0 if final_veto else 0.0, "{}", base.run_id, base.source),
    ]

    # compact debug ctx to help tuning
    debug_ctx = {
        "mfi_val": feats["mfi_val"], "mfi_slope": feats["mfi_slope"],
        "obv_above_ema": feats["obv_above_ema"], "price_hh": feats["price_hh"], "obv_hh": feats["obv_hh"],
        "rvol": feats["rvol_now"], "vol_cv20": feats["vol_cv20"], "vol_rank60": feats["vol_rank60"],
        "voi_long_build": feats["voi_long_build"], "voi_short_build": feats["voi_short_build"],
        "voi_short_cover": feats["voi_short_cover"], "voi_long_unwind": feats["voi_long_unwind"], "voi_mag": feats["voi_mag"],
        "spread_bps": feats["spread_bps"], "turnover_est": feats["turnover_est"],
        "illiquid_flag": feats["illiquid_flag"], "roll_trap": feats["roll_trap"],
        "in_session": feats["in_session"], "near_open": feats["near_open"], "near_close": feats["near_close"],
    }
    rows.append((symbol, kind, tf, ts, "FLOW.debug_ctx", 0.0, json.dumps(debug_ctx), base.run_id, base.source))

    if cfg["write_scenarios_debug"]:
        for name, sc in fired:
            rows.append((symbol, kind, tf, ts, f"FLOW.scenario.{name}", float(sc), "{}", base.run_id, base.source))

    write_values(rows)
    return (ts, final_score, final_veto)