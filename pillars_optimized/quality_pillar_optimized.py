# pillars/quality_pillar.py
from __future__ import annotations
import json, math, configparser, ast, functools
from typing import Optional, Tuple, Dict, Any, List
import numpy as np
import pandas as pd

from utils.db import get_db_connection
from .common import (
    ema, atr, adx, resample, write_values, last_metric, clamp, obv_series,
    TZ, DEFAULT_INI, BaseCfg, min_bars_for_tf, ensure_min_bars, maybe_trim_last_bar
)

# -----------------------------
# Optimized Safe Scenario Evaluator
# -----------------------------
_ALLOWED_BOOL_OPS = {ast.And, ast.Or}
_ALLOWED_UNARY_BOOL = {ast.Not}
_ALLOWED_CMP_OPS  = {ast.Lt, ast.Gt, ast.Le, ast.Ge, ast.Eq, ast.NotEq}
_ALLOWED_NUM_OPS  = {ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod, ast.FloorDiv, ast.Pow, ast.USub, ast.UAdd}

@functools.lru_cache(maxsize=2048)
def _compile_rule(expr: str):
    """OPTIMIZATION: Compiles the string expression into an AST object ONCE."""
    if not expr or not expr.strip():
        return None
    try:
        return ast.parse(expr, mode="eval")
    except Exception:
        return None

def _safe_eval(expr_or_tree: str | ast.AST, scope: Dict[str, Any]) -> bool:
    """Evaluates expression safely against scope using cached AST."""
    if not expr_or_tree: return False

    if isinstance(expr_or_tree, str):
        tree = _compile_rule(expr_or_tree)
        if not tree: return False
    else:
        tree = expr_or_tree

    def _eval(node):
        if isinstance(node, ast.Expression): return _eval(node.body)
        if isinstance(node, ast.BoolOp):
            vals = [_eval(v) for v in node.values]
            return all(vals) if isinstance(node.op, ast.And) else any(vals)
        if isinstance(node, ast.UnaryOp):
            val = _eval(node.operand)
            if isinstance(node.op, ast.Not): return not val
            if isinstance(node.op, ast.USub): return -val
            return val
        if isinstance(node, ast.BinOp):
            a, b = _eval(node.left), _eval(node.right)
            if isinstance(node.op, ast.Add): return a + b
            if isinstance(node.op, ast.Sub): return a - b
            if isinstance(node.op, ast.Mult): return a * b
            if isinstance(node.op, ast.Div): return a / b if b!=0 else 0
            return 0
        if isinstance(node, ast.Compare):
            a, b = _eval(node.left), _eval(node.comparators[0])
            op = node.ops[0]
            if isinstance(op, ast.Lt): return a < b
            if isinstance(op, ast.Gt): return a > b
            if isinstance(op, ast.Le): return a <= b
            if isinstance(op, ast.Ge): return a >= b
            if isinstance(op, ast.Eq): return a == b
            if isinstance(op, ast.NotEq): return a != b
            return False
        if isinstance(node, ast.Name): return scope.get(node.id, 0.0)
        if isinstance(node, ast.Constant): return node.value
        return False

    try:
        return bool(_eval(tree))
    except Exception:
        return False

# -----------------------------
# Helpers
# -----------------------------
def _wick_body_eval(last_row: pd.Series) -> Tuple[float, bool, float]:
    body = float(abs(last_row["close"] - last_row["open"]))
    upper = float(last_row["high"] - max(last_row["close"], last_row["open"]))
    lower = float(min(last_row["close"], last_row["open"]) - last_row["low"])
    wick = upper + lower
    ratio = wick / (body if body > 0 else 1e-9)
    pts = -10.0 if ratio > 2.0 else (5.0 if ratio < 1.0 else 0.0)
    return float(pts), bool(ratio > 2.0), float(ratio)

def _near_level(price: float, level: Optional[float], atr_last: float, near_atr: float) -> bool:
    if level is None or atr_last <= 0:
        return False
    return abs(price - float(level)) <= (near_atr * atr_last)

def _rsi(series: pd.Series, n:int=14) -> pd.Series:
    d = series.diff()
    up = d.clip(lower=0.0)
    dn = (-d).clip(lower=0.0)
    rs = up.ewm(alpha=1.0/n, adjust=False).mean() / (dn.ewm(alpha=1.0/n, adjust=False).mean().replace(0, np.nan))
    out = 100 - 100/(1+rs)
    return out.fillna(50.0)

def _higher_tf(tf: str) -> str:
    tf = tf.lower()
    if tf in {"15m", "15"}: return "60m" # Adjusted logic
    if tf in {"25m","25"}:   return "65m"
    if tf in {"65m","65","60m","60"}:   return "240m"
    if tf in {"125m","125"}: return "1d"
    return "1d"

def _safe_num(x: Any, default: float = 0.0) -> float:
    try:
        fx = float(x)
        return fx if np.isfinite(fx) else default
    except Exception:
        return default

# -----------------------------
# Config
# -----------------------------
def _cfg(path=DEFAULT_INI) -> dict:
    cp = configparser.ConfigParser(inline_comment_prefixes=(";", "#"), interpolation=None, strict=False)
    cp.read(path)
    return {
        "atr_win":  cp.getint("quality", "atr_win",  fallback=14),
        "gap_sigma":cp.getfloat("quality", "gap_sigma", fallback=3.0),
        "vahval_break_vol_mult": cp.getfloat("quality", "vahval_break_vol_mult", fallback=2.0),
        "atr_veto_pctile": cp.getfloat("quality", "atr_veto_pctile", fallback=0.90),
        "vol_cv_good": cp.getfloat("quality", "vol_cv_good", fallback=0.50),
        "vol_cv_bad":  cp.getfloat("quality", "vol_cv_bad",  fallback=1.50),
        "near_vahval_atr": cp.getfloat("quality", "near_vahval_atr", fallback=0.25),
        "rules_mode": cp.get("quality", "rules_mode", fallback="additive").strip().lower(),
        "clamp_low": cp.getfloat("quality", "clamp_low", fallback=0.0),
        "clamp_high": cp.getfloat("quality", "clamp_high", fallback=100.0),
        "write_scenarios_debug": cp.getboolean("quality", "write_scenarios_debug", fallback=False),
        "min_bars": cp.getint("quality", "min_bars", fallback=120),
        "bb_strong": cp.getfloat("thresholds", "bb_strong", fallback=6.5),
        "ml_enabled": cp.getboolean("quality_ml", "enabled", fallback=False),
        "ml_blend_weight": cp.getfloat("quality_ml", "blend_weight", fallback=0.35),
        "ml_source_table": cp.get("quality_ml", "source_table", fallback="").strip(),
        "ml_callback": cp.get("quality_ml", "callback", fallback="").strip(),
        "ml_soften_veto_if_prob_ge": cp.get("quality_ml", "soften_veto_if_prob_ge", fallback="").strip(),
        "ml_veto_if_prob_lt": cp.get("quality_ml", "veto_if_prob_lt", fallback="").strip(),
        "cp": cp,
    }

# -----------------------------
# Feature builder (Context Aware)
# -----------------------------
def _build_features(dtf: pd.DataFrame, symbol: str, kind: str, tf: str, cfg: dict, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    o = dtf["open"]; c = dtf["close"]; h = dtf["high"]; l = dtf["low"]; v = dtf["volume"]
    last = dtf.iloc[-1]; prev = dtf.iloc[-2] if len(dtf) > 1 else dtf.iloc[-1]

    # ATR & ATR%
    ATR = atr(h, l, c, cfg["atr_win"])
    atr_last = _safe_num(ATR.iloc[-1], 0.0)
    atr_pct_series = (ATR / c.replace(0, np.nan)) * 100.0
    atr_pct = _safe_num(atr_pct_series.iloc[-1], 0.0)
    atr_pct_avg_20 = _safe_num(atr_pct_series.rolling(20).mean().iloc[-1] if len(atr_pct_series.dropna()) else atr_pct, atr_pct)

    # ADX
    a14, dip, dim = adx(h, l, c, 14)
    adx14 = _safe_num(a14.iloc[-1], 0.0)
    adx9, _, _ = adx(h, l, c, 9)
    adx9_val = _safe_num(adx9.iloc[-1], 0.0)
    adx14_prev = _safe_num(a14.iloc[-2], adx14) if len(a14) > 1 else adx14

    # Returns sigma (60)
    rets = c.pct_change().dropna()
    sigma60 = _safe_num(rets.rolling(60).std(ddof=1).iloc[-1] if len(rets) >= 60 else rets.std(ddof=1), 0.0)

    # Gap ratio
    prev_close = _safe_num(prev["close"], _safe_num(last["close"], 0.0))
    gap_ratio = abs(_safe_num(last["close"], prev_close) - prev_close) / (prev_close if prev_close else 1e-9)

    # Volume stats
    vol_mean20 = _safe_num(v.rolling(20).mean().iloc[-1] if len(v) >= 20 else v.mean(), 0.0)
    vol_sd20   = _safe_num(v.rolling(20).std(ddof=1).iloc[-1] if len(v) >= 20 else v.std(ddof=1), 0.0)
    vol_spike_2sd = bool(_safe_num(v.iloc[-1], 0.0) > (vol_mean20 + 2.0 * vol_sd20))
    vol_cv20 = _safe_num((v.rolling(20).std(ddof=1) / v.rolling(20).mean()).iloc[-1] if len(v) >= 20 else 1.0, 1.0)

    # Relative Volume (Required for INI 'rvol_now >= 1.2')
    rvol_now = _safe_num(v.iloc[-1], 0.0) / max(1e-9, vol_mean20)

    # Wick/body + ratio
    _, wick_bad, wick_ratio = _wick_body_eval(last)

    # --- METRICS VIA CONTEXT (DB Optimization) ---
    def get_meta(key, tf_override=None):
        k = key if tf_override is None else f"{key}|{tf_override}"
        # 1. Try specific TF key in context
        if context and k in context: return context[k]
        # 2. Try global key in context
        if context and key in context: return context[key]
        # 3. Fallback (slow)
        return last_metric(symbol, kind, (tf_override or tf), key)

    poc = get_meta("VP.POC")
    vah = get_meta("VP.VAH")
    val = get_meta("VP.VAL")
    bb_score = float(get_meta("BB.score") or 0.0)

    htf = _higher_tf(tf)
    vah_htf = get_meta("VP.VAH", tf_override=htf)

    # Pivot Points (Required for INI 'near_r1', 'between_r1_s1')
    pivot_r1 = get_meta("PIVOT.r1")
    pivot_s1 = get_meta("PIVOT.s1")

    # TL Metrics (Required for INI)
    tl_val = float(get_meta("TL.valuation") or 0.0)
    tl_dur = float(get_meta("TL.durability") or 0.0)

    # OBV Z-Score (Required for INI 'z_obv')
    obv = obv_series(c, v)
    obv_d = obv.diff()
    mu = _safe_num(obv_d.ewm(span=5, adjust=False).mean().iloc[-1] if len(obv_d.dropna()) else 0.0, 0.0)
    sd = _safe_num(obv_d.rolling(20).std(ddof=1).iloc[-1] if len(obv_d.dropna()) else 0.0, 0.0)
    z_obv = ( _safe_num(obv_d.iloc[-1], 0.0) - mu ) / (sd if sd > 0 else 1e9)

    # Calculations
    price = _safe_num(last["close"], 0.0)
    near_poc = bool(poc is not None and abs(price - float(poc)) / (atr_last if atr_last else 1e-9) <= 0.5)
    near_vah = _near_level(price, vah, atr_last, cfg["near_vahval_atr"])
    near_val = _near_level(price, val, atr_last, cfg["near_vahval_atr"])

    # Pivot Proximity
    near_r1 = _near_level(price, pivot_r1, atr_last, 0.5)
    near_s1 = _near_level(price, pivot_s1, atr_last, 0.5)

    vol_ok_vahval = bool(_safe_num(v.iloc[-1], 0.0) >= (cfg["vahval_break_vol_mult"] * vol_mean20)) if vol_mean20 > 0 else True
    inside_va = bool((val is not None and vah is not None) and (float(val) <= price <= float(vah)))

    # Bollinger (20, 2.0)
    n_bb, k_bb = 20, 2.0
    mavg = c.rolling(n_bb, min_periods=n_bb//2).mean()
    mstd = c.rolling(n_bb, min_periods=n_bb//2).std(ddof=1)
    bb_up = mavg + k_bb*mstd
    bb_lo = mavg - k_bb*mstd
    bb_width = (bb_up - bb_lo)
    bb_width_pct = _safe_num((bb_width.iloc[-1] / (_safe_num(c.iloc[-1], 1e-9))) * 100.0 if pd.notna(bb_width.iloc[-1]) else 0.0, 0.0)

    # Distance outside bands
    up_val = _safe_num(bb_up.iloc[-1] if pd.notna(bb_up.iloc[-1]) else c.iloc[-1], _safe_num(c.iloc[-1], 0.0))
    lo_val = _safe_num(bb_lo.iloc[-1] if pd.notna(bb_lo.iloc[-1]) else c.iloc[-1], _safe_num(c.iloc[-1], 0.0))
    above = max(0.0, _safe_num(c.iloc[-1], 0.0) - up_val)
    below = max(0.0, lo_val - _safe_num(c.iloc[-1], 0.0))
    outside_px = above if above > 0 else below
    close_dist_bb = (outside_px / (atr_last if atr_last else 1e-9)) if outside_px > 0 else 0.0

    # Candle Patterns
    body = abs(_safe_num(last["close"], 0.0) - _safe_num(last["open"], 0.0))
    rng = max(1e-9, _safe_num(last["high"], 0.0) - _safe_num(last["low"], 0.0))
    prev_open = _safe_num(prev["open"], _safe_num(prev["close"], 0.0)); prev_close2 = _safe_num(prev["close"], prev_open)

    engulfing_bull = bool(
        (_safe_num(last["close"], 0.0) > _safe_num(last["open"], 0.0)) and (prev_close2 < prev_open) and
        (_safe_num(last["close"], 0.0) >= max(prev_open, prev_close2)) and
        (_safe_num(last["open"], 0.0)  <= min(prev_open, prev_close2))
    )
    doji_candle = bool((body / rng) < 0.1)
    upper_wick = _safe_num(last["high"], 0.0) - max(_safe_num(last["close"], 0.0), _safe_num(last["open"], 0.0))
    lower_wick = min(_safe_num(last["close"], 0.0), _safe_num(last["open"], 0.0)) - _safe_num(last["low"], 0.0)
    hammer_bullish = bool((lower_wick >= 2.0*body) and (upper_wick <= 0.3*body) and (_safe_num(last["close"], 0.0) >= _safe_num(last["open"], 0.0)))

    reversal_candle = (engulfing_bull or hammer_bullish)

    # Logic: close_r1_break_fail
    close_r1_break_fail = False
    if pivot_r1 is not None:
        r1 = float(pivot_r1)
        if _safe_num(last["high"], 0.0) > r1 and _safe_num(last["close"], 0.0) < r1:
            close_r1_break_fail = True

    # Logic: between_r1_s1
    between_r1_s1 = False
    if pivot_r1 is not None and pivot_s1 is not None:
        if _safe_num(last["low"], 0.0) > float(pivot_s1) and _safe_num(last["high"], 0.0) < float(pivot_r1):
            between_r1_s1 = True

    # RSI bullish divergence vs N bars back
    rsi14 = _rsi(c, 14)
    N = 5
    rsi_bull_div = bool((_safe_num(c.iloc[-1], 0.0) < _safe_num(c.iloc[-N], _safe_num(c.iloc[-1], 0.0))) and (_safe_num(rsi14.iloc[-1], 50.0) > _safe_num(rsi14.iloc[-N], 50.0)))

    feats: Dict[str, Any] = {
        "open": _safe_num(last["open"], 0.0), "close": _safe_num(last["close"], 0.0),
        "high": _safe_num(last["high"], 0.0), "low": _safe_num(last["low"], 0.0),
        "volume": _safe_num(last["volume"], 0.0), "price": _safe_num(last["close"], 0.0),

        # anchors
        "poc": float(poc) if poc is not None else float("nan"),
        "vah": float(vah) if vah is not None else float("nan"),
        "val": float(val) if val is not None else float("nan"),
        "vah_htf": float(vah_htf) if vah_htf is not None else float("nan"),
        "inside_va": inside_va,
        "bb_score": _safe_num(bb_score, 0.0),

        # proximity & volume gating
        "near_poc": near_poc, "near_vah": near_vah, "near_val": near_val,
        "near_r1": near_r1, "near_s1": near_s1,
        "vol_ok_vahval": vol_ok_vahval,
        "vah_val_vol_spike": vol_ok_vahval, # Alias for INI

        # ADX
        "adx14": adx14, "adx14_prev": adx14_prev, "adx9": adx9_val,

        # ATR%
        "atr_pct": atr_pct, "atr_pct_avg_20": atr_pct_avg_20,

        # volatility / gaps
        "sigma60": sigma60, "gap_ratio": gap_ratio,

        # volume stats
        "volume_avg_20": vol_mean20, "vol_cv20": vol_cv20,
        "vol_spike_2sd": vol_spike_2sd, "rvol_now": rvol_now,
        "z_obv": z_obv,

        # wick & shape
        "wick_ratio": float(wick_ratio),

        # BB info
        "bb_upper_band": up_val, "bb_lower_band": lo_val,
        "bb_width_pct": bb_width_pct, "close_dist_bb": close_dist_bb,

        # patterns
        "engulfing_bull": engulfing_bull,
        "doji_candle": doji_candle,
        "hammer_bullish": hammer_bullish,
        "reversal_candle": reversal_candle,
        "close_r1_break_fail": close_r1_break_fail,

        # RSI divergence
        "rsi_bull_div": rsi_bull_div,

        # External/Calc
        "tl_val": tl_val, "tl_dur": tl_dur,
        "between_r1_s1": between_r1_s1,
    }
    return feats


# -----------------------------
# Scenario engine
# -----------------------------
def _run_scenarios(features: Dict[str, Any], cfg: dict) -> Tuple[float, bool, bool, Dict[str, float], List[tuple]]:
    cp = cfg["cp"]
    total = 0.0
    veto = False
    reversal = False
    parts: Dict[str, float] = {}
    fired: List[tuple[str, float]] = []

    # expose thresholds to expressions
    features["bb_strong"] = float(cfg["bb_strong"])

    for section in ['bullish', 'bearish']:
        if not cp.has_section(section):
            continue

        for name, when_str in cp.items(section):
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

            score = float(cp.get(section, name, fallback="0").split('when')[0].strip())
            total += score
            parts[name] = score
            fired.append((name, score))


    return total, veto, reversal, parts, fired


# -----------------------------
# Public API
# -----------------------------
def score_quality(symbol: str, kind: str, tf: str, df5: pd.DataFrame, base: BaseCfg,
                  ini_path=DEFAULT_INI, context: Optional[Dict[str, Any]] = None):
    """
    Scenario-driven Quality pillar.
    Optimized for Context Injection.
    """

    # 1. Check if Orchestrator already provided correct TF
    dftf = df5
    if tf != "15m" and pd.infer_freq(dftf.index) != tf:
         dftf = resample(df5, tf)

    dftf = maybe_trim_last_bar(dftf)
    if not ensure_min_bars(dftf, tf):
        return None

    cfg = _cfg(ini_path)

    # Build Features (with Context)
    feats = _build_features(dftf, symbol, kind, tf, cfg, context)

    # Run Engine
    total, veto, reversal, parts, fired = _run_scenarios(feats, cfg)
    score = float(clamp(total, cfg["clamp_low"], cfg["clamp_high"]))
    ts = dftf.index[-1].to_pydatetime().replace(tzinfo=TZ)

    # ----- Optional ML blend -----
    ml_prob = None
    final_score = score
    final_veto = bool(veto)

    try:
        if cfg.get("ml_enabled", False):
            w = max(0.0, min(1.0, float(cfg.get("ml_blend_weight", 0.35))))

            # Path A: DB-driven
            tbl = cfg.get("ml_source_table", "")
            if tbl:
                with get_db_connection() as conn, conn.cursor() as cur:
                    cur.execute(f"SELECT prob_long, prob_short FROM {tbl} WHERE symbol=%s AND tf=%s AND ts=%s LIMIT 1", (symbol, tf, ts))
                    r = cur.fetchone()
                if r:
                    ml_prob = max(_safe_num(r[0], 0.5), 1.0 - _safe_num(r[1], 0.5))

            # Path B: Python callback
            if ml_prob is None:
                cb = cfg.get("ml_callback", "")
                if cb:
                    mod_name, _, fn_name = cb.rpartition(".")
                    if mod_name and fn_name:
                        import importlib
                        mod = importlib.import_module(mod_name)
                        fn  = getattr(mod, fn_name)
                        ml_prob = float(fn(symbol=symbol, kind=kind, tf=tf, ts=ts, features=feats))
                        ml_prob = max(0.0, min(1.0, ml_prob))

            # Blend + guard-rails
            if ml_prob is not None:
                base_prob = float(score) / 100.0
                blended = (1.0 - w) * base_prob + w * ml_prob
                final_score = round(100.0 * blended, 2)

                thr_soft = cfg.get("ml_soften_veto_if_prob_ge", "").strip()
                if thr_soft and blended >= float(thr_soft):
                     final_veto = False

                thr_hard = cfg.get("ml_veto_if_prob_lt", "").strip()
                if thr_hard and blended < float(thr_hard):
                     final_veto = True

    except Exception:
        pass

    # Write
    rows = [
        (symbol, kind, tf, ts, "QUAL.score", float(score), json.dumps({}), base.run_id, base.source),
        (symbol, kind, tf, ts, "QUAL.veto_flag", 1.0 if veto else 0.0, "{}", base.run_id, base.source),
        (symbol, kind, tf, ts, "QUAL.reversal_flag", 1.0 if reversal else 0.0, "{}", base.run_id, base.source),

        # blended
        (symbol, kind, tf, ts, "QUAL.score_final", float(final_score), "{}", base.run_id, base.source),
        (symbol, kind, tf, ts, "QUAL.veto_final", 1.0 if final_veto else 0.0, "{}", base.run_id, base.source),
    ]

    # Debug: compact context for tuning
    debug_ctx = {
        "adx14": feats["adx14"],
        "atr_pct": feats["atr_pct"],
        "bb_width_pct": feats["bb_width_pct"],
        "wick_ratio": feats["wick_ratio"],
        "vol_cv20": feats["vol_cv20"],
        "near_vah": feats["near_vah"],
        "bb_score": feats["bb_score"],
    }
    rows.append((symbol, kind, tf, ts, "QUAL.debug_ctx", 0.0, json.dumps(debug_ctx), base.run_id, base.source))

    if cfg["write_scenarios_debug"]:
        for name, sc in fired:
            rows.append((symbol, kind, tf, ts, f"QUAL.scenario.{name}", float(sc), "{}", base.run_id, base.source))

    write_values(rows)
    return (ts, final_score, final_veto, reversal)