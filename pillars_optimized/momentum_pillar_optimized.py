# pillars/momentum_pillar.py
from __future__ import annotations
import json, math, configparser, ast, functools
from typing import Optional, Tuple, Dict, List, Any
import numpy as np
import pandas as pd

from utils.db import get_db_connection
from .common import *
from .common import (
    ema, atr, adx, obv_series, bb_width_pct, resample, write_values, clamp,
    TZ, DEFAULT_INI, BaseCfg, min_bars_for_tf, ensure_min_bars, maybe_trim_last_bar, last_metric
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
    """OPTIMIZATION: Compiles the string expression into an AST object ONCE and caches it."""
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
# Local helpers
# -----------------------------
def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    if len(close) < 3:
        return pd.Series([50.0] * len(close), index=close.index, dtype=float)
    d = close.diff()
    g = d.clip(lower=0)
    l = (-d).clip(lower=0)
    ema_g = g.ewm(alpha=1 / max(1, n), adjust=False).mean()
    ema_l = l.ewm(alpha=1 / max(1, n), adjust=False).mean().replace(0, np.nan)
    rs = ema_g / ema_l
    out = 100 - 100 / (1 + rs)
    return out.fillna(50.0)

def _rmi(close: pd.Series, lb: int = 14, m: int = 5) -> pd.Series:
    if len(close) < m + 3:
        return pd.Series([50.0] * len(close), index=close.index, dtype=float)
    diffm = close.diff(m)
    up = diffm.clip(lower=0)
    dn = (-diffm).clip(lower=0)
    ema_up = up.ewm(span=max(2, lb), adjust=False).mean()
    ema_dn = dn.ewm(span=max(2, lb), adjust=False).mean().replace(0, np.nan)
    r = ema_up / ema_dn
    out = 100 - 100 / (1 + r)
    return out.fillna(50.0)

def _count_sign_flips(s: pd.Series, look: int = 12) -> int:
    if s is None or len(s.dropna()) < 2: return 0
    x = np.sign(s.tail(max(2, look)).fillna(0.0).values)
    return int(np.sum(np.abs(np.diff(x)) > 0))

def _safe_num(x: Any, default: float = 0.0) -> float:
    try:
        fx = float(x)
        return fx if np.isfinite(fx) else default
    except Exception:
        return default

# -----------------------------
# Config
# -----------------------------
def _cfg(path: str = DEFAULT_INI) -> dict:
    cp = configparser.ConfigParser(inline_comment_prefixes=(";", "#"), interpolation=None, strict=False)
    cp.read(path)
    return {
        "rsi_fast":  cp.getint("momentum", "rsi_fast",  fallback=5),
        "rsi_std":   cp.getint("momentum", "rsi_std",   fallback=14),
        "rmi_lb":    cp.getint("momentum", "rmi_lb",    fallback=14),
        "rmi_m":     cp.getint("momentum", "rmi_m",     fallback=5),
        "atr_win":   cp.getint("momentum", "atr_win",   fallback=14),
        "low_vol_thr":  cp.getfloat("momentum", "low_vol_thr",  fallback=3.0),
        "mid_vol_thr":  cp.getfloat("momentum", "mid_vol_thr",  fallback=6.0),
        "rules_mode": cp.get("momentum", "rules_mode", fallback="additive").lower(),
        "clamp_low": cp.getfloat("momentum", "clamp_low", fallback=0.0),
        "clamp_high": cp.getfloat("momentum", "clamp_high", fallback=100.0),
        "write_scenarios_debug": cp.getboolean("momentum", "write_scenarios_debug", fallback=False),
        "min_bars": cp.getint("momentum", "min_bars", fallback=120),
        "vol_avg_win": cp.getint("momentum", "vol_avg_win", fallback=20),
        "bb_win": cp.getint("momentum", "bb_win", fallback=20),
        "bb_k": cp.getfloat("momentum", "bb_k", fallback=2.0),
        "div_lookback": cp.getint("momentum", "div_lookback", fallback=5),
        "ml_enabled": cp.getboolean("momentum_ml", "enabled", fallback=False),
        "ml_blend_weight": cp.getfloat("momentum_ml", "blend_weight", fallback=0.35),
        "ml_source_table": cp.get("momentum_ml", "source_table", fallback="").strip(),
        "ml_callback": cp.get("momentum_ml", "callback", fallback="").strip(),
        "ml_veto_if_prob_lt": cp.get("momentum_ml", "veto_if_prob_lt", fallback="").strip(),
        "_ini_path": path,
        "cp": cp,
    }

# -----------------------------
# Feature builder (Context Aware)
# -----------------------------
def _mom_features(dtf: pd.DataFrame, symbol: str, kind: str, tf: str, cfg: dict, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    c = dtf["close"]; h = dtf["high"]; l = dtf["low"]
    o = dtf["open"];  v = dtf.get("volume", pd.Series(index=dtf.index, dtype=float)).fillna(0)

    ATR = atr(h, l, c, cfg["atr_win"])
    atr_val = _safe_num(ATR.iloc[-1], 0.0)
    px_now  = _safe_num(c.iloc[-1], 1.0)
    atr_pct_now = float((atr_val / max(1e-9, px_now)) * 100.0)
    atr_avg_20 = _safe_num(ATR.rolling(20).mean().iloc[-1] if len(ATR) >= 20 else atr_val, atr_val)

    rsi_fast = _rsi(c, cfg["rsi_fast"])
    rsi_std  = _rsi(c, cfg["rsi_std"])
    rmi      = _rmi(c, lb=cfg["rmi_lb"], m=cfg["rmi_m"])

    macd_line = ema(c, 12) - ema(c, 26)
    macd_sig  = macd_line.ewm(span=9, adjust=False).mean()
    hist      = macd_line - macd_sig
    hist_ema  = hist.ewm(span=5, adjust=False).mean()
    hist_diff = _safe_num(hist.diff().iloc[-1] if len(hist) > 1 else 0.0, 0.0)
    zero_cross_up = (len(macd_line) > 1) and (macd_line.iloc[-2] <= 0 <= macd_line.iloc[-1])
    zero_cross_down = (len(macd_line) > 1) and (macd_line.iloc[-2] >= 0 >= macd_line.iloc[-1])

    obv   = obv_series(c, v)
    obv_d = obv.diff()
    mu = _safe_num(obv_d.ewm(span=5, adjust=False).mean().iloc[-1] if len(obv_d.dropna()) else 0.0, 0.0)
    sd = _safe_num(obv_d.rolling(20).std(ddof=1).iloc[-1] if len(obv_d.dropna()) else 0.0, 0.0)
    z_obv = ( _safe_num(obv_d.iloc[-1], 0.0) - mu ) / (sd if sd > 0 else 1e9)

    vol_avg_win = int(cfg.get("vol_avg_win", 20))
    v_avg = v.rolling(vol_avg_win).mean()
    rvol_now = _safe_num(v.iloc[-1], 0.0) / max(1.0, _safe_num(v_avg.iloc[-1] if len(v_avg.dropna()) else 1.0, 1.0))

    tp = (h + l + c) / 3.0
    rmf = tp * v
    pos = (tp > tp.shift(1)).astype(float) * rmf
    neg = (tp < tp.shift(1)).astype(float) * rmf
    posn = pos.rolling(14, min_periods=7).sum()
    negn = neg.replace(0, np.nan).rolling(14, min_periods=7).sum()
    mfi_ratio = posn / negn
    ratio_val = _safe_num(mfi_ratio.iloc[-1] if len(mfi_ratio.dropna()) else 1.0, 1.0)
    mfi_now = float(100 - (100 / (1 + ratio_val)))
    mfi_up  = bool((mfi_ratio.diff().iloc[-1] or 0) > 0) if len(mfi_ratio.dropna()) else False

    roc3 = c.pct_change(3)
    roc_atr_ratio = abs(_safe_num(roc3.iloc[-1], 0.0)) / max(1e-9, atr_val / max(1e-9, px_now))

    a14, dip, dim = adx(h, l, c, 14)
    a9,  _,  _    = adx(h, l, c, 9)
    adx_rising = _safe_num(a14.diff().iloc[-1] if len(a14) > 1 else 0.0, 0.0) > 0
    di_plus  = _safe_num(dip.iloc[-1] if len(dip) else 0.0, 0.0)
    di_minus = _safe_num(dim.iloc[-1] if len(dim) else 0.0, 0.0)
    di_plus_gt = di_plus > di_minus

    bw = bb_width_pct(c, n=cfg.get("bb_win", 20), k=cfg.get("bb_k", 2.0))
    bw_now = _safe_num(bw.iloc[-1] if len(bw.dropna()) else 0.0, 0.0)
    bw_prev = _safe_num(bw.iloc[-2] if len(bw.dropna()) > 1 else bw_now, bw_now)
    tail = bw.tail(120).dropna()
    bb_width_pct_rank = float((tail <= bw_now).mean() * 100.0) if len(tail) >= 20 else 50.0
    squeeze_flag = int(bb_width_pct_rank <= 20.0)

    n = max(1, int(cfg.get("div_lookback", 5)))
    close_prev_n = _safe_num(c.iloc[-n] if len(c) > n else c.iloc[-1], px_now)
    low_prev_n   = _safe_num(l.iloc[-n] if len(l) > n else l.iloc[-1], _safe_num(l.iloc[-1], px_now))
    high_prev_n  = _safe_num(h.iloc[-n] if len(h) > n else h.iloc[-1], _safe_num(h.iloc[-1], px_now))
    rsi_prev5    = _safe_num(rsi_fast.iloc[-2] if len(rsi_fast) > 1 else rsi_fast.iloc[-1], 50.0)
    rsi5         = _safe_num(rsi_fast.iloc[-1] if len(rsi_fast) else 50.0, 50.0)
    rsi_prev_std_n = _safe_num(rsi_std.iloc[-n] if len(rsi_std) > n else rsi_std.iloc[-1], 50.0)
    rmi_now      = _safe_num(rmi.iloc[-1] if len(rmi) else 50.0, 50.0)
    rmi_prev_n   = _safe_num(rmi.iloc[-n] if len(rmi) > n else rmi_now, rmi_now)
    macd_hist_prev_n = _safe_num(hist.iloc[-n] if len(hist) > n else hist.iloc[-1], 0.0)
    bb_width_prev_n  = _safe_num(bw.iloc[-n] if len(bw) > n else bw_now, bw_now)
    ema50 = _safe_num(ema(c, 50).iloc[-1], px_now)
    flips_hist = _count_sign_flips(hist, look=12)

    # --- FETCH EXTERNAL METRICS (Context or DB) ---
    def get_meta(key, tf_override=None):
        """Helper to get metrics from context or fallback to DB."""
        # Check for TF-specific key first
        tf_key = f"{key}|{tf_override or tf}"
        if context and tf_key in context:
            return float(context[tf_key])

        # Check for global key
        if context and key in context:
            return float(context[key])

        # Fallback to database
        v = last_metric(symbol, kind, tf_override or tf, key)
        return float(v) if v is not None else 0.0

    tl_mom = get_meta("TL.momentum")
    tl_mom_prev = get_meta("TL.momentum.prev_d")
    roc21 = get_meta("ROC21")
    day_atr = get_meta("Day ATR")

    # Pivot points for the current timeframe
    pivot_r1 = get_meta("pivot_r1", tf)
    pivot_s1 = get_meta("pivot_s1", tf)
    pivot_r2 = get_meta("pivot_r2", tf)
    pivot_s2 = get_meta("pivot_s2", tf)

    # Calculate distances and relationships
    price_above_r1 = px_now > pivot_r1 if pivot_r1 else False
    price_below_s1 = px_now < pivot_s1 if pivot_s1 else False

    feat: Dict[str, float | int | bool] = {
        "open": _safe_num(o.iloc[-1], px_now), "close": px_now,
        "high": _safe_num(h.iloc[-1], px_now), "low": _safe_num(l.iloc[-1], px_now),
        "volume": _safe_num(v.iloc[-1], 0.0),
        "volume_avg_20": _safe_num(v.rolling(20).mean().iloc[-1] if len(v) >= 20 else v.mean(), 0.0),
        "rvol_now": float(rvol_now),
        "atr_pct": float(atr_pct_now), "atr_avg_20": float(atr_avg_20),
        "rsi_fast": float(_safe_num(rsi_fast.iloc[-1] if len(rsi_fast) else 50.0, 50.0)),
        "rsi_std": float(_safe_num(rsi_std.iloc[-1] if len(rsi_std) else 50.0, 50.0)),
        "rsi5": float(rsi5), "rsi_prev5": float(rsi_prev5),
        "rsi_prev_std_n": float(rsi_prev_std_n),
        "rmi": float(rmi_now), "rmi_now": float(rmi_now), "rmi_prev_n": float(rmi_prev_n),
        "macd_line": float(_safe_num(macd_line.iloc[-1], 0.0)),
        "macd_sig": float(_safe_num(macd_sig.iloc[-1], 0.0)),
        "hist": float(_safe_num(hist.iloc[-1], 0.0)),
        "macd_hist": float(_safe_num(hist.iloc[-1], 0.0)),
        "macd_hist_prev_n": float(macd_hist_prev_n),
        "hist_ema": float(_safe_num(hist_ema.iloc[-1], 0.0)),
        "hist_diff": float(hist_diff),
        "zero_cross_up": bool(zero_cross_up), "zero_cross_down": bool(zero_cross_down),
        "z_obv": float(z_obv), "mfi_now": float(mfi_now), "mfi_up": bool(mfi_up),
        "adx14": float(_safe_num(a14.iloc[-1], 0.0)), "adx9": float(_safe_num(a9.iloc[-1], 0.0)),
        "adx_rising": bool(adx_rising),
        "di_plus": float(di_plus), "di_minus": float(di_minus), "di_plus_gt": bool(di_plus_gt),
        "roc_atr_ratio": float(roc_atr_ratio),
        "bb_width_pct": float(bw_now), "bb_width": float(bw_now),
        "bb_width_pct_prev": float(bw_prev), "bb_width_prev_n": float(bb_width_prev_n),
        "bb_width_pct_rank": float(bb_width_pct_rank),
        "squeeze_flag": int(squeeze_flag),
        "close_prev_n": float(close_prev_n), "low_prev_n": float(low_prev_n), "high_prev_n": float(high_prev_n),
        "ema50": float(ema50), "whipsaw_flips": float(flips_hist),

        # External Metrics Mapped for INI
        "tl_mom": float(tl_mom),
        "tl_mom_prev": float(tl_mom_prev),
        "roc21": float(roc21),
        "day_atr": float(day_atr),

        # Pivot Data
        "pivot_r1": pivot_r1,
        "pivot_s1": pivot_s1,
        "pivot_r2": pivot_r2,
        "pivot_s2": pivot_s2,
        "price_above_r1": price_above_r1,
        "price_below_s1": price_below_s1
    }
    return feat

# -----------------------------
# Public API
# -----------------------------
def score_momentum(symbol: str, kind: str, tf: str, df5: pd.DataFrame, base: BaseCfg,
                   ini_path: str = DEFAULT_INI, context: Optional[Dict[str, Any]] = None):

    # 1. Check Orchestrator Resampling
    dftf = df5
    if tf != "15m" and pd.infer_freq(dftf.index) != tf:
         dftf = resample(df5, tf)

    dftf = maybe_trim_last_bar(dftf)
    cfg = _cfg(ini_path)

    if not ensure_min_bars(dftf, tf) or len(dftf) < cfg.get("min_bars", 120):
        return None

    F = _mom_features(dftf, symbol, kind, tf, cfg, context)
    total_score = 0
    parts = {}

    for section in ['bullish', 'bearish']:
        if not cfg['cp'].has_section(section):
            continue
        for name, when_str in cfg['cp'].items(section):
            when_tree = _compile_rule(when_str)
            if when_tree and _safe_eval(when_tree, F):
                score = float(cfg['cp'].get(section, name, fallback="0").split('when')[0].strip())
                total_score += score
                parts[f"SCN.{name}"] = score

    total_score = float(clamp(total_score, cfg.get("clamp_low", 0.0), cfg.get("clamp_high", 100.0)))
    ts = dftf.index[-1].to_pydatetime().replace(tzinfo=TZ)

    # ML Blend Logic (Optional)
    ml_prob = None
    ml_score = None
    final_score = total_score
    final_veto = False

    try:
        if cfg.get("ml_enabled", False):
            w = float(cfg.get("ml_blend_weight", 0.35))
            w = max(0.0, min(1.0, w))

            # Path A: DB Source
            tbl = cfg.get("ml_source_table", "")
            if tbl:
                with get_db_connection() as conn, conn.cursor() as cur:
                    cur.execute(f"SELECT prob_long, prob_short FROM {tbl} WHERE symbol=%s AND tf=%s AND ts=%s LIMIT 1", (symbol, tf, ts))
                    r = cur.fetchone()
                if r:
                    ml_prob = max(_safe_num(r[0], 0.5), 1.0 - _safe_num(r[1], 0.5))

            # Path B: Callback
            if ml_prob is None:
                cb = cfg.get("ml_callback", "")
                if cb:
                    mod_name, _, fn_name = cb.rpartition(".")
                    if mod_name and fn_name:
                        import importlib
                        mod = importlib.import_module(mod_name)
                        fn  = getattr(mod, fn_name)
                        ml_prob = float(fn(symbol=symbol, kind=kind, tf=tf, ts=ts, frame=dftf, features=F))
                        ml_prob = max(0.0, min(1.0, ml_prob))

            # Blend
            if ml_prob is not None:
                base_prob = float(total_score) / 100.0
                blended   = (1.0 - w) * base_prob + w * ml_prob
                final_score = round(100.0 * blended, 2)

                thr_txt = cfg.get("ml_veto_if_prob_lt", "").strip()
                if thr_txt and blended < float(thr_txt):
                     final_veto = True

            if ml_prob is not None:
                 ml_ctx = {"ml_prob": ml_prob, "blend_weight": cfg.get("ml_blend_weight", 0.35)}
                 parts["ml_ctx"] = 0.0 # Placeholder for parts dict, full json below

    except Exception:
        pass

    rows = [
        (symbol, kind, tf, ts, "MOM.score", float(total_score), json.dumps({"rules_mode": cfg.get("rules_mode")}), base.run_id, base.source),
        (symbol, kind, tf, ts, "MOM.score_final", float(final_score), "{}", base.run_id, base.source),
        (symbol, kind, tf, ts, "MOM.veto_final",  1.0 if final_veto else 0.0, "{}", base.run_id, base.source),
    ]

    for k, v in parts.items():
        rows.append((symbol, kind, tf, ts, f"MOM.{k}", float(v), "{}", base.run_id, base.source))

    if "ml_ctx" in locals() and ml_ctx:
         rows.append((symbol, kind, tf, ts, "MOM.ml_ctx", 0.0, json.dumps(ml_ctx), base.run_id, base.source))

    write_values(rows)
    return (ts, float(final_score), bool(final_veto))