# pillars/trend_pillar.py
from __future__ import annotations
import json, math, configparser, ast, functools
from typing import Optional, Tuple, Dict, Any, List
import numpy as np
import pandas as pd

from utils.db import get_db_connection
from .common import (
    ema, atr, adx, resample, write_values, last_metric, clamp,
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
    if not expr or not expr.strip(): return None
    try: return ast.parse(expr, mode="eval")
    except Exception: return None

def _safe_eval(expr_or_tree: str | ast.AST, scope: Dict[str, Any]) -> bool:
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

    try: return bool(_eval(tree))
    except Exception: return False

# -----------------------------
# Helpers & Math
# -----------------------------
def _supertrend(high, low, close, period=10, multiplier=3.0):
    """Vectorized SuperTrend."""
    # Basic ATR
    tr1 = pd.DataFrame(high - low)
    tr2 = pd.DataFrame(abs(high - close.shift(1)))
    tr3 = pd.DataFrame(abs(low - close.shift(1)))
    frames = [tr1, tr2, tr3]
    tr = pd.concat(frames, axis=1, join='outer').max(axis=1)
    atr = tr.ewm(alpha=1/period).mean()

    # HL2
    hl2 = (high + low) / 2

    # Fast iterative implementation
    m = len(close)
    dir_ = np.zeros(m)

    # Convert to numpy for speed
    hl2_v = hl2.values
    atr_v = atr.values
    close_v = close.values

    # Upper/Lower arrays
    basic_upper = hl2_v + (multiplier * atr_v)
    basic_lower = hl2_v - (multiplier * atr_v)

    final_upper = np.copy(basic_upper)
    final_lower = np.copy(basic_lower)

    # We need to iterate because current value depends on previous trend
    for i in range(1, m):
        if basic_upper[i] < final_upper[i-1] or close_v[i-1] > final_upper[i-1]:
            final_upper[i] = basic_upper[i]
        else:
            final_upper[i] = final_upper[i-1]

        if basic_lower[i] > final_lower[i-1] or close_v[i-1] < final_lower[i-1]:
            final_lower[i] = basic_lower[i]
        else:
            final_lower[i] = final_lower[i-1]

        # Direction: 1 = Uptrend, -1 = Downtrend
        prev_dir = dir_[i-1] if i > 0 else 1

        if prev_dir == 1:
            if close_v[i] < final_lower[i]:
                dir_[i] = -1
            else:
                dir_[i] = 1
        else:
            if close_v[i] > final_upper[i]:
                dir_[i] = 1
            else:
                dir_[i] = -1

    return pd.Series(dir_, index=close.index)

def _safe_num(x, default=0.0):
    try:
        v = float(x)
        return v if np.isfinite(v) else default
    except:
        return default

def _rsi(series: pd.Series, win: int = 14) -> pd.Series:
    if series is None or len(series) < 3:
        return pd.Series([50.0] * len(series), index=series.index, dtype=float) if series is not None else pd.Series(dtype=float)
    delta = series.diff()
    up = delta.clip(lower=0.0)
    down = (-delta).clip(lower=0.0)
    roll_up = up.ewm(alpha=1.0 / max(1, win), adjust=False).mean()
    roll_down = down.ewm(alpha=1.0 / max(1, win), adjust=False).mean().replace(0, np.nan)
    rs = roll_up / roll_down
    out = 100.0 - (100.0 / (1.0 + rs))
    return out.fillna(50.0)

# -----------------------------
# Config
# -----------------------------
def _cfg(path=DEFAULT_INI) -> dict:
    cp = configparser.ConfigParser(inline_comment_prefixes=(";", "#"), strict=False)
    cp.read(path)

    return {
        "ema_short": cp.getint("trend", "ema_short", fallback=20),
        "ema_mid": cp.getint("trend", "ema_mid", fallback=50),
        "ema_long": cp.getint("trend", "ema_long", fallback=200),
        "adx_len": cp.getint("trend", "adx_len", fallback=14),
        "st_period": cp.getint("trend", "st_period", fallback=10),
        "st_mult": cp.getfloat("trend", "st_mult", fallback=3.0),

        "rules_mode": cp.get("trend", "rules_mode", fallback="additive").lower(),
        "clamp_low": cp.getfloat("trend", "clamp_low", fallback=0.0),
        "clamp_high": cp.getfloat("trend", "clamp_high", fallback=100.0),
        "min_bars": cp.getint("trend", "min_bars", fallback=120),

        # Penalty thresholds (ATR% of price)
        "atr_penalty_8":  cp.getfloat("trend", "atr_penalty_8",  fallback=8.0),
        "atr_penalty_10": cp.getfloat("trend", "atr_penalty_10", fallback=10.0),
        "atr_penalty_15": cp.getfloat("trend", "atr_penalty_15", fallback=15.0),
        "poc_align_bonus": cp.getfloat("trend", "poc_align_bonus", fallback=5.0),
        "roc_win":  cp.getint("trend", "roc_win",  fallback=5),
        "atr_win":  cp.getint("trend", "atr_win",  fallback=14),
        "adx_main": cp.getint("trend", "adx_main", fallback=14),
        "adx_fast": cp.getint("trend", "adx_fast", fallback=9),
        "vol_avg_win": cp.getint("trend", "vol_avg_win", fallback=20),
        "bb_win": cp.getint("trend", "bb_win", fallback=20),
        "bb_k": cp.getfloat("trend", "bb_k", fallback=2.0),
        "kc_win": cp.getint("trend", "kc_win", fallback=20),
        "kc_mult": cp.getfloat("trend", "kc_mult", fallback=1.5),

        "ml_enabled": cp.getboolean("trend_ml", "enabled", fallback=False),
        "ml_blend_weight": cp.getfloat("trend_ml", "blend_weight", fallback=0.35),
        "ml_source_table": cp.get("trend_ml", "source_table", fallback="").strip(),
        "ml_callback": cp.get("trend_ml", "callback", fallback="").strip(),
        "ml_veto_if_prob_lt": cp.get("trend_ml", "veto_if_prob_lt", fallback="").strip(),
        "cp": cp,
        "_ini_path": path
    }

def _load_scenarios(cfg: dict) -> Tuple[str, List[dict]]:
    cp = cfg["cp"]
    names = []
    if cp.has_section("trend_scenarios"):
        raw = cp.get("trend_scenarios", "list", fallback="")
        names = [n.strip() for n in raw.replace("\n"," ").split(",") if n.strip()]
    return cfg["rules_mode"], names

# -----------------------------
# Feature Builder
# -----------------------------
def _build_features(dftf: pd.DataFrame, cfg: dict, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    c = dftf["close"]; h = dftf["high"]; l = dftf["low"]; o = dftf["open"]
    v = dftf.get("volume", pd.Series(index=dftf.index, dtype=float)).fillna(0)

    # 1. EMAs
    e1 = ema(c, cfg["ema_short"])
    e2 = ema(c, cfg["ema_mid"])
    e3 = ema(c, cfg["ema_long"])

    # 2. SuperTrend
    st_dir = _supertrend(h, l, c, cfg["st_period"], cfg["st_mult"])

    # 3. ADX
    adx_v, pdi, mdi = adx(h, l, c, cfg["adx_len"])
    adx9, _, _      = adx(h, l, c, cfg["adx_fast"])

    # 4. MACD
    macd_line = ema(c, 12) - ema(c, 26)
    macd_sig  = macd_line.ewm(span=9, adjust=False).mean()
    hist      = macd_line - macd_sig

    # 5. ATR & ROC
    ATR = atr(h, l, c, cfg["atr_win"])
    atr_val = _safe_num(ATR.iloc[-1], 0.0)
    roc = c.pct_change(cfg["roc_win"])

    # 6. BB & KC (Squeeze)
    bb_mid = c.rolling(cfg["bb_win"]).mean()
    bb_std = c.rolling(cfg["bb_win"]).std(ddof=0)
    bb_up  = bb_mid + cfg["bb_k"] * bb_std
    bb_lo  = bb_mid - cfg["bb_k"] * bb_std

    kc_mid = ema(c, cfg["kc_win"])
    kc_up  = kc_mid + cfg["kc_mult"] * ATR
    kc_lo  = kc_mid - cfg["kc_mult"] * ATR

    # Last values
    last_c = float(c.iloc[-1])
    last_e1 = float(e1.iloc[-1])
    last_e2 = float(e2.iloc[-1])
    last_e3 = float(e3.iloc[-1])

    # Slopes (normalized)
    def _slope(s): return (s.iloc[-1] - s.iloc[-5]) / s.iloc[-1] * 100 if len(s) > 5 else 0.0
    slope_e1 = _slope(e1)
    slope_e2 = _slope(e2)
    slope_e3 = _slope(e3)

    # Context injection
    def get_meta(k):
        if context and k in context: return float(context[k])
        # Fallback logic would go here, but avoiding DB hits
        return 0.0

    pivot_p = get_meta("pivot_p")
    poc = get_meta("VP.POC")

    # Helper checks
    squeeze = 1 if (bb_up.iloc[-1] - bb_lo.iloc[-1]) < (kc_up.iloc[-1] - kc_lo.iloc[-1]) else 0
    bb_w_now = (bb_up.iloc[-1] - bb_lo.iloc[-1]) / max(1e-9, last_c) * 100.0
    bb_w_prev = (bb_up.iloc[-2] - bb_lo.iloc[-2]) / max(1e-9, c.iloc[-2]) * 100.0 if len(c) > 1 else bb_w_now

    poc_dist_atr = float(abs(last_c - poc) / max(1e-9, atr_val)) if poc > 0 else 0.0

    feats = {
        "close": last_c,
        "ema_short": last_e1, "ema_mid": last_e2, "ema_long": last_e3,
        "slope_short": slope_e1, "slope_mid": slope_e2, "slope_long": slope_e3,

        "st_dir": int(st_dir.iloc[-1]),
        "adx": float(adx_v.iloc[-1]),
        "pdi": float(pdi.iloc[-1]),
        "mdi": float(mdi.iloc[-1]),
        "dip_gt_dim": bool(pdi.iloc[-1] > mdi.iloc[-1]),
        "adx9": float(adx9.iloc[-1]),
        "adx14": float(adx_v.iloc[-1]),

        "macd_line": float(macd_line.iloc[-1]),
        "macd_sig": float(macd_sig.iloc[-1]),
        "hist_diff": float(hist.diff().iloc[-1]) if len(hist) > 1 else 0.0,

        "atr_pct": float(atr_val / max(1e-9, last_c) * 100.0),
        "roc_abs_over_atr_ratio": float(abs(roc.iloc[-1]) / max(1e-9, atr_val / last_c)) if len(roc) else 0.0,

        "squeeze_flag": int(squeeze),
        "bb_width_pct": float(bb_w_now),
        "bb_width_pct_prev": float(bb_w_prev),

        # Booleans for easy rules
        "price_gt_ema_short": bool(last_c > last_e1),
        "price_gt_ema_mid":   bool(last_c > last_e2),
        "price_gt_ema_long":  bool(last_c > last_e3),
        "ema_stack_bull":     bool(last_e1 > last_e2 > last_e3),
        "ema_stack_bear":     bool(last_e1 < last_e2 < last_e3),
        "adx_trend_strength": bool(float(adx_v.iloc[-1]) > 25),
        "supertrend_bull":    bool(st_dir.iloc[-1] == 1),
        "pivot_p": pivot_p,
        "above_pivot": bool(pivot_p > 0 and last_c > pivot_p),
        "poc_dist_atr": poc_dist_atr,
    }

    # Previous values for checks
    feats["adx14_prev"] = float(adx_v.iloc[-2]) if len(adx_v) > 1 else feats["adx14"]
    feats["adx9_prev"]  = float(adx9.iloc[-2])  if len(adx9) > 1  else feats["adx9"]

    return feats

# -----------------------------
# Scenario Engine
# -----------------------------
def _run_scenarios(features: Dict[str, Any], cfg: dict, names: List[str]) -> Tuple[float, bool, Dict[str, float], List[tuple]]:
    cp = cfg["cp"]
    total = 0.0
    veto = False
    parts = {}
    fired = []

    for name in names:
        sec = f"trend_scenario.{name}"
        if not cp.has_section(sec): continue

        when_tree = _compile_rule(cp.get(sec, "when", fallback=""))
        if when_tree and _safe_eval(when_tree, features):
            sc = float(cp.get(sec, "score", fallback="0"))
            total = sc if cfg["rules_mode"] == "override" else (total + sc)
            parts[name] = sc
            fired.append((name, sc))

            if cp.getboolean(sec, "set_veto", fallback=False):
                veto = True

            # Bonus
            bonus_tree = _compile_rule(cp.get(sec, "bonus_when", fallback=""))
            if bonus_tree and _safe_eval(bonus_tree, features):
                b = float(cp.get(sec, "bonus", fallback="0"))
                total += b
                parts[f"{name}.bonus"] = b
                fired.append((f"{name}.bonus", b))

    return total, veto, parts, fired

# -----------------------------
# Public API
# -----------------------------
def score_trend(symbol: str, kind: str, tf: str, df5: pd.DataFrame, base: BaseCfg,
                ini_path=DEFAULT_INI, context: Optional[Dict[str, Any]] = None):

    # 1. Data Prep
    # Resample if needed (Orchestrator might have passed 5m, we need TF)
    # Optimized: Check freq first
    dftf = df5
    if tf != "15m" and pd.infer_freq(dftf.index) != tf:
        dftf = resample(df5, tf)

    dftf = maybe_trim_last_bar(dftf)
    if not ensure_min_bars(dftf, tf): return None

    cfg = _cfg(ini_path)
    if len(dftf) < cfg["min_bars"]: return None

    # 2. Build Features (Context Aware)
    feats = _build_features(dftf, cfg, context)

    # 3. Run Scenarios
    _, names = _load_scenarios(cfg)
    total, veto, parts, fired = _run_scenarios(feats, cfg, names)

    score = float(clamp(total, cfg["clamp_low"], cfg["clamp_high"]))
    ts = dftf.index[-1].to_pydatetime().replace(tzinfo=TZ)

    # 4. ML Blend
    final_score = score
    final_veto = veto
    ml_prob = None

    try:
        if cfg["ml_enabled"]:
            w = max(0.0, min(1.0, float(cfg["ml_blend_weight"])))
            # ... Standard ML Logic Placeholder ...
            # If enabled, load model -> predict -> blend
            pass
    except:
        pass

    # 5. Write Output
    rows = [
        (symbol, kind, tf, ts, "TREND.score", score, json.dumps({}), base.run_id, base.source),
        (symbol, kind, tf, ts, "TREND.veto_flag", 1.0 if veto else 0.0, "{}", base.run_id, base.source),
        (symbol, kind, tf, ts, "TREND.score_final", final_score, "{}", base.run_id, base.source),
        (symbol, kind, tf, ts, "TREND.veto_final", 1.0 if final_veto else 0.0, "{}", base.run_id, base.source),
    ]

    # Debug Context
    debug_ctx = {
        "ema_stack_bull": feats["ema_stack_bull"],
        "st_dir": feats["st_dir"],
        "adx": feats["adx"],
        "slope_short": feats["slope_short"]
    }
    rows.append((symbol, kind, tf, ts, "TREND.debug_ctx", 0.0, json.dumps(debug_ctx), base.run_id, base.source))

    if fired:
        for n, s in fired:
            rows.append((symbol, kind, tf, ts, f"TREND.scenario.{n}", float(s), "{}", base.run_id, base.source))

    write_values(rows)
    return (ts, final_score, final_veto)