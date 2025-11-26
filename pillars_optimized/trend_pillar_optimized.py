# pillars/trend_pillar.py
from __future__ import annotations
import json, math, configparser, ast, functools
from typing import Optional, Tuple, Dict, Any, List
import numpy as np
import pandas as pd
from utils.db import get_db_connection
from .common import (
    ema, atr, adx, bb_width_pct, resample, write_values, last_metric, clamp,
    TZ, DEFAULT_INI, BaseCfg, min_bars_for_tf, ensure_min_bars, maybe_trim_last_bar
)

# -----------------------------
# Optimized Rule Engine
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

def _eval_number(expr: str, vars: Dict[str, Any]) -> float:
    if not expr: return 0.0
    try:
        return float(eval(expr, {"__builtins__": {}}, vars))
    except:
        return 0.0

# -----------------------------
# Config
# -----------------------------
def _cfg(path=DEFAULT_INI):
    cp = configparser.ConfigParser(inline_comment_prefixes=(";", "#"), strict=False)
    cp.read(path)

    def _parse_tuple(key, default):
        s = cp.get("trend", key, fallback=default)
        return tuple(float(x.strip()) for x in s.split(","))

    return {
        "ma_short": cp.getint("trend", "ma_short", fallback=20),
        "ma_long": cp.getint("trend", "ma_long", fallback=50),
        "adx_win": cp.getint("trend", "adx_win", fallback=14),
        "adx_thresh": cp.getfloat("trend", "adx_thresh", fallback=25.0),
        "adx_slope_win": cp.getint("trend", "adx_slope_win", fallback=5),
        "psar_step": cp.getfloat("trend", "psar_step", fallback=0.02),
        "psar_max": cp.getfloat("trend", "psar_max", fallback=0.20),
        "rules_mode": cp.get("trend", "rules_mode", fallback="additive").strip().lower(),
        "clamp_low": cp.getfloat("trend", "clamp_low", fallback=0.0),
        "clamp_high": cp.getfloat("trend", "clamp_high", fallback=100.0),
        "write_scenarios_debug": cp.getboolean("trend", "write_scenarios_debug", fallback=False),
        "cp": cp
    }


# -----------------------------
# Helpers
# -----------------------------
def _psar(high, low, step, max_step):
    psar = low.copy()
    bull = True
    af = step
    hp = high.iloc[0]
    lp = low.iloc[0]

    for i in range(2, len(high)):
        if bull:
            psar.iloc[i] = psar.iloc[i-1] + af * (hp - psar.iloc[i-1])
        else:
            psar.iloc[i] = psar.iloc[i-1] + af * (lp - psar.iloc[i-1])

        if bull:
            if low.iloc[i] < psar.iloc[i]:
                bull = False
                psar.iloc[i] = hp
                lp = low.iloc[i]
                af = step
            else:
                if high.iloc[i] > hp:
                    hp = high.iloc[i]
                    af = min(af + step, max_step)
        else:
            if high.iloc[i] > psar.iloc[i]:
                bull = True
                psar.iloc[i] = lp
                hp = high.iloc[i]
                af = step
            else:
                if low.iloc[i] < lp:
                    lp = low.iloc[i]
                    af = min(af + step, max_step)
    return psar

def _slope_rank(s: pd.Series, n: int) -> float:
    if len(s) < n: return 0.5
    slopes = s.rolling(n).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0], raw=True)
    return slopes.rank(pct=True).iloc[-1] or 0.5

# -----------------------------
# Core Scorer (Context Optimized)
# -----------------------------
def _trend_score(dtf: pd.DataFrame, symbol: str, kind: str, tf: str, cfg: dict, context: Optional[Dict[str, Any]] = None) -> Tuple[float, bool, Dict[str, float], Dict[str, Any]]:
    c=dtf["close"]; h=dtf["high"]; l=dtf["low"]

    # --- MA Alignment ---
    ema_s = ema(c, cfg["ma_short"])
    ema_l = ema(c, cfg["ma_long"])
    ma_up = ema_s.iloc[-1] > ema_l.iloc[-1]
    ma_confirm = (ema_s.iloc[-1] > ema_s.iloc[-2]) if ma_up else (ema_s.iloc[-1] < ema_s.iloc[-2])
    ma_dist_pct = (ema_s.iloc[-1] - ema_l.iloc[-1]) / max(1e-9, ema_l.iloc[-1]) * 100.0

    ma_pts = 0.0
    if abs(ma_dist_pct) > 0.1:  ma_pts += 10.0
    if ma_confirm:              ma_pts += 10.0
    if ma_up and c.iloc[-1] > ema_s.iloc[-1]: ma_pts += 5.0
    if not ma_up and c.iloc[-1] < ema_s.iloc[-1]: ma_pts += 5.0

    # --- ADX Strength ---
    ADX = adx(h, l, c, cfg["adx_win"])
    adx_val = ADX["ADX_14"].iloc[-1]
    adx_slope_rank = _slope_rank(ADX["ADX_14"], cfg["adx_slope_win"])

    adx_pts = 0.0
    if adx_val > cfg["adx_thresh"]:
        adx_pts = 10.0 + 15.0 * adx_slope_rank
    else: # trending down or sideways
        adx_pts = 10.0 * (1.0 - adx_slope_rank)

    # --- PSAR Confirmation ---
    psar = _psar(h, l, cfg["psar_step"], cfg["psar_max"])
    psar_up = c.iloc[-1] > psar.iloc[-1]
    psar_confirm = (psar.iloc[-1] > psar.iloc[-2]) if psar_up else (psar.iloc[-1] < psar.iloc[-2])

    psar_pts = 0.0
    if psar_confirm: psar_pts += 15.0
    if (psar_up and ma_up) or (not psar_up and not ma_up): psar_pts += 10.0

    # --- Pivot Context ---
    # Using pre-calculated pivot points from context
    price = c.iloc[-1]
    r1 = context.get('pivot_r1') if context else None
    s1 = context.get('pivot_s1') if context else None

    pivot_pts = 10.0
    if ma_up and r1 is not None and price > r1:
        pivot_pts = 15.0
    elif not ma_up and s1 is not None and price < s1:
        pivot_pts = 15.0

    base_total = ma_pts + adx_pts + psar_pts + pivot_pts
    base_score = float(clamp(base_total, 0, 100))
    base_veto = bool(adx_val < 15.0)

    # ---- Scenario Rules ----
    vars = {
        "ma_up": ma_up, "ma_confirm": ma_confirm, "ma_dist_pct": abs(ma_dist_pct),
        "adx": adx_val, "adx_slope_rank": adx_slope_rank, "adx_strong": adx_val > cfg["adx_thresh"],
        "psar_up": psar_up, "psar_confirm": psar_confirm,
        "price_gt_r1": (r1 is not None and price > r1),
        "price_lt_s1": (s1 is not None and price < s1),
    }

    delta = 0.0; rule_veto = False; hits = []

    cp = cfg['cp']
    for section in ['bullish', 'bearish']:
        if not cp.has_section(section):
            continue
        for name, when_str in cp.items(section):
            when_tree = _compile_rule(when_str)
            if when_tree and _safe_eval(when_tree, vars):
                score = float(cp.get(section, name, fallback="0").split('when')[0].strip())
                delta += score
                hits.append({"name": name, "points": score})

    if cfg["rules_mode"] == "override": score = delta
    else: score = base_score + delta

    score = float(clamp(score, cfg["clamp_low"], cfg["clamp_high"]))
    veto_flag = bool(base_veto or rule_veto)

    parts = {
        "MA": ma_pts, "ADX": adx_pts, "PSAR": psar_pts, "Pivots": pivot_pts,
        "adx_val": adx_val, "adx_slope": adx_slope_rank,
        "ma_dist": ma_dist_pct, "rules_delta": delta,
    }
    debug_ctx = {"rules_mode": cfg["rules_mode"], "hits": hits}
    return score, veto_flag, parts, debug_ctx

# -----------------------------
# Public API
# -----------------------------
def score_trend(symbol: str, kind: str, tf: str, df5: pd.DataFrame, base: BaseCfg,
                ini_path=DEFAULT_INI, context: Optional[Dict[str, Any]] = None):

    dftf = resample(df5, tf)
    dftf = maybe_trim_last_bar(dftf)
    if not ensure_min_bars(dftf, tf): return None

    cfg = _cfg(ini_path)
    score, veto, parts, dbg = _trend_score(dftf, symbol, kind, tf, cfg, context)
    ts = dftf.index[-1].to_pydatetime().replace(tzinfo=TZ)

    rows = [
        (symbol, kind, tf, ts, "TREND.score", float(score), json.dumps({}), base.run_id, base.source),
        (symbol, kind, tf, ts, "TREND.veto_flag", 1.0 if veto else 0.0, "{}", base.run_id, base.source),

        # Parts
        (symbol, kind, tf, ts, "TREND.MA", float(parts["MA"]), "{}", base.run_id, base.source),
        (symbol, kind, tf, ts, "TREND.ADX", float(parts["ADX"]), "{}", base.run_id, base.source),
        (symbol, kind, tf, ts, "TREND.PSAR", float(parts["PSAR"]), "{}", base.run_id, base.source),
        (symbol, kind, tf, ts, "TREND.Pivots", float(parts["Pivots"]), "{}", base.run_id, base.source),
        (symbol, kind, tf, ts, "TREND.adx_val", float(parts["adx_val"]), "{}", base.run_id, base.source),
    ]

    if cfg.get("write_scenarios_debug"):
        rows.append((symbol, kind, tf, ts, "TREND.rules_hits", float(len(dbg.get("hits",[]))), json.dumps(dbg), base.run_id, base.source))

    write_values(rows)
    return (ts, score, veto)