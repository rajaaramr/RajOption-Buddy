from __future__ import annotations

import ast
import configparser
import functools
import json
import math
from pathlib import Path
from typing import Optional, Tuple, Dict, List, Any

import numpy as np
import pandas as pd

from utils.db import get_db_connection
from pillars.common import (
    ema,
    atr,
    adx,
    obv_series,
    bb_width_pct,
    resample,
    write_values,
    clamp,
    TZ,
    BaseCfg,
    min_bars_for_tf,
    ensure_min_bars,
    maybe_trim_last_bar,
)

# ---------------------------------------------------------------------
# Paths / defaults
# ---------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_INI = str(BASE_DIR / "momentum_scenarios.ini")


# ---------------------------------------------------------------------
# Optimized Safe Scenario Evaluator
# ---------------------------------------------------------------------
@functools.lru_cache(maxsize=2048)
def _compile_rule(expr: str):
    """Compile string expression into AST once & cache."""
    if not expr or not expr.strip():
        return None
    try:
        return ast.parse(expr, mode="eval")
    except Exception:
        return None


def _safe_eval(expr_or_tree: str | ast.AST, scope: Dict[str, Any]) -> bool:
    """Evaluate expression safely against scope using cached AST."""
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

        if isinstance(node, ast.BoolOp):
            vals = [_eval(v) for v in node.values]
            if isinstance(node.op, ast.And):
                return all(vals)
            if isinstance(node.op, ast.Or):
                return any(vals)
            return False

        if isinstance(node, ast.UnaryOp):
            val = _eval(node.operand)
            if isinstance(node.op, ast.Not):
                return not val
            if isinstance(node.op, ast.USub):
                return -val
            if isinstance(node.op, ast.UAdd):
                return +val
            return val

        if isinstance(node, ast.BinOp):
            a, b = _eval(node.left), _eval(node.right)
            if isinstance(node.op, ast.Add):
                return a + b
            if isinstance(node.op, ast.Sub):
                return a - b
            if isinstance(node.op, ast.Mult):
                return a * b
            if isinstance(node.op, ast.Div):
                return a / b if b != 0 else 0.0
            if isinstance(node.op, ast.Mod):
                return a % b if b != 0 else 0.0
            if isinstance(node.op, ast.FloorDiv):
                return a // b if b != 0 else 0.0
            if isinstance(node.op, ast.Pow):
                return a ** b
            return 0.0

        if isinstance(node, ast.Compare):
            a = _eval(node.left)
            b = _eval(node.comparators[0])
            op = node.ops[0]
            if isinstance(op, ast.Lt):
                return a < b
            if isinstance(op, ast.Gt):
                return a > b
            if isinstance(op, ast.Le):
                return a <= b
            if isinstance(op, ast.Ge):
                return a >= b
            if isinstance(op, ast.Eq):
                return a == b
            if isinstance(op, ast.NotEq):
                return a != b
            return False

        if isinstance(node, ast.Name):
            return scope.get(node.id, 0.0)

        if isinstance(node, ast.Constant):
            return node.value

        return False

    try:
        return bool(_eval(tree))
    except Exception:
        return False


# ---------------------------------------------------------------------
# Local helpers
# ---------------------------------------------------------------------
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
    if s is None or len(s.dropna()) < 2:
        return 0
    x = np.sign(s.tail(max(2, look)).fillna(0.0).values)
    return int(np.sum(np.abs(np.diff(x)) > 0))


def _safe_num(x: Any, default: float = 0.0) -> float:
    try:
        fx = float(x)
        return fx if np.isfinite(fx) else default
    except Exception:
        return default


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
def _cfg(path: str | None = None) -> dict:
    cp = configparser.ConfigParser(
        inline_comment_prefixes=(";", "#"),
        interpolation=None,
        strict=False,
    )

    if not path:
        path = DEFAULT_INI

    cp.read(path)

    ml_enabled = cp.getboolean("momentum_ml", "enabled", fallback=False)
    ml_blend_weight = cp.getfloat("momentum_ml", "blend_weight", fallback=0.35)
    ml_target_name = cp.get("momentum_ml", "target_name", fallback="momentum_ml.target.bull_2pct_4h").strip()
    ml_version = cp.get("momentum_ml", "version", fallback="xgb_v1").strip()
    ml_calibration_table = cp.get(
        "momentum_ml", "calibration_table", fallback="indicators.momentum_calibration_4h"
    ).strip()
    ml_veto_if_prob_lt = cp.get("momentum_ml", "veto_if_prob_lt", fallback="").strip()

    return {
        "rsi_fast": cp.getint("momentum", "rsi_fast", fallback=5),
        "rsi_std": cp.getint("momentum", "rsi_std", fallback=14),
        "rmi_lb": cp.getint("momentum", "rmi_lb", fallback=14),
        "rmi_m": cp.getint("momentum", "rmi_m", fallback=5),
        "atr_win": cp.getint("momentum", "atr_win", fallback=14),
        "low_vol_thr": cp.getfloat("momentum", "low_vol_thr", fallback=3.0),
        "mid_vol_thr": cp.getfloat("momentum", "mid_vol_thr", fallback=6.0),
        "rules_mode": cp.get("momentum", "rules_mode", fallback="additive").lower(),
        "clamp_low": cp.getfloat("momentum", "clamp_low", fallback=0.0),
        "clamp_high": cp.getfloat("momentum", "clamp_high", fallback=100.0),
        "write_scenarios_debug": cp.getboolean("momentum", "write_scenarios_debug", fallback=False),
        "min_bars": cp.getint("momentum", "min_bars", fallback=120),
        "vol_avg_win": cp.getint("momentum", "vol_avg_win", fallback=20),
        "bb_win": cp.getint("momentum", "bb_win", fallback=20),
        "bb_k": cp.getfloat("momentum", "bb_k", fallback=2.0),
        "div_lookback": cp.getint("momentum", "div_lookback", fallback=5),
        # ML config (Flow-style)
        "ml_enabled": ml_enabled,
        "ml_blend_weight": ml_blend_weight,
        "ml_target_name": ml_target_name,
        "ml_version": ml_version,
        "ml_calibration_table": ml_calibration_table,
        "ml_veto_if_prob_lt": ml_veto_if_prob_lt,
        # raw cp + ini path for any advanced use
        "_ini_path": path,
        "cp": cp,
    }


# ---------------------------------------------------------------------
# Scenario loader
# ---------------------------------------------------------------------
def _load_mom_scenarios(cfg: dict) -> Tuple[str, List[dict]]:
    cp = cfg["cp"]
    rules_mode = cfg["rules_mode"]

    names: List[str] = []
    if cp.has_section("mom_scenarios"):
        raw = cp.get("mom_scenarios", "list", fallback="")
        raw = raw.replace("\n", " ")
        names = [n.strip() for n in raw.split(",") if n.strip()]

    scenarios: List[dict] = []
    for n in names:
        sec = f"mom_scenario.{n}"
        if not cp.has_section(sec):
            continue
        scenarios.append(
            {
                "name": n,
                "when": cp.get(sec, "when", fallback=""),
                "score": cp.getfloat(sec, "score", fallback=0.0),
                "bonus_when": cp.get(sec, "bonus_when", fallback=""),
                "bonus": cp.getfloat(sec, "bonus", fallback=0.0),
            }
        )
    return rules_mode, scenarios


# ---------------------------------------------------------------------
# Feature builder
# ---------------------------------------------------------------------
def _mom_features(dtf: pd.DataFrame, cfg: dict) -> Dict[str, Any]:
    c = dtf["close"]
    h = dtf["high"]
    l = dtf["low"]
    o = dtf["open"]
    v = dtf.get("volume", pd.Series(index=dtf.index, dtype=float)).fillna(0)

    # ATR and ATR%
    ATR = atr(h, l, c, cfg["atr_win"])
    atr_val = _safe_num(ATR.iloc[-1], 0.0)
    px_now = _safe_num(c.iloc[-1], 1.0)
    atr_pct_now = float((atr_val / max(1e-9, px_now)) * 100.0)
    atr_avg_20 = _safe_num(ATR.rolling(20).mean().iloc[-1] if len(ATR) >= 20 else atr_val, atr_val)

    # RSI / RMI
    rsi_fast = _rsi(c, cfg["rsi_fast"])
    rsi_std = _rsi(c, cfg["rsi_std"])
    rmi = _rmi(c, lb=cfg["rmi_lb"], m=cfg["rmi_m"])

    # MACD
    macd_line = ema(c, 12) - ema(c, 26)
    macd_sig = macd_line.ewm(span=9, adjust=False).mean()
    hist = macd_line - macd_sig
    hist_ema = hist.ewm(span=5, adjust=False).mean()
    hist_diff = _safe_num(hist.diff().iloc[-1] if len(hist) > 1 else 0.0, 0.0)
    zero_cross_up = (len(macd_line) > 1) and (macd_line.iloc[-2] <= 0 <= macd_line.iloc[-1])
    zero_cross_down = (len(macd_line) > 1) and (macd_line.iloc[-2] >= 0 >= macd_line.iloc[-1])

    # OBV / z-score
    obv = obv_series(c, v)
    obv_d = obv.diff()
    mu = _safe_num(obv_d.ewm(span=5, adjust=False).mean().iloc[-1] if len(obv_d.dropna()) else 0.0, 0.0)
    sd = _safe_num(
        obv_d.rolling(20).std(ddof=1).iloc[-1] if len(obv_d.dropna()) else 0.0,
        0.0,
    )
    z_obv = (_safe_num(obv_d.iloc[-1], 0.0) - mu) / (sd if sd > 0 else 1e9)

    # Relative volume
    vol_avg_win = int(cfg.get("vol_avg_win", 20))
    v_avg = v.rolling(vol_avg_win).mean()
    rvol_now = _safe_num(v.iloc[-1], 0.0) / max(
        1.0,
        _safe_num(v_avg.iloc[-1] if len(v_avg.dropna()) else 1.0, 1.0),
    )

    # MFI-ish
    tp = (h + l + c) / 3.0
    rmf = tp * v
    pos = (tp > tp.shift(1)).astype(float) * rmf
    neg = (tp < tp.shift(1)).astype(float) * rmf
    posn = pos.rolling(14, min_periods=7).sum()
    negn = neg.replace(0, np.nan).rolling(14, min_periods=7).sum()
    mfi_ratio = posn / negn
    ratio_val = _safe_num(mfi_ratio.iloc[-1] if len(mfi_ratio.dropna()) else 1.0, 1.0)
    mfi_now = float(100 - (100 / (1 + ratio_val)))
    mfi_up = bool((mfi_ratio.diff().iloc[-1] or 0) > 0) if len(mfi_ratio.dropna()) else False

    # ROC vs ATR
    roc3 = c.pct_change(3)
    roc_atr_ratio = abs(_safe_num(roc3.iloc[-1], 0.0)) / max(1e-9, atr_val / max(1e-9, px_now))

    # ADX/DI
    a14, dip, dim = adx(h, l, c, 14)
    a9, _, _ = adx(h, l, c, 9)
    adx_rising = _safe_num(a14.diff().iloc[-1] if len(a14) > 1 else 0.0, 0.0) > 0
    di_plus = _safe_num(dip.iloc[-1] if len(dip) else 0.0, 0.0)
    di_minus = _safe_num(dim.iloc[-1] if len(dim) else 0.0, 0.0)
    di_plus_gt = di_plus > di_minus

    # Bollinger width
    bw = bb_width_pct(c, n=cfg.get("bb_win", 20), k=cfg.get("bb_k", 2.0))
    bw_now = _safe_num(bw.iloc[-1] if len(bw.dropna()) else 0.0, 0.0)
    bw_prev = _safe_num(bw.iloc[-2] if len(bw.dropna()) > 1 else bw_now, bw_now)
    tail = bw.tail(120).dropna()
    bb_width_pct_rank = float((tail <= bw_now).mean() * 100.0) if len(tail) >= 20 else 50.0
    squeeze_flag = int(bb_width_pct_rank <= 20.0)

    # Lookback N
    n = max(1, int(cfg.get("div_lookback", 5)))
    close_prev_n = _safe_num(c.iloc[-n] if len(c) > n else c.iloc[-1], px_now)
    low_prev_n = _safe_num(l.iloc[-n] if len(l) > n else l.iloc[-1], _safe_num(l.iloc[-1], px_now))
    high_prev_n = _safe_num(h.iloc[-n] if len(h) > n else h.iloc[-1], _safe_num(h.iloc[-1], px_now))
    rsi_prev5 = _safe_num(rsi_fast.iloc[-2] if len(rsi_fast) > 1 else rsi_fast.iloc[-1], 50.0)
    rsi5 = _safe_num(rsi_fast.iloc[-1] if len(rsi_fast) else 50.0, 50.0)
    rsi_prev_std_n = _safe_num(rsi_std.iloc[-n] if len(rsi_std) > n else rsi_std.iloc[-1], 50.0)
    rmi_now = _safe_num(rmi.iloc[-1] if len(rmi) else 50.0, 50.0)
    rmi_prev_n = _safe_num(rmi.iloc[-n] if len(rmi) > n else rmi_now, rmi_now)
    macd_hist_prev_n = _safe_num(hist.iloc[-n] if len(hist) > n else hist.iloc[-1], 0.0)
    bb_width_prev_n = _safe_num(bw.iloc[-n] if len(bw) > n else bw_now, bw_now)

    # EMA anchor
    ema50 = _safe_num(ema(c, 50).iloc[-1], px_now)

    # Whipsaw
    flips_hist = _count_sign_flips(hist, look=12)

    return {
        "open": _safe_num(o.iloc[-1], px_now),
        "close": px_now,
        "high": _safe_num(h.iloc[-1], px_now),
        "low": _safe_num(l.iloc[-1], px_now),
        "volume": _safe_num(v.iloc[-1], 0.0),
        "volume_avg_20": _safe_num(v.rolling(20).mean().iloc[-1] if len(v) >= 20 else v.mean(), 0.0),
        "rvol_now": float(rvol_now),
        "atr_pct": float(atr_pct_now),
        "atr_avg_20": float(atr_avg_20),
        "rsi_fast": float(_safe_num(rsi_fast.iloc[-1] if len(rsi_fast) else 50.0, 50.0)),
        "rsi_std": float(_safe_num(rsi_std.iloc[-1] if len(rsi_std) else 50.0, 50.0)),
        "rsi5": float(rsi5),
        "rsi_prev5": float(rsi_prev5),
        "rsi_prev_std_n": float(rsi_prev_std_n),
        "rmi": float(rmi_now),
        "rmi_now": float(rmi_now),
        "rmi_prev_n": float(rmi_prev_n),
        "macd_line": float(_safe_num(macd_line.iloc[-1], 0.0)),
        "macd_sig": float(_safe_num(macd_sig.iloc[-1], 0.0)),
        "hist": float(_safe_num(hist.iloc[-1], 0.0)),
        "macd_hist": float(_safe_num(hist.iloc[-1], 0.0)),
        "macd_hist_prev_n": float(macd_hist_prev_n),
        "hist_ema": float(_safe_num(hist_ema.iloc[-1], 0.0)),
        "hist_diff": float(hist_diff),
        "zero_cross_up": bool(zero_cross_up),
        "zero_cross_down": bool(zero_cross_down),
        "z_obv": float(z_obv),
        "mfi_now": float(mfi_now),
        "mfi_up": bool(mfi_up),
        "adx14": float(_safe_num(a14.iloc[-1], 0.0)),
        "adx9": float(_safe_num(a9.iloc[-1], 0.0)),
        "adx_rising": bool(adx_rising),
        "di_plus": float(di_plus),
        "di_minus": float(di_minus),
        "di_plus_gt": bool(di_plus_gt),
        "roc_atr_ratio": float(roc_atr_ratio),
        "bb_width_pct": float(bw_now),
        "bb_width": float(bw_now),
        "bb_width_pct_prev": float(bw_prev),
        "bb_width_prev_n": float(bb_width_prev_n),
        "bb_width_pct_rank": float(bb_width_pct_rank),
        "squeeze_flag": int(squeeze_flag),
        "close_prev_n": float(close_prev_n),
        "low_prev_n": float(low_prev_n),
        "high_prev_n": float(high_prev_n),
        "ema50": float(ema50),
        "whipsaw_flips": float(flips_hist),
    }


# ---------------------------------------------------------------------
# Base momentum scorer (rules engine only)
# ---------------------------------------------------------------------
def _momentum_score_base(dtf: pd.DataFrame, cfg: dict) -> Tuple[float, Dict[str, float], bool]:
    c = dtf["close"]
    h = dtf["high"]
    l = dtf["low"]
    v = dtf.get("volume", pd.Series(0, index=dtf.index))

    # Vol regime via ATR%
    ATR = atr(h, l, c, cfg["atr_win"])
    atr_pct = _safe_num((ATR.iloc[-1] / max(1e-9, c.iloc[-1])) * 100.0, 0.0)

    if atr_pct < cfg["low_vol_thr"]:
        rsi_thr = 60
    elif atr_pct < cfg["mid_vol_thr"]:
        rsi_thr = 65
    else:
        rsi_thr = 70

    rsi_fast = _rsi(c, cfg["rsi_fast"])
    rsi_std = _rsi(c, cfg["rsi_std"])

    # RMI adaptive
    rmi_lb = 9 if atr_pct > cfg["mid_vol_thr"] else (21 if atr_pct < cfg["low_vol_thr"] else cfg["rmi_lb"])
    rmi_m = 3 if atr_pct > cfg["mid_vol_thr"] else cfg["rmi_m"]
    rmi = _rmi(c, lb=rmi_lb, m=rmi_m)

    # RMI vs RSI slope divergence (±10)
    rmi_slope = _safe_num(rmi.diff().iloc[-1] if len(rmi) > 1 else 0.0, 0.0)
    rsi_slope = _safe_num(rsi_std.diff().iloc[-1] if len(rsi_std) > 1 else 0.0, 0.0)
    if rmi_slope > rsi_slope > 0:
        rmi_pts = 10
    elif rmi_slope < rsi_slope < 0:
        rmi_pts = -10
    else:
        rmi_pts = 0

    # RSI vol-aware score (0–15)
    rsi_val = _safe_num(rsi_std.iloc[-1] if len(rsi_std) else 50.0, 50.0)
    if rsi_val >= rsi_thr:
        rsi_pts = 15
    elif rsi_val >= (rsi_thr - 5):
        rsi_pts = 8
    else:
        rsi_pts = 0

    # MACD + volume
    macd_line = ema(c, 12) - ema(c, 26)
    macd_sig = macd_line.ewm(span=9, adjust=False).mean()
    hist = macd_line - macd_sig
    hist_ema = hist.ewm(span=5, adjust=False).mean()

    obv = obv_series(c, v)
    obv_d = obv.diff()
    mu = _safe_num(obv_d.ewm(span=5, adjust=False).mean().iloc[-1] if len(obv_d.dropna()) else 0.0, 0.0)
    sd = _safe_num(
        obv_d.rolling(20).std(ddof=1).iloc[-1] if len(obv_d.dropna()) else 0.0,
        0.0,
    )
    z = (_safe_num(obv_d.iloc[-1], 0.0) - mu) / (sd if sd > 0 else 1e9)

    tp = (h + l + c) / 3.0
    rmf = tp * v
    pos = (tp > tp.shift(1)).astype(float) * rmf
    neg = (tp < tp.shift(1)).astype(float) * rmf
    mfi_ratio = pos.rolling(14, min_periods=7).sum() / neg.replace(0, np.nan).rolling(14, min_periods=7).sum()
    mfi_up = (mfi_ratio.diff().iloc[-1] or 0) > 0 if len(mfi_ratio.dropna()) else False

    vol_pts = 15 if z >= 2.0 else (10 if z >= 1.0 else 0)
    if vol_pts > 0 and mfi_up:
        vol_pts = min(20, vol_pts + 5)

    zero_cross_now = (len(macd_line) > 1) and (
        (macd_line.iloc[-2] <= 0 <= macd_line.iloc[-1])
        or (macd_line.iloc[-2] >= 0 >= macd_line.iloc[-1])
    )
    hist_above_ema = _safe_num(hist.iloc[-1], 0.0) > _safe_num(hist_ema.iloc[-1], 0.0) if len(hist) else False

    if zero_cross_now and z >= 2.0:
        macd_pts = 30
    elif hist_above_ema:
        macd_pts = 15
    else:
        macd_pts = 10 if zero_cross_now else 0

    flips = _count_sign_flips(hist, look=12)
    whipsaw_flag = flips >= 4
    whipsaw_pen = -10 if whipsaw_flag else 0

    roc = c.pct_change(3)
    ratio = abs(_safe_num(roc.iloc[-1], 0.0)) / max(
        1e-9,
        _safe_num(ATR.iloc[-1], 0.0) / max(1e-9, _safe_num(c.iloc[-1], 1.0)),
    )
    if ratio >= 1.0:
        roc_pts = 15
    elif ratio >= 0.7:
        roc_pts = 8
    else:
        roc_pts = 0

    a14, dip, dim = adx(h, l, c, 14)
    a9, _, _ = adx(h, l, c, 9)
    adx_ok = (
        _safe_num(a9.iloc[-1], 0.0) > _safe_num(a14.iloc[-1], 0.0)
        and _safe_num(a14.diff().iloc[-1], 0.0) > 0
        and _safe_num(dip.iloc[-1], 0.0) > _safe_num(dim.iloc[-1], 0.0)
    )
    di_close = abs(_safe_num(dip.iloc[-1], 0.0) - _safe_num(dim.iloc[-1], 0.0)) < 2.0
    if adx_ok:
        adx_pts = 15
    elif _safe_num(a14.diff().iloc[-1], 0.0) > 0 and di_close:
        adx_pts = -5
    else:
        adx_pts = 0

    bw = bb_width_pct(c, n=20, k=2.0)
    if len(bw.dropna()) > 40:
        p20 = (
            float(np.nanpercentile(bw.tail(120).dropna(), 20))
            if len(bw.dropna()) >= 50
            else float(bw.dropna().quantile(0.2))
        )
        squeeze = _safe_num(bw.iloc[-1], 0.0) <= p20
    else:
        squeeze = False

    ctx_bonus = 10 if (squeeze and hist_above_ema and rsi_pts > 0) else 0

    total = rmi_pts + rsi_pts + macd_pts + roc_pts + adx_pts + vol_pts + ctx_bonus + whipsaw_pen
    score = float(clamp(total, 0, 100))

    parts = {
        "rmi_pts": float(rmi_pts),
        "rsi_pts": float(rsi_pts),
        "macd_pts": float(macd_pts),
        "roc_pts": float(roc_pts),
        "adx_pts": float(adx_pts),
        "vol_pts": float(vol_pts),
        "ctx_bonus": float(ctx_bonus),
        "whipsaw_flips": float(flips),
        "atr_pct": float(atr_pct),
    }
    return score, parts, whipsaw_flag


# ---------------------------------------------------------------------
# Public API – rules + ML + fused
# ---------------------------------------------------------------------
def score_momentum(
    symbol: str,
    kind: str,          # 'futures' or 'spot'
    tf: str,            # '15m','30m','60m','120m'
    df5: pd.DataFrame,  # base frame (15m in your case, 5m if you add later)
    base: BaseCfg,
    ini_path: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None,
    mode: str = "both",   # 'rules', 'ml', 'both'
):
    """
    Momentum pillar:
      - Computes rules score from base → tf.
      - Optionally fuses with ML (if ml_enabled + ml_pillars rows exist).
      - DB writes are mode-aware (rules vs ml vs both).
    """
    mode = (mode or "both").lower()
    if mode not in ("rules", "ml", "both"):
        mode = "both"

    # --- Resample & guardrails ---
    dftf = df5
    if tf != "15m":
        dftf = resample(df5, tf)

    dftf = maybe_trim_last_bar(dftf)
    cfg = _cfg(ini_path)

    min_bars_cfg = cfg.get("min_bars", 120)
    if not ensure_min_bars(dftf, tf) or len(dftf) < min_bars_cfg:
        print(
            f"[MOM] {symbol} {kind} {tf}: skipping scoring – "
            f"len={len(dftf)}, min_bars_cfg={min_bars_cfg}"
        )
        return None

    # --- Base rules engine ---
    base_score, parts, whipsaw_flag = _momentum_score_base(dftf, cfg)

    # --- Scenario layer (rules-only) ---
    try:
        rules_mode, scenarios = _load_mom_scenarios(cfg)
    except Exception:
        rules_mode, scenarios = (cfg.get("rules_mode", "additive"), [])

    scen_total = 0.0
    F = None
    if scenarios:
        F = _mom_features(dftf, cfg)
        for sc in scenarios:
            try:
                when_expr = sc.get("when", "")
                if when_expr and _safe_eval(when_expr, F):
                    scen_total += sc["score"]
                    parts[f"SCN.{sc['name']}"] = float(sc["score"])
                    bonus_expr = sc.get("bonus_when", "")
                    if bonus_expr and _safe_eval(bonus_expr, F):
                        scen_total += sc["bonus"]
                        parts[f"SCN.{sc['name']}.bonus"] = float(sc["bonus"])
            except Exception:
                continue

    if rules_mode == "override":
        rules_score = scen_total
    else:
        rules_score = base_score + scen_total

    rules_score = float(
        clamp(rules_score, cfg.get("clamp_low", 0.0), cfg.get("clamp_high", 100.0))
    )
    ts = dftf.index[-1].to_pydatetime().replace(tzinfo=TZ)

    # --- Rules veto ---
    rules_veto = bool(whipsaw_flag)

    # ----------------------------
    # ML: read from ml_pillars + calibration
    # ----------------------------
    ml_enabled = cfg.get("ml_enabled", False)
    ml_prob_raw: Optional[float] = None
    ml_prob_cal: Optional[float] = None
    ml_score: Optional[float] = None
    fused_prob: Optional[float] = None
    fused_score: Optional[float] = None
    fused_veto = rules_veto  # start from rules veto, ML can only tighten

    # ML min-bars (stricter than rules)
    ml_min_bars_map = {"15m": 60, "30m": 40, "60m": 20, "120m": 10, "240m": 5}
    ml_min_bars = ml_min_bars_map.get(tf, 60)

    if ml_enabled and mode in ("ml", "both") and len(dftf) >= ml_min_bars:
        target_name = cfg.get("ml_target_name")
        version = cfg.get("ml_version", "xgb_v1")
        calib_table = cfg.get("ml_calibration_table", "indicators.momentum_calibration_4h")
        w = float(cfg.get("ml_blend_weight", 0.35))
        w = max(0.0, min(1.0, w))

        market_type = kind  # 'futures' / 'spot'

        try:
            with get_db_connection() as conn, conn.cursor() as cur:
                # 1) Read raw prob from indicators.ml_pillars
                cur.execute(
                    """
                    SELECT prob_up, prob_down
                      FROM indicators.ml_pillars
                     WHERE pillar      = %s
                       AND symbol      = %s
                       AND market_type = %s
                       AND tf          = %s
                       AND target_name = %s
                       AND version     = %s
                       AND ts          = %s
                     ORDER BY ts DESC
                     LIMIT 1
                    """,
                    ("momentum", symbol, market_type, tf, target_name, version, ts),
                )
                row = cur.fetchone()

                if row:
                    prob_up_db = _safe_num(row[0], 0.5)
                    prob_dn_db = _safe_num(row[1], 0.5)
                    ml_prob_raw = max(0.0, min(1.0, prob_up_db))

                    # 2) Calibrate: map prob_up → p_up_cal using bucket table
                    cur.execute(
                        f"""
                        SELECT p_min, p_max, realized_up_rate
                          FROM {calib_table}
                         WHERE pillar      = %s
                           AND target_name = %s
                           AND version     = %s
                           AND tf          = %s
                           AND %s >= p_min
                           AND %s <  p_max
                         ORDER BY p_max
                         LIMIT 1
                        """,
                        ("momentum", target_name, version, tf, ml_prob_raw, ml_prob_raw),
                    )
                    crow = cur.fetchone()
                    if crow:
                        ml_prob_cal = float(crow[2])
                    else:
                        ml_prob_cal = ml_prob_raw  # fallback

                    ml_prob_cal = max(0.0, min(1.0, ml_prob_cal))
                    ml_score = round(100.0 * ml_prob_cal, 2)

                    # 3) Blend with rules → fused prob
                    base_prob = float(rules_score) / 100.0
                    fused_prob = (1.0 - w) * base_prob + w * ml_prob_cal
                    fused_prob = max(0.0, min(1.0, fused_prob))
                    fused_score = round(100.0 * fused_prob, 2)

                    # 4) ML veto threshold (if configured)
                    thr_txt = cfg.get("ml_veto_if_prob_lt", "").strip()
                    if thr_txt:
                        try:
                            thr = float(thr_txt)
                            if 0.0 <= thr <= 1.0 and fused_prob < thr:
                                fused_veto = True
                        except Exception:
                            pass

        except Exception as e:
            print(f"[MOM] {symbol} {kind} {tf}: ML block failed: {e}")
            ml_prob_raw = None
            ml_prob_cal = None
            fused_score = None
            fused_veto = rules_veto

    # If ML disabled or missing, fused = rules
    if fused_score is None:
        fused_score = float(rules_score)
        fused_prob = float(rules_score) / 100.0
        fused_veto = rules_veto

    # --- Build debug context ---
    debug_ctx = {
        "rules_score": float(rules_score),
        "rules_veto": bool(rules_veto),
        "ml_prob_raw": ml_prob_raw,
        "ml_prob_cal": ml_prob_cal,
        "fused_prob": fused_prob,
        "fused_score": fused_score,
    }

    # --- DB writes, mode-dependent ---
    rows = []

    # rules-only metrics
    if mode in ("rules", "both"):
        rows.extend(
            [
                (symbol, kind, tf, ts, "MOM.score", float(rules_score), "{}", base.run_id, base.source),
                (
                    symbol,
                    kind,
                    tf,
                    ts,
                    "MOM.veto_flag",
                    1.0 if rules_veto else 0.0,
                    "{}",
                    base.run_id,
                    base.source,
                ),
                (
                    symbol,
                    kind,
                    tf,
                    ts,
                    "MOM.debug_ctx",
                    0.0,
                    json.dumps(debug_ctx),
                    base.run_id,
                    base.source,
                ),
                (
                    symbol,
                    kind,
                    tf,
                    ts,
                    "MOM.whipsaw_flag",
                    1.0 if whipsaw_flag else 0.0,
                    "{}",
                    base.run_id,
                    base.source,
                ),
            ]
        )

        # scenario + base parts (for debugging)
        for k, v in parts.items():
            rows.append((symbol, kind, tf, ts, f"MOM.{k}", float(v), "{}", base.run_id, base.source))

    # ML + fused metrics
    if mode in ("ml", "both") and ml_enabled:
        ml_ctx = {
            "pillar": "momentum",
            "target_name": cfg.get("ml_target_name"),
            "version": cfg.get("ml_version", "xgb_v1"),
            "ml_prob_raw": ml_prob_raw,
            "ml_prob_cal": ml_prob_cal,
            "blend_weight": cfg.get("ml_blend_weight", 0.35),
        }

        p_up = ml_prob_cal if ml_prob_cal is not None else (ml_prob_raw if ml_prob_raw is not None else fused_prob)
        p_up = max(0.0, min(1.0, p_up if p_up is not None else 0.5))
        p_down = 1.0 - p_up

        rows.extend(
            [
                (
                    symbol,
                    kind,
                    tf,
                    ts,
                    "MOM.ml_score",
                    ml_score if ml_score is not None else 0.0,
                    "{}",
                    base.run_id,
                    base.source,
                ),
                (symbol, kind, tf, ts, "MOM.ml_p_up_cal", p_up, "{}", base.run_id, base.source),
                (symbol, kind, tf, ts, "MOM.ml_p_down_cal", p_down, "{}", base.run_id, base.source),
                (symbol, kind, tf, ts, "MOM.fused_score", float(fused_score), "{}", base.run_id, base.source),
                (
                    symbol,
                    kind,
                    tf,
                    ts,
                    "MOM.fused_veto",
                    1.0 if fused_veto else 0.0,
                    "{}",
                    base.run_id,
                    base.source,
                ),
                (
                    symbol,
                    kind,
                    tf,
                    ts,
                    "MOM.ml_ctx",
                    0.0,
                    json.dumps(ml_ctx),
                    base.run_id,
                    base.source,
                ),
            ]
        )

    if rows:
        write_values(rows)

    print(
        f"[MOM] {symbol} {kind} {tf} @ {ts}: "
        f"rules={rules_score:.2f}, fused={fused_score:.2f}, veto={fused_veto}"
    )

    return ts, float(fused_score), bool(fused_veto)
