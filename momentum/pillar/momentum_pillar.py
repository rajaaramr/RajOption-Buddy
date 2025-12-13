# pillars/momentum_pillar.py
from __future__ import annotations

import ast
import configparser
import functools
import json
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
    write_values,
    clamp,
    TZ,
    BaseCfg,
    ensure_min_bars,
    maybe_trim_last_bar,
)

# Prefer NSE-aligned resample if present
try:
    from pillars.common import resample_nse as _resample_tf  # expects offset='9h15min' inside
except Exception:
    from pillars.common import resample as _resample_tf

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_INI = str(BASE_DIR / "momentum_scenarios.ini")


# ---------------------------------------------------------------------
# OHLCV normalization
# ---------------------------------------------------------------------
def _ensure_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or len(df) == 0:
        raise ValueError("Empty frame passed to momentum pillar")

    if isinstance(df, pd.Series):
        c = df.astype(float)
        out = pd.DataFrame(index=c.index)
        out["open"] = c
        out["high"] = c
        out["low"] = c
        out["close"] = c
        out["volume"] = 0.0
        return out

    out = df.copy()

    # Flatten MultiIndex columns: [('open','first'), ...] -> ['open', ...]
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = [c[0] if isinstance(c, tuple) else c for c in out.columns]

    cols = set(out.columns)

    if "close" not in cols:
        raise KeyError("Momentum pillar expected a 'close' column but did not find one")

    o = out["open"] if "open" in cols else out["close"]
    h = out["high"] if "high" in cols else out["close"]
    l = out["low"] if "low" in cols else out["close"]
    c = out["close"]
    v = out["volume"] if "volume" in cols else pd.Series(0.0, index=out.index)

    canon = pd.DataFrame(
        {
            "open": o.astype(float),
            "high": h.astype(float),
            "low": l.astype(float),
            "close": c.astype(float),
            "volume": v.astype(float),
        },
        index=out.index,
    ).sort_index()

    return canon


def _agg_from_base_candles(df15: pd.DataFrame, tf: str) -> pd.DataFrame:
    df15 = _ensure_ohlcv(df15)
    freq = tf.replace("m", "T")

    out = (
        df15[["open", "high", "low", "close", "volume"]]
        .resample(freq, label="right", closed="right")
        .agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }
        )
        .dropna(subset=["close"])
        .astype(float)
        .sort_index()
    )
    return out


# ---------------------------------------------------------------------
# Safe evaluator
# ---------------------------------------------------------------------
@functools.lru_cache(maxsize=2048)
def _compile_rule(expr: str):
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
                return a**b
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
# Small helpers
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
        "min_bars": cp.getint("momentum", "min_bars", fallback=120),
        "vol_avg_win": cp.getint("momentum", "vol_avg_win", fallback=20),
        "bb_win": cp.getint("momentum", "bb_win", fallback=20),
        "bb_k": cp.getfloat("momentum", "bb_k", fallback=2.0),
        "div_lookback": cp.getint("momentum", "div_lookback", fallback=5),
        # ML config
        "ml_enabled": ml_enabled,
        "ml_blend_weight": ml_blend_weight,
        "ml_target_name": ml_target_name,
        "ml_version": ml_version,
        "ml_calibration_table": ml_calibration_table,
        "ml_veto_if_prob_lt": ml_veto_if_prob_lt,
        "cp": cp,
        "_ini_path": path,
    }


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


# ============================================================
# Context-aware feature builder
# ============================================================
def _ctx_get(context: Optional[Dict[str, Any]], *keys: str, default=None):
    if not context:
        return default

    # allow context["frames"] dict or direct keys
    frames = context.get("frames") if isinstance(context, dict) else None

    for k in keys:
        if k in context:
            return context.get(k)
        if isinstance(frames, dict) and k in frames:
            return frames.get(k)

    return default


def _ctx_series(context: Optional[Dict[str, Any]], key: str) -> Optional[pd.Series]:
    if not context:
        return None
    sdict = context.get("series")
    if isinstance(sdict, dict) and key in sdict:
        s = sdict.get(key)
        if isinstance(s, pd.Series) and len(s) > 0:
            return s
    return None


def _mom_features(dtf: pd.DataFrame, cfg: dict, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Build features with a priority order:
      1) Use context (precomputed indicators from indicators_worker frames)
      2) Fallback to compute from OHLCV
    """
    dtf = _ensure_ohlcv(dtf)
    c, h, l, o, v = dtf["close"], dtf["high"], dtf["low"], dtf["open"], dtf["volume"]
    px_now = _safe_num(c.iloc[-1], 1.0)

    # ATR
    atr_val = _ctx_get(context, "atr_14", "ATR.14", "atr", default=None)
    if atr_val is None:
        ATR = atr(h, l, c, cfg["atr_win"])
        atr_val = _safe_num(ATR.iloc[-1], 0.0)
    atr_val = _safe_num(atr_val, 0.0)
    atr_pct_now = float((atr_val / max(1e-9, px_now)) * 100.0)

    # RSI std + fast
    rsi_std_val = _ctx_get(context, "rsi_14", "rsi_std", "RSI.14", default=None)
    if rsi_std_val is None:
        rsi_std_s = _rsi(c, cfg["rsi_std"])
        rsi_std_val = _safe_num(rsi_std_s.iloc[-1], 50.0)
    rsi_std_val = _safe_num(rsi_std_val, 50.0)

    rsi_fast_val = _ctx_get(context, "rsi_5", "rsi_fast", "RSI.5", default=None)
    if rsi_fast_val is None:
        rsi_fast_s = _rsi(c, cfg["rsi_fast"])
        rsi_fast_val = _safe_num(rsi_fast_s.iloc[-1], 50.0)
    rsi_fast_val = _safe_num(rsi_fast_val, 50.0)

    # RMI (usually not in frames; still allow context)
    rmi_val = _ctx_get(context, "rmi", "RMI", default=None)
    if rmi_val is None:
        rmi_s = _rmi(c, lb=cfg["rmi_lb"], m=cfg["rmi_m"])
        rmi_val = _safe_num(rmi_s.iloc[-1], 50.0)
    rmi_val = _safe_num(rmi_val, 50.0)

    # MACD line/signal/hist
    macd_line_val = _ctx_get(context, "macd_line", "macd", "MACD.line", default=None)
    macd_sig_val = _ctx_get(context, "macd_sig", "macd_signal", "MACD.signal", default=None)
    hist_val = _ctx_get(context, "macd_hist", "hist", "MACD.hist", default=None)

    if macd_line_val is None or macd_sig_val is None or hist_val is None:
        macd_line = ema(c, 12) - ema(c, 26)
        macd_sig = macd_line.ewm(span=9, adjust=False).mean()
        hist = macd_line - macd_sig
        macd_line_val = _safe_num(macd_line.iloc[-1], 0.0)
        macd_sig_val = _safe_num(macd_sig.iloc[-1], 0.0)
        hist_val = _safe_num(hist.iloc[-1], 0.0)
        hist_series = hist
    else:
        macd_line_val = _safe_num(macd_line_val, 0.0)
        macd_sig_val = _safe_num(macd_sig_val, 0.0)
        hist_val = _safe_num(hist_val, 0.0)
        hist_series = _ctx_series(context, "macd_hist")  # optional

    # ADX/DI
    adx14_val = _ctx_get(context, "adx_14", "adx14", "ADX.14", default=None)
    di_plus_val = _ctx_get(context, "di_plus", "plus_di", "DI+.14", default=None)
    di_minus_val = _ctx_get(context, "di_minus", "minus_di", "DI-.14", default=None)

    if adx14_val is None or di_plus_val is None or di_minus_val is None:
        a14, dip, dim = adx(h, l, c, 14)
        adx14_val = _safe_num(a14.iloc[-1], 0.0)
        di_plus_val = _safe_num(dip.iloc[-1], 0.0)
        di_minus_val = _safe_num(dim.iloc[-1], 0.0)
    else:
        adx14_val = _safe_num(adx14_val, 0.0)
        di_plus_val = _safe_num(di_plus_val, 0.0)
        di_minus_val = _safe_num(di_minus_val, 0.0)

    # BB width %
    bw_val = _ctx_get(context, "bb_width_pct", "BB.width_pct", default=None)
    if bw_val is None:
        bw = bb_width_pct(c, n=cfg.get("bb_win", 20), k=cfg.get("bb_k", 2.0))
        bw_val = _safe_num(bw.iloc[-1], 0.0)
        bw_series = bw
    else:
        bw_val = _safe_num(bw_val, 0.0)
        bw_series = _ctx_series(context, "bb_width_pct")

    # OBV z-score (usually not in frames; still compute from OHLCV)
    obv = obv_series(c, v)
    obv_d = obv.diff()
    mu = _safe_num(obv_d.ewm(span=5, adjust=False).mean().iloc[-1] if len(obv_d.dropna()) else 0.0, 0.0)
    sd = _safe_num(obv_d.rolling(20).std(ddof=1).iloc[-1] if len(obv_d.dropna()) else 0.0, 0.0)
    z_obv = (_safe_num(obv_d.iloc[-1], 0.0) - mu) / (sd if sd > 0 else 1e9)

    # RVOL
    vol_avg_win = int(cfg.get("vol_avg_win", 20))
    v_avg = v.rolling(vol_avg_win).mean()
    rvol_now = _safe_num(v.iloc[-1], 0.0) / max(1.0, _safe_num(v_avg.iloc[-1] if len(v_avg.dropna()) else 1.0, 1.0))

    # Whipsaw flips: need hist series; if not available, reconstruct minimal from OHLCV
    if hist_series is None:
        macd_line = ema(c, 12) - ema(c, 26)
        macd_sig = macd_line.ewm(span=9, adjust=False).mean()
        hist_series = macd_line - macd_sig
    flips_hist = _count_sign_flips(hist_series, look=12)

    # “prev_n” values (needed by some scenarios)
    n = max(1, int(cfg.get("div_lookback", 5)))
    close_prev_n = _safe_num(c.iloc[-n] if len(c) > n else c.iloc[-1], px_now)

    # If context provides series, use it; else fallback to compute from df series
    rsi_std_series = _ctx_series(context, "rsi_14")
    if rsi_std_series is None:
        rsi_std_series = _rsi(c, cfg["rsi_std"])
    rsi_prev_std_n = _safe_num(rsi_std_series.iloc[-n] if len(rsi_std_series) > n else rsi_std_series.iloc[-1], 50.0)

    rmi_series = _ctx_series(context, "rmi")
    if rmi_series is None:
        rmi_series = _rmi(c, lb=cfg["rmi_lb"], m=cfg["rmi_m"])
    rmi_prev_n = _safe_num(rmi_series.iloc[-n] if len(rmi_series) > n else rmi_series.iloc[-1], 50.0)

    bw_prev_n = 0.0
    if bw_series is not None and len(bw_series) > 0:
        bw_prev_n = _safe_num(bw_series.iloc[-n] if len(bw_series) > n else bw_series.iloc[-1], bw_val)
    else:
        bw_prev_n = bw_val

    feats = {
        "open": _safe_num(o.iloc[-1], px_now),
        "high": _safe_num(h.iloc[-1], px_now),
        "low": _safe_num(l.iloc[-1], px_now),
        "close": float(px_now),
        "volume": _safe_num(v.iloc[-1], 0.0),

        "atr_pct": float(atr_pct_now),

        "rsi_fast": float(rsi_fast_val),
        "rsi_std": float(rsi_std_val),

        "rmi": float(rmi_val),
        "rmi_now": float(rmi_val),
        "rmi_prev_n": float(rmi_prev_n),

        "macd_line": float(macd_line_val),
        "macd_sig": float(macd_sig_val),
        "hist": float(hist_val),

        "z_obv": float(z_obv),
        "rvol_now": float(rvol_now),

        "adx14": float(adx14_val),
        "di_plus": float(di_plus_val),
        "di_minus": float(di_minus_val),
        "di_plus_gt": bool(di_plus_val > di_minus_val),

        "bb_width_pct": float(bw_val),

        "close_prev_n": float(close_prev_n),
        "rsi_prev_std_n": float(rsi_prev_std_n),
        "bb_width_prev_n": float(bw_prev_n),

        "whipsaw_flips": float(flips_hist),
    }

    return feats


# ---------------------------------------------------------------------
# Base rules scoring using features dict (so context can help!)
# ---------------------------------------------------------------------
def _momentum_score_base_from_feats(F: Dict[str, Any], cfg: dict) -> Tuple[float, Dict[str, float], bool]:
    atr_pct = float(F.get("atr_pct", 0.0))

    if atr_pct < cfg["low_vol_thr"]:
        rsi_thr = 60
    elif atr_pct < cfg["mid_vol_thr"]:
        rsi_thr = 65
    else:
        rsi_thr = 70

    rsi_val = float(F.get("rsi_std", 50.0))
    if rsi_val >= rsi_thr:
        rsi_pts = 15
    elif rsi_val >= (rsi_thr - 5):
        rsi_pts = 8
    else:
        rsi_pts = 0

    hist_val = float(F.get("hist", 0.0))
    flips = int(F.get("whipsaw_flips", 0))
    whipsaw_flag = flips >= 4
    whipsaw_pen = -10 if whipsaw_flag else 0

    z = float(F.get("z_obv", 0.0))
    vol_pts = 15 if z >= 2.0 else (10 if z >= 1.0 else 0)

    # MACD points (basic, keep your earlier logic if you want)
    macd_pts = 15 if hist_val > 0 else 0

    # ADX points
    adx14 = float(F.get("adx14", 0.0))
    di_plus = float(F.get("di_plus", 0.0))
    di_minus = float(F.get("di_minus", 0.0))
    if adx14 > 18 and di_plus > di_minus:
        adx_pts = 15
    else:
        adx_pts = 0

    total = rsi_pts + macd_pts + adx_pts + vol_pts + whipsaw_pen
    score = float(clamp(total, 0, 100))

    parts = {
        "rsi_pts": float(rsi_pts),
        "macd_pts": float(macd_pts),
        "adx_pts": float(adx_pts),
        "vol_pts": float(vol_pts),
        "whipsaw_flips": float(flips),
        "atr_pct": float(atr_pct),
    }
    return score, parts, whipsaw_flag


# ---------------------------------------------------------------------
# Public API – batch-friendly
# ---------------------------------------------------------------------
def score_momentum(
    symbol: str,
    kind: str,           # 'futures' or 'spot'
    tf: str,             # '15m','30m','60m','120m','240m'
    df5: pd.DataFrame,   # base (15m) frame or full frame
    base: BaseCfg,
    ini_path: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None,
    mode: str = "both",            # 'rules','ml','both'
    write_to_db: bool = True,      # batch support
    ignore_min_bars: bool = False  # backfill override
):
    mode = (mode or "both").lower()
    if mode not in ("rules", "ml", "both"):
        mode = "both"

    # --- Resample NSE-aligned ---
    dftf = df5
    if tf != "15m":
        dftf = _resample_tf(df5, tf)
        try:
            dftf = _ensure_ohlcv(dftf)
        except KeyError:
            dftf = _agg_from_base_candles(df5, tf)
    else:
        dftf = _ensure_ohlcv(dftf)

    dftf = maybe_trim_last_bar(dftf)
    cfg = _cfg(ini_path)

    if not ignore_min_bars:
        min_bars_cfg = cfg.get("min_bars", 120)
        if (not ensure_min_bars(dftf, tf)) or (len(dftf) < min_bars_cfg):
            return None

    # --- Features (context-first) ---
    F = _mom_features(dftf, cfg, context=context)

    # --- Base rules from features ---
    base_score, parts, whipsaw_flag = _momentum_score_base_from_feats(F, cfg)

    # --- Scenario layer (uses same features dict) ---
    try:
        rules_mode, scenarios = _load_mom_scenarios(cfg)
    except Exception:
        rules_mode, scenarios = (cfg.get("rules_mode", "additive"), [])

    scen_total = 0.0
    fired: List[str] = []

    for sc in scenarios:
        try:
            when_expr = sc.get("when", "")
            if when_expr and _safe_eval(when_expr, F):
                scen_total += float(sc["score"])
                parts[f"SCN.{sc['name']}"] = float(sc["score"])
                fired.append(sc["name"])

                bonus_expr = sc.get("bonus_when", "")
                if bonus_expr and _safe_eval(bonus_expr, F):
                    scen_total += float(sc["bonus"])
                    parts[f"SCN.{sc['name']}.bonus"] = float(sc["bonus"])
        except Exception:
            continue

    rules_score = scen_total if rules_mode == "override" else (base_score + scen_total)
    rules_score = float(clamp(rules_score, cfg.get("clamp_low", 0.0), cfg.get("clamp_high", 100.0)))

    ts = dftf.index[-1].to_pydatetime().replace(tzinfo=TZ)
    rules_veto = bool(whipsaw_flag)

    # ----------------------------
    # ML: same as your earlier block
    # ----------------------------
    ml_enabled = cfg.get("ml_enabled", False)
    ml_prob_raw: Optional[float] = None
    ml_prob_cal: Optional[float] = None
    ml_score: Optional[float] = None
    fused_prob: Optional[float] = None
    fused_score: Optional[float] = None
    fused_veto = rules_veto

    ml_min_bars_map = {"15m": 60, "30m": 40, "60m": 20, "120m": 10, "240m": 5}
    ml_min_bars = ml_min_bars_map.get(tf, 60)

    if ml_enabled and mode in ("ml", "both") and len(dftf) >= ml_min_bars:
        target_name = cfg.get("ml_target_name")
        version = cfg.get("ml_version", "xgb_v1")
        calib_table = cfg.get("ml_calibration_table", "indicators.momentum_calibration_4h")
        w = float(cfg.get("ml_blend_weight", 0.35))
        w = max(0.0, min(1.0, w))
        market_type = kind

        try:
            with get_db_connection() as conn, conn.cursor() as cur:
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
                    ml_prob_raw = max(0.0, min(1.0, prob_up_db))

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
                    ml_prob_cal = float(crow[2]) if crow else ml_prob_raw
                    ml_prob_cal = max(0.0, min(1.0, ml_prob_cal))
                    ml_score = round(100.0 * ml_prob_cal, 2)

                    base_prob = float(rules_score) / 100.0
                    fused_prob = (1.0 - w) * base_prob + w * ml_prob_cal
                    fused_prob = max(0.0, min(1.0, fused_prob))
                    fused_score = round(100.0 * fused_prob, 2)

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

    if fused_score is None:
        fused_score = float(rules_score)
        fused_prob = float(rules_score) / 100.0
        fused_veto = rules_veto

    debug_ctx = {
        "rules_score": float(rules_score),
        "rules_veto": bool(rules_veto),
        "ml_prob_raw": ml_prob_raw,
        "ml_prob_cal": ml_prob_cal,
        "fused_prob": fused_prob,
        "fused_score": fused_score,
        "scenarios_fired": fired,
        "used_context": bool(context),
    }

    rows: List[tuple] = []

    if mode in ("rules", "both"):
        rows.extend(
            [
                (symbol, kind, tf, ts, "MOM.score", float(rules_score), "{}", base.run_id, base.source),
                (symbol, kind, tf, ts, "MOM.veto_flag", 1.0 if rules_veto else 0.0, "{}", base.run_id, base.source),
                (symbol, kind, tf, ts, "MOM.debug_ctx", 0.0, json.dumps(debug_ctx), base.run_id, base.source),
            ]
        )
        for k, v in parts.items():
            rows.append((symbol, kind, tf, ts, f"MOM.{k}", float(v), "{}", base.run_id, base.source))

    if mode in ("ml", "both") and ml_enabled:
        p_up = ml_prob_cal if ml_prob_cal is not None else (ml_prob_raw if ml_prob_raw is not None else fused_prob)
        p_up = max(0.0, min(1.0, float(p_up if p_up is not None else 0.5)))
        p_down = 1.0 - p_up

        rows.extend(
            [
                (symbol, kind, tf, ts, "MOM.ml_score", float(ml_score or 0.0), "{}", base.run_id, base.source),
                (symbol, kind, tf, ts, "MOM.ml_p_up_cal", p_up, "{}", base.run_id, base.source),
                (symbol, kind, tf, ts, "MOM.ml_p_down_cal", p_down, "{}", base.run_id, base.source),
                (symbol, kind, tf, ts, "MOM.fused_score", float(fused_score), "{}", base.run_id, base.source),
                (symbol, kind, tf, ts, "MOM.fused_veto", 1.0 if fused_veto else 0.0, "{}", base.run_id, base.source),
            ]
        )

    if write_to_db and rows:
        write_values(rows)

    return ts, float(fused_score), bool(fused_veto), rows
