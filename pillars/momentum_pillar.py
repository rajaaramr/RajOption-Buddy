# pillars/momentum_pillar.py
from __future__ import annotations
import json, math, configparser
from typing import Optional, Tuple, Dict, List, Any
import numpy as np
import pandas as pd

# common helpers you already have:
# ema, atr, adx, obv_series, bb_width_pct, resample, write_values, clamp, TZ, DEFAULT_INI, BaseCfg
from .common import *
from pillars.common import min_bars_for_tf, ensure_min_bars, maybe_trim_last_bar


# -----------------------------
# Local helpers
# -----------------------------
def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    if len(close) < 3:
        return pd.Series([50.0] * len(close), index=close.index, dtype=float)
    d = close.diff()
    g = d.clip(lower=0)
    l = (-d).clip(lower=0)
    # use EMA for smoother RSI; guard zeros
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


# -----------------------------
# Config (base + scenario + ML)
# -----------------------------
def _cfg(path: str = DEFAULT_INI) -> dict:
    cp = configparser.ConfigParser(inline_comment_prefixes=(";", "#"), interpolation=None, strict=False)
    cp.read(path)
    return {
        # Core params
        "rsi_fast":  cp.getint("momentum", "rsi_fast",  fallback=5),
        "rsi_std":   cp.getint("momentum", "rsi_std",   fallback=14),
        "rmi_lb":    cp.getint("momentum", "rmi_lb",    fallback=14),
        "rmi_m":     cp.getint("momentum", "rmi_m",     fallback=5),
        "atr_win":   cp.getint("momentum", "atr_win",   fallback=14),

        # Regime thresholds
        "low_vol_thr":  cp.getfloat("momentum", "low_vol_thr",  fallback=3.0),
        "mid_vol_thr":  cp.getfloat("momentum", "mid_vol_thr",  fallback=6.0),

        # Scenario engine controls
        "rules_mode": cp.get("momentum", "rules_mode", fallback="additive").lower(),  # additive | override
        "clamp_low": cp.getfloat("momentum", "clamp_low", fallback=0.0),
        "clamp_high": cp.getfloat("momentum", "clamp_high", fallback=100.0),
        "write_scenarios_debug": cp.getboolean("momentum", "write_scenarios_debug", fallback=False),
        "min_bars": cp.getint("momentum", "min_bars", fallback=120),

        # Extra feature windows
        "vol_avg_win": cp.getint("momentum", "vol_avg_win", fallback=20),
        "bb_win": cp.getint("momentum", "bb_win", fallback=20),
        "bb_k": cp.getfloat("momentum", "bb_k", fallback=2.0),
        "div_lookback": cp.getint("momentum", "div_lookback", fallback=5),

        # --- ML controls (flag-driven; no-op if disabled) ---
        "ml_enabled": cp.getboolean("momentum_ml", "enabled", fallback=False),
        "ml_blend_weight": cp.getfloat("momentum_ml", "blend_weight", fallback=0.35),
        "ml_source_table": cp.get("momentum_ml", "source_table", fallback="").strip(),
        "ml_callback": cp.get("momentum_ml", "callback", fallback="").strip(),
        "ml_veto_if_prob_lt": cp.get("momentum_ml", "veto_if_prob_lt", fallback="").strip(),

        "_ini_path": path,
        "cp": cp,
    }


# -----------------------------
# Scenario loader (namespaced)
# -----------------------------
def _load_mom_scenarios(path: str) -> Tuple[str, List[dict]]:
    """
    Looks for:
      [mom_scenarios]
      list = name1, name2, ...
      [mom_scenario.name1]
      when = <expr>
      score = +10
      bonus_when = <expr>  ; optional
      bonus = +5
    """
    cp = configparser.ConfigParser(inline_comment_prefixes=(";", "#"), interpolation=None, strict=False)
    cp.read(path)

    rules_mode = cp.get("momentum", "rules_mode", fallback="additive").lower()

    names: List[str] = []
    if cp.has_section("mom_scenarios"):
        raw = cp.get("mom_scenarios", "list", fallback="")
        names = [n.strip() for n in raw.split(",") if n.strip()]

    scenarios: List[dict] = []
    for n in names:
        sec = f"mom_scenario.{n}"
        if not cp.has_section(sec):
            continue
        scenarios.append({
            "name": n,
            "when": cp.get(sec, "when", fallback=""),
            "score": cp.getfloat(sec, "score", fallback=0.0),
            "bonus_when": cp.get(sec, "bonus_when", fallback=""),
            "bonus": cp.getfloat(sec, "bonus", fallback=0.0),
        })
    return rules_mode, scenarios


# -----------------------------
# Feature builder (covers INI refs)
# -----------------------------
def _mom_features(dtf: pd.DataFrame, cfg: dict) -> Dict[str, float | int | bool]:
    c = dtf["close"]; h = dtf["high"]; l = dtf["low"]
    o = dtf["open"];  v = dtf.get("volume", pd.Series(index=dtf.index, dtype=float)).fillna(0)

    # ATR & ATR%
    ATR = atr(h, l, c, cfg["atr_win"])
    atr_val = _safe_num(ATR.iloc[-1], 0.0)
    px_now  = _safe_num(c.iloc[-1], 1.0)
    atr_pct_now = float((atr_val / max(1e-9, px_now)) * 100.0)
    atr_avg_20 = _safe_num(ATR.rolling(20).mean().iloc[-1] if len(ATR) >= 20 else atr_val, atr_val)

    # RSI fast/std
    rsi_fast = _rsi(c, cfg["rsi_fast"])
    rsi_std  = _rsi(c, cfg["rsi_std"])

    # RMI
    rmi     = _rmi(c, lb=cfg["rmi_lb"], m=cfg["rmi_m"])

    # MACD / histogram
    macd_line = ema(c, 12) - ema(c, 26)
    macd_sig  = macd_line.ewm(span=9, adjust=False).mean()
    hist      = macd_line - macd_sig
    hist_ema  = hist.ewm(span=5, adjust=False).mean()
    hist_diff = _safe_num(hist.diff().iloc[-1] if len(hist) > 1 else 0.0, 0.0)
    zero_cross_up = (len(macd_line) > 1) and (macd_line.iloc[-2] <= 0 <= macd_line.iloc[-1])
    zero_cross_down = (len(macd_line) > 1) and (macd_line.iloc[-2] >= 0 >= macd_line.iloc[-1])

    # OBV z-score
    obv  = obv_series(c, v)
    obv_d = obv.diff()
    mu = _safe_num(obv_d.ewm(span=5, adjust=False).mean().iloc[-1] if len(obv_d.dropna()) else 0.0, 0.0)
    sd = _safe_num(obv_d.rolling(20).std(ddof=1).iloc[-1] if len(obv_d.dropna()) else 0.0, 0.0)
    z_obv = ( _safe_num(obv_d.iloc[-1], 0.0) - mu ) / (sd if sd > 0 else 1e9)

    # Relative volume
    vol_avg_win = int(cfg.get("vol_avg_win", 20))
    v_avg = v.rolling(vol_avg_win).mean()
    rvol_now = _safe_num(v.iloc[-1], 0.0) / max(1.0, _safe_num(v_avg.iloc[-1] if len(v_avg.dropna()) else 1.0, 1.0))

    # Simple MFI-ish up/down
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

    # ROC vs ATR
    roc3 = c.pct_change(3)
    roc_atr_ratio = abs(_safe_num(roc3.iloc[-1], 0.0)) / max(1e-9, atr_val / max(1e-9, px_now))

    # ADX/DI
    a14, dip, dim = adx(h, l, c, 14)
    a9,  _,  _    = adx(h, l, c, 9)
    adx_rising = _safe_num(a14.diff().iloc[-1] if len(a14) > 1 else 0.0, 0.0) > 0
    di_plus  = _safe_num(dip.iloc[-1] if len(dip) else 0.0, 0.0)
    di_minus = _safe_num(dim.iloc[-1] if len(dim) else 0.0, 0.0)
    di_plus_gt = di_plus > di_minus

    # Bollinger width (% of price) + rank
    bw = bb_width_pct(c, n=cfg.get("bb_win", 20), k=cfg.get("bb_k", 2.0))
    bw_now = _safe_num(bw.iloc[-1] if len(bw.dropna()) else 0.0, 0.0)
    bw_prev = _safe_num(bw.iloc[-2] if len(bw.dropna()) > 1 else bw_now, bw_now)
    tail = bw.tail(120).dropna()
    bb_width_pct_rank = float((tail <= bw_now).mean() * 100.0) if len(tail) >= 20 else 50.0
    squeeze_flag = int(bb_width_pct_rank <= 20.0)

    # Lookback N features for “_prev_n”
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

    # EMA anchor
    ema50 = _safe_num(ema(c, 50).iloc[-1], px_now)

    # Whipsaw flips on hist
    flips_hist = _count_sign_flips(hist, look=12)

    feat: Dict[str, float | int | bool] = {
        # Price & volume
        "open": _safe_num(o.iloc[-1], px_now), "close": px_now,
        "high": _safe_num(h.iloc[-1], px_now), "low": _safe_num(l.iloc[-1], px_now),
        "volume": _safe_num(v.iloc[-1], 0.0),
        "volume_avg_20": _safe_num(v.rolling(20).mean().iloc[-1] if len(v) >= 20 else v.mean(), 0.0),
        "rvol_now": float(rvol_now),

        # Vol & ATR
        "atr_pct": float(atr_pct_now),
        "atr_avg_20": float(atr_avg_20),

        # RSI/RMI
        "rsi_fast": float(_safe_num(rsi_fast.iloc[-1] if len(rsi_fast) else 50.0, 50.0)),
        "rsi_std": float(_safe_num(rsi_std.iloc[-1] if len(rsi_std) else 50.0, 50.0)),
        "rsi5": float(rsi5), "rsi_prev5": float(rsi_prev5),
        "rsi_prev_std_n": float(rsi_prev_std_n),
        "rmi": float(rmi_now), "rmi_now": float(rmi_now), "rmi_prev_n": float(rmi_prev_n),

        # MACD & histogram
        "macd_line": float(_safe_num(macd_line.iloc[-1], 0.0)),
        "macd_sig": float(_safe_num(macd_sig.iloc[-1], 0.0)),
        "hist": float(_safe_num(hist.iloc[-1], 0.0)),
        "macd_hist": float(_safe_num(hist.iloc[-1], 0.0)),
        "macd_hist_prev_n": float(macd_hist_prev_n),
        "hist_ema": float(_safe_num(hist_ema.iloc[-1], 0.0)),
        "hist_diff": float(hist_diff),
        "zero_cross_up": bool(zero_cross_up),
        "zero_cross_down": bool(zero_cross_down),

        # OBV/MFI
        "z_obv": float(z_obv),
        "mfi_now": float(mfi_now),
        "mfi_up": bool(mfi_up),

        # ADX/DI
        "adx14": float(_safe_num(a14.iloc[-1], 0.0)),
        "adx9": float(_safe_num(a9.iloc[-1], 0.0)),
        "adx_rising": bool(adx_rising),
        "di_plus": float(di_plus),
        "di_minus": float(di_minus),
        "di_plus_gt": bool(di_plus_gt),

        # ROC/ATR
        "roc_atr_ratio": float(roc_atr_ratio),

        # Bands / squeeze / widths
        "bb_width_pct": float(bw_now),
        "bb_width": float(bw_now),               # alias for your INI
        "bb_width_pct_prev": float(bw_prev),
        "bb_width_prev_n": float(bb_width_prev_n),
        "bb_width_pct_rank": float(bb_width_pct_rank),
        "squeeze_flag": int(squeeze_flag),

        # Lookback comparators
        "close_prev_n": float(close_prev_n),
        "low_prev_n": float(low_prev_n),
        "high_prev_n": float(high_prev_n),

        # EMA anchor
        "ema50": float(ema50),

        # Whipsaw
        "whipsaw_flips": float(flips_hist),
    }
    return feat


def _eval_expr(expr: str, F: dict) -> bool:
    if not expr:
        return False
    # We purposefully restrict builtins to None. Expressions should reference only F keys.
    return bool(eval(expr, {"__builtins__": None}, {k: F[k] for k in F}))


# -----------------------------
# Base momentum scorer (unchanged behavior)
# -----------------------------
def _momentum_score_base(dtf: pd.DataFrame, cfg: dict) -> Tuple[float, Dict[str, float], bool]:
    c = dtf["close"]; h = dtf["high"]; l = dtf["low"]; v = dtf["volume"]

    # Vol regime via ATR%
    ATR = atr(h, l, c, cfg["atr_win"])
    atr_pct = _safe_num((ATR.iloc[-1] / max(1e-9, c.iloc[-1])) * 100.0, 0.0)

    # Vol-aware RSI thresholds
    if atr_pct < cfg["low_vol_thr"]:
        rsi_thr = 60
    elif atr_pct < cfg["mid_vol_thr"]:
        rsi_thr = 65
    else:
        rsi_thr = 70

    rsi_fast = _rsi(c, cfg["rsi_fast"])
    rsi_std  = _rsi(c, cfg["rsi_std"])

    # RMI adaptive
    rmi_lb = 9 if atr_pct > cfg["mid_vol_thr"] else (21 if atr_pct < cfg["low_vol_thr"] else cfg["rmi_lb"])
    rmi_m  = 3 if atr_pct > cfg["mid_vol_thr"] else cfg["rmi_m"]
    rmi    = _rmi(c, lb=rmi_lb, m=rmi_m)

    # RMI vs RSI slope divergence (±10)
    rmi_slope = _safe_num(rmi.diff().iloc[-1] if len(rmi) > 1 else 0.0, 0.0)
    rsi_slope = _safe_num(rsi_std.diff().iloc[-1] if len(rsi_std) > 1 else 0.0, 0.0)
    rmi_pts = 10 if (rmi_slope > rsi_slope > 0) else (-10 if (rmi_slope < rsi_slope < 0) else 0)

    # RSI vol-aware score (0–15)
    rsi_val = _safe_num(rsi_std.iloc[-1] if len(rsi_std) else 50.0, 50.0)
    if rsi_val >= rsi_thr:
        rsi_pts = 15
    elif rsi_val >= (rsi_thr - 5):
        rsi_pts = 8
    else:
        rsi_pts = 0

    # MACD hist vs EMA(hist) with volume-surge gating
    macd_line = ema(c, 12) - ema(c, 26)
    macd_sig  = macd_line.ewm(span=9, adjust=False).mean()
    hist      = macd_line - macd_sig
    hist_ema  = hist.ewm(span=5, adjust=False).mean()

    # Volume burst via OBVΔ z-score
    obv  = obv_series(c, v)
    obv_d = obv.diff()
    mu = _safe_num(obv_d.ewm(span=5, adjust=False).mean().iloc[-1] if len(obv_d.dropna()) else 0.0, 0.0)
    sd = _safe_num(obv_d.rolling(20).std(ddof=1).iloc[-1] if len(obv_d.dropna()) else 0.0, 0.0)
    z  = (_safe_num(obv_d.iloc[-1], 0.0) - mu) / (sd if sd > 0 else 1e9)

    # MFI rising bonus (same proxy as features)
    tp = (h + l + c) / 3.0
    rmf = tp * v
    pos = (tp > tp.shift(1)).astype(float) * rmf
    neg = (tp < tp.shift(1)).astype(float) * rmf
    mfi_ratio = (pos.rolling(14, min_periods=7).sum() /
                 neg.replace(0, np.nan).rolling(14, min_periods=7).sum())
    mfi_up = (mfi_ratio.diff().iloc[-1] or 0) > 0 if len(mfi_ratio.dropna()) else False

    vol_pts = 15 if z >= 2.0 else (10 if z >= 1.0 else 0)
    if vol_pts > 0 and mfi_up:
        vol_pts = min(20, vol_pts + 5)

    # MACD points:
    zero_cross_now = (len(macd_line) > 1) and ((macd_line.iloc[-2] <= 0 <= macd_line.iloc[-1]) or \
                                               (macd_line.iloc[-2] >= 0 >= macd_line.iloc[-1]))
    hist_above_ema = _safe_num(hist.iloc[-1], 0.0) > _safe_num(hist_ema.iloc[-1], 0.0) if len(hist) else False
    if zero_cross_now and z >= 2.0:
        macd_pts = 30
    elif hist_above_ema:
        macd_pts = 15
    else:
        macd_pts = 10 if zero_cross_now else 0

    # Whipsaw penalty / flag
    flips = _count_sign_flips(hist, look=12)
    whipsaw_flag = flips >= 4
    whipsaw_pen = -10 if whipsaw_flag else 0

    # ROC vs ATR adaptive (0–15)
    roc = c.pct_change(3)
    ratio = abs(_safe_num(roc.iloc[-1], 0.0)) / max(1e-9, _safe_num(ATR.iloc[-1], 0.0) / max(1e-9, _safe_num(c.iloc[-1], 1.0)))
    if ratio >= 1.0:
        roc_pts = 15
    elif ratio >= 0.7:
        roc_pts = 8
    else:
        roc_pts = 0

    # ADX slope + DI filter (0–15)
    a14, dip, dim = adx(h, l, c, 14)
    a9,  _,  _    = adx(h, l, c, 9)
    adx_ok = (_safe_num(a9.iloc[-1], 0.0) > _safe_num(a14.iloc[-1], 0.0)) and \
             (_safe_num(a14.diff().iloc[-1], 0.0) > 0) and (_safe_num(dip.iloc[-1], 0.0) > _safe_num(dim.iloc[-1], 0.0))
    di_close = abs(_safe_num(dip.iloc[-1], 0.0) - _safe_num(dim.iloc[-1], 0.0)) < 2.0
    adx_pts = 15 if adx_ok else (-5 if (_safe_num(a14.diff().iloc[-1], 0.0) > 0 and di_close) else 0)

    # Context bonus (post-squeeze breakout)
    bw = bb_width_pct(c, n=20, k=2.0)
    if len(bw.dropna()) > 40:
        p20 = float(np.nanpercentile(bw.tail(120).dropna(), 20)) if len(bw.dropna()) >= 50 else float(bw.dropna().quantile(0.2))
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


# -----------------------------
# Public API (with scenarios + optional ML blend)
# -----------------------------
def score_momentum(symbol: str, kind: str, tf: str, df5: pd.DataFrame, base: BaseCfg, ini_path: str = DEFAULT_INI):
    cfg = _cfg(ini_path)
    dftf = resample(df5, tf)
    dftf = maybe_trim_last_bar(dftf)
    if not ensure_min_bars(dftf, tf) or len(dftf) < cfg.get("min_bars", 120):
        return None

    # Base model
    base_score, parts, whipsaw_flag = _momentum_score_base(dftf, cfg)
    total_score = base_score

    # Scenarios (optional)
    try:
        rules_mode, scenarios = _load_mom_scenarios(cfg["_ini_path"])
    except Exception:
        rules_mode, scenarios = (cfg.get("rules_mode", "additive"), [])

    if scenarios:
        F = _mom_features(dftf, cfg)
        scen_total = 0.0
        for sc in scenarios:
            try:
                if _eval_expr(sc["when"], F):
                    scen_total += sc["score"]
                    parts[f"SCN.{sc['name']}"] = float(sc["score"])
                    if sc.get("bonus_when") and _eval_expr(sc["bonus_when"], F):
                        scen_total += sc["bonus"]
                        parts[f"SCN.{sc['name']}.bonus"] = float(sc["bonus"])
            except Exception:
                # defensive: skip broken scenario
                continue

        total_score = scen_total if rules_mode == "override" else (base_score + scen_total)
        total_score = float(clamp(total_score, cfg.get("clamp_low", 0.0), cfg.get("clamp_high", 100.0)))

    ts = dftf.index[-1].to_pydatetime().replace(tzinfo=TZ)

    # ----- Optional ML blend (pattern mirrors FLOW pillar) -----
    ml_prob = None
    ml_score = None
    final_score = total_score
    final_veto = False  # momentum pillar typically doesn't set its own veto; we expose optional veto_if_prob_lt

    try:
        if cfg.get("ml_enabled", False):
            w = float(cfg.get("ml_blend_weight", 0.35))
            w = max(0.0, min(1.0, w))

            # Path A: DB source (per-symbol/tf/ts predictions)
            tbl = cfg.get("ml_source_table", "")
            if tbl:
                from utils.db import get_db_connection
                with get_db_connection() as conn, conn.cursor() as cur:
                    cur.execute(f"""
                        SELECT prob_long, prob_short
                          FROM {tbl}
                         WHERE symbol=%s AND tf=%s AND ts=%s
                         ORDER BY ts DESC
                         LIMIT 1
                    """, (symbol, tf, ts))
                    r = cur.fetchone()
                if r:
                    prob_long  = _safe_num(r[0], 0.5)
                    prob_short = _safe_num(r[1], 0.5)
                    ml_prob = max(prob_long, 1.0 - prob_short)  # reconcile
                    ml_score = round(100.0 * ml_prob, 2)

            # Path B: Python callback (compute now)
            if ml_prob is None:
                cb = cfg.get("ml_callback", "")
                if cb:
                    mod_name, _, fn_name = cb.rpartition(".")
                    if mod_name and fn_name:
                        import importlib
                        mod = importlib.import_module(mod_name)
                        fn  = getattr(mod, fn_name)
                        ml_prob = float(fn(symbol=symbol, kind=kind, tf=tf, ts=ts,
                                          frame=dftf, features=_mom_features(dftf, cfg)))
                        if not np.isfinite(ml_prob): ml_prob = 0.0
                        ml_prob = max(0.0, min(1.0, ml_prob))
                        ml_score = round(100.0 * ml_prob, 2)

            # Blend if available
            if ml_prob is not None:
                base_prob = float(total_score) / 100.0
                blended   = (1.0 - w) * base_prob + w * ml_prob
                final_score = round(100.0 * blended, 2)

                # Optional: derive a veto if prob is too low (guard rails)
                thr_txt = cfg.get("ml_veto_if_prob_lt", "").strip()
                if thr_txt:
                    try:
                        thr = float(thr_txt)
                        if 0.0 <= thr <= 1.0 and blended < thr:
                            final_veto = True
                    except Exception:
                        pass

    except Exception:
        # keep pillar robust even if ML path errors
        pass

    rows = [
        (symbol, kind, tf, ts, "MOM.score", float(total_score), json.dumps({
            "rules_mode": cfg.get("rules_mode", "additive")
        }), base.run_id, base.source),

        # blended outputs
        (symbol, kind, tf, ts, "MOM.score_final", float(final_score), "{}", base.run_id, base.source),
        (symbol, kind, tf, ts, "MOM.veto_final",  1.0 if final_veto else 0.0, "{}", base.run_id, base.source),

        # diagnostics
        (symbol, kind, tf, ts, "MOM.whipsaw_flag", 1.0 if whipsaw_flag else 0.0, "{}", base.run_id, base.source),

        # debug parts (base)
        (symbol, kind, tf, ts, "MOM.RMI_adaptive", float(parts.get("rmi_pts", 0.0)), "{}", base.run_id, base.source),
        (symbol, kind, tf, ts, "MOM.RSI_vol", float(parts.get("rsi_pts", 0.0)), "{}", base.run_id, base.source),
        (symbol, kind, tf, ts, "MOM.MACD_hist_ema", float(parts.get("macd_pts", 0.0)), "{}", base.run_id, base.source),
        (symbol, kind, tf, ts, "MOM.ROC_adaptive", float(parts.get("roc_pts", 0.0)), "{}", base.run_id, base.source),
        (symbol, kind, tf, ts, "MOM.ADX_slope_DI", float(parts.get("adx_pts", 0.0)), "{}", base.run_id, base.source),
        (symbol, kind, tf, ts, "MOM.Volume_burst", float(parts.get("vol_pts", 0.0)), "{}", base.run_id, base.source),
        (symbol, kind, tf, ts, "MOM.Context_bonus", float(parts.get("ctx_bonus", 0.0)), "{}", base.run_id, base.source),
        (symbol, kind, tf, ts, "MOM.whipsaw_flips", float(parts.get("whipsaw_flips", 0.0)), "{}", base.run_id, base.source),
        (symbol, kind, tf, ts, "MOM.atr_pct", float(parts.get("atr_pct", 0.0)), "{}", base.run_id, base.source),
    ]

    # Optional: write scenario contributions
    if cfg.get("write_scenarios_debug", False):
        for k, v in parts.items():
            if k.startswith("SCN."):
                rows.append((symbol, kind, tf, ts, f"MOM.{k}", float(v), "{}", base.run_id, base.source))

    # Optional: record ML context when enabled
    try:
        if cfg.get("ml_enabled", False):
            ml_ctx = {"ml_prob": ml_prob, "blend_weight": cfg.get("ml_blend_weight", 0.35)}
            rows.append((symbol, kind, tf, ts, "MOM.ml_ctx", 0.0, json.dumps(ml_ctx), base.run_id, base.source))
    except Exception:
        pass

    write_values(rows)
    return (ts, float(final_score), bool(final_veto))
