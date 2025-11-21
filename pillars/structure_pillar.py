# pillars/structure_pillar.py
from __future__ import annotations
import json, os, math, configparser
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
from datetime import timezone
import numpy as np
import pandas as pd

from .common import (  # keep contracts identical to other pillars
    DEFAULT_INI, BaseCfg, TZ, resample, write_values, last_metric, clamp
)
from pillars.common import ensure_min_bars, maybe_trim_last_bar

# -----------------------------
# Local TA helpers (scoped)
# -----------------------------
def _ema(s: pd.Series, n:int) -> pd.Series: return s.ewm(span=n, adjust=False).mean()

def _true_range(h: pd.Series, l: pd.Series, c: pd.Series) -> pd.Series:
    pc = c.shift(1)
    return pd.concat([(h-l).abs(), (h-pc).abs(), (l-pc).abs()], axis=1).max(axis=1)

def _atr(h: pd.Series, l: pd.Series, c: pd.Series, n:int=14) -> pd.Series:
    return _true_range(h,l,c).ewm(alpha=1.0/n, adjust=False).mean()

def _bb_width_pct(close: pd.Series, n:int=20, k:float=2.0) -> pd.Series:
    ma = close.rolling(n, min_periods=max(5, n//2)).mean()
    sd = close.rolling(n, min_periods=max(5, n//2)).std(ddof=1)
    upper = ma + k*sd; lower = ma - k*sd
    width = (upper - lower) / (ma.replace(0,np.nan).abs())
    return 100.0 * width

def _safe_num(x: Any, default: float = 0.0) -> float:
    try:
        fx = float(x)
        return fx if np.isfinite(fx) else default
    except Exception:
        return default

# -----------------------------
# Config
# -----------------------------
@dataclass
class StructCfg:
    metric_prefix: str
    min_bars: int
    # weights for subcomponents (sum ~1.0)
    w_stop: float; w_reward: float; w_trigger: float; w_path: float; w_anchor: float
    # thresholds
    near_anchor_atr: float; min_stop_atr: float; max_stop_atr: float
    rr_bins: Tuple[float,float,float]
    vol_surge_k: float; squeeze_pct: float
    wick_clean_thr: float; wick_dirty_thr: float
    veto_rr_floor: float
    # ML
    ml_enabled: bool; ml_blend_weight: float
    ml_source_table: str; ml_callback: str
    ml_soften_veto_if_prob_ge: Optional[float]; ml_veto_if_prob_lt: Optional[float]
    # pass parser for future
    cp: configparser.ConfigParser

def _csv(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]

def load_cfg(ini_path: str = DEFAULT_INI, section: str = "structure") -> StructCfg:
    cp = configparser.ConfigParser(inline_comment_prefixes=(";","#"), interpolation=None, strict=False)
    cp.read(ini_path)
    sect = section

    metric_prefix = cp.get(sect, "metric_prefix", fallback="STRUCT")
    rr_bins = tuple(float(x) for x in _csv(cp.get(sect, "rr_bins", fallback="1.0,1.5,2.0"))[:3]) or (1.0,1.5,2.0)

    # ML block (optional)
    ml_enabled = cp.getboolean(f"{sect}_ml", "enabled", fallback=False)
    ml_blend_weight = cp.getfloat(f"{sect}_ml", "blend_weight", fallback=0.35)
    ml_source_table = cp.get(f"{sect}_ml", "source_table", fallback="").strip()
    ml_callback = cp.get(f"{sect}_ml", "callback", fallback="").strip()

    def _opt_thr(key: str) -> Optional[float]:
        raw = cp.get(f"{sect}_ml", key, fallback="").strip()
        if not raw: return None
        try:
            v = float(raw)
            return v if 0.0 <= v <= 1.0 else None
        except Exception:
            return None

    return StructCfg(
        metric_prefix=metric_prefix,
        min_bars=cp.getint(sect, "min_bars", fallback=120),
        w_stop=cp.getfloat(sect, "w_stop", fallback=0.25),
        w_reward=cp.getfloat(sect, "w_reward", fallback=0.25),
        w_trigger=cp.getfloat(sect, "w_trigger", fallback=0.20),
        w_path=cp.getfloat(sect, "w_path", fallback=0.10),
        w_anchor=cp.getfloat(sect, "w_anchor", fallback=0.20),
        near_anchor_atr=cp.getfloat(sect, "near_anchor_atr", fallback=0.75),
        min_stop_atr=cp.getfloat(sect, "min_stop_atr", fallback=0.5),
        max_stop_atr=cp.getfloat(sect, "max_stop_atr", fallback=3.0),
        rr_bins=rr_bins,
        vol_surge_k=cp.getfloat(sect, "vol_surge_k", fallback=2.0),
        squeeze_pct=cp.getfloat(sect, "squeeze_pct", fallback=25.0),
        wick_clean_thr=cp.getfloat(sect, "wick_clean_thr", fallback=1.0),
        wick_dirty_thr=cp.getfloat(sect, "wick_dirty_thr", fallback=2.0),
        veto_rr_floor=cp.getfloat(sect, "veto_rr_floor", fallback=0.9),
        ml_enabled=ml_enabled,
        ml_blend_weight=max(0.0, min(1.0, ml_blend_weight)),
        ml_source_table=ml_source_table,
        ml_callback=ml_callback,
        ml_soften_veto_if_prob_ge=_opt_thr("soften_veto_if_prob_ge"),
        ml_veto_if_prob_lt=_opt_thr("veto_if_prob_lt"),
        cp=cp,
    )

# -----------------------------
# Anchors / bias / sub-scorers
# -----------------------------
def _anchors(symbol: str, kind: str, tf: str) -> Dict[str, Optional[float]]:
    return {
        "poc": last_metric(symbol, kind, tf, "VP.POC"),
        "val": last_metric(symbol, kind, tf, "VP.VAL"),
        "vah": last_metric(symbol, kind, tf, "VP.VAH"),
        "bb_score": last_metric(symbol, kind, tf, "BB.score"),
    }

def _dir_bias(close: pd.Series) -> int:
    if len(close) < 50: return 0
    ema10, ema20, ema50 = _ema(close,10), _ema(close,20), _ema(close,50)
    if float(ema10.iloc[-1]) > float(ema20.iloc[-1]) > float(ema50.iloc[-1]): return +1
    if float(ema10.iloc[-1]) < float(ema20.iloc[-1]) < float(ema50.iloc[-1]): return -1
    return 0

def _wick_body_ratio(o: float, h: float, l: float, c: float) -> float:
    body = abs(c - o); full = (h - l)
    if full <= 0: return 0.0
    upper = max(0.0, h - max(c,o))
    lower = max(0.0, min(c,o) - l)
    return float((upper + lower) / max(1e-9, body))

def _stop_target(close: float, atr: float, A: Dict[str, Optional[float]], dir_: int, cfg: StructCfg) -> Tuple[Optional[float], Optional[float]]:
    poc, val, vah = A.get("poc"), A.get("val"), A.get("vah")
    if dir_ > 0:
        candidates = [x for x in [val, poc] if x is not None and x < close]
        stop = max(candidates) if candidates else close - cfg.min_stop_atr*atr
        target = vah if (vah is not None and vah > close) else (close + 2.0*atr)
    elif dir_ < 0:
        candidates = [x for x in [vah, poc] if x is not None and x > close]
        stop = min(candidates) if candidates else close + cfg.min_stop_atr*atr
        target = val if (val is not None and val < close) else (close - 2.0*atr)
    else:
        return None, None
    return float(stop), float(target)

def _rr_score(rr: float, bins: Tuple[float,float,float]) -> float:
    a,b,c = bins
    if rr >= c: return 25.0
    if rr >= b: return 18.0
    if rr >= a: return 10.0
    return 0.0

def _trigger_score(d: pd.DataFrame, atr: float, dir_: int, A: Dict[str, Optional[float]], cfg: StructCfg) -> float:
    if len(d) < 30 or dir_ == 0: return 8.0
    c = d["close"]; v = d["volume"]
    vah, val = A.get("vah"), A.get("val")
    vavg = float(v.rolling(20, min_periods=10).mean().iloc[-1])
    vnow = float(v.iloc[-1])
    now = float(c.iloc[-1]); prev = float(c.iloc[-2]) if len(c) > 1 else now
    pts = 8.0
    if dir_ > 0 and vah is not None and now > float(vah) >= prev and vnow > cfg.vol_surge_k * max(1e-9, vavg):
        pts = 20.0
    elif dir_ < 0 and val is not None and now < float(val) <= prev and vnow > cfg.vol_surge_k * max(1e-9, vavg):
        pts = 20.0
    elif vnow > 1.2 * max(1e-9, vavg):
        pts = 12.0

    bb = _bb_width_pct(c, 20, 2.0)
    if len(bb.dropna()) >= 40:
        pct = float(pd.Series(bb).rank(pct=True).iloc[-1])
        if pct <= cfg.squeeze_pct/100.0:
            pts += 5.0
    return clamp(pts, 0.0, 100.0)

def _path_score(o: float, h: float, l: float, c: float, cfg: StructCfg) -> float:
    r = _wick_body_ratio(o,h,l,c)
    if r <= cfg.wick_clean_thr: return 10.0
    if r >= cfg.wick_dirty_thr: return 0.0
    t0, t1 = cfg.wick_clean_thr, cfg.wick_dirty_thr
    return float(10.0 * (1.0 - (r - t0) / max(1e-9, (t1 - t0))))

def _anchor_conf(close: float, atr: float, A: Dict[str, Optional[float]], cfg: StructCfg) -> float:
    near = 0
    for k in ("poc","val","vah"):
        v = A.get(k)
        if v is None: continue
        if abs(close - float(v)) <= cfg.near_anchor_atr * atr:
            near += 1
    if near >= 2: return 20.0
    if near == 1: return 12.0
    return 6.0

def _obstruction_penalty(close: float, target: float, A: Dict[str, Optional[float]]) -> float:
    poc = A.get("poc")
    if poc is None: return 0.0
    lo, hi = sorted([close, target])
    return -5.0 if (lo < float(poc) < hi) else 0.0

# -----------------------------
# Per-TF core score
# -----------------------------
def _score_for_direction(d: pd.DataFrame, A: Dict[str, Optional[float]], dir_: int, cfg: StructCfg) -> Tuple[float, Dict[str,float], bool]:
    c = d["close"]; h = d["high"]; l = d["low"]; o = d["open"]
    atr = _safe_num(_atr(h,l,c,14).iloc[-1], 0.0)
    if atr <= 0 or dir_ == 0:
        return 50.0, {"stop":10,"reward":10,"trigger":10,"path":10,"anchor":10,"rr":1.0,"stop_atr":1.0}, False

    close = float(c.iloc[-1])
    stop, target = _stop_target(close, atr, A, dir_, cfg)
    if stop is None or target is None:
        return 40.0, {"stop":5,"reward":5,"trigger":10,"path":10,"anchor":10,"rr":0.0,"stop_atr":0.0}, True

    risk = abs(close - stop); move = abs(target - close)
    stop_atr = (risk / atr) if atr > 0 else 0.0
    rr = (move / risk) if risk > 0 else 0.0

    # 0–25
    if cfg.min_stop_atr <= stop_atr <= cfg.max_stop_atr: stop_score = 25.0
    else:
        if stop_atr < cfg.min_stop_atr:
            stop_score = max(0.0, 25.0 * (stop_atr / max(cfg.min_stop_atr, 1e-9)))
        else:
            stop_score = max(0.0, 25.0 * (cfg.max_stop_atr / max(stop_atr, 1e-9)))

    reward_score  = _rr_score(rr, cfg.rr_bins)          # 0–25
    trigger_score = _trigger_score(d, atr, dir_, A, cfg) # 0–20 (+5 squeeze)
    path_score    = _path_score(float(o.iloc[-1]), float(h.iloc[-1]), float(l.iloc[-1]), close, cfg)
    path_score    = max(0.0, min(10.0, path_score + _obstruction_penalty(close, target, A)))
    anchor_score  = _anchor_conf(close, atr, A, cfg)    # 0–20

    veto = False
    if rr < cfg.veto_rr_floor: veto = True
    if stop_atr < 0.2 or stop_atr > 5.0: veto = True

    total = (
        cfg.w_stop   * stop_score +
        cfg.w_reward * reward_score +
        cfg.w_trigger* trigger_score +
        cfg.w_path   * path_score +
        cfg.w_anchor * anchor_score
    )
    parts = {
        "stop": stop_score, "reward": reward_score, "trigger": trigger_score,
        "path": path_score, "anchor": anchor_score, "rr": rr, "stop_atr": stop_atr
    }
    return float(clamp(total, 0.0, 100.0)), parts, veto

# -----------------------------
# Public API (single TF)
# -----------------------------
def score_structure(symbol: str, kind: str, tf: str, df5: pd.DataFrame, base: BaseCfg, ini_path: str = DEFAULT_INI, section: str = "structure"):
    """
    Computes Structure pillar for a single timeframe from 5m data.
    Writes:
      P.score, P.veto_flag, P.score_final, P.veto_final
      P.stop / reward / trigger / path / anchor / rr / stop_atr (debug)
    Returns: (ts, score_final, veto_final, parts_dict)
    """
    cfg = load_cfg(ini_path, section=section)
    dftf = resample(df5, tf)
    dftf = maybe_trim_last_bar(dftf)
    if not ensure_min_bars(dftf, tf) or len(dftf) < cfg.min_bars:
        return None

    A = _anchors(symbol, kind, tf)
    bias = _dir_bias(dftf["close"])
    up_s, up_p, up_v = _score_for_direction(dftf, A, +1, cfg)
    dn_s, dn_p, dn_v = _score_for_direction(dftf, A, -1, cfg)

    if bias > 0: direction, score, parts, veto = (+1, up_s, up_p, up_v)
    elif bias < 0: direction, score, parts, veto = (-1, dn_s, dn_p, dn_v)
    else:
        direction, score, parts, veto = ((+1, up_s, up_p, up_v) if up_s >= dn_s else (-1, dn_s, dn_p, dn_v))

    ts = dftf.index[-1].to_pydatetime().replace(tzinfo=TZ)

    # ---- Optional ML blend ----
    final_score = float(score)
    final_veto  = bool(veto)
    ml_prob = None

    try:
        if cfg.ml_enabled:
            w = cfg.ml_blend_weight

            # Path A: DB-driven probability
            if cfg.ml_source_table:
                from utils.db import get_db_connection
                with get_db_connection() as conn, conn.cursor() as cur:
                    cur.execute(f"""
                        SELECT prob_long, prob_short
                          FROM {cfg.ml_source_table}
                         WHERE symbol=%s AND tf=%s AND ts=%s
                         ORDER BY ts DESC
                         LIMIT 1
                    """, (symbol, tf, ts))
                    r = cur.fetchone()
                if r:
                    prob_long  = _safe_num(r[0], 0.5)
                    prob_short = _safe_num(r[1], 0.5)
                    ml_prob = max(prob_long, 1.0 - prob_short)

            # Path B: Python callback
            if ml_prob is None and cfg.ml_callback:
                mod_name, _, fn_name = cfg.ml_callback.rpartition(".")
                if mod_name and fn_name:
                    import importlib
                    mod = importlib.import_module(mod_name)
                    fn  = getattr(mod, fn_name)
                    ml_prob = float(fn(symbol=symbol, kind=kind, tf=tf, ts=ts, parts=parts, anchors=A, direction=direction))
                    if not np.isfinite(ml_prob): ml_prob = 0.0
                    ml_prob = max(0.0, min(1.0, ml_prob))

            if ml_prob is not None:
                base_prob = final_score / 100.0
                blended = (1.0 - w) * base_prob + w * ml_prob
                final_score = round(100.0 * blended, 2)

                t_soft = cfg.ml_soften_veto_if_prob_ge
                t_hard = cfg.ml_veto_if_prob_lt
                if t_soft is not None and blended >= t_soft:
                    final_veto = False
                if t_hard is not None and blended < t_hard:
                    final_veto = True
    except Exception:
        # be robust to ML errors
        pass

    P = cfg.metric_prefix
    ctx = {
        "dir": int(direction),
        "anchors": {k: (float(v) if v is not None else None) for k,v in A.items()},
        "rr": float(parts.get("rr", 0.0)),
        "stop_atr": float(parts.get("stop_atr", 0.0)),
        "variant": P,
        "ml_prob": ml_prob,
    }

    rows = [
        (symbol, kind, tf, ts, f"{P}.stop",     float(parts["stop"]),   json.dumps(ctx), base.run_id, base.source),
        (symbol, kind, tf, ts, f"{P}.reward",   float(parts["reward"]), json.dumps(ctx), base.run_id, base.source),
        (symbol, kind, tf, ts, f"{P}.trigger",  float(parts["trigger"]),json.dumps(ctx), base.run_id, base.source),
        (symbol, kind, tf, ts, f"{P}.path",     float(parts["path"]),   json.dumps(ctx), base.run_id, base.source),
        (symbol, kind, tf, ts, f"{P}.anchor",   float(parts["anchor"]), json.dumps(ctx), base.run_id, base.source),
        (symbol, kind, tf, ts, f"{P}.rr",       float(parts["rr"]),     json.dumps(ctx), base.run_id, base.source),
        (symbol, kind, tf, ts, f"{P}.stop_atr", float(parts["stop_atr"]),json.dumps(ctx), base.run_id, base.source),

        (symbol, kind, tf, ts, f"{P}.score",       float(score),      "{}", base.run_id, base.source),
        (symbol, kind, tf, ts, f"{P}.veto_flag",   1.0 if bool(veto) else 0.0, "{}", base.run_id, base.source),
        (symbol, kind, tf, ts, f"{P}.score_final", float(final_score),"{}", base.run_id, base.source),
        (symbol, kind, tf, ts, f"{P}.veto_final",  1.0 if bool(final_veto) else 0.0, "{}", base.run_id, base.source),
    ]
    write_values(rows)
    return (ts, final_score, final_veto, parts)
