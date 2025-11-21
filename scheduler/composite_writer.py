# File: scheduler/composite_writer.py
# Purpose: Compute composite confidence per symbol/TF and upsert into signals.confidence_composite
# Build: r5_dyn_regime + subscores + symbols filter

import json
import math
import configparser
from datetime import datetime, timezone
from typing import Dict, Any, List, Tuple, Optional, Iterable

from utils.db import get_db_connection


# ---------- Utils ----------
def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))

def lin_map(x: float, x0: float, y0: float, x1: float, y1: float) -> float:
    if x1 == x0:
        return (y0 + y1) / 2
    y = y0 + (x - x0) * (y1 - y0) / (x1 - x0)
    lo, hi = (y0, y1) if y0 <= y1 else (y1, y0)
    return clamp(y, lo, hi)

def percentile_rank(series: Optional[List[Optional[float]]], v: Optional[float]) -> float:
    if not series or v is None:
        return 50.0
    s = [x for x in series if x is not None]
    if not s:
        return 50.0
    s.sort()
    count = sum(1 for x in s if x <= v)
    return 100.0 * count / len(s)

def piecewise_adx_to_score(a: Optional[float]) -> float:
    a = a or 0.0
    if a <= 15:  return 30.0 * a / 15.0
    if a <= 25:  return lin_map(a, 15, 30, 25, 60)
    if a <= 35:  return lin_map(a, 25, 60, 35, 85)
    if a <= 40:  return lin_map(a, 35, 85, 40, 100)
    return 100.0

def rsi_to_score(rsi: Optional[float]) -> float:
    if rsi is None: return 50.0
    if rsi <= 40:  return lin_map(rsi, 0, 0, 40, 25)
    if rsi <= 50:  return lin_map(rsi, 40, 25, 50, 50)
    if rsi <= 60:  return lin_map(rsi, 50, 50, 60, 75)
    if rsi <= 100: return lin_map(rsi, 60, 75, 100, 100)
    return 50.0

def mfi_to_score(mfi: Optional[float]) -> float:
    return rsi_to_score(mfi)

def macd_regime_to_score(line: Optional[float], signal: Optional[float],
                         hist: Optional[float], prev_hist: Optional[float]) -> float:
    if None in (line, signal, hist, prev_hist):
        return 50.0
    bull = (line > signal) and (hist > prev_hist)
    bear = (line < signal) and (hist < prev_hist)
    if bull: return 75.0
    if bear: return 25.0
    return 50.0

def ema_ladder_score(e5: Optional[float], e10: Optional[float],
                     e20: Optional[float], e25: Optional[float]) -> float:
    if None in (e5, e10, e20, e25): return 50.0
    cnt = 0
    if e5  > e10: cnt += 1
    if e10 > e20: cnt += 1
    if e20 > e25: cnt += 1
    if e5 > e20:  cnt += 1
    return 25.0 * cnt

def ema_slope_bonus(ema20_now: Optional[float], ema20_prev3: Optional[float],
                    price_now: Optional[float]) -> float:
    if None in (ema20_now, ema20_prev3, price_now) or (price_now or 0) <= 0:
        return 0.0
    slope = max(0.0, (ema20_now - ema20_prev3) / price_now)
    x = 5000.0 * slope
    bonus = 15.0 * clamp(math.log1p(x), 0.0, 1.5) / 1.5
    return bonus

def atr_penalty(atr_pct: Optional[float]) -> float:
    if atr_pct is None: return 0.0
    if atr_pct <= 0.04: return 0.0
    if atr_pct <= 0.06: return 5.0
    if atr_pct <= 0.10: return 15.0
    return 20.0


# ---------- Scoring buckets ----------
def score_trend(emas: Dict[str, Optional[float]],
                price_now: Optional[float],
                ema20_prev3: Optional[float]) -> float:
    e5, e10, e20, e25 = (emas.get('5'), emas.get('10'), emas.get('20'), emas.get('25'))
    base  = ema_ladder_score(e5, e10, e20, e25)
    bonus = ema_slope_bonus(e20, ema20_prev3, price_now)
    return clamp(base + bonus, 0.0, 100.0)

def surprise_factor(adx: Optional[float],
                    roc_series: Optional[List[Optional[float]]], roc_now: Optional[float],
                    macd_hist: Optional[float], macd_hist_prev: Optional[float]) -> float:
    if adx is None or roc_series is None or roc_now is None:
        return 0.0
    roc_pr = percentile_rank(roc_series, roc_now)
    rising = (macd_hist is not None and macd_hist_prev is not None and macd_hist > macd_hist_prev)
    if adx <= 18.0 and roc_pr >= 95.0 and rising:
        return 6.0 + 4.0 * ((roc_pr - 95.0) / 5.0)
    return 0.0

def score_momentum(rsi: Optional[float],
                   macd_line: Optional[float], macd_signal: Optional[float],
                   macd_hist: Optional[float], macd_hist_prev: Optional[float],
                   roc_series: Optional[List[Optional[float]]], roc_now: Optional[float],
                   adx_for_surprise: Optional[float]) -> float:
    rsi_sc  = rsi_to_score(rsi)
    macd_sc = macd_regime_to_score(macd_line, macd_signal, macd_hist, macd_hist_prev)
    roc_pct = percentile_rank(roc_series, roc_now)
    roc_sc  = clamp(roc_pct, 0.0, 100.0)
    base = clamp(0.4*rsi_sc + 0.4*macd_sc + 0.2*roc_sc, 0.0, 100.0)
    bonus = surprise_factor(adx_for_surprise, roc_series, roc_now, macd_hist, macd_hist_prev)
    return clamp(base + bonus, 0.0, 100.0)

def score_quality(adx: Optional[float], atr_pct: Optional[float]) -> float:
    adx_sc  = piecewise_adx_to_score(adx)
    penalty = atr_penalty(atr_pct)
    return clamp(adx_sc - penalty, 0.0, 100.0)

def score_flow(market_type: str,
               oi_score_final: Optional[float],
               mfi: Optional[float],
               obv_delta_series: Optional[List[Optional[float]]],
               obv_delta_now: Optional[float]) -> float:
    mfi_sc  = mfi_to_score(mfi)
    obv_pct = percentile_rank(obv_delta_series, obv_delta_now)
    obv_sc  = clamp(obv_pct, 0.0, 100.0)
    cash_flow = 0.5*mfi_sc + 0.5*obv_sc
    if market_type == 'futures' and oi_score_final is not None:
        return clamp(0.6*oi_score_final + 0.4*cash_flow, 0.0, 100.0)
    return clamp(cash_flow, 0.0, 100.0)

def score_structure(bb_in_zone: bool, bb_score: Optional[float],
                    close: Optional[float], poc: Optional[float],
                    val: Optional[float], vah: Optional[float],
                    trend_score_val: float) -> float:
    s = 50.0
    if bb_in_zone and bb_score is not None:
        s = 85.0 + clamp(bb_score/10.0, 0.0, 1.0) * 15.0
    elif bb_score is not None:
        s = min(60.0, 40.0 + 3.0*bb_score)

    if (close is not None) and (poc is not None) and (val is not None) and (vah is not None):
        uptrend = trend_score_val >= 60.0
        downtrend = trend_score_val <= 40.0
        if uptrend and (poc <= close <= vah):
            s += lin_map(close, poc, 5.0, vah, 10.0)
        elif downtrend and (val <= close <= poc):
            s += lin_map(close, poc, 5.0, val, 10.0)

    return clamp(s, 0.0, 100.0)


# ---------- Consensus & Veto ----------
def consensus_bonus_across_tfs(
    tf_to_triplet: Dict[str, Tuple[Optional[float], Optional[float], Optional[float], Optional[float], Optional[bool]]],
    max_bonus: float
) -> Tuple[float, Dict[str, str]]:

    if not tf_to_triplet:
        return 0.0, {}

    dirs: Dict[str, str] = {}
    for tf, (tr, mo, adx, rsi, bb_in_zone) in tf_to_triplet.items():
        tr  = tr  if tr  is not None else 0.0
        mo  = mo  if mo  is not None else 0.0
        adx = adx if adx is not None else 0.0
        rsi = rsi if rsi is not None else 50.0
        in_zone = bool(bb_in_zone) if bb_in_zone is not None else False

        if adx >= 25 and tr >= 60 and mo >= 60:
            dirs[tf] = 'bullish'
        elif adx >= 25 and tr <= 40 and mo <= 40:
            dirs[tf] = 'bearish'
        elif adx < 20 and 45 <= rsi <= 55 and in_zone:
            dirs[tf] = 'range'
        else:
            dirs[tf] = 'neutral'

    bull_cnt = sum(1 for d in dirs.values() if d == 'bullish')
    bear_cnt = sum(1 for d in dirs.values() if d == 'bearish')
    rng_cnt  = sum(1 for d in dirs.values() if d == 'range')

    bonus = 0.0
    if bull_cnt >= 3 or bear_cnt >= 3: bonus = max_bonus
    elif bull_cnt >= 2 or bear_cnt >= 2: bonus = 0.6 * max_bonus
    elif rng_cnt >= 2: bonus = 0.5 * max_bonus

    return bonus, dirs

def adjust_veto_thresholds(veto_cfg: Dict[str, float], atr_pct: Optional[float]) -> Dict[str, float]:
    vs = dict(veto_cfg)
    if atr_pct is None:
        return vs
    if atr_pct <= 0.03:
        for k in ("short_rsi_gt","short_mfi_gt","short_rmi_gt"): vs[k] += 2.0
        for k in ("long_rsi_lt","long_mfi_lt","long_rmi_lt"):   vs[k] -= 2.0
    elif atr_pct >= 0.08:
        for k in ("short_rsi_gt","short_mfi_gt","short_rmi_gt"): vs[k] += 8.0
        for k in ("long_rsi_lt","long_mfi_lt","long_rmi_lt"):   vs[k] -= 8.0
    return vs

def compute_veto(oi_buildup_label: Optional[str],
                 rsi_15m: Optional[float], rmi_15m: Optional[float], mfi_15m: Optional[float],
                 adx_15m: Optional[float],
                 veto_cfg: Dict[str, float]) -> Tuple[float, Dict[str, List[str]]]:
    flags: Dict[str, List[str]] = {}
    penalty = 0.0
    vs = veto_cfg
    veto_short, veto_long = [], []

    if oi_buildup_label == 'SHORT_BUILDUP':
        if rsi_15m is not None and rsi_15m > vs['short_rsi_gt']: veto_short.append(f"RSI>{vs['short_rsi_gt']}(15m)")
        if rmi_15m is not None and rmi_15m > vs['short_rmi_gt']: veto_short.append(f"RMI>{vs['short_rmi_gt']}(15m)")
        if mfi_15m is not None and mfi_15m > vs['short_mfi_gt']: veto_short.append(f"MFI>{vs['short_mfi_gt']}(15m)")
    if oi_buildup_label == 'LONG_BUILDUP':
        if rsi_15m is not None and rsi_15m < vs['long_rsi_lt']: veto_long.append(f"RSI<{vs['long_rsi_lt']}(15m)")
        if rmi_15m is not None and rmi_15m < vs['long_rmi_lt']: veto_long.append(f"RMI<{vs['long_rmi_lt']}(15m)")
        if mfi_15m is not None and mfi_15m < vs['long_mfi_lt']: veto_long.append(f"MFI<{vs['long_mfi_lt']}(15m)")

    if veto_short:
        penalty += vs['veto_penalty']; flags['veto_short'] = veto_short
    if veto_long:
        penalty += vs['veto_penalty']; flags['veto_long'] = veto_long

    if adx_15m is not None and adx_15m < vs['adx_min']:
        penalty += vs['chop_penalty']
        flags['low_adx'] = [f"ADX<{vs['adx_min']}(15m)"]

    return penalty, flags

def render_notes(symbol: str,
                 oi_buildup_label: Optional[str], oi_strength: Optional[float],
                 veto_flags: Dict[str, Any],
                 trend: float, momentum: float, structure: float) -> str:
    bits: List[str] = []
    if oi_buildup_label:
        bits.append(f"{oi_buildup_label.replace('_',' ')} (strength {oi_strength or 0:.0f})")
    if 'veto_short' in veto_flags:
        bits.append("avoid short: " + ", ".join(veto_flags['veto_short']))
    if 'veto_long' in veto_flags:
        bits.append("avoid long: " + ", ".join(veto_flags['veto_long']))
    if not bits:
        bits.append(f"OK: Trend {trend:.0f}, Momentum {momentum:.0f}, Structure {structure:.0f}")
    return " | ".join(bits)

def get_db():
    return get_db_connection()

def load_config(path: str="indicators.ini") -> Dict[str, Any]:
    cfg = configparser.ConfigParser(); cfg.read(path)

    weights = {
        "trend": float(cfg.get("composite.weights", "trend", fallback="0.25")),
        "momentum": float(cfg.get("composite.weights", "momentum", fallback="0.25")),
        "quality": float(cfg.get("composite.weights", "quality", fallback="0.15")),
        "flow": float(cfg.get("composite.weights", "flow", fallback="0.15")),
        "structure": float(cfg.get("composite.weights", "structure", fallback="0.20")),
        "consensus_bonus_max": float(cfg.get("composite.weights", "consensus_bonus_max", fallback="10")),
    }
    veto = {
        "short_rsi_gt": float(cfg.get("composite.veto","short_rsi_gt",fallback="60")),
        "short_rmi_gt": float(cfg.get("composite.veto","short_rmi_gt",fallback="55")),
        "short_mfi_gt": float(cfg.get("composite.veto","short_mfi_gt",fallback="55")),
        "long_rsi_lt":  float(cfg.get("composite.veto","long_rsi_lt",fallback="40")),
        "long_rmi_lt":  float(cfg.get("composite.veto","long_rmi_lt",fallback="45")),
        "long_mfi_lt":  float(cfg.get("composite.veto","long_mfi_lt",fallback="45")),
        "adx_min":      float(cfg.get("composite.veto","adx_min",fallback="20")),
        "veto_penalty": float(cfg.get("composite.veto","veto_penalty",fallback="10")),
        "chop_penalty": float(cfg.get("composite.veto","chop_penalty",fallback="5")),
    }
    tfs_list = [t.strip() for t in cfg.get("composite.tfs","list",fallback="25m,65m,125m").split(",")]
    w_str = cfg.get("composite.tfs","mtf_weights",fallback="25m:0.3,65m:0.5,125m:0.2")
    mtf_weights: Dict[str, float] = {}
    for kv in w_str.split(","):
        k, v = kv.split(":"); mtf_weights[k.strip()] = float(v)
    general = {"score_cap": float(cfg.get("composite.general","score_cap",fallback="100"))}

    dyn = {
        "enable_dynamic_weights": cfg.getboolean("composite.dynamic","enable_dynamic_weights",fallback=True),
        "regime_tf": cfg.get("composite.dynamic","regime_tf",fallback="125m"),
        "trend_adx_min": float(cfg.get("composite.dynamic","trend_adx_min",fallback="25")),
        "range_adx_max": float(cfg.get("composite.dynamic","range_adx_max",fallback="20")),
        "boost_trend": float(cfg.get("composite.dynamic","boost_trend",fallback="0.05")),
        "boost_momentum": float(cfg.get("composite.dynamic","boost_momentum",fallback="0.05")),
        "boost_structure": float(cfg.get("composite.dynamic","boost_structure",fallback="0.05")),
        "boost_flow": float(cfg.get("composite.dynamic","boost_flow",fallback="0.03")),
    }
    return {"weights":weights, "veto":veto, "tfs":tfs_list, "mtf_weights":mtf_weights, "general":general, "dynamic": dyn}

def fetch_symbols_and_market_types(conn):
    cfg = configparser.ConfigParser(); cfg.read("indicators.ini")
    fut_tbl = cfg.get("sources","futures_table", fallback="market.futures_candles")
    spot_tbl= cfg.get("sources","spot_table",    fallback="market.spot_candles")
    sym_col = cfg.get("sources","symbol_col",    fallback="symbol")
    ts_col  = cfg.get("sources","ts_col",        fallback="ts")

    pairs: List[Tuple[str,str]] = []
    with conn.cursor() as cur:
        cur.execute(f"""
            SELECT DISTINCT {sym_col} AS symbol, 'futures' AS market_type
            FROM {fut_tbl}
            WHERE {ts_col} >= now() - interval '5 days'
        """)
        pairs += [(r[0], r[1]) for r in cur.fetchall()]

    with conn.cursor() as cur:
        cur.execute(f"""
            SELECT DISTINCT {sym_col} AS symbol, 'spot' AS market_type
            FROM {spot_tbl}
            WHERE {ts_col} >= now() - interval '5 days'
        """)
        pairs += [(r[0], r[1]) for r in cur.fetchall()]

    if not pairs:
        raise RuntimeError("No symbols found in spot/futures tables in last 5 days.")
    return pairs

def fetch_indicator_snapshot(conn, symbol: str, market_type: str, tf: str):
    cfg = configparser.ConfigParser(); cfg.read("indicators.ini")
    view  = cfg.get("sources","indicators_view", fallback="indicators.values")
    sym   = cfg.get("sources","symbol_col",    fallback="symbol")
    tscol = cfg.get("sources","ts_col",        fallback="ts")
    tfcol = cfg.get("sources","timeframe_col", fallback="interval")

    with conn.cursor() as cur:
        cur.execute(f"""
            WITH latest AS (
              SELECT MAX({tscol}) AS ts
              FROM {view}
              WHERE {sym}=%s AND market_type=%s AND {tfcol}=%s
            )
            SELECT
              l.ts,
              MAX(val) FILTER (WHERE metric='CLOSE')       AS close,
              MAX(val) FILTER (WHERE metric='EMA.5')       AS ema5,
              MAX(val) FILTER (WHERE metric='EMA.10')      AS ema10,
              MAX(val) FILTER (WHERE metric='EMA.20')      AS ema20,
              MAX(val) FILTER (WHERE metric='EMA.25')      AS ema25,
              MAX(val) FILTER (WHERE metric='RSI')         AS rsi,
              MAX(val) FILTER (WHERE metric='RMI')         AS rmi,
              MAX(val) FILTER (WHERE metric='MACD.line')   AS macd_line,
              MAX(val) FILTER (WHERE metric='MACD.signal') AS macd_signal,
              MAX(val) FILTER (WHERE metric='MACD.hist')   AS macd_hist,
              MAX(val) FILTER (WHERE metric='ADX')         AS adx,
              MAX(val) FILTER (WHERE metric='ATR')         AS atr_abs,
              MAX(val) FILTER (WHERE metric='MFI')         AS mfi,
              MAX(val) FILTER (WHERE metric='OBV.delta')   AS obv_delta_now,
              MAX(val) FILTER (WHERE metric='BB.in_zone')  AS bb_in_zone,
              MAX(val) FILTER (WHERE metric='BB.score')    AS bb_score
            FROM latest l
            JOIN {view} v
              ON v.{sym}=%s AND v.market_type=%s AND v.{tfcol}=%s AND v.{tscol}=l.ts
            GROUP BY l.ts
        """, (symbol, market_type, tf, symbol, market_type, tf))
        row = cur.fetchone()
        if not row:
            return None

    (ts, close, ema5, ema10, ema20, ema25, rsi, rmi,
     macd_line, macd_signal, macd_hist, adx, atr_abs, mfi, obv_delta,
     bb_in_zone, bb_score) = row

    atr_pct = (atr_abs / close) if (close not in (None, 0)) and (atr_abs is not None) else None

    return {
        "ts": ts,
        "close": close,
        "ema": {"5": ema5, "10": ema10, "20": ema20, "25": ema25},
        "ema20_prev3": ema20,  # placeholder
        "rsi": rsi, "rmi": rmi,
        "macd_line": macd_line, "macd_signal": macd_signal,
        "macd_hist": macd_hist, "macd_hist_prev": macd_hist,
        "roc_now": None, "roc_series": None,
        "adx": adx,
        "atr_pct": atr_pct,
        "mfi": mfi,
        "obv_delta_now": obv_delta,
        "obv_delta_series": None,
        "vp": {"poc": None, "val": None, "vah": None},
        "bb": {"in_zone": bool(bb_in_zone) if bb_in_zone is not None else False, "score": bb_score},
        "rsi_15m": None, "rmi_15m": None, "mfi_15m": None, "adx_15m": None
    }

def fetch_oi_row(conn, symbol: str, tf: str):
    cfg = configparser.ConfigParser(); cfg.read("indicators.ini")
    view   = cfg.get("sources","indicators_view",  fallback="indicators.values")
    symcol = cfg.get("sources","symbol_col",       fallback="symbol")
    tscol  = cfg.get("sources","ts_col",           fallback="ts")

    with conn.cursor() as cur:
        cur.execute(f"""
            SELECT val FROM {view}
            WHERE {symcol}=%s AND source='conf_oi'
              AND (metric = %s OR metric = %s)
            ORDER BY {tscol} DESC LIMIT 1
        """, (symbol, f"CONF_OI.score.{tf}", f"conf_oi.score.{tf}"))
        r = cur.fetchone()
        if r:
            score = float(r[0])
        else:
            cur.execute(f"""
                SELECT val FROM {view}
                WHERE {symcol}=%s AND source='conf_oi'
                  AND (metric = %s OR metric = %s)
                ORDER BY {tscol} DESC LIMIT 1
            """, (symbol, f"CONF_OI.prob.{tf}", f"conf_oi.prob.{tf}"))
            r = cur.fetchone()
            if not r:
                return None
            score = float(r[0]) * 100.0

        cur.execute(f"""
            SELECT context FROM {view}
            WHERE {symcol}=%s AND source='conf_oi' AND metric='OI.buildup_label'
            ORDER BY {tscol} DESC LIMIT 1
        """, (symbol,))
        label = None
        rr = cur.fetchone()
        if rr and rr[0]:
            try:
                c = rr[0] if isinstance(rr[0], dict) else json.loads(rr[0])
                label = c.get("label")
            except Exception:
                pass

        cur.execute(f"""
            SELECT val FROM {view}
            WHERE {symcol}=%s AND source='conf_oi' AND metric='OI.buildup_strength'
            ORDER BY {tscol} DESC LIMIT 1
        """, (symbol,))
        rr = cur.fetchone()
        strength = float(rr[0]) if rr else None

        return {"score_final": score, "buildup_label": label, "buildup_strength": strength}

    return None

def upsert_composite(conn, row: Dict[str, Any]) -> None:
    sql = """
    INSERT INTO signals.confidence_composite
    (symbol, market_type, tf, ts,
     trend, momentum, quality, flow, structure,
     consensus_bonus, final_score,
     oi_buildup_label, oi_buildup_strength,
     consensus_flags, veto_flags, notes,
     run_id, source)
    VALUES
    (%(symbol)s, %(market_type)s, %(tf)s, %(ts)s,
     %(trend)s, %(momentum)s, %(quality)s, %(flow)s, %(structure)s,
     %(consensus_bonus)s, %(final_score)s,
     %(oi_buildup_label)s, %(oi_buildup_strength)s,
     %(consensus_flags)s, %(veto_flags)s, %(notes)s,
     %(run_id)s, %(source)s)
    ON CONFLICT (symbol, market_type, tf, ts, source)
    DO UPDATE SET
      trend=EXCLUDED.trend,
      momentum=EXCLUDED.momentum,
      quality=EXCLUDED.quality,
      flow=EXCLUDED.flow,
      structure=EXCLUDED.structure,
      consensus_bonus=EXCLUDED.consensus_bonus,
      final_score=EXCLUDED.final_score,
      oi_buildup_label=EXCLUDED.oi_buildup_label,
      oi_buildup_strength=EXCLUDED.oi_buildup_strength,
      consensus_flags=EXCLUDED.consensus_flags,
      veto_flags=EXCLUDED.veto_flags,
      notes=EXCLUDED.notes,
      run_id=EXCLUDED.run_id,
      computed_at=now()
    """
    with conn.cursor() as cur:
        cur.execute(sql, row)

def upsert_subscores(conn, row: Dict[str, Any]) -> None:
    """Populate signals.confidence_subscores too."""
    sql = """
    INSERT INTO signals.confidence_subscores
    (symbol, market_type, tf, ts,
     trend, momentum, quality, flow, structure,
     oi_buildup_label, oi_buildup_strength,
     details, run_id, source)
    VALUES
    (%(symbol)s, %(market_type)s, %(tf)s, %(ts)s,
     %(trend)s, %(momentum)s, %(quality)s, %(flow)s, %(structure)s,
     %(oi_buildup_label)s, %(oi_buildup_strength)s,
     %(details)s, %(run_id)s, %(source)s)
    ON CONFLICT (symbol, market_type, tf, ts, source)
    DO UPDATE SET
      trend=EXCLUDED.trend,
      momentum=EXCLUDED.momentum,
      quality=EXCLUDED.quality,
      flow=EXCLUDED.flow,
      structure=EXCLUDED.structure,
      oi_buildup_label=EXCLUDED.oi_buildup_label,
      oi_buildup_strength=EXCLUDED.oi_buildup_strength,
      details=EXCLUDED.details,
      run_id=EXCLUDED.run_id,
      computed_at=now()
    """
    with conn.cursor() as cur:
        cur.execute(sql, row)


# Fusion helper
def fuse_confidence_score(spot_score, fut_score, oi_score, buildup_label=None):
    weights = {"spot_comp": 0.4, "fut_comp": 0.4, "oi_score": 0.2}
    score_inputs = {"spot_comp": spot_score, "fut_comp": fut_score, "oi_score": oi_score}
    available = {k: v for k, v in score_inputs.items() if v is not None}
    total_weight = sum(weights[k] for k in available)
    if not available or total_weight == 0:
        return None, []
    final_score = sum((v * (weights[k] / total_weight)) for k, v in available.items())
    veto_flags = []
    if buildup_label in ("short buildup", "long unwinding"):
        if spot_score and spot_score > 60 and fut_score and fut_score > 60:
            veto_flags.append(f"OI says {buildup_label}")
    return round(final_score, 2), veto_flags


# ---------- Dynamic pillar weights ----------
def dynamic_pillar_weights(base: Dict[str, float], regime: str, dyn_cfg: Dict[str, Any]) -> Dict[str, float]:
    w = dict(base)
    if not dyn_cfg.get("enable_dynamic_weights", True):
        return w
    bt = dyn_cfg["boost_trend"]; bm = dyn_cfg["boost_momentum"]
    bs = dyn_cfg["boost_structure"]; bf = dyn_cfg["boost_flow"]
    if regime == "trend":
        w["trend"]     += bt
        w["momentum"]  += bm
        w["structure"]  = max(0.0, w["structure"] - bs)
    elif regime == "range":
        w["structure"] += bs
        w["flow"]      += bf
        w["trend"]      = max(0.0, w["trend"] - bt)
        w["momentum"]   = max(0.0, w["momentum"] - bm)
    s = sum(w.values()) or 1.0
    for k in w: w[k] = w[k] / s
    return w

def detect_regime_for_symbol(tf_snaps: Dict[str, Dict[str, Any]], dyn_cfg: Dict[str, Any]) -> str:
    pref = dyn_cfg.get("regime_tf", "125m")
    adx = tf_snaps.get(pref, {}).get("adx")
    if adx is None and tf_snaps:
        k = sorted(tf_snaps.keys(), key=lambda x: (len(x), x))[-1]
        adx = tf_snaps[k].get("adx")
    if adx is None: return "neutral"
    if adx >= dyn_cfg.get("trend_adx_min", 25.0): return "trend"
    if adx <= dyn_cfg.get("range_adx_max", 20.0): return "range"
    return "neutral"


# ---------- Main compute ----------
def compute_for_tf(symbol: str, market_type: str, tf: str,
                   snap: Dict[str, Any],
                   oi_row: Optional[Dict[str, Any]],
                   weights: Dict[str, float],
                   veto_cfg: Dict[str, float],
                   score_cap: float,
                   consensus_ctx: Dict[str, Tuple[Optional[float], Optional[float], Optional[float], Optional[float], Optional[bool]]],
                   dyn_weights_regime: Optional[str],
                   dyn_cfg: Dict[str, Any]) -> Dict[str, Any]:

    eff_weights = dynamic_pillar_weights(weights, dyn_weights_regime or "neutral", dyn_cfg)

    trend_sc  = score_trend(snap.get('ema', {}), snap.get('close'), snap.get('ema20_prev3'))
    mo_sc     = score_momentum(
                    snap.get('rsi'),
                    snap.get('macd_line'), snap.get('macd_signal'),
                    snap.get('macd_hist'), snap.get('macd_hist_prev'),
                    snap.get('roc_series'), snap.get('roc_now'),
                    snap.get('adx'))
    qual_sc   = score_quality(snap.get('adx'), snap.get('atr_pct'))
    flow_sc   = score_flow(market_type,
                           (oi_row.get('score_final') if oi_row else None),
                           snap.get('mfi'),
                           snap.get('obv_delta_series'), snap.get('obv_delta_now'))
    struct_sc = score_structure(bool(snap.get('bb', {}).get('in_zone')),
                                snap.get('bb', {}).get('score'),
                                snap.get('close'),
                                snap.get('vp', {}).get('poc'),
                                snap.get('vp', {}).get('val'),
                                snap.get('vp', {}).get('vah'),
                                trend_sc)

    base = (eff_weights['trend']*trend_sc + eff_weights['momentum']*mo_sc +
            eff_weights['quality']*qual_sc + eff_weights['flow']*flow_sc +
            eff_weights['structure']*struct_sc)

    consensus_ctx[tf] = (trend_sc, mo_sc, snap.get('adx'), snap.get('rsi'), snap.get('bb', {}).get('in_zone'))

    veto_adj = adjust_veto_thresholds(veto_cfg, snap.get('atr_pct'))
    penalty, veto_flags = compute_veto(
        (oi_row.get('buildup_label') if oi_row else None),
        snap.get('rsi_15m'), snap.get('rmi_15m'), snap.get('mfi_15m'), snap.get('adx_15m'),
        veto_adj
    )

    final_score = clamp(base - penalty, 0.0, score_cap)

    notes = render_notes(
        symbol,
        (oi_row.get('buildup_label') if oi_row else None),
        (oi_row.get('buildup_strength') if oi_row else None),
        veto_flags, trend_sc, mo_sc, struct_sc
    )

    return {
        "trend": round(trend_sc, 2),
        "momentum": round(mo_sc, 2),
        "quality": round(qual_sc, 2),
        "flow": round(flow_sc, 2),
        "structure": round(struct_sc, 2),
        "consensus_bonus": 0.0,
        "final_score": round(final_score, 2),
        "veto_flags": json.dumps(veto_flags),
        "notes": notes
    }

def _filter_pairs(pairs: Iterable[Tuple[str,str]], symbols: Optional[List[str]]) -> List[Tuple[str,str]]:
    if not symbols:
        return list(pairs)
    want = set(s.upper() for s in symbols)
    return [(s, mt) for (s, mt) in pairs if s.upper() in want]

def run(symbols: Optional[List[str]] = None):
    """Accepts optional symbols list (fixes worker crash)."""
    print("CompositeWriter build: r5_dyn_regime")
    cfg = load_config()
    weights, veto_cfg, tfs, score_cap = cfg['weights'], cfg['veto'], cfg['tfs'], cfg['general']['score_cap']
    dyn_cfg = cfg['dynamic']

    conn = get_db()
    try:
        pairs = fetch_symbols_and_market_types(conn)
        pairs = _filter_pairs(pairs, symbols)

        run_id = f"conf_comp_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"

        for symbol, market_type in pairs:
            # Phase 1: fetch all TF snapshots up-front to determine regime
            tf_snaps: Dict[str, Dict[str, Any]] = {}
            for tf in tfs:
                snap = fetch_indicator_snapshot(conn, symbol, market_type, tf)
                if snap and snap.get("ts"):
                    tf_snaps[tf] = snap
            if not tf_snaps:
                continue

            regime = detect_regime_for_symbol(tf_snaps, dyn_cfg)

            # Phase 2: score per TF
            tf_results: Dict[Tuple[str, str], Dict[str, Any]] = {}
            consensus_ctx: Dict[str, Tuple[Optional[float], Optional[float], Optional[float], Optional[float], Optional[bool]]] = {}

            for tf, snap in tf_snaps.items():
                oi_row = fetch_oi_row(conn, symbol, tf) if market_type == "futures" else None
                res = compute_for_tf(symbol, market_type, tf, snap, oi_row,
                                     weights, veto_cfg, score_cap,
                                     consensus_ctx, regime, dyn_cfg)
                tf_results[(market_type, tf)] = {"ts": snap["ts"], "res": res, "oi_row": oi_row}

                # write subscores table
                upsert_subscores(conn, {
                    "symbol": symbol,
                    "market_type": market_type,
                    "tf": tf,
                    "ts": snap["ts"],
                    "trend": res["trend"],
                    "momentum": res["momentum"],
                    "quality": res["quality"],
                    "flow": res["flow"],
                    "structure": res["structure"],
                    "oi_buildup_label": (oi_row.get("buildup_label") if oi_row else None),
                    "oi_buildup_strength": (oi_row.get("buildup_strength") if oi_row else None),
                    "details": json.dumps({"regime": regime}),
                    "run_id": run_id,
                    "source": "conf_composite"
                })

                # write composite (initial, before consensus)
                upsert_composite(conn, {
                    "symbol": symbol,
                    "market_type": market_type,
                    "tf": tf,
                    "ts": snap["ts"],
                    **res,
                    "oi_buildup_label": (oi_row.get("buildup_label") if oi_row else None),
                    "oi_buildup_strength": (oi_row.get("buildup_strength") if oi_row else None),
                    "consensus_flags": "{}",
                    "run_id": run_id,
                    "source": "conf_composite"
                })

            # Phase 3: apply consensus bonus
            bonus, flags = consensus_bonus_across_tfs(consensus_ctx, weights["consensus_bonus_max"])
            for tf in tf_snaps.keys():
                key = (market_type, tf)
                if key not in tf_results:
                    continue
                r = tf_results[key]["res"]
                new_score = clamp(r["final_score"] + bonus, 0.0, score_cap)
                r["consensus_bonus"] = round(bonus, 2)
                r["final_score"] = round(new_score, 2)

                upsert_composite(conn, {
                    "symbol": symbol,
                    "market_type": market_type,
                    "tf": tf,
                    "ts": tf_results[key]["ts"],
                    **r,
                    "oi_buildup_label": (tf_results[key]["oi_row"].get("buildup_label") if tf_results[key]["oi_row"] else None),
                    "oi_buildup_strength": (tf_results[key]["oi_row"].get("buildup_strength") if tf_results[key]["oi_row"] else None),
                    "consensus_flags": json.dumps(flags),
                    "run_id": run_id,
                    "source": "conf_composite"
                })

            # ---- Fusion across spot + futures ----
            for tf in tfs:
                spot_key, fut_key = ("spot", tf), ("futures", tf)
                if spot_key not in tf_results or fut_key not in tf_results:
                    continue

                spot_score = tf_results[spot_key]["res"]["final_score"]
                fut_score  = tf_results[fut_key]["res"]["final_score"]
                oi_row     = tf_results[fut_key]["oi_row"]

                oi_score = oi_row.get("score_final") if oi_row else None
                buildup_label = oi_row.get("buildup_label") if oi_row else None

                fused_score, fuse_veto_flags = fuse_confidence_score(spot_score, fut_score, oi_score, buildup_label)

                if fused_score is not None:
                    upsert_composite(conn, {
                        "symbol": symbol,
                        "market_type": "fused",
                        "tf": tf,
                        "ts": tf_results[fut_key]["ts"],
                        "trend": None, "momentum": None, "quality": None, "flow": None, "structure": None,
                        "consensus_bonus": 0.0,
                        "final_score": fused_score,
                        "oi_buildup_label": buildup_label,
                        "oi_buildup_strength": (oi_row.get("buildup_strength") if oi_row else None),
                        "consensus_flags": "{}",
                        "veto_flags": json.dumps(fuse_veto_flags),
                        "notes": f"Fusion spot+fut+OI → {fused_score}",
                        "run_id": run_id,
                        "source": "conf_fusion_v1"
                    })

        conn.commit()
        print("✅ Composite + Fusion confidence updated.")
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

# Optional alias so worker can call this if present
def run_for_symbols(symbols: List[str]):
    return run(symbols=symbols)

if __name__ == "__main__":
    run()
