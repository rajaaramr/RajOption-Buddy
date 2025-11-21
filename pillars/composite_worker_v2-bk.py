# composite_worker_v2.py
from __future__ import annotations
import os, json, math, configparser
from typing import Dict, List, Optional, Iterable
from datetime import datetime, timezone
import numpy as np
import psycopg2.extras as pgx

from utils.db import get_db_connection
from pillars.common import (
    TZ, DEFAULT_INI, load_base_cfg, load_5m, resample, write_values,
    last_metric, clamp, now_ts, maybe_trim_last_bar, ensure_min_bars
)
from pillars.trend_pillar import score_trend
from pillars.momentum_pillar import score_momentum
from pillars.quality_pillar import score_quality
from pillars.flow_pillar import score_flow
from pillars.risk_pillar import score_risk

# If structure_pillar.py lives in another package (e.g., scheduler.structure_pillar),
# adjust this import accordingly.
from pillars.structure_pillar import process_symbol as run_structure_for_symbol

DEFAULT_COMP_INI = os.getenv("COMPOSITE_V2_INI", "composite_v2.ini")

# ===== DB writer for composite table =====
def _insert_composite(rows: List[tuple]) -> int:
    if not rows: return 0
    sql = """
        INSERT INTO indicators.composite_v2
            (symbol, market_type, interval, ts,
             trend, momentum, quality, flow, structure, risk,
             quality_veto, flow_veto, structure_veto, risk_veto,
             final_score, final_prob, context, run_id, source)
        VALUES %s
        ON CONFLICT (symbol, market_type, interval, ts) DO UPDATE
           SET trend=EXCLUDED.trend, momentum=EXCLUDED.momentum, quality=EXCLUDED.quality,
               flow=EXCLUDED.flow, structure=EXCLUDED.structure, risk=EXCLUDED.risk,
               quality_veto=EXCLUDED.quality_veto, flow_veto=EXCLUDED.flow_veto,
               structure_veto=EXCLUDED.structure_veto, risk_veto=EXCLUDED.risk_veto,
               final_score=EXCLUDED.final_score, final_prob=EXCLUDED.final_prob,
               context=EXCLUDED.context, run_id=EXCLUDED.run_id, source=EXCLUDED.source
    """
    with get_db_connection() as conn, conn.cursor() as cur:
        pgx.execute_values(cur, sql, rows, page_size=1000)
        conn.commit()
        return len(rows)

# ===== INI loader =====
def _load_comp_cfg(path: str = DEFAULT_COMP_INI) -> dict:
    cp = configparser.ConfigParser(inline_comment_prefixes=(";","#"), interpolation=None, strict=False)
    cp.read(path)
    w = {
        "trend":     cp.getfloat("composite","w_trend",     fallback=0.22),
        "momentum":  cp.getfloat("composite","w_momentum",  fallback=0.22),
        "quality":   cp.getfloat("composite","w_quality",   fallback=0.18),
        "flow":      cp.getfloat("composite","w_flow",      fallback=0.14),
        "structure": cp.getfloat("composite","w_structure", fallback=0.14),
        "risk":      cp.getfloat("composite","w_risk",      fallback=0.10),
    }
    s = sum(w.values()) or 1.0
    w = {k: v/s for k,v in w.items()}

    tfw_raw = cp.get("composite","tf_weights",fallback="25m:0.3,65m:0.3,125m:0.4")
    tfw: Dict[str,float] = {}
    for part in tfw_raw.split(","):
        if ":" in part:
            k,v = part.split(":",1)
            try: tfw[k.strip()] = float(v.strip())
            except: pass
    tfw_sum = sum(tfw.values()) or 1.0
    tfw = {k: v/tfw_sum for k,v in tfw.items()}
    return {"pillar_w": w, "tf_w": tfw}

# ===== Helpers =====
def _score_to_prob(s: float)->float:
    z = (s - 50.0) / 12.0
    p = 1.0 / (1.0 + math.exp(-z))
    return float(clamp(p, 0.0, 1.0))

def _blend_mtf(values: Dict[str, float], weights: Dict[str,float]) -> Optional[float]:
    # normalize over present TFs only
    num = den = 0.0
    for tf, v in values.items():
        if v is None: 
            continue
        w = float(weights.get(tf, 0.0))
        if w <= 0:
            continue
        num += w * float(v); den += w
    return (num/den) if den > 0 else None

# ===== Core =====
def process_symbol(symbol: str, *, kind: str,
                   pillars_ini: str = DEFAULT_INI,
                   comp_ini: str = DEFAULT_COMP_INI) -> int:
    base = load_base_cfg(pillars_ini)
    cfg  = _load_comp_cfg(comp_ini)

    # 1) Load 5m once
    df5 = load_5m(symbol, kind, base.lookback_days)
    if df5.empty:
        print(f"⚠️ {kind}:{symbol} no candles")
        return 0

    # 2) Trigger STRUCTURE (it writes its own per-TF and MTF metrics)
    try:
        run_structure_for_symbol(symbol, kind=kind)
    except Exception as e:
        print(f"⚠️ STRUCT compute failed for {kind}:{symbol}: {e}")

    # 3) Run pillars per TF; collect for MTF blends
    per_tf = {"TREND": {}, "MOMENTUM": {}, "QUALITY": {}, "FLOW": {}, "RISK": {}}
    veto_tf = {"QUALITY": {}, "FLOW": {}, "RISK": {}}

    for tf in base.tfs:
        dftf = resample(df5, tf)
        dftf = maybe_trim_last_bar(dftf)
        if not ensure_min_bars(dftf, tf):
            # <<< FIX: skip this TF instead of returning None >>>
            continue

        # Trend
        try:
            t_res = score_trend(symbol, kind, tf, df5, base)
            if t_res:
                _, t_score = t_res
                per_tf["TREND"][tf] = float(t_score)
        except Exception as e:
            print(f"⚠️ TREND {kind}:{symbol}:{tf} → {e}")

        # Momentum
        try:
            m_res = score_momentum(symbol, kind, tf, df5, base)
            if m_res:
                _, m_score = m_res
                per_tf["MOMENTUM"][tf] = float(m_score)
        except Exception as e:
            print(f"⚠️ MOMENTUM {kind}:{symbol}:{tf} → {e}")

        # Quality
        try:
            q_res = score_quality(symbol, kind, tf, df5, base)
            if q_res:
                _, q_score, q_veto, _rev = q_res
                per_tf["QUALITY"][tf] = float(q_score)
                veto_tf["QUALITY"][tf] = bool(q_veto)
        except Exception as e:
            print(f"⚠️ QUALITY {kind}:{symbol}:{tf} → {e}")

        # Flow
        try:
            f_res = score_flow(symbol, kind, tf, df5, base)
            if f_res:
                _, f_score, f_veto = f_res
                per_tf["FLOW"][tf] = float(f_score)
                veto_tf["FLOW"][tf] = bool(f_veto)
        except Exception as e:
            print(f"⚠️ FLOW {kind}:{symbol}:{tf} → {e}")

        # Risk
        try:
            r_res = score_risk(symbol, kind, tf, df5, base)
            if r_res:
                _, r_score, r_veto = r_res
                per_tf["RISK"][tf] = float(r_score)
                veto_tf["RISK"][tf] = bool(r_veto)
        except Exception as e:
            print(f"⚠️ RISK {kind}:{symbol}:{tf} → {e}")

    # 4) MTF per pillar (in-memory). STRUCTURE MTF read from DB.
    Wtf = cfg["tf_w"]
    mtf_trend    = _blend_mtf(per_tf["TREND"],    Wtf)
    mtf_momentum = _blend_mtf(per_tf["MOMENTUM"], Wtf)
    mtf_quality  = _blend_mtf(per_tf["QUALITY"],  Wtf)
    mtf_flow     = _blend_mtf(per_tf["FLOW"],     Wtf)
    mtf_risk     = _blend_mtf(per_tf["RISK"],     Wtf)

    # STRUCTURE MTF (already written by structure_pillar)
    struct_mtf  = last_metric(symbol, kind, "MTF", "STRUCT.score")
    struct_veto = last_metric(symbol, kind, "MTF", "STRUCT.veto_flag")
    mtf_structure  = float(struct_mtf) if struct_mtf is not None else None
    veto_structure = bool((struct_veto or 0.0) >= 0.5)

    # Veto OR across TFs
    veto_quality = any(veto_tf["QUALITY"].values()) if veto_tf["QUALITY"] else False
    veto_flow    = any(veto_tf["FLOW"].values())    if veto_tf["FLOW"]    else False
    veto_risk    = any(veto_tf["RISK"].values())    if veto_tf["RISK"]    else False

    # Neutral 50 for missing MTFs
    def nz(x): return 50.0 if x is None else float(x)

    w = cfg["pillar_w"]
    final_score = clamp(
        w["trend"]    * nz(mtf_trend) +
        w["momentum"] * nz(mtf_momentum) +
        w["quality"]  * nz(mtf_quality) +
        w["flow"]     * nz(mtf_flow) +
        w["structure"]* nz(mtf_structure) +
        w["risk"]     * nz(mtf_risk),
        0, 100
    )
    hard_veto = bool(veto_quality or veto_flow or veto_structure or veto_risk)
    if hard_veto:
        final_score = 0.0
    final_prob = _score_to_prob(final_score)

    ts = now_ts()
    ctx = {
        "pillar_w": w,
        "tf_w": Wtf,
        "per_tf": per_tf,
        "vetos": {
            "quality": bool(veto_quality),
            "flow": bool(veto_flow),
            "structure": bool(veto_structure),
            "risk": bool(veto_risk),
        }
    }

    rows = [(
        symbol, kind, "MTF", ts,
        float(nz(mtf_trend)), float(nz(mtf_momentum)), float(nz(mtf_quality)),
        float(nz(mtf_flow)), float(nz(mtf_structure)), float(nz(mtf_risk)),
        bool(veto_quality), bool(veto_flow), bool(veto_structure), bool(veto_risk),
        float(final_score), float(final_prob), json.dumps(ctx),
        os.getenv("RUN_ID","comp_v2"), os.getenv("SRC","composite_v2")
    )]
    n = _insert_composite(rows) if rows else 0
    print(f"✅ COMP2 {kind}:{symbol} → score={final_score:.1f} prob={final_prob:.2f} veto={hard_veto}")
    return int(n)

# ===== Runner =====
def run(symbols: Optional[List[str]] = None, kinds: Iterable[str] = ("futures","spot")) -> int:
    # discover symbols from active webhooks
    if symbols is None:
        with get_db_connection() as conn, conn.cursor() as cur:
            cur.execute("""
                SELECT DISTINCT symbol
                  FROM webhooks.webhook_alerts
                 WHERE status IN ('INDICATOR_PROCESS','SIGNAL_PROCESS','DATA_PROCESSING')
            """)
            rows = cur.fetchall()
        symbols = [r[0] for r in rows or []]

    total = 0
    for s in symbols:
        for k in kinds:
            try:
                total += process_symbol(s, kind=k)
            except Exception as e:
                print(f"❌ COMP2 {k}:{s} → {e}")
    print(f"🎯 COMP2 wrote {total} row(s) across {len(symbols)} symbol(s)")
    return total

if __name__ == "__main__":
    import sys
    args = [a for a in sys.argv[1:]]
    syms = [a for a in args if not a.startswith("--")]
    ks = [a.split("=",1)[1] for a in args if a.startswith("--kinds=")]
    kinds = tuple(ks[0].split(",")) if ks else ("futures","spot")
    run(syms or None, kinds=kinds)
