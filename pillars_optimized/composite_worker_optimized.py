# scheduler/composite_worker_optimized.py
from __future__ import annotations
import os, json, math, traceback, configparser, concurrent.futures
from typing import Dict, List, Optional, Iterable, Tuple, Any
from datetime import datetime, timezone, timedelta
import pandas as pd
import psycopg2.extras as pgx
from utils.db import get_db_connection

TZ = timezone.utc

# --- Config Loader ---
try:
    from utils.configs import get_config_parser
except Exception:
    def get_config_parser():
        cp = configparser.ConfigParser()
        if os.path.exists("configs/data.ini"):
            cp.read("configs/data.ini")
        return cp

CFG = get_config_parser()

BASE_INTERVAL = "15m"
TF_SET = ["15m","30m","60m","120m","240m"]
UNIVERSE_NAME = os.getenv("UNIVERSE_NAME", CFG.get("universe","name",fallback="largecaps_v1"))
COMP_LOOKBACK_DAYS = int(os.getenv("COMP_LOOKBACK_DAYS", CFG.get("composite","lookback_days_15m",fallback="5")))
RUN_ID = os.getenv("RUN_ID", datetime.now(TZ).strftime("%Y%m%dT%H%M%SZ_comp"))
DEFAULT_COMP_INI = "pillars_optimized/composite_optimized.ini"
MAX_WORKERS = int(os.getenv("COMP_WORKERS", "4")) # Parallel workers

# --- Pillar Imports ---
from pillars_optimized.common import clamp, now_ts, ensure_min_bars
from pillars_optimized.trend_pillar_optimized import score_trend
from pillars_optimized.momentum_pillar_optimized import score_momentum
from pillars_optimized.quality_pillar_optimized import score_quality
from pillars_optimized.flow_pillar_optimized import score_flow
from pillars_optimized.risk_pillar_optimized import score_risk
from pillars_optimized.structure_pillar_optimized import score_structure
from pillars_optimized.ml_pillar_optimized import score_ml # Placeholder
# from pillars_optimized.confidence_pillar_optimized import score_confidence # Placeholder
# from pillars_optimized.strategy_selector_optimized import select_strategy

def _table_name(kind:str)->str:
    return "market.spot_candles" if kind=="spot" else "market.futures_candles"

def _score_to_prob(s: float) -> float:
    z = (s - 50.0) / 12.0
    p = 1.0 / (1.0 + math.exp(-z))
    return float(clamp(p, 0.0, 1.0))

def _load_comp_cfg(path: str=DEFAULT_COMP_INI)->dict:
    cp = configparser.ConfigParser(inline_comment_prefixes=(";","#"), interpolation=None, strict=False)
    cp.read(path)
    w = {
        "trend":cp.getfloat("composite.weights","trend",fallback=0.20),
        "momentum":cp.getfloat("composite.weights","momentum",fallback=0.20),
        "quality":cp.getfloat("composite.weights","quality",fallback=0.10),
        "flow":cp.getfloat("composite.weights","flow",fallback=0.10),
        "structure":cp.getfloat("composite.weights","structure",fallback=0.10),
        "risk":cp.getfloat("composite.weights","risk",fallback=0.10),
        "ml":cp.getfloat("composite.weights","ml",fallback=0.10),
        "confidence":cp.getfloat("composite.weights","confidence",fallback=0.10),
    }
    s = sum(w.values()) or 1.0
    w = {k:v/s for k,v in w.items()}

    tfw_raw = cp.get("composite.tfs","mtf_weights",fallback="15m:0.1,30m:0.3,60m:0.3,90m:0.2,120m:0.1")
    tfw: Dict[str,float] = {}
    for part in tfw_raw.split(","):
        if ":" in part:
            k,v = part.split(":",1)
            try: tfw[k.strip()] = float(v.strip())
            except: pass

    return {"pillar_w":w, "tf_w":tfw}

def _load_intra(symbol:str, kind:str, interval:str, lookback_days:int)->pd.DataFrame:
    tbl = _table_name(kind)
    cutoff = datetime.now(TZ) - timedelta(days=lookback_days)
    with get_db_connection() as conn, conn.cursor() as cur:
        cur.execute(
            f"""SELECT ts,(open)::float8,(high)::float8,(low)::float8,(close)::float8,COALESCE(volume,0)::float8
                  FROM {tbl}
                 WHERE symbol=%s AND interval=%s AND ts >= %s
                 ORDER BY ts ASC""",
            (symbol, interval, cutoff),
        )
        rows = cur.fetchall()
    if not rows:
        return pd.DataFrame(columns=["open","high","low","close","volume"])
    df = pd.DataFrame(rows, columns=["ts","open","high","low","close","volume"])
    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    df = df.dropna(subset=["ts"]).set_index("ts").sort_index()
    for c in ("open","high","low","close","volume"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def _load_all_tfs(symbol:str, kind:str, lookback_days:int)->Dict[str,pd.DataFrame]:
    df15 = _load_intra(symbol, kind, "15m", lookback_days)
    out: Dict[str,pd.DataFrame] = {"15m": df15}
    if df15.empty: return out

    def _agg(df:pd.DataFrame, minutes:int)->pd.DataFrame:
        rule = f"{minutes}T"
        return df.resample(rule, label="right", closed="right").agg({
            "open":"first","high":"max","low":"min","close":"last","volume":"sum"
        }).dropna()

    for tf, mins in {"30m":30,"60m":60,"120m":120,"240m":240}.items():
        out[tf] = _agg(df15, mins)
    return out

def _fetch_context(symbol: str, kind: str) -> Dict[str, Any]:
    """
    Optimization: Fetch Pivots, VP, and other anchors ONCE per symbol.
    This prevents N+1 queries inside the pillars.
    """
    table = "indicators.futures_frames" if kind == "futures" else "indicators.spot_frames"

    cols = [
        "pivot_p", "pivot_r1", "pivot_s1", "pivot_r2", "pivot_s2",
        "vp_poc", "vp_vah", "vp_val", "bb_score"
    ]

    with get_db_connection() as conn, conn.cursor() as cur:
        cur.execute("""
            SELECT column_name FROM information_schema.columns
            WHERE table_name = %s AND column_name = ANY(%s)
        """, (table.split('.')[-1], cols))
        existing = {row[0] for row in cur.fetchall()}

    if not existing: return {}

    sel_cols = ", ".join(existing)

    with get_db_connection() as conn, conn.cursor() as cur:
        cur.execute(f"""
            SELECT {sel_cols} FROM {table}
            WHERE symbol=%s AND interval='15m'
            ORDER BY ts DESC LIMIT 1
        """, (symbol,))
        row = cur.fetchone()

    if not row: return {}

    ctx = {}
    col_list = list(existing)
    for i, col in enumerate(col_list):
        val = row[i]
        if val is not None:
            if col == "vp_poc": ctx["VP.POC"] = float(val)
            elif col == "vp_vah": ctx["VP.VAH"] = float(val)
            elif col == "vp_val": ctx["VP.VAL"] = float(val)
            elif col == "bb_score": ctx["BB.score"] = float(val)
            else: ctx[col] = float(val)

    # Fetch last 3 days of IV
    with get_db_connection() as conn:
        df_iv = pd.read_sql("""
            SELECT trade_date, AVG(iv) as avg_iv
            FROM raw_ingest.daily_options
            WHERE symbol = %s AND iv IS NOT NULL
            GROUP BY trade_date
            ORDER BY trade_date DESC
            LIMIT 3
        """, conn, params=(symbol,))
        ctx["daily_iv"] = df_iv.to_dict('records')

    # Fetch confidence state
    with get_db_connection() as conn:
        df_conf = pd.read_sql("""
            SELECT *
            FROM indicators.confidence_state
            WHERE symbol = %s AND interval = '15m'
            ORDER BY ts DESC
            LIMIT 1
        """, conn, params=(symbol,))
        if not df_conf.empty:
            ctx["confidence"] = df_conf.iloc[0].to_dict()

    # --- Fetch additional data for Flow Pillar ---

    # Fetch latest futures buildup
    with get_db_connection() as conn:
        df_fut = pd.read_sql("""
            SELECT buildup
            FROM raw_ingest.daily_futures
            WHERE symbol = %s
            ORDER BY trade_date DESC, expiry DESC
            LIMIT 1
        """, conn, params=(symbol,))
        if not df_fut.empty:
            ctx["daily_futures_buildup"] = df_fut.iloc[0]['buildup']

    # Fetch aggregated options OI change percentage
    with get_db_connection() as conn:
        df_opts = pd.read_sql("""
            SELECT option_type, AVG(oi_change_pct) as avg_oi_change_pct
            FROM raw_ingest.daily_options
            WHERE symbol = %s
              AND trade_date = (SELECT MAX(trade_date) FROM raw_ingest.daily_options WHERE symbol = %s)
            GROUP BY option_type
        """, conn, params=(symbol, symbol))

        for _, row in df_opts.iterrows():
            if row['option_type'] == 'CE':
                ctx['daily_options_call_oi_change_pct_avg'] = row['avg_oi_change_pct']
            elif row['option_type'] == 'PE':
                ctx['daily_options_put_oi_change_pct_avg'] = row['avg_oi_change_pct']


    return ctx

def _insert_composite(rows: List[tuple])->int:
    if not rows: return 0
    sql = """
      INSERT INTO indicators.composite_v2
        (symbol, market_type, interval, ts,
         trend, momentum, quality, flow, structure, risk, ml, confidence,
         quality_veto, flow_veto, structure_veto, risk_veto,
         final_score, final_prob, context, run_id, source)
      VALUES %s
      ON CONFLICT (symbol, market_type, interval, ts) DO UPDATE
        SET trend=EXCLUDED.trend, momentum=EXCLUDED.momentum, quality=EXCLUDED.quality,
            flow=EXCLUDED.flow, structure=EXCLUDED.structure, risk=EXCLUDED.risk,
            ml=EXCLUDED.ml, confidence=EXCLUDED.confidence,
            quality_veto=EXCLUDED.quality_veto, flow_veto=EXCLUDED.flow_veto,
            structure_veto=EXCLUDED.structure_veto, risk_veto=EXCLUDED.risk_veto,
            final_score=EXCLUDED.final_score, final_prob=EXCLUDED.final_prob,
            context=EXCLUDED.context, run_id=EXCLUDED.run_id, source=EXCLUDED.source
    """
    with get_db_connection() as conn, conn.cursor() as cur:
        pgx.execute_values(cur, sql, rows, page_size=1000)
        conn.commit()
        return len(rows)

def _blend_mtf(values: Dict[str,float], Wtf:Dict[str,float])->Optional[float]:
    num = den = 0.0
    for tf,v in values.items():
        if v is None: continue
        w = float(Wtf.get(tf,0.0))
        if w<=0: continue
        num += w*float(v); den += w
    return (num/den) if den>0 else None

def _score_per_kind(symbol:str, kind:str, Wtf:Dict[str,float])->Tuple[Dict[str,float], Dict[str,bool]]:
    dfs = _load_all_tfs(symbol, kind, COMP_LOOKBACK_DAYS)
    out: Dict[str,float] = {"trend":None,"momentum":None,"quality":None,"flow":None,"structure":None,"risk":None, "ml":None, "confidence":None}
    veto: Dict[str,bool] = {"quality":False,"flow":False,"structure":False,"risk":False}

    if dfs.get("15m", pd.DataFrame()).empty:
        return out, veto

    context = _fetch_context(symbol, kind)

    per_tf = {"TREND":{}, "MOMENTUM":{}, "QUALITY":{}, "FLOW":{}, "RISK":{}, "STRUCT":{}, "ML":{}, "CONFIDENCE":{}}
    veto_tf = {"QUALITY":{}, "FLOW":{}, "RISK":{}, "STRUCT":{}}

    from pillars_optimized.common import BaseCfg
    base_cfg = BaseCfg(run_id=RUN_ID, source="composite_internal")

    for tf in TF_SET:
        dftf = dfs.get(tf, pd.DataFrame())
        if dftf.empty or not ensure_min_bars(dftf, tf): continue

        try:
            r = score_trend(symbol, kind, tf, dftf, base_cfg, context=context)
            if r: per_tf["TREND"][tf] = float(r[1])
        except Exception: pass

        try:
            r = score_momentum(symbol, kind, tf, dftf, base_cfg, context=context)
            if r: per_tf["MOMENTUM"][tf] = float(r[1])
        except Exception: pass

        try:
            r = score_quality(symbol, kind, tf, dftf, base_cfg, context=context)
            if r:
                per_tf["QUALITY"][tf] = float(r[1])
                veto_tf["QUALITY"][tf] = bool(r[2])
        except Exception: pass

        try:
            r = score_flow(symbol, kind, tf, dftf, base_cfg, context=context)
            if r:
                per_tf["FLOW"][tf] = float(r[1])
                veto_tf["FLOW"][tf] = bool(r[2])
        except Exception: pass

        try:
            r = score_risk(symbol, kind, tf, dftf, base_cfg, context=context)
            if r:
                per_tf["RISK"][tf] = float(r[1])
                veto_tf["RISK"][tf] = bool(r[2])
        except Exception: pass

        try:
            r = score_structure(symbol, kind, tf, dftf, base_cfg, context=context)
            if r:
                per_tf["STRUCT"][tf] = float(r[1])
                veto_tf["STRUCT"][tf] = bool(r[2])
        except Exception: pass

        try:
            r = score_ml(symbol, kind, tf, dftf, base_cfg, context=context)
            if r: per_tf["ML"][tf] = float(r[1])
        except Exception: pass

        # try:
        #     r = score_confidence(symbol, kind, tf, dftf, base_cfg, context=context)
        #     if r: per_tf["CONFIDENCE"][tf] = float(r[1])
        # except Exception: pass


    out["trend"]     = _blend_mtf(per_tf["TREND"],    Wtf)
    out["momentum"]  = _blend_mtf(per_tf["MOMENTUM"], Wtf)
    out["quality"]   = _blend_mtf(per_tf["QUALITY"],  Wtf)
    out["flow"]      = _blend_mtf(per_tf["FLOW"],     Wtf)
    out["risk"]      = _blend_mtf(per_tf["RISK"],     Wtf)
    out["structure"] = _blend_mtf(per_tf["STRUCT"],   Wtf)
    out["ml"]        = _blend_mtf(per_tf["ML"], Wtf)
    out["confidence"] = context.get("confidence", {}).get("conf_total", 0.5) * 100.0 if context else 50.0

    veto["quality"]   = any(veto_tf["QUALITY"].values())
    veto["flow"]      = any(veto_tf["FLOW"].values())
    veto["risk"]      = any(veto_tf["RISK"].values())
    veto["structure"] = any(veto_tf["STRUCT"].values())

    return out, veto

def _merge_spot_fut(spot, fut, v_spot, v_fut, bias):
    nz = lambda x: 50.0 if x is None else float(x)

    merged = {}
    for k in ["trend", "momentum", "quality", "risk", "structure", "ml", "confidence"]:
        s_val = spot.get(k); f_val = fut.get(k)
        if s_val is not None and f_val is not None:
            merged[k] = bias * f_val + (1-bias) * s_val
        else:
            merged[k] = f_val if f_val is not None else s_val

    merged["flow"] = fut.get("flow") if fut.get("flow") is not None else spot.get("flow")

    veto = {
        k: (v_spot.get(k,False) or v_fut.get(k,False))
        for k in ["quality", "flow", "structure", "risk"]
    }
    return merged, veto

def process_symbol_unified(symbol:str, comp_ini:str=DEFAULT_COMP_INI)->int:
    cfg = _load_comp_cfg(comp_ini)
    Wtf, w = cfg["tf_w"], cfg["pillar_w"]

    spot_scores, spot_veto = _score_per_kind(symbol, "spot", Wtf)
    fut_scores,  fut_veto  = _score_per_kind(symbol, "futures", Wtf)

    def have(d): return any(v is not None for v in d.values())

    if have(fut_scores) and not have(spot_scores):
        merged, veto, src_mode = fut_scores, fut_veto, "futures_only"
    elif have(spot_scores) and not have(fut_scores):
        merged, veto, src_mode = spot_scores, spot_veto, "spot_only"
    else:
        merged, veto = _merge_spot_fut(spot_scores, fut_scores, spot_veto, fut_veto, 0.6)
        src_mode = "merged"

    nz = lambda x: 50.0 if x is None else float(x)

    final_scores = {k: nz(merged.get(k)) for k in w}
    base_score = sum(w[k] * final_scores[k] for k in w)

    hard_veto = any(veto.values())
    final_score = 0.0 if hard_veto else base_score
    final_prob  = _score_to_prob(final_score)

    ctx = {
        "pillar_w": w,
        "per_kind": {"spot": spot_scores, "futures": fut_scores},
        "veto": veto,
        "final_scores": final_scores
    }

    # Call the strategy selector
    # suggested_strategies = select_strategy(symbol, {"final_score": final_score, **final_scores}, ctx)
    # ctx["suggested_strategies"] = suggested_strategies

    ts_mtf = now_ts()
    rows = [(
        symbol, "unified", "MTF", ts_mtf,
        final_scores.get("trend"), final_scores.get("momentum"), final_scores.get("quality"),
        final_scores.get("flow"),  final_scores.get("structure"), final_scores.get("risk"),
        final_scores.get("ml"), final_scores.get("confidence"),
        veto["quality"], veto["flow"], veto["structure"], veto["risk"],
        float(final_score), float(final_prob), json.dumps(ctx),
        RUN_ID, "composite_v2_unified"
    )]

    _insert_composite(rows)
    print(f"✅ {symbol} -> {final_score:.1f} (Veto={hard_veto})")
    return 1

def run(symbols: Optional[List[str]]=None):
    if symbols is None:
        from scheduler.indicators_worker import fetch_batch_universe
        rows = fetch_batch_universe(limit=200)
        symbols = [r["symbol"] for r in rows]

    if not symbols:
        print("No symbols to process.")
        return

    print(f"🚀 Starting Composite V2 for {len(symbols)} symbols with {MAX_WORKERS} workers...")

    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_symbol_unified, s): s for s in symbols}

        for future in concurrent.futures.as_completed(futures):
            sym = futures[future]
            try:
                future.result()
            except Exception as e:
                print(f"❌ Error {sym}: {e}")

if __name__ == "__main__":
    import sys
    syms = None
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        if "," in arg: syms = arg.split(",")
        else: syms = [arg]
    run(syms)