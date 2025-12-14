# scheduler/composite_worker_v2.py  — unified per-symbol composite (15m base)
from __future__ import annotations
import os, json, math, traceback, configparser
from typing import Dict, List, Optional, Iterable, Tuple, Any
from datetime import datetime, timezone, timedelta
import pandas as pd
import psycopg2.extras as pgx
from utils.db import get_db_connection

TZ = timezone.utc

# cfg loader (safe)
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
DEFAULT_COMP_INI = os.getenv("COMPOSITE_V2_INI", "composite_v2.ini")

# Pillars
from pillars.common import clamp, now_ts, ensure_min_bars
from Trend.Pillar.trend_pillar import score_trend
from pillars.momentum_pillar import score_momentum
from pillars.quality_pillar import score_quality
from pillars.flow_pillar import score_flow
from pillars.risk_pillar import score_risk
from pillars.structure_pillar import process_symbol as run_structure_for_symbol

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
        "trend":cp.getfloat("composite","w_trend",fallback=0.22),
        "momentum":cp.getfloat("composite","w_momentum",fallback=0.22),
        "quality":cp.getfloat("composite","w_quality",fallback=0.18),
        "flow":cp.getfloat("composite","w_flow",fallback=0.14),
        "structure":cp.getfloat("composite","w_structure",fallback=0.14),
        "risk":cp.getfloat("composite","w_risk",fallback=0.10),
    }
    s = sum(w.values()) or 1.0
    w = {k:v/s for k,v in w.items()}
    w_nl = cp.getfloat("composite","w_nl",fallback=0.12)
    tfw_raw = cp.get("composite","tf_weights",fallback="15m:0.30,30m:0.30,60m:0.25,120m:0.10,240m:0.05")
    tfw: Dict[str,float] = {}
    for part in tfw_raw.split(","):
        if ":" in part:
            k,v = part.split(":",1)
            try: tfw[k.strip()] = float(v.strip())
            except: pass
    tfw_sum = sum(tfw.values()) or 1.0
    tfw = {k:v/tfw_sum for k,v in tfw.items()}
    # bias when merging spot+fut pillars (slight futures tilt)
    merge_bias_fut = cp.getfloat("composite","merge_bias_fut",fallback=0.6)  # 0..1
    return {"pillar_w":w,"w_nl":w_nl,"tf_w":tfw,"merge_bias_fut":merge_bias_fut}

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
    df["volume"] = df["volume"].fillna(0.0)
    df = df.dropna(subset=["open","high","low","close"]).astype(
        {"open":"float64","high":"float64","low":"float64","close":"float64","volume":"float64"}
    )
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

def fetch_latest_oi_conf(symbol:str, kind:str)->Optional[float]:
    with get_db_connection() as conn, conn.cursor() as cur:
        cur.execute("""SELECT val FROM indicators.values
                        WHERE symbol=%s AND market_type=%s AND interval='MTF'
                          AND metric='OI_CONF.score.mtf'
                        ORDER BY ts DESC LIMIT 1""",(symbol,kind))
        row = cur.fetchone()
        if row: return float(row[0])
        cur.execute("""SELECT val FROM indicators.values
                        WHERE symbol=%s AND market_type=%s AND interval='MTF'
                          AND metric='CONF_NL.score.mtf'
                        ORDER BY ts DESC LIMIT 1""",(symbol,kind))
        row = cur.fetchone()
        return float(row[0]) if row else None

def _insert_composite(rows: List[tuple])->int:
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
    out: Dict[str,float] = {"trend":None,"momentum":None,"quality":None,"flow":None,"structure":None,"risk":None}  # type: ignore
    veto: Dict[str,bool] = {"quality":False,"flow":False,"structure":False,"risk":False}

    if dfs.get("15m", pd.DataFrame()).empty:
        return out, veto

    # structure pillar writes metrics itself
    try: run_structure_for_symbol(symbol, kind=kind)
    except Exception as e: print(f"⚠️ STRUCT {kind}:{symbol} → {e}")

    per_tf = {"TREND":{}, "MOMENTUM":{}, "QUALITY":{}, "FLOW":{}, "RISK":{}}
    veto_tf = {"QUALITY":{}, "FLOW":{}, "RISK":{}}

    for tf in TF_SET:
        dftf = dfs.get(tf, pd.DataFrame())
        if dftf.empty or not ensure_min_bars(dftf, tf): continue
        try:
            r=score_trend(symbol,kind,tf,dftf,None);      outv = per_tf["TREND"]
            if r: outv[tf]=float(r[1])
        except Exception as e: print(f"⚠️ TREND {kind}:{symbol}:{tf} → {e}")
        try:
            r=score_momentum(symbol,kind,tf,dftf,None);   outv = per_tf["MOMENTUM"]
            if r: outv[tf]=float(r[1])
        except Exception as e: print(f"⚠️ MOMENTUM {kind}:{symbol}:{tf} → {e}")
        try:
            r=score_quality(symbol,kind,tf,dftf,None)
            if r:
                per_tf["QUALITY"][tf]=float(r[1]); veto_tf["QUALITY"][tf]=bool(r[2])
        except Exception as e: print(f"⚠️ QUALITY {kind}:{symbol}:{tf} → {e}")
        try:
            r=score_flow(symbol,kind,tf,dftf,None)
            if r:
                per_tf["FLOW"][tf]=float(r[1]); veto_tf["FLOW"][tf]=bool(r[2])
        except Exception as e: print(f"⚠️ FLOW {kind}:{symbol}:{tf} → {e}")
        try:
            r=score_risk(symbol,kind,tf,dftf,None)
            if r:
                per_tf["RISK"][tf]=float(r[1]); veto_tf["RISK"][tf]=bool(r[2])
        except Exception as e: print(f"⚠️ RISK {kind}:{symbol}:{tf} → {e}")

    # read structure MTF
    def _last(metric:str)->Optional[float]:
        with get_db_connection() as conn, conn.cursor() as cur:
            cur.execute("""SELECT val FROM indicators.values
                            WHERE symbol=%s AND market_type=%s AND interval='MTF' AND metric=%s
                            ORDER BY ts DESC LIMIT 1""",(symbol,kind,metric))
            row = cur.fetchone()
            return float(row[0]) if row else None
    out["structure"] = _last("STRUCT.score")
    veto["structure"] = bool(((_last("STRUCT.veto_flag") or 0.0))>=0.5)

    # blend MTF for other pillars
    out["trend"]    = _blend_mtf(per_tf["TREND"],    Wtf)
    out["momentum"] = _blend_mtf(per_tf["MOMENTUM"], Wtf)
    out["quality"]  = _blend_mtf(per_tf["QUALITY"],  Wtf)
    out["flow"]     = _blend_mtf(per_tf["FLOW"],     Wtf)
    out["risk"]     = _blend_mtf(per_tf["RISK"],     Wtf)

    veto["quality"] = any(veto_tf["QUALITY"].values())
    veto["flow"]    = any(veto_tf["FLOW"].values())
    veto["risk"]    = any(veto_tf["RISK"].values())

    return out, veto

def _merge_spot_fut(spot:Dict[str,float], fut:Dict[str,float], veto_spot:Dict[str,bool], veto_fut:Dict[str,bool], merge_bias_fut:float)->Tuple[Dict[str,float], Dict[str,bool]]:
    nz = lambda x: 50.0 if x is None else float(x)
    # futures-leaning merge for market-informed pillars
    merged = {
        "trend":    merge_bias_fut*nz(fut["trend"])    + (1-merge_bias_fut)*nz(spot["trend"]),
        "momentum": merge_bias_fut*nz(fut["momentum"]) + (1-merge_bias_fut)*nz(spot["momentum"]),
        "quality":  merge_bias_fut*nz(fut["quality"])  + (1-merge_bias_fut)*nz(spot["quality"]),
        "flow":     (fut["flow"] if fut["flow"] is not None else spot["flow"]),
        "structure":(fut["structure"] if fut["structure"] is not None else spot["structure"]),
        "risk":     merge_bias_fut*nz(fut["risk"])     + (1-merge_bias_fut)*nz(spot["risk"]),
    }
    # veto if any kind vetoes
    veto = {
        "quality": bool(veto_spot["quality"] or veto_fut["quality"]),
        "flow":    bool(veto_spot["flow"]    or veto_fut["flow"]),
        "structure": bool(veto_spot["structure"] or veto_fut["structure"]),
        "risk":    bool(veto_spot["risk"]    or veto_fut["risk"]),
    }
    return merged, veto

def process_symbol_unified(symbol:str, comp_ini:str=DEFAULT_COMP_INI)->int:
    cfg = _load_comp_cfg(comp_ini)
    Wtf, w, w_nl, bias = cfg["tf_w"], cfg["pillar_w"], float(cfg["w_nl"]), float(cfg["merge_bias_fut"])

    # per-kind scores
    spot_scores, spot_veto = _score_per_kind(symbol, "spot", Wtf)
    fut_scores,  fut_veto  = _score_per_kind(symbol, "futures", Wtf)

    # OI confidence: prefer futures, fallback spot, then 50
    nl = fetch_latest_oi_conf(symbol, "futures")
    if nl is None:
        nl = fetch_latest_oi_conf(symbol, "spot")
    if nl is None:
        try:
            import scheduler.update_confidence_oi as uoi
            uoi.run_for_symbols([symbol], kind="futures")
            nl = fetch_latest_oi_conf(symbol, "futures")
        except Exception:
            pass
    nl_score = 50.0 if nl is None else float(nl)

    # if only one side has data, use it; else merge
    def have(d:Dict[str,float])->bool: return any(v is not None for v in d.values())
    if have(fut_scores) and not have(spot_scores):
        merged, veto = fut_scores, fut_veto
        src_mode = "futures_only"
    elif have(spot_scores) and not have(fut_scores):
        merged, veto = spot_scores, spot_veto
        src_mode = "spot_only"
    else:
        merged, veto = _merge_spot_fut(spot_scores, fut_scores, spot_veto, fut_veto, bias)
        src_mode = "merged"

    nz = lambda x: 50.0 if x is None else float(x)
    base_score = (
        w["trend"]    * nz(merged["trend"]) +
        w["momentum"] * nz(merged["momentum"]) +
        w["quality"]  * nz(merged["quality"]) +
        w["flow"]     * nz(merged["flow"]) +
        w["structure"]* nz(merged["structure"]) +
        w["risk"]     * nz(merged["risk"])
    )
    fused = (base_score + w_nl * nl_score) / (1.0 + w_nl)
    fused = clamp(fused, 0, 100)

    hard_veto = bool(veto["quality"] or veto["flow"] or veto["structure"] or veto["risk"])
    final_score = 0.0 if hard_veto else fused
    final_prob  = _score_to_prob(final_score)

    ctx = {
        "pillar_w": w,
        "tf_w": Wtf,
        "per_kind": {"spot": spot_scores, "futures": fut_scores},
        "veto": veto,
        "merge": {"mode": src_mode, "bias_futures": bias},
        "nonlinear": {"score": nl_score, "metric": "OI_CONF.score.mtf", "w_nl": w_nl},
        "tfs_used": TF_SET,
        "base_interval": BASE_INTERVAL
    }

    ts_mtf = now_ts()
    rows = [(
        symbol, "unified", "MTF", ts_mtf,
        nz(merged["trend"]), nz(merged["momentum"]), nz(merged["quality"]),
        nz(merged["flow"]),  nz(merged["structure"]), nz(merged["risk"]),
        bool(veto["quality"]), bool(veto["flow"]), bool(veto["structure"]), bool(veto["risk"]),
        float(final_score), float(final_prob), json.dumps(ctx),
        os.getenv("RUN_ID","comp_v2"), os.getenv("SRC","composite_v2_unified")
    )]
    n = _insert_composite(rows)
    print(f"✅ COMP2 UNIFIED {symbol} → {final_score:.1f} p={final_prob:.2f} veto={hard_veto} ({src_mode}, NL={nl_score:.1f})")
    return n

# --- universe gating (unified) ---
def _get_newest_any_ts(conn, symbol:str)->Optional[datetime]:
    with conn.cursor() as cur:
        cur.execute("""
          SELECT GREATEST(
            COALESCE((SELECT max(ts) FROM market.spot_candles    WHERE symbol=%s AND interval IN ('15m','30m','60m','120m','240m')), 'epoch'::timestamptz),
            COALESCE((SELECT max(ts) FROM market.futures_candles WHERE symbol=%s AND interval IN ('15m','30m','60m','120m','240m')), 'epoch'::timestamptz)
          )""",(symbol,symbol))
        row = cur.fetchone()
    return (pd.to_datetime(row[0], utc=True).to_pydatetime() if row and row[0] else None)

def _fetch_batch_universe(limit:int=200)->List[str]:
    sql = """
      WITH u AS (
        SELECT u.symbol,
               COALESCE((SELECT max(ts) FROM market.spot_candles    sc WHERE sc.symbol=u.symbol AND sc.interval IN ('15m','30m','60m','120m','240m')),'epoch'::timestamptz) AS newest_spot,
               COALESCE((SELECT max(ts) FROM market.futures_candles fc WHERE fc.symbol=u.symbol AND fc.interval IN ('15m','30m','60m','120m','240m')),'epoch'::timestamptz) AS newest_fut,
               u.last_comp_at, u.last_comp_spot_at, u.last_comp_fut_at
          FROM reference.symbol_universe u
         WHERE u.universe_name=%s
      )
      SELECT symbol
        FROM u
       WHERE (last_comp_at IS NULL OR last_comp_at < GREATEST(newest_spot,newest_fut))
          OR (last_comp_at IS NULL AND (COALESCE(last_comp_spot_at,'epoch'::timestamptz) < newest_spot
                                     OR COALESCE(last_comp_fut_at,'epoch'::timestamptz) < newest_fut))
       ORDER BY symbol
       LIMIT %s
    """
    with get_db_connection() as conn, conn.cursor() as cur:
        cur.execute(sql,(UNIVERSE_NAME,limit))
        return [r[0] for r in cur.fetchall()]

def _set_universe_status(conn, symbol:str, status:str, gate_ts:Optional[datetime]=None)->None:
    # prefer unified columns; fall back silently
    tries = []
    tries.append(("UPDATE reference.symbol_universe SET comp_status=%s, comp_status_ts=NOW()"
                  + (", last_comp_at=%s" if gate_ts else "") + " WHERE symbol=%s",
                  ([status, gate_ts, symbol] if gate_ts else [status, symbol])))
    # best-effort legacy mirrors
    if gate_ts:
        tries.append(("UPDATE reference.symbol_universe SET last_comp_spot_at=GREATEST(COALESCE(last_comp_spot_at,'epoch'::timestamptz), %s) WHERE symbol=%s",
                      [gate_ts, symbol]))
        tries.append(("UPDATE reference.symbol_universe SET last_comp_fut_at=GREATEST(COALESCE(last_comp_fut_at,'epoch'::timestamptz), %s) WHERE symbol=%s",
                      [gate_ts, symbol]))
    with conn.cursor() as cur:
        for sql,args in tries:
            try: cur.execute(sql, args)
            except Exception: pass
    conn.commit()

# --- runner ---
def run(symbols: Optional[List[str]]=None)->int:
    if symbols is None:
        symbols = _fetch_batch_universe(limit=200)
    total = 0
    for s in symbols:
        try:
            with get_db_connection() as conn:
                _set_universe_status(conn, s, "COMP_RUNNING")
            wrote = process_symbol_unified(s)
            with get_db_connection() as conn:
                latest = _get_newest_any_ts(conn, s) or datetime.now(TZ)
                _set_universe_status(conn, s, "COMP_DONE", gate_ts=latest)
            total += wrote
        except Exception as e:
            with get_db_connection() as conn:
                _set_universe_status(conn, s, "COMP_ERROR")
            print(f"❌ COMP2 UNIFIED {s} → {e}\n{traceback.format_exc()}")
            continue
    print(f"🎯 COMP2 UNIFIED wrote {total} row(s) across {len(symbols)} symbol(s)")
    return total

def _parse_flags(argv: List[str])->dict:
    out: Dict[str,Any] = {}
    it = iter(argv)
    for tok in it:
        if tok == "--symbols": out["symbols"] = next(it, "")
        elif tok == "--limit": out["limit"] = int(next(it, "200"))
    return out

if __name__ == "__main__":
    import sys
    flags = _parse_flags(sys.argv[1:])
    syms = flags.get("symbols","")
    if syms:
        symbols = [s.strip().upper() for s in syms.split(",") if s.strip()]
    else:
        symbols = None
    run(symbols)
