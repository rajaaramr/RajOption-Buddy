# scheduler/structure_runner.py
from __future__ import annotations
import os, json, configparser
from typing import Iterable, List, Optional, Dict
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd
from utils.db import get_db_connection
from pillars.structure_pillar import load_cfg, score_structure
from pillars.common import TZ, ensure_min_bars

DEFAULT_INI = os.getenv("STRUCTURE_INI", "structure.ini")

TF_LIST = ["15m","30m","60m","120m","240m"]  # edit if needed

def _load_5m(symbol: str, kind: str, lookback_days: int) -> pd.DataFrame:
    table = "market.futures_candles" if kind == "futures" else "market.spot_candles"
    cutoff = datetime.now(TZ) - timedelta(days=lookback_days)
    with get_db_connection() as conn, conn.cursor() as cur:
        cur.execute(f"""
            SELECT ts, open, high, low, close, volume
              FROM {table}
             WHERE symbol=%s AND interval='5m' AND ts >= %s
             ORDER BY ts ASC
        """, (symbol, cutoff))
        rows = cur.fetchall()
    if not rows: return pd.DataFrame()
    df = pd.DataFrame(rows, columns=["ts","open","high","low","close","volume"])
    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    df = df.dropna(subset=["ts"]).set_index("ts").sort_index()
    for c in ["open","high","low","close","volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(how="any").loc[~df.index.duplicated(keep="last")]

def _safe_num(val, default=0.0) -> float:
    try:
        return float(val) if val is not None else default
    except (ValueError, TypeError):
        return default

def _blend(scores: Dict[str,float], weights: Dict[str,float]) -> float:
    if not scores: return 0.0
    s=w=0.0
    for tf,v in scores.items():
        wt = float(weights.get(tf, 0.0))
        s += wt*v; w += wt
    return float(s/w) if w>0 else float(sum(scores.values())/len(scores))

def run(symbols: Optional[List[str]] = None, kinds: Iterable[str] = ("futures","spot"), ini_path: str = DEFAULT_INI) -> int:
    # discover symbols if not passed
    if symbols is None:
        with get_db_connection() as conn, conn.cursor() as cur:
            cur.execute("""
                SELECT DISTINCT symbol
                  FROM webhooks.webhook_alerts
                 WHERE status IN ('INDICATOR_PROCESS','SIGNAL_PROCESS','DATA_PROCESSING')
            """)
            rows = cur.fetchall()
        symbols = [r[0] for r in rows or []]

    # multi-variant support via [structure_runner]
    cp = configparser.ConfigParser(inline_comment_prefixes=(";","#"), interpolation=None, strict=False)
    cp.read(ini_path)
    variants = []
    if cp.has_section("structure_runner"):
        raw = cp.get("structure_runner", "variants", fallback="structure")
        variants = [x.strip() for x in raw.split(",") if x.strip()]
    else:
        variants = ["structure"]

    total = 0
    for s in symbols:
        for k in kinds:
            df5 = _load_5m(s, k, lookback_days=cp.getint("structure", "lookback_days", fallback=120))
            if df5.empty:
                print(f"⚠️ structure: no data for {k}:{s}")
                continue

            for sect in variants:
                cfg = load_cfg(ini_path, section=sect)

                per_tf_score_final = {}
                per_tf_veto_final  = {}

                for tf in cp.get(sect, "tfs", fallback="25m,65m,125m").split(","):
                    tf = tf.strip()
                    if not tf: continue
                    res = score_structure(s, k, tf, df5, base=type("B", (), {"run_id": os.getenv("RUN_ID","struct_run"), "source": os.getenv("SRC","structure")})(), ini_path=ini_path, section=sect)
                    if res is None: 
                        continue
                    _, score_final, veto_final, _ = res
                    per_tf_score_final[tf] = float(score_final)
                    per_tf_veto_final[tf]  = bool(veto_final)

                if not per_tf_score_final:
                    continue

                # MTF roll-up
                mtf_score = _blend(per_tf_score_final, cfg.cp.get(sect, "mtf_weights", fallback="25m:0.3,65m:0.3,125m:0.4"))
                # parse weights string to dict
                wmap={}
                for part in cfg.cp.get(sect,"mtf_weights",fallback="25m:0.3,65m:0.3,125m:0.4").split(","):
                    if ":" in part:
                        k1,v1=part.split(":",1); 
                        try: wmap[k1.strip()]=float(v1.strip())
                        except: pass
                ws = sum(wmap.values()) or 1.0
                wmap = {k1:v1/ws for k1,v1 in wmap.items()}

                mtf_score = sum(per_tf_score_final.get(t,0.0)*wmap.get(t,0.0) for t in per_tf_score_final.keys())
                if sum(wmap.values())==0: mtf_score = float(sum(per_tf_score_final.values())/len(per_tf_score_final))
                mtf_veto  = any(per_tf_veto_final.values())

                # ML on MTF (reuse TF ML settings)
                mtf_score_final = mtf_score
                mtf_veto_final  = mtf_veto

                try:
                    cfg2 = load_cfg(ini_path, section=sect)
                    if cfg2.ml_enabled:
                        base_prob = mtf_score / 100.0
                        ml_prob = None
                        if cfg2.ml_source_table:
                            from utils.db import get_db_connection
                            with get_db_connection() as conn, conn.cursor() as cur:
                                cur.execute(f"""
                                    SELECT prob_long, prob_short
                                      FROM {cfg2.ml_source_table}
                                     WHERE symbol=%s AND tf=%s
                                     ORDER BY ts DESC
                                     LIMIT 1
                                """, (s, "MTF"))
                                r = cur.fetchone()
                            if r:
                                prob_long=_safe_num(r[0],0.5); prob_short=_safe_num(r[1],0.5)
                                ml_prob = max(prob_long, 1.0 - prob_short)
                        if ml_prob is None and cfg2.ml_callback:
                            import importlib
                            mod_name,_,fn_name = cfg2.ml_callback.rpartition(".")
                            if mod_name and fn_name:
                                mod=importlib.import_module(mod_name)
                                fn=getattr(mod, fn_name)
                                ml_prob = float(fn(symbol=s, kind=k, tf="MTF", ts=df5.index[-1].to_pydatetime().replace(tzinfo=TZ), parts={}, anchors={}, direction=0))
                                if not np.isfinite(ml_prob): ml_prob = 0.0
                                ml_prob = max(0.0, min(1.0, ml_prob))
                        if ml_prob is not None:
                            w = cfg2.ml_blend_weight
                            blended = (1.0 - w)*base_prob + w*ml_prob
                            mtf_score_final = round(100.0 * blended, 2)
                            if cfg2.ml_soften_veto_if_prob_ge is not None and blended >= cfg2.ml_soften_veto_if_prob_ge:
                                mtf_veto_final = False
                            if cfg2.ml_veto_if_prob_lt is not None and blended < cfg2.ml_veto_if_prob_lt:
                                mtf_veto_final = True
                except Exception:
                    pass

                # write MTF rows
                from pillars.common import write_values
                ts_latest = df5.index[-1].to_pydatetime().replace(tzinfo=TZ)
                P = cfg.metric_prefix + "_MTF"
                rows = [
                    (s, k, "MTF", ts_latest, f"{P}.score",       float(mtf_score),       "{}", os.getenv("RUN_ID","struct_run"), os.getenv("SRC","structure")),
                    (s, k, "MTF", ts_latest, f"{P}.veto_flag",   1.0 if mtf_veto else 0.0, "{}", os.getenv("RUN_ID","struct_run"), os.getenv("SRC","structure")),
                    (s, k, "MTF", ts_latest, f"{P}.score_final", float(mtf_score_final),  "{}", os.getenv("RUN_ID","struct_run"), os.getenv("SRC","structure")),
                    (s, k, "MTF", ts_latest, f"{P}.veto_final",  1.0 if mtf_veto_final else 0.0, "{}", os.getenv("RUN_ID","struct_run"), os.getenv("SRC","structure")),
                ]
                write_values(rows)
                total += 1

    print(f"✅ structure_runner wrote rows for {total} variant-batches")
    return total

if __name__ == "__main__":
    import sys
    args = [a for a in sys.argv[1:]]
    syms = [a for a in args if not a.startswith("--")]
    run(syms or None)
