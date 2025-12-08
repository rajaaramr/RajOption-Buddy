# momentum/ml/momentum_bucket_stats.py

from __future__ import annotations
import argparse, datetime as dt

import numpy as np
import pandas as pd

from utils.db import get_db_engine, get_db_connection

PILLAR = "momentum"
DEFAULT_TARGET = "momentum_ml.target.bull_2pct_4h"
DEFAULT_VERSION = "xgb_v1"
CALIB_TABLE = "indicators.momentum_calibration_4h"


def _parse_args():
    ap = argparse.ArgumentParser(description="Momentum ML bucket stats + calibration.")
    ap.add_argument("--target-name", default=DEFAULT_TARGET)
    ap.add_argument("--version", default=DEFAULT_VERSION)
    ap.add_argument("--tf", default="15m")
    ap.add_argument("--market-type", default="futures")
    ap.add_argument("--bins", type=int, default=10)
    ap.add_argument("--lookback-days", type=int, default=180)
    return ap.parse_args()


def main():
    args = _parse_args()
    engine = get_db_engine()
    cutoff = dt.datetime.utcnow() - dt.timedelta(days=args.lookback_days)

    sql = """
        SELECT prob_up, future_ret_pct
          FROM indicators.ml_pillars
         WHERE pillar = %(pillar)s
           AND target_name = %(target_name)s
           AND version = %(version)s
           AND market_type = %(market_type)s
           AND tf = %(tf)s
           AND ts >= %(cutoff)s
    """
    df = pd.read_sql(sql, engine, params={
        "pillar": PILLAR,
        "target_name": args.target_name,
        "version": args.version,
        "market_type": args.market_type,
        "tf": args.tf,
        "cutoff": cutoff,
    })
    if df.empty:
        raise RuntimeError("No ml_pillars rows for calibration.")

    df = df.dropna(subset=["prob_up", "future_ret_pct"])
    df["hit_up"] = (df["future_ret_pct"] >= 0.02).astype(int)

    bins = np.linspace(0.0, 1.0, args.bins + 1)
    df["bucket"] = np.digitize(df["prob_up"], bins, right=False) - 1
    df = df[(df["bucket"] >= 0) & (df["bucket"] < args.bins)]

    rows = []
    for b in range(args.bins):
        sub = df[df["bucket"] == b]
        if sub.empty:
            continue
        p_min = float(bins[b])
        p_max = float(bins[b + 1])
        avg_p = float(sub["prob_up"].mean())
        realized = float(sub["hit_up"].mean())
        n = int(len(sub))
        rows.append((PILLAR, args.target_name, args.version, args.tf, b, p_min, p_max, avg_p, realized, n))

    if not rows:
        raise RuntimeError("No non-empty buckets; cannot calibrate.")

    with get_db_connection() as conn, conn.cursor() as cur:
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {CALIB_TABLE} (
                pillar          text,
                target_name     text,
                version         text,
                tf              text,
                bucket_id       integer,
                p_min           double precision,
                p_max           double precision,
                avg_p           double precision,
                realized_up_rate double precision,
                n               integer,
                PRIMARY KEY (pillar, target_name, version, tf, bucket_id)
            );
            """
        )
        cur.execute(
            f"""
            DELETE FROM {CALIB_TABLE}
             WHERE pillar = %s
               AND target_name = %s
               AND version = %s
               AND tf = %s;
            """,
            (PILLAR, args.target_name, args.version, args.tf),
        )
        cur.executemany(
            f"""
            INSERT INTO {CALIB_TABLE} (
                pillar, target_name, version, tf,
                bucket_id, p_min, p_max, avg_p, realized_up_rate, n
            )
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s);
            """,
            rows,
        )
        conn.commit()
        # --- Update ML calibration freshness on symbol_universe ---
    from utils.db import get_db_connection  # already imported at top, so optional here too

    with get_db_connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            UPDATE reference.symbol_universe
               SET mom_ml_calib_last_at = now()
             WHERE enabled = TRUE;
            """
        )
        conn.commit()

    print("[MOM_CALIB] Updated mom_ml_calib_last_at on reference.symbol_universe")

    print(f"[MOM_CALIB] Wrote {len(rows)} buckets into {CALIB_TABLE}.")


if __name__ == "__main__":
    main()
