# ml/flow_bucket_stats.py

import argparse
import os
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

from utils.db import get_db_connection

# You can tweak these if needed
DEFAULT_PILLAR      = "flow"
DEFAULT_MARKET_TYPE = "futures"
DEFAULT_TF          = "15m"
DEFAULT_TARGET_NAME = "flow_ml.target.ret_2pct_4h"   # matches your ml_targets.ini
DEFAULT_VERSION     = "xgb_v1"                       # matches flow_train.py
UP_THRESHOLD_PCT    = 2.0                            # 2% UP in 4h

def load_flow_ml_rows(
    pillar: str,
    market_type: str,
    tf: str,
    target_name: str,
    version: str,
    min_date: str | None = None,
) -> pd.DataFrame:
    sql = """
        SELECT
            symbol,
            market_type,
            pillar,
            tf,
            ts,
            target_name,
            version,
            prob_up      AS p_up,
            prob_down    AS p_down,
            future_ret_pct
        FROM indicators.ml_pillars
        WHERE pillar      = %(pillar)s
          AND market_type = %(market_type)s
          AND tf          = %(tf)s
          AND target_name = %(target_name)s
          AND version     = %(version)s
          AND prob_up IS NOT NULL
          AND future_ret_pct IS NOT NULL
        ORDER BY ts
    """

    params = {
        "pillar": pillar,
        "market_type": market_type,
        "tf": tf,
        "target_name": target_name,
        "version": version,
    }

    with get_db_connection() as conn:
        # 🔍 DEBUG: what do we have in ml_pillars?
        with conn.cursor() as cur:
            cur.execute("""
                SELECT pillar, market_type, tf, target_name, version, 
                       COUNT(*) 
                  FROM indicators.ml_pillars
                 GROUP BY 1,2,3,4,5
                 ORDER BY 1,2,3,4,5
            """)
            print("[DEBUG ml_pillars groups]:")
            for row in cur.fetchall():
                print("  ", row)

        # now the real query
        df = pd.read_sql(sql, conn, params=params)


    if min_date:
        df = df[df["ts"] >= pd.to_datetime(min_date)]

    return df


def compute_buckets(
    df: pd.DataFrame,
    n_buckets: int = 9,
    up_threshold_pct: float = 2.0,
) -> pd.DataFrame:
    """
    - Buckets by p_up into n_buckets
    - is_up = 1 if future_ret_pct >= up_threshold_pct
    - realized_up_rate = mean(is_up) per bucket
    """

    if df.empty:
        raise RuntimeError("Empty dataframe passed to compute_buckets()")

    # Work on a copy
    df = df.copy()

    # Keep only rows with valid probs + future_return
    df = df.dropna(subset=["p_up", "future_ret_pct"])
    if df.empty:
        raise RuntimeError("No rows with non-null p_up & future_ret_pct")

    # 1) Quantile buckets on model probability
    df["bucket"] = pd.qcut(
        df["p_up"],
        q=n_buckets,
        labels=False,
        duplicates="drop",
    ) + 1  # make buckets 1..N

    # 2) Define what "UP" means for realized rate (>= +2% in 4h)
    df["is_up"] = (df["future_ret_pct"] >= up_threshold_pct).astype(int)

    # 3) Aggregate per bucket
    grp = df.groupby("bucket", as_index=False).agg(
        p_min=("p_up", "min"),
        p_max=("p_up", "max"),
        avg_p=("p_up", "mean"),
        realized_up_rate=("is_up", "mean"),
        n=("is_up", "size"),
    )

    # Nice ordering
    grp = grp.sort_values("bucket").reset_index(drop=True)
    return grp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_name", default=DEFAULT_TARGET_NAME)
    parser.add_argument("--version", default=DEFAULT_VERSION)
    parser.add_argument("--pillar", default=DEFAULT_PILLAR)
    parser.add_argument("--market-type", default=DEFAULT_MARKET_TYPE)
    parser.add_argument("--tf", default=DEFAULT_TF)
    parser.add_argument("--min-date", default=None, help="YYYY-MM-DD (optional)")
    parser.add_argument("--n-buckets", type=int, default=9)
    parser.add_argument(
        "--out-csv",
        default="flow_bucket_stats.csv",
        help="Path to write CSV with bucket stats",
    )

    args = parser.parse_args()

    print(
        f"[FLOW_BUCKETS] Loading ml_pillars for "
        f"pillar={args.pillar}, target={args.target_name}, version={args.version}"
    )

    df = load_flow_ml_rows(
        target_name=args.target_name,
        version=args.version,
        pillar=args.pillar,
        market_type=args.market_type,
        tf=args.tf,
        min_date=args.min_date,
    )

    buckets_df = compute_buckets(df, n_buckets=args.n_buckets)

    # Pretty print
    pd.set_option("display.float_format", lambda x: f"{x:.12f}")
    print("\n[FLOW_BUCKETS] Bucket stats:\n")
    print(buckets_df.to_string(index=False))

    # Save to CSV for calibration step
    buckets_df.to_csv(args.out_csv, index=False)
    print(f"\n[FLOW_BUCKETS] Saved → {args.out_csv}")


if __name__ == "__main__":
    main()
