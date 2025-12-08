# ml_strategy/flow/flow_calibration_seed.py

import os
import argparse
import pandas as pd
from psycopg2.extras import execute_values

from utils.db import get_db_connection


DEFAULT_CSV = os.path.join(
    "data",
    "flow_buckets_bull_2pct_4h.csv",  # adjust if you used a different name
)


def seed_flow_calibration(
    csv_path: str = DEFAULT_CSV,
    target_name: str = "flow_ml.target.bull_2pct_4h",
    version: str = "xgb_v1",
) -> None:
    print(f"[FLOW_CAL] Loading bucket CSV -> {csv_path}")
    df = pd.read_csv(csv_path)

    required_cols = {"bucket", "p_min", "p_max", "avg_p", "realized_up_rate", "n"}
    missing = required_cols - set(df.columns)
    if missing:
        raise RuntimeError(f"CSV is missing columns: {missing}")

    rows = []
    for _, row in df.iterrows():
        rows.append(
            (
                target_name,
                version,
                int(row["bucket"]),
                float(row["p_min"]),
                float(row["p_max"]),
                float(row["avg_p"]),
                float(row["realized_up_rate"]),
                int(row["n"]),
            )
        )

    if not rows:
        print("[FLOW_CAL] No rows to insert, aborting.")
        return

    sql = """
        INSERT INTO indicators.flow_calibration_4h (
            target_name,
            version,
            bucket,
            p_min,
            p_max,
            avg_p,
            realized_up_rate,
            n
        )
        VALUES %s
        ON CONFLICT (target_name, version, bucket)
        DO UPDATE
           SET p_min            = EXCLUDED.p_min,
               p_max            = EXCLUDED.p_max,
               avg_p            = EXCLUDED.avg_p,
               realized_up_rate = EXCLUDED.realized_up_rate,
               n                = EXCLUDED.n,
               created_at       = now()
    """

    with get_db_connection() as conn, conn.cursor() as cur:
        execute_values(cur, sql, rows, page_size=100)
        conn.commit()

    print(f"[FLOW_CAL] Upserted {len(rows)} rows into indicators.flow_calibration_4h")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        default=DEFAULT_CSV,
        help="Path to buckets CSV (from flow_bucket_stats --out-csv ...)",
    )
    parser.add_argument(
        "--target-name",
        default="flow_ml.target.bull_2pct_4h",
        help="Target name, must match ml_pillars target_name",
    )
    parser.add_argument(
        "--version",
        default="xgb_v1",
        help="Model version (same as in ml_pillars)",
    )
    args = parser.parse_args()

    seed_flow_calibration(
        csv_path=args.csv,
        target_name=args.target_name,
        version=args.version,
    )


if __name__ == "__main__":
    main()
