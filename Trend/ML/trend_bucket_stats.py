# Trend/ML/trend_bucket_stats.py

import pandas as pd
import numpy as np
from psycopg2.extras import execute_values
from utils.db import get_db_connection

CALIBRATION_TABLE = "indicators.trend_calibration_4h"

def run_bucket_stats():
    print(f"[TREND_CALIBRATION] Computing stats for {CALIBRATION_TABLE}...")

    # Fetch Predictions vs Realized
    # We join ml_pillars with actual future returns (which we can compute or if stored)
    # trend_train.py writes 'future_ret_pct' to ml_pillars during training (in-sample).
    # For robust calibration, we should ideally use out-of-sample or the full set if we trust it.
    # Let's use whatever is in indicators.ml_pillars where future_ret_pct is not null (0.0 often default if backfilled without future knowledge)
    # The backfiller writes 0.0 for future_ret. The trainer writes actuals.
    # We filter for non-zero or rely on rows that have it.

    sql = """
        SELECT prob_up, future_ret_pct
        FROM indicators.ml_pillars
        WHERE pillar='trend'
          AND future_ret_pct IS NOT NULL
          AND future_ret_pct != 0
    """

    with get_db_connection() as conn:
        df = pd.read_sql(sql, conn)

    if df.empty:
        print("No data found for calibration.")
        return

    # Define buckets (0.0 to 1.0)
    bins = np.linspace(0, 1, 11) # 10 bins: 0-0.1, ...
    df['bucket'] = pd.cut(df['prob_up'], bins, include_lowest=True)

    stats = df.groupby('bucket', observed=False).agg(
        count=('prob_up', 'count'),
        avg_prob=('prob_up', 'mean'),
        realized_up_rate=('future_ret_pct', lambda x: (x > 2.0).mean()) # Threshold matches target definition roughly
    ).reset_index()

    # Prepare rows
    rows = []
    for _, row in stats.iterrows():
        if row['count'] == 0: continue

        # Parse IntervalIndex for min/max
        interval = row['bucket']
        min_p = interval.left
        max_p = interval.right
        cal_p = row['realized_up_rate']

        rows.append((float(min_p), float(max_p), float(cal_p), int(row['count'])))

    # Write to DB
    # Create table if not exists
    create_sql = f"""
        CREATE TABLE IF NOT EXISTS {CALIBRATION_TABLE} (
            min_prob FLOAT,
            max_prob FLOAT,
            realized_up_rate FLOAT,
            sample_count INT,
            updated_at TIMESTAMP DEFAULT NOW(),
            PRIMARY KEY (min_prob, max_prob)
        );
        TRUNCATE TABLE {CALIBRATION_TABLE};
    """

    ins_sql = f"""
        INSERT INTO {CALIBRATION_TABLE} (min_prob, max_prob, realized_up_rate, sample_count)
        VALUES %s
    """

    with get_db_connection() as conn, conn.cursor() as cur:
        cur.execute(create_sql)
        execute_values(cur, ins_sql, rows)
        conn.commit()

    print(f"Updated {CALIBRATION_TABLE} with {len(rows)} buckets.")
    print(stats)

if __name__ == "__main__":
    run_bucket_stats()
