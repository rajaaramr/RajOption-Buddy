# scheduler/confidence_worker.py
from __future__ import annotations
import os
from typing import List, Dict, Any
import pandas as pd
from utils.db import get_db_connection
import psycopg2.extras as pgx

from pillars_optimized.confidence_features import (
    load_confidence_config,
    compute_price_vol_conf_features,
    compute_oi_conf_features,
    aggregate_mtf_oi_conf,
    build_confidence_row,
)
from pillars_optimized.common import resample, ensure_min_bars, maybe_trim_last_bar, now_ts
from scheduler.indicators_worker import fetch_batch_universe


def _load_data(symbol: str, kind: str, lookback_days: int) -> Dict[str, pd.DataFrame]:
    """Loads all necessary data for a symbol."""
    frames_table = "indicators.spot_frames" if kind == "spot" else "indicators.futures_frames"
    candles_table = "market.spot_candles" if kind == "spot" else "market.futures_candles"
    cutoff = pd.Timestamp.now(tz='UTC') - pd.Timedelta(days=lookback_days)

    with get_db_connection() as conn:
        # Load candles
        df_candles = pd.read_sql(f"""
            SELECT ts, open, high, low, close, volume
            FROM {candles_table}
            WHERE symbol = %s AND interval = '15m' AND ts >= %s
            ORDER BY ts ASC
        """, conn, params=(symbol, cutoff), index_col='ts')

        # Load frames
        df_frames = pd.read_sql(f"""
            SELECT *
            FROM {frames_table}
            WHERE symbol = %s AND interval = '15m' AND ts >= %s
            ORDER BY ts ASC
        """, conn, params=(symbol, cutoff), index_col='ts')

    # Merge the two dataframes
    df_merged = df_candles.join(df_frames, how='inner')

    return {"15m": df_merged}

def _insert_confidence_state(rows: List[Dict[str, Any]]):
    """Bulk inserts rows into the indicators.confidence_state table."""
    if not rows:
        return

    cols = list(rows[0].keys())
    sql = f"""
        INSERT INTO indicators.confidence_state ({', '.join(cols)})
        VALUES %s
        ON CONFLICT (symbol, interval, ts) DO UPDATE SET
        {', '.join([f"{c} = EXCLUDED.{c}" for c in cols if c not in ['symbol', 'interval', 'ts']])}
    """

    with get_db_connection() as conn, conn.cursor() as cur:
        pgx.execute_values(cur, sql, [tuple(row.values()) for row in rows])
        conn.commit()


def process_symbol(symbol: str, kind: str, ini_path: str = "indicators.ini"):
    """
    Processes a single symbol to calculate and store confidence scores.
    """
    conf_cfg, oi_cfg = load_confidence_config(ini_path)
    data = _load_data(symbol, kind, conf_cfg.lookback_days)

    if "15m" not in data or data["15m"].empty:
        print(f"No 15m data for {symbol}, skipping.")
        return

    df15 = data["15m"]
    rows_to_insert = []

    per_tf_conf = {}

    for tf in conf_cfg.tfs:
        df_tf = resample(df15, tf)
        df_tf = maybe_trim_last_bar(df_tf)
        if not ensure_min_bars(df_tf, tf):
            continue

        price_vol_features = compute_price_vol_conf_features(df_tf, conf_cfg)
        oi_features = compute_oi_conf_features(df_tf, conf_cfg, oi_cfg)

        per_tf_conf[tf] = oi_features['oi_conf_score'].iloc[-1]

        latest_ts = df_tf.index[-1]
        latest_pv_row = price_vol_features.iloc[-1]
        latest_oi_row = oi_features.iloc[-1]

        row = build_confidence_row(
            symbol, tf, latest_ts, latest_pv_row, latest_oi_row,
            per_tf_conf[tf], 0.0, conf_cfg
        )
        rows_to_insert.append(row)

    oi_conf_mtf, mtf_align_score = aggregate_mtf_oi_conf(per_tf_conf, conf_cfg)

    final_rows_to_insert = []
    for row in rows_to_insert:
        # The `build_confidence_row` creates the full row, but the `conf_total`
        # in it is based on the single-TF OI score. We need to recalculate it
        # using the `mtf_oi_conf`.

        conf_price_vol = row['conf_price_vol']

        # Use the correct Bayesian blending formula from confidence_features.py
        if conf_cfg.oi_integration == "bayes":
            # Convert probabilities to logits
            logit_pv = np.log(conf_price_vol / max(1e-6, 1 - conf_price_vol))
            logit_oi_mtf = np.log(mtf_oi_conf / max(1e-6, 1 - mtf_oi_conf))

            # Weighted average in logit space
            combined_logit = (1 - conf_cfg.bayes_m) * logit_pv + conf_cfg.bayes_m * logit_oi_mtf

            # Convert back to probability
            conf_total = 1.0 / (1.0 + np.exp(-combined_logit))
        else: # Fallback to simple average
            conf_total = 0.5 * (conf_price_vol + mtf_oi_conf)

        # Update the row with the corrected final score and MTF alignment
        row['conf_total'] = float(conf_total)
        row['mtf_oi_align_score'] = float(mtf_align_score)
        final_rows_to_insert.append(row)

    _insert_confidence_state(final_rows_to_insert)
    print(f"✅ Confidence scores calculated for {symbol}")

def run(symbols: Optional[List[str]] = None):
    """
    Main runner for the confidence worker.
    """
    if symbols is None:
        rows = fetch_batch_universe(limit=50)
        symbols = [r["symbol"] for r in rows]

    for symbol in symbols:
        try:
            process_symbol(symbol, "futures")
        except Exception as e:
            print(f"❌ Error processing {symbol}: {e}")
            traceback.print_exc()

if __name__ == "__main__":
    run()
