from __future__ import annotations

import argparse
import datetime as dt
import logging
import sys
import time
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import pandas as pd
from sqlalchemy import create_engine

from utils.db import get_db_connection
from pillars.common import resample, BaseCfg, write_values
from momentum.pillar.momentum_features_optimized import MomentumFeatureEngine
from momentum.ml.momentum_fuse_backfill import MomentumFuser

# ---------------------------------
# LOGGING
# ---------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------
# CONFIG
# ---------------------------------
BASE_TF = "15m"
TF_LIST = ["15m", "30m", "60m", "120m", "240m"]
WARMUP_DAYS = 20  # Sufficient warmup for EMA50, BB20, etc.

# ---------------------------------
# HELPER: Fuser (ML Lookup)
# ---------------------------------
def _get_ml_probs(symbol: str, market_type: str, tfs: List[str], start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> pd.DataFrame:
    """
    Fetch ML probabilities from indicators.ml_pillars for the given window.
    Returns DataFrame indexed by (tf, ts) with column 'prob_up'.
    """
    # Always fetch from futures?
    # Usually ML is stored under market_type='futures' even for spot?
    # Or strict match?
    # Let's try strict match first, if empty, maybe futures?
    # For now, stick to request: "Spot Rules to join with Futures ML Data".

    target_kind = 'futures' # As per Flow fix

    sql = """
        SELECT tf, ts, prob_up
          FROM indicators.ml_pillars
         WHERE pillar = 'momentum'
           AND symbol = %s
           AND market_type = %s
           AND ts BETWEEN %s AND %s
           AND target_name = 'momentum_ml.target.bull_2pct_4h'
    """
    with get_db_connection() as conn:
        df = pd.read_sql(sql, conn, params=(symbol, target_kind, start_ts, end_ts))

    if df.empty:
        return pd.DataFrame()

    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df.set_index(["tf", "ts"])


def _load_calibration(target_name="momentum_ml.target.bull_2pct_4h") -> pd.DataFrame:
    """Load calibration table: p_min, p_max, realized_up_rate."""
    sql = """
        SELECT p_min, p_max, realized_up_rate
          FROM indicators.momentum_calibration_4h
         WHERE target_name = %s
    """
    try:
        with get_db_connection() as conn:
            df = pd.read_sql(sql, conn, params=(target_name,))
        return df
    except Exception:
        return pd.DataFrame()


def _calibrate_prob(prob: float, calib_df: pd.DataFrame) -> float:
    if pd.isna(prob):
        return prob
    # Find row where p_min <= prob < p_max
    # Naive iteration is slow but safe. Vectorized logic preferred.
    # But here we do it per-row or vectorized?
    # Let's do vectorized in main loop if possible, or apply.
    # For backfill speed, vector lookup is better.
    return prob # Placeholder, we'll do it vectorized in process_symbol


# ---------------------------------
# CLI
# ---------------------------------
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Momentum pillar backfill (Vectorized)")

    g_sym = p.add_mutually_exclusive_group(required=True)
    g_sym.add_argument("--symbol", help="Single symbol")
    g_sym.add_argument("--all-symbols", action="store_true", help="All enabled symbols")

    p.add_argument("--kind", choices=["futures", "spot"], required=True)
    p.add_argument("--mode", choices=["rules", "ml", "both"], default="both")

    p.add_argument("--from-date", help="YYYY-MM-DD start date")
    p.add_argument("--lookback-days", type=int, default=None)
    p.add_argument("--momentum-ini", help="Path to momentum_scenarios.ini")
    p.add_argument("--run-id", default=None)
    p.add_argument("--max-symbols", type=int, default=None)

    return p.parse_args()


# ---------------------------------
# Universe
# ---------------------------------
def _load_universe(kind: str, symbol: Optional[str] = None, max_symbols: Optional[int] = None) -> pd.DataFrame:
    cols = [
        "symbol",
        "mom_rules_last_spot_ts",
        "mom_rules_last_fut_ts",
        "mom_ml_last_spot_ts",
        "mom_ml_last_fut_ts",
        "enabled",
    ]
    sql = f"SELECT {', '.join(cols)} FROM reference.symbol_universe WHERE enabled = TRUE"
    params = []
    if symbol:
        sql += " AND symbol = %s"
        params.append(symbol)
    sql += " ORDER BY symbol"
    if max_symbols:
        sql += f" LIMIT {max_symbols}"

    with get_db_connection() as conn:
        df = pd.read_sql(sql, conn, params=params)
    return df


# ---------------------------------
# Data Loader
# ---------------------------------
def _load_15m_candles(symbol: str, kind: str, start_ts: pd.Timestamp) -> pd.DataFrame:
    table = "market.futures_candles" if kind == "futures" else "market.spot_candles"

    # Warmup
    fetch_from = start_ts - pd.Timedelta(days=WARMUP_DAYS)

    sql = f"""
        SELECT ts, open, high, low, close, volume
          FROM {table}
         WHERE symbol = %s
           AND ts >= %s
         ORDER BY ts
    """
    with get_db_connection() as conn:
        df = pd.read_sql(sql, conn, params=(symbol, fetch_from))

    if df.empty:
        return pd.DataFrame()

    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df.set_index("ts").sort_index()


# ---------------------------------
# Logic
# ---------------------------------
def _process_symbol(
    row: pd.Series,
    kind: str,
    args: argparse.Namespace,
    now: dt.datetime,
    engine: MomentumFeatureEngine,
    fuser: MomentumFuser,
    calib_df: pd.DataFrame
) -> Dict[str, Optional[pd.Timestamp]]:

    symbol = row["symbol"]

    # 1. Determine Time Window
    # Get last runs
    if kind == "futures":
        last_rules = pd.to_datetime(row["mom_rules_last_fut_ts"]).tz_localize("UTC") if pd.notnull(row["mom_rules_last_fut_ts"]) else None
        last_ml = pd.to_datetime(row["mom_ml_last_fut_ts"]).tz_localize("UTC") if pd.notnull(row["mom_ml_last_fut_ts"]) else None
    else:
        last_rules = pd.to_datetime(row["mom_rules_last_spot_ts"]).tz_localize("UTC") if pd.notnull(row["mom_rules_last_spot_ts"]) else None
        last_ml = pd.to_datetime(row["mom_ml_last_spot_ts"]).tz_localize("UTC") if pd.notnull(row["mom_ml_last_spot_ts"]) else None

    # Determine start_ts
    start_ts = None
    if args.from_date:
        start_ts = pd.Timestamp(args.from_date, tz="UTC")
    elif args.lookback_days:
        start_ts = pd.Timestamp(now, tz="UTC") - pd.Timedelta(days=args.lookback_days)
    else:
        # Incremental
        # Min of rules/ml or just rules if mode=rules?
        # Simplify: take min of existing checkpoints, or default to 90 days ago if None
        ts_list = [t for t in [last_rules, last_ml] if t is not None]
        if ts_list:
            start_ts = min(ts_list)
        else:
            start_ts = pd.Timestamp(now, tz="UTC") - pd.Timedelta(days=90)

    logger.info(f"Processing {symbol} ({kind}) from {start_ts}...")

    # 2. Load Data
    df15 = _load_15m_candles(symbol, kind, start_ts)
    if df15.empty:
        logger.warning(f"{symbol}: No data found.")
        return {}

    # 3. Load ML Probs (Batch)
    # We fetch ML probs for the entire range + TFs
    ml_probs_df = _get_ml_probs(symbol, kind, TF_LIST, df15.index[0], df15.index[-1])

    # 4. Loop Timeframes
    latest_ts_map = {}

    for tf in TF_LIST:
        # Resample
        if tf == "15m":
            dftf = df15.copy()
        else:
            dftf = resample(df15, tf)

        if dftf.empty:
            continue

        # Filter to requested start_ts (after resampling, so we have valid history)
        dftf = dftf[dftf.index >= start_ts]
        if dftf.empty:
            continue

        # Feature Engine
        try:
            feats = engine.compute_features(dftf)
        except Exception as e:
            logger.error(f"{symbol} {tf}: Feature computation failed: {e}")
            continue

        # Prepare Results
        # feats has 'MOM.score', 'MOM.veto_flag'

        # 5. Fusion Step
        # Lookup ML prob
        # ml_probs_df index: (tf, ts)

        # We need to align ml_probs with feats
        # Create a temp series for mapping
        if not ml_probs_df.empty and tf in ml_probs_df.index.get_level_values(0):
            # Extract ML for this TF
            ml_slice = ml_probs_df.xs(tf, level="tf") # Index is now ts
            # Join?
            # feats index is ts
            feats = feats.join(ml_slice["prob_up"].rename("raw_prob"), how="left")
        else:
            feats["raw_prob"] = np.nan

        # Calibrate ML Prob
        # We use calib_df: p_min, p_max, realized_up_rate
        # Vectorized bucket lookup using pd.cut?
        feats["calib_prob"] = feats["raw_prob"] # Default uncalibrated

        if not calib_df.empty:
            # Simple apply for calibration
            def get_calib(p):
                if pd.isna(p): return p
                # Simple linear scan of calib_df (sorted by p_max)
                for r in calib_df.itertuples():
                    if p < r.p_max:
                        return r.realized_up_rate
                return p # Fallback

            feats["calib_prob"] = feats["raw_prob"].apply(get_calib)

        # Fusion Formula
        # fused = (1-w)*rules + w*ml
        # w = 0.35 default
        w = 0.35
        # rules_score is 0..100. Convert to prob 0..1
        feats["rules_prob"] = feats["MOM.score"] / 100.0

        # If ML is NaN, fused = rules
        feats["ml_prob_final"] = feats["calib_prob"].fillna(feats["rules_prob"]) # If missing, use rules

        # Let's apply weight only if ML exists (raw_prob not null)
        # Using np.where
        has_ml = feats["raw_prob"].notnull()

        feats["fused_prob"] = feats["rules_prob"] # Default
        feats.loc[has_ml, "fused_prob"] = (1.0 - w) * feats.loc[has_ml, "rules_prob"] + w * feats.loc[has_ml, "calib_prob"]

        feats["MOM.fused_score"] = (feats["fused_prob"].clip(0, 1) * 100.0).round(2)
        feats["MOM.ml_p_up_cal"] = feats["calib_prob"] # Store for fusion script

        # 6. Apply Sigmoid & Bucket (from Fusion Script logic)
        # We can use MomentumFuser directly here!
        # compute_fusion_metrics expects DF with MOM.fused_score, MOM.ml_p_up_cal
        feats = fuser.compute_fusion_metrics(feats)

        # 7. Write to DB
        # Prepare rows
        rows = []
        run_id = args.run_id or f"MOM_BF_{now.strftime('%Y%m%d')}"

        for ts, row_data in feats.iterrows():
            ts_str = ts.to_pydatetime()

            # Base Rules
            rows.append((symbol, kind, tf, ts_str, "MOM.score", float(row_data["MOM.score"]), "{}", run_id, "mom_opt"))
            rows.append((symbol, kind, tf, ts_str, "MOM.veto_flag", float(row_data["MOM.veto_flag"]), "{}", run_id, "mom_opt"))

            # Fused
            rows.append((symbol, kind, tf, ts_str, "MOM.fused_score", float(row_data["MOM.fused_score"]), "{}", run_id, "mom_opt"))
            rows.append((symbol, kind, tf, ts_str, "MOM.ml_p_up_cal", float(row_data["MOM.ml_p_up_cal"]) if pd.notnull(row_data["MOM.ml_p_up_cal"]) else 0.0, "{}", run_id, "mom_opt"))

            # Final & Bucket
            rows.append((symbol, kind, tf, ts_str, "MOM.score_final", float(row_data["MOM.score_final"]), "{}", run_id, "mom_opt"))
            rows.append((symbol, kind, tf, ts_str, "MOM.ml_bucket", float(row_data["MOM.ml_bucket"]), "{}", run_id, "mom_opt"))

            # ML Score (0-100)
            if pd.notnull(row_data.get("calib_prob")):
                rows.append((symbol, kind, tf, ts_str, "MOM.ml_score", float(row_data["calib_prob"]*100.0), "{}", run_id, "mom_opt"))

        if rows:
            write_values(rows)
            latest_ts_map[tf] = feats.index[-1]

    # Return latest timestamps for Universe update
    if not latest_ts_map:
        return {}

    max_ts = max(latest_ts_map.values())
    return {
        "rules_last_ts": max_ts,
        "ml_last_ts": max_ts
    }

# ---------------------------------
# Status Update
# ---------------------------------
def _update_universe(kind: str, updates: Dict[str, Dict[str, pd.Timestamp]]):
    if not updates:
        return

    now_utc = dt.datetime.now(dt.timezone.utc)

    with get_db_connection() as conn, conn.cursor() as cur:
        for symbol, data in updates.items():
            ts = data.get("rules_last_ts")
            if ts is None: continue

            logger.info(f"[AUDIT] Updating {symbol} {kind} -> {ts}")

            if kind == "futures":
                cur.execute("""
                    UPDATE reference.symbol_universe
                       SET mom_rules_last_fut_ts = %s,
                           mom_rules_last_run_at = %s,
                           mom_ml_last_fut_ts = %s,
                           mom_ml_last_run_at = %s
                     WHERE symbol = %s
                """, (ts, now_utc, ts, now_utc, symbol))
            else:
                cur.execute("""
                    UPDATE reference.symbol_universe
                       SET mom_rules_last_spot_ts = %s,
                           mom_rules_last_run_at = %s,
                           mom_ml_last_spot_ts = %s,
                           mom_ml_last_run_at = %s
                     WHERE symbol = %s
                """, (ts, now_utc, ts, now_utc, symbol))
        conn.commit()

# ---------------------------------
# MAIN
# ---------------------------------
def main():
    args = _parse_args()
    now = dt.datetime.now(dt.timezone.utc)

    # Init Engine
    engine = MomentumFeatureEngine(ini_path=args.momentum_ini)
    fuser = MomentumFuser(symbol="DUMMY")

    # Load Calibration once
    calib_df = _load_calibration()

    # Universe
    df_univ = _load_universe(args.kind, args.symbol, args.max_symbols)
    logger.info(f"Loaded {len(df_univ)} symbols.")

    updates = {}

    for row in df_univ.itertuples():
        try:
            row_s = pd.Series(row._asdict())
            res = _process_symbol(row_s, args.kind, args, now, engine, fuser, calib_df)
            if res:
                updates[row.symbol] = res
        except Exception as e:
            logger.error(f"Error processing {row.symbol}: {e}", exc_info=True)

    _update_universe(args.kind, updates)
    logger.info("Backfill Complete.")

if __name__ == "__main__":
    main()
