# ml/flow_train.py

import os
import json
import configparser
import argparse
from dataclasses import dataclass
from typing import List, Optional, Tuple
from datetime import datetime, timedelta, timezone, time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from psycopg2.extras import execute_values
from pandas.tseries.offsets import DateOffset

from utils.db import get_db_connection
from pillars.common import TZ
from flow.pillar.flow_features_optimized import FlowFeatureEngine, load_daily_futures, load_external_metrics

ML_TARGETS_INI = "ml_targets.ini"

# ---------------------------------------------------------------------
# Optional status hook (safe no-op if not implemented)
# ---------------------------------------------------------------------
try:
    from flow.runtime.utils import update_flow_ml_jobs_status  # type: ignore
except Exception:
    def update_flow_ml_jobs_status(*args, **kwargs):
        print("[FLOW_ML] update_flow_ml_jobs_status not wired; skipping status update")


# ------------------------------------------------------------
# 1) Target config
# ------------------------------------------------------------

@dataclass
class FlowMLTarget:
    """
    Configuration for a single FLOW ML target.
    """
    name: str                   # full section name, e.g. "flow_ml.target.ret_2pct_4h"
    enabled: bool
    mode: str                   # 'direction_threshold'
    base_tf: str                # '15m'
    horizon_bars: int
    threshold_up_pct: float
    threshold_down_pct: float
    min_hold_bars: int
    market_type: str = "futures"
    version: str = "xgb_v1"     # model version tag


def load_ml_targets_for_flow(
    ini_path: str = ML_TARGETS_INI,
) -> List[FlowMLTarget]:
    """
    Read [flow_ml.target.*] sections from ml_targets.ini and return FlowMLTarget objects.
    """
    cp = configparser.ConfigParser(
        inline_comment_prefixes=(";", "#"),
        interpolation=None,
        strict=False,
    )
    cp.read(ini_path)

    targets: List[FlowMLTarget] = []

    for sec in cp.sections():
        if not sec.startswith("flow_ml.target."):
            continue

        enabled = cp.getboolean(sec, "enabled", fallback=True)
        mode = cp.get(sec, "mode", fallback="direction_threshold").strip()
        base_tf = cp.get(sec, "base_tf", fallback="15m").strip()
        horizon_bars = cp.getint(sec, "horizon_bars", fallback=16)

        threshold_up_pct = cp.getfloat(sec, "threshold_up_pct", fallback=2.0)
        threshold_down_pct = cp.getfloat(sec, "threshold_down_pct", fallback=-2.0)
        min_hold_bars = cp.getint(sec, "min_hold_bars", fallback=0)

        market_type = cp.get(sec, "market_type", fallback="futures").strip()

        targets.append(
            FlowMLTarget(
                name=sec,  # use full section name as target_name
                enabled=enabled,
                mode=mode,
                base_tf=base_tf,
                horizon_bars=horizon_bars,
                threshold_up_pct=threshold_up_pct,
                threshold_down_pct=threshold_down_pct,
                min_hold_bars=min_hold_bars,
                market_type=market_type,
            )
        )

    return targets


# ------------------------------------------------------------
# 2) Load features + future returns
# ------------------------------------------------------------

def load_flow_features_and_returns(
    target: FlowMLTarget,
    lookback_days: int = 180,
    force_all_data: bool = False,
) -> pd.DataFrame:
    """
    Build the training dataframe for a given FLOW ML target.

    - Uses market.futures_candles as price source (15m)
    - Uses indicators.futures_frames as indicator feature source (same TF)
    - Computes future_ret_pct using LEAD(close, horizon_bars)
    """

    if target.market_type != "futures":
        raise NotImplementedError(
            f"Only market_type='futures' supported for now, got {target.market_type!r}"
        )

    base_tf = target.base_tf
    if base_tf not in ("15m",):
        raise NotImplementedError(
            f"Only base_tf='15m' supported initially, got {base_tf!r}"
        )

    if force_all_data:
        # Load essentially everything (e.g. 5 years)
        cutoff = datetime.now(TZ) - timedelta(days=365 * 5)
        print(f"[FLOW_ML] Loading ALL data (up to 5 years) for target={target.name}")
    else:
        cutoff = datetime.now(TZ) - timedelta(days=lookback_days)

    sql = """
    WITH base AS (
        SELECT
            f.symbol,
            f.ts,
            f.close,
            f.volume,
            f.oi,
            LEAD(f.close, %(horizon_bars)s)
                OVER (PARTITION BY f.symbol ORDER BY f.ts) AS future_close
        FROM market.futures_candles f
        WHERE f.interval = %(base_tf)s
          AND f.ts >= %(cutoff)s
          AND EXTRACT(ISODOW FROM f.ts) BETWEEN 1 AND 5   -- Mon–Fri
          AND f.ts::time >= TIME '09:15:00'               -- NSE open
          AND f.ts::time <= TIME '15:30:00'               -- NSE close
    )

      SELECT
            b.symbol,
            b.ts,
            %(base_tf)s AS tf,
            'futures'   AS market_type,

            -- price & target
            b.close,
            b.volume,
            b.oi,
            b.future_close,
            CASE
                WHEN b.future_close IS NULL THEN NULL
                ELSE 100.0 * (b.future_close - b.close) / NULLIF(b.close, 0)
            END AS future_ret_pct,

            -- === Indicator features from indicators.futures_frames ===
            fr.rsi,
            fr.roc,
            fr.adx,
            fr.plus_di,
            fr.minus_di,
            fr.macd,
            fr.macd_signal,
            fr.macd_hist,
            fr.atr_14,
            fr.cci,
            fr.rmi,

            fr.obv,
            fr.obv_ema,
            fr.obv_delta,
            fr.obv_zs,
            fr.obv_zl,
            fr.obv_zh,
            fr.mfi_14,

            fr.vwap_session,
            fr.vwap_rolling_20,
            fr.vwap_cumulative,

            fr.bb_diag_vol_pct,
            fr.bb_diag_csv_z,
            fr.bb_diag_obv_delta,
            fr.bb_diag_block_len,
            fr.bb_diag_vwap,

            fr.vp_val,
            fr.vp_vah,
            fr.vp_poc,

            fr.ema_5,
            fr.ema_10,
            fr.ema_20,
            fr.ema_50,

            fr.stoch_k,
            fr.stoch_d,

            fr.pivot_p,
            fr.pivot_r1,
            fr.pivot_s1,
            fr.pivot_r2,
            fr.pivot_s2,
            fr.pivot_r3,
            fr.pivot_s3,
            fr.pivot_r1_dist_pct,
            fr.pivot_s1_dist_pct

        FROM base b
        LEFT JOIN indicators.futures_frames fr
          ON fr.symbol   = b.symbol
         AND fr.interval = %(base_tf)s
         AND fr.ts       = b.ts

        ORDER BY b.symbol, b.ts
    """

    params = {
        "horizon_bars": int(target.horizon_bars),
        "base_tf": base_tf,
        "cutoff": cutoff,
    }
    with get_db_connection() as conn:
        df = pd.read_sql(sql, conn, params=params)

    if df.empty:
        raise RuntimeError(
            f"No training data returned for FLOW ML target={target.name}"
        )

    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df = df.set_index("ts").sort_index()

    # ------------------------------------------------------------------
    # 🛡 NSE 4h guardrail: ensure future horizon stays within same day
    # and does NOT cross beyond 15:30
    # ------------------------------------------------------------------
    if target.base_tf != "15m":
        raise NotImplementedError(
            f"Guardrail currently assumes base_tf='15m', got {target.base_tf!r}"
        )

    bar_minutes = 15
    horizon_minutes = bar_minutes * int(target.horizon_bars)

    # expected future timestamp in *continuous time*
    df["ts_future"] = df.index + pd.Timedelta(minutes=horizon_minutes)

    same_day = df.index.date == df["ts_future"].dt.date
    within_close = df["ts_future"].dt.time <= time(15, 30)

    mask = same_day & within_close

    # any row that violates the 4h-intraday constraint → treat as no label
    df.loc[~mask, ["future_close", "future_ret_pct"]] = np.nan

    # we need valid forward return ONLY after applying guardrail
    df = df.dropna(subset=["future_ret_pct"])

    # ------------------------------------------------------------------
    # 3) Enrich with Advanced Flow Features (VOI, Daily, Session)
    # ------------------------------------------------------------------
    print(f"[FLOW_ML] Enriching with Advanced Flow Features for {len(df)} rows...")

    # We need to process per-symbol because FlowFeatureEngine is symbol-aware (daily futures merge)
    enriched_chunks = []

    # Get unique symbols in the dataset
    symbols = df["symbol"].unique()

    for sym in symbols:
        # Slice for this symbol
        df_sym = df[df["symbol"] == sym].copy()
        if df_sym.empty:
            continue

        # Load Daily Data for this symbol
        daily_df = load_daily_futures(sym)

        # Load External Metrics
        start_ts = df_sym.index[0]
        end_ts = df_sym.index[-1]
        metrics_df = load_external_metrics(sym, start_ts, end_ts)
        # Filter metrics for current TF if possible, though engine handles it
        if not metrics_df.empty and "interval" in metrics_df.columns:
             metrics_df = metrics_df[metrics_df['interval'] == base_tf]

        # Initialize Engine
        engine = FlowFeatureEngine(sym, "futures")

        # Compute Features
        # The engine expects a dataframe with OHLCV+OI. 'df_sym' has them from SQL.
        # It returns the dataframe with NEW columns appended.
        df_enriched = engine.compute_all_features(df_sym, daily_df, metrics_df)

        enriched_chunks.append(df_enriched)

    if enriched_chunks:
        df = pd.concat(enriched_chunks)
        df = df.sort_index()

    return df


# ------------------------------------------------------------
# 2.1) Model/feature saver for infer_daily
# ------------------------------------------------------------

def save_flow_models_and_features(
    target_name: str,
    version: str,
    model_up,
    model_dn,
    feature_cols: list,
):
    """
    Save UP/DOWN models + feature list in the format expected by flow_infer_daily.py.
    """
    base_dir = Path(__file__).resolve().parent  # .../flow/ml
    models_dir = (base_dir / ".." / ".." / "models" / "flow").resolve()
    models_dir.mkdir(parents=True, exist_ok=True)

    up_path = models_dir / f"{target_name}__{version}__up.pkl"
    dn_path = models_dir / f"{target_name}__{version}__down.pkl"
    meta_path = models_dir / f"{target_name}__{version}__features.json"

    joblib.dump(model_up, up_path)
    joblib.dump(model_dn, dn_path)

    with open(meta_path, "w") as f:
        json.dump({"features": list(feature_cols)}, f, indent=2)

    print(f"[FLOW_ML] Saved UP model   → {up_path}")
    print(f"[FLOW_ML] Saved DOWN model → {dn_path}")
    print(f"[FLOW_ML] Saved features   → {meta_path}")


# ------------------------------------------------------------
# 3) Label builder (bullish 2%+ for now)
# ------------------------------------------------------------

def build_up_label_direction_threshold(
    df: pd.DataFrame,
    target: FlowMLTarget,
) -> pd.Series:
    """
    For mode='direction_threshold':
      y = 1 if future_ret_pct >= threshold_up_pct
          0 otherwise
    """
    if "future_ret_pct" not in df.columns:
        raise ValueError("DataFrame missing 'future_ret_pct' column")

    y = (df["future_ret_pct"] >= target.threshold_up_pct).astype(int)
    return y


# ------------------------------------------------------------
# 4) Training Logic (Production vs Backfill)
# ------------------------------------------------------------

def prepare_features_and_labels(
    df: pd.DataFrame, target: FlowMLTarget
) -> Tuple[pd.DataFrame, pd.Series, List[str], pd.DataFrame]:
    """
    Common helper:
    1. Build labels
    2. Filter NA
    3. Select numeric features
    4. Fill NaN/Inf
    Returns (X, y, feature_cols, df_clean)
    """
    # 1) Build labels
    if target.mode == "direction_threshold":
        y = build_up_label_direction_threshold(df, target)
    else:
        raise NotImplementedError(
            f"Unsupported mode={target.mode} for target={target.name}"
        )

    # 2) Filter rows with valid label
    mask = y.notna()
    df_clean = df.loc[mask].copy()
    y_clean = y.loc[mask]

    if df_clean.empty:
        return pd.DataFrame(), pd.Series(), [], pd.DataFrame()

    # 3) Build feature matrix X
    drop_cols = {
        "future_ret_pct",
        "future_close",
        "market_type",
        "tf",
        "symbol",
        "ts_future",
    }

    features_df = df_clean.drop(columns=[c for c in drop_cols if c in df_clean.columns])
    features_df = features_df.select_dtypes(include=["number", "bool"])
    feature_cols = list(features_df.columns)

    X = features_df.copy()
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    return X, y_clean, feature_cols, df_clean


def train_model_production(target: FlowMLTarget, df: pd.DataFrame):
    """
    Mode: PRODUCTION
    - Trains on ALL provided data.
    - Saves .pkl models for runtime usage.
    - DOES NOT write historical predictions to DB (avoids pollution).
    """
    print(f"[FLOW_ML] [PRODUCTION] Training final model for target={target.name}")

    X, y, feature_cols, _ = prepare_features_and_labels(df, target)

    if X.empty:
        print(f"[FLOW_ML] No usable features for target={target.name}, skipping")
        return

    # Train XGBoost
    model = XGBClassifier(
        max_depth=4,
        n_estimators=200,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        n_jobs=4,
    )
    model.fit(X, y.values)

    # Save artifact
    save_flow_models_and_features(
        target_name=target.name,
        version=target.version,
        model_up=model,
        model_dn=model,
        feature_cols=feature_cols,
    )

    # Optional: Print metrics (Accuracy on train set - purely informational)
    preds = model.predict(X)
    acc = (preds == y.values).mean()
    print(f"[FLOW_ML] [PRODUCTION] In-sample Accuracy: {acc:.2%}")


def train_model_backfill(
    target: FlowMLTarget,
    df: pd.DataFrame,
    step_months: int = 1,
    initial_train_months: int = 3
):
    """
    Mode: BACKFILL (Rolling Walk-Forward)
    - Loop: Train[Past] -> Predict[Next Month]
    - Does NOT save .pkl models.
    - Writes OOF probabilities to indicators.ml_pillars.
    """
    print(f"[FLOW_ML] [BACKFILL] Starting Rolling Walk-Forward for {target.name}...")

    # Ensure index is sorted
    df = df.sort_index()
    if df.empty:
        print("[FLOW_ML] No data to backfill.")
        return

    start_ts = df.index.min()
    max_ts = df.index.max()

    # Define start of the "Prediction" phase
    # First training window = [start_ts, current_cursor)
    # First prediction window = [current_cursor, current_cursor + step)

    # Align cursor to start of a month for cleaner boundaries?
    # Or just add offsets. Let's use offsets from start_ts.

    current_cursor = start_ts + DateOffset(months=initial_train_months)

    all_oof_rows = []

    # Expanding Window Loop
    while current_cursor < max_ts:
        next_cursor = current_cursor + DateOffset(months=step_months)

        # 1. Define Slices
        # Train: Everything before current_cursor
        train_mask = df.index < current_cursor
        # Test: Everything in [current_cursor, next_cursor)
        test_mask = (df.index >= current_cursor) & (df.index < next_cursor)

        df_train = df.loc[train_mask]
        df_test = df.loc[test_mask]

        if df_train.empty or df_test.empty:
            print(f"[FLOW_ML] Skipping window {current_cursor.date()} -> {next_cursor.date()} (insufficient data)")
            current_cursor = next_cursor
            continue

        print(f"[FLOW_ML] Window: Train up to {current_cursor.date()} | Predict {current_cursor.date()} to {next_cursor.date()}")

        # 2. Train Model
        X_train, y_train, feature_cols, _ = prepare_features_and_labels(df_train, target)
        if X_train.empty:
            current_cursor = next_cursor
            continue

        model = XGBClassifier(
            max_depth=4,
            n_estimators=200,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="logloss",
            n_jobs=4,
        )
        model.fit(X_train, y_train.values)

        # 3. Predict on Test (OOF)
        # Use prepare_features_and_labels to get X_test and the aligned df_test_clean
        X_test, _, _, df_test_clean = prepare_features_and_labels(df_test, target)

        # Re-align X_test columns to match X_train
        missing_cols = set(feature_cols) - set(X_test.columns)
        for c in missing_cols:
            X_test[c] = 0.0
        X_test = X_test[feature_cols] # Enforce order

        if X_test.empty:
            current_cursor = next_cursor
            continue

        prob_long = model.predict_proba(X_test)[:, 1]
        prob_short = 1.0 - prob_long

        # 4. Collect Results
        context_json = json.dumps({
            "base_tf": target.base_tf,
            "horizon_bars": target.horizon_bars,
            "features": feature_cols,
            "mode": "backfill_oof"
        })

        # Safely iterate over the aligned dataframe rows
        for (idx, row), p_long, p_short in zip(df_test_clean.iterrows(), prob_long, prob_short):
            all_oof_rows.append((
                row["symbol"],
                target.market_type,
                "flow",
                row["tf"],
                idx.to_pydatetime(),
                target.name,
                target.version,
                float(p_long),
                float(p_short),
                float(row["future_ret_pct"]),
                context_json
            ))

        # Move cursor
        current_cursor = next_cursor

    # 5. Bulk Write to DB
    if not all_oof_rows:
        print(f"[FLOW_ML] No OOF predictions generated for {target.name}")
        return

    print(f"[FLOW_ML] Writing {len(all_oof_rows)} OOF rows to indicators.ml_pillars for {target.name}...")

    sql = """
        INSERT INTO indicators.ml_pillars
            (symbol, market_type, pillar, tf, ts,
             target_name, version,
             prob_up, prob_down, future_ret_pct, context)
        VALUES %s
        ON CONFLICT (symbol, market_type, pillar, tf, ts, target_name, version)
        DO UPDATE
           SET prob_up       = EXCLUDED.prob_up,
               prob_down     = EXCLUDED.prob_down,
               future_ret_pct = EXCLUDED.future_ret_pct,
               context        = EXCLUDED.context,
               created_at     = now()
    """

    with get_db_connection() as conn, conn.cursor() as cur:
        execute_values(cur, sql, all_oof_rows, page_size=1000)
        conn.commit()

    print("[FLOW_ML] Backfill complete.")


# ------------------------------------------------------------
# 5) Orchestrator
# ------------------------------------------------------------

def run_flow_ml_pipeline(
    targets_ini: str = ML_TARGETS_INI,
    lookback_days: int = 180,
    only_target: Optional[str] = None,
    mode: str = "production",
    step_months: int = 1,
):
    """
    Main entry point.
    """
    targets = load_ml_targets_for_flow(targets_ini)
    if not targets:
        print(f"[FLOW_ML] No targets found in {targets_ini}")
        return

    for target in targets:
        if not target.enabled:
            continue
        if only_target and target.name != only_target:
            continue

        print(f"--- Processing Target: {target.name} (Mode: {mode.upper()}) ---")

        # 1. Load Data
        # For backfill, we force loading ALL data to enable the walk-forward loop
        force_all = (mode == "backfill")

        try:
            df = load_flow_features_and_returns(
                target,
                lookback_days=lookback_days,
                force_all_data=force_all
            )
        except RuntimeError as e:
            print(f"[FLOW_ML] Data load error: {e}")
            continue

        # 2. Dispatch
        if mode == "production":
            train_model_production(target, df)
        elif mode == "backfill":
            train_model_backfill(
                target,
                df,
                step_months=step_months,
                initial_train_months=3 # First 3 months for initial training
            )
        else:
            print(f"Unknown mode: {mode}")

# ------------------------------------------------------------
# 6) CLI
# ------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets-ini", default=ML_TARGETS_INI,
                        help="Path to ml_targets.ini")
    parser.add_argument("--lookback-days", type=int, default=180,
                        help="Lookback days for production training")
    parser.add_argument(
        "--only-target",
        default=None,
        help="Exact target section name",
    )
    parser.add_argument(
        "--mode",
        choices=["production", "backfill"],
        default="production",
        help="Mode: 'production' (train all, save model) or 'backfill' (rolling OOF, save DB)"
    )
    parser.add_argument(
        "--step-months",
        type=int,
        default=1,
        help="For backfill mode: Step size in months for rolling window"
    )

    args = parser.parse_args()

    run_flow_ml_pipeline(
        targets_ini=args.targets_ini,
        lookback_days=args.lookback_days,
        only_target=args.only_target,
        mode=args.mode,
        step_months=args.step_months,
    )


if __name__ == "__main__":
    main()
