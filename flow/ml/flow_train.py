# ml/flow_train.py

import os
import json
import configparser
from dataclasses import dataclass
from typing import List, Optional
from datetime import datetime, timedelta, timezone, time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from psycopg2.extras import execute_values

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

    Example INI section:

        [flow_ml.target.ret_2pct_4h]
        enabled            = true
        mode               = direction_threshold
        base_tf            = 15m
        horizon_bars       = 16
        threshold_up_pct   = 2.0
        threshold_down_pct = -2.0
        min_hold_bars      = 4
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

    cutoff = datetime.now(TZ) - timedelta(days=lookback_days)

    sql = """
    WITH base AS (
        SELECT
            f.symbol,
            f.ts,
            f.open,
            f.high,
            f.low,
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
            b.open,
            b.high,
            b.low,
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
# 4) Orchestrator: train + write to DB
# ------------------------------------------------------------

def train_and_write_all_flow_models(
    targets_ini: str = ML_TARGETS_INI,
    lookback_days: int = 180,
    only_target: Optional[str] = None,
):
    """
    1) Load targets from ml_targets.ini
    2) For each enabled target (or one filtered by only_target):
         - Load features + future_ret_pct
         - Build labels
         - Train XGB model
         - Save model to disk
         - Save models+features for infer_daily
         - Write in-sample probabilities to indicators.ml_pillars
    """
    targets = load_ml_targets_for_flow(targets_ini)
    if not targets:
        print(f"[FLOW_ML] No [flow_ml.target.*] sections found in {targets_ini}")
        return

    for target in targets:
        if not target.enabled:
            continue
        if only_target and target.name != only_target:
            continue

        print(
            f"[FLOW_ML] Training target={target.name} "
            f"base_tf={target.base_tf} horizon_bars={target.horizon_bars}"
        )

        # 1) Load training dataframe
        df = load_flow_features_and_returns(target, lookback_days=lookback_days)

        # 2) Build labels (bullish direction for v1)
        if target.mode == "direction_threshold":
            y = build_up_label_direction_threshold(df, target)
        else:
            raise NotImplementedError(
                f"Unsupported mode={target.mode} for target={target.name}"
            )

        mask = y.notna()
        df = df.loc[mask]
        y = y.loc[mask]

        # 3) Build feature matrix X
        drop_cols = {
            "future_ret_pct",
            "future_close",
            "market_type",
            "tf",
            # explicitly never used as features:
            "symbol",
        }

        # Start by dropping obvious non-feature columns
        features_df = df.drop(columns=[c for c in drop_cols if c in df.columns])

        # Keep ONLY numeric / bool columns (drops any datetime, object etc.)
        features_df = features_df.select_dtypes(include=["number", "bool"])

        feature_cols = list(features_df.columns)

        X = features_df.copy()
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        if X.empty:
            print(f"[FLOW_ML] No usable features for target={target.name}, skipping")
            continue

        # 4) Train XGBoost model
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

        # 5) Save model to disk in legacy JSON format (optional)
        os.makedirs("models/flow", exist_ok=True)
        model_path = os.path.join("models", "flow", f"{target.name}_{target.version}.json")
        model.save_model(model_path)
        print(f"[FLOW_ML] Saved model → {model_path}")

        # 5b) Save models + features in the format expected by flow_infer_daily.py
        # For now we use the same classifier for "up" and "down" (we don't train a separate
        # down model yet). If we later add a dedicated down-model, we can wire it here.
        save_flow_models_and_features(
            target_name=target.name,
            version=target.version,
            model_up=model,
            model_dn=model,
            feature_cols=feature_cols,
        )

        # 6) In-sample probabilities (v1)
        prob_long = model.predict_proba(X)[:, 1]
        prob_short = 1.0 - prob_long

        # 7) Prepare rows for indicators.ml_pillars
        rows = []
        for (ts, row), p_long, p_short in zip(df.iterrows(), prob_long, prob_short):
            rows.append(
                (
                    row["symbol"],
                    target.market_type,   # 'futures'
                    "flow",               # pillar
                    row["tf"],
                    ts.to_pydatetime(),
                    target.name,          # target_name (full INI section)
                    target.version,
                    float(p_long),
                    float(p_short),
                    float(row["future_ret_pct"]),
                    json.dumps(
                        {
                            "base_tf": target.base_tf,
                            "horizon_bars": target.horizon_bars,
                            "features": feature_cols,
                        }
                    ),
                )
            )

        if not rows:
            print(f"[FLOW_ML] No rows to write for target={target.name}")
            continue

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
            execute_values(cur, sql, rows, page_size=1000)
            conn.commit()

            # mark calibration time (if wired)
            update_flow_ml_jobs_status(conn, calib_at=datetime.now(timezone.utc))

        print(
            f"[FLOW_ML] Wrote {len(rows)} rows into indicators.ml_pillars "
            f"for target={target.name}"
        )


# ------------------------------------------------------------
# 5) CLI
# ------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--targets-ini", default=ML_TARGETS_INI,
                        help="Path to ml_targets.ini")
    parser.add_argument("--lookback-days", type=int, default=180)
    parser.add_argument(
        "--only-target",
        default=None,
        help="Exact target section name, e.g. flow_ml.target.bull_2pct_4h",
    )
    args = parser.parse_args()

    train_and_write_all_flow_models(
        targets_ini=args.targets_ini,
        lookback_days=args.lookback_days,
        only_target=args.only_target,
    )


if __name__ == "__main__":
    main()
