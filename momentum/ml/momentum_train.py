# momentum/ml/momentum_train.py

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
from momentum.pillar.momentum_features_optimized import MomentumFeatureEngine

ML_TARGETS_INI = "momentum/ml/ml_targets.ini"

# ------------------------------------------------------------
# 1) Target config
# ------------------------------------------------------------

@dataclass
class MomentumMLTarget:
    """
    Configuration for a single MOMENTUM ML target.
    """
    name: str                   # full section name, e.g. "momentum_ml.target.bull_2pct_4h"
    enabled: bool
    mode: str                   # 'direction_threshold'
    base_tf: str                # '15m'
    horizon_bars: int
    threshold_up_pct: float
    threshold_down_pct: float
    min_hold_bars: int
    market_type: str = "futures"
    version: str = "xgb_v1"     # model version tag


def load_ml_targets_for_momentum(
    ini_path: str = ML_TARGETS_INI,
) -> List[MomentumMLTarget]:
    """
    Read [momentum_ml.target.*] sections from ml_targets.ini
    """
    # Adjust path if relative
    if not os.path.isabs(ini_path) and not os.path.exists(ini_path):
         # try relative to file
         base = Path(__file__).resolve().parent
         ini_path = str(base / "ml_targets.ini")

    cp = configparser.ConfigParser(
        inline_comment_prefixes=(";", "#"),
        interpolation=None,
        strict=False,
    )
    cp.read(ini_path)

    targets: List[MomentumMLTarget] = []

    for sec in cp.sections():
        if not sec.startswith("momentum_ml.target."):
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
            MomentumMLTarget(
                name=sec,
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

def load_momentum_features_and_returns(
    target: MomentumMLTarget,
    lookback_days: int = 180,
) -> pd.DataFrame:
    """
    Build the training dataframe for a given MOMENTUM ML target.
    Uses MomentumFeatureEngine to compute features from raw candles.
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

    # Fetch RAW candles (OHLCV)
    sql = """
    SELECT
        symbol,
        ts,
        open, high, low, close, volume,
        LEAD(close, %(horizon_bars)s)
            OVER (PARTITION BY symbol ORDER BY ts) AS future_close
    FROM market.futures_candles
    WHERE interval = %(base_tf)s
      AND ts >= %(cutoff)s
      AND EXTRACT(ISODOW FROM ts) BETWEEN 1 AND 5   -- Mon–Fri
      AND ts::time >= TIME '09:15:00'               -- NSE open
      AND ts::time <= TIME '15:30:00'               -- NSE close
    ORDER BY symbol, ts
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
            f"No training data returned for MOMENTUM ML target={target.name}"
        )

    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df = df.set_index("ts").sort_index()

    # Calc return
    df["future_ret_pct"] = (df["future_close"] - df["close"]) / df["close"] * 100.0

    # Guardrail: Ensure horizon is same-day
    bar_minutes = 15
    horizon_minutes = bar_minutes * int(target.horizon_bars)
    df["ts_future"] = df.index + pd.Timedelta(minutes=horizon_minutes)

    same_day = df.index.date == df["ts_future"].dt.date
    within_close = df["ts_future"].dt.time <= time(15, 30)

    mask = same_day & within_close
    df.loc[~mask, "future_ret_pct"] = np.nan

    df = df.dropna(subset=["future_ret_pct"])

    # ------------------------------------------------------------------
    # Enrich with MomentumFeatureEngine
    # ------------------------------------------------------------------
    print(f"[MOM_ML] Enriching with Momentum Features for {len(df)} rows...")

    engine = MomentumFeatureEngine() # Config loaded from default INI or passed

    enriched_chunks = []
    symbols = df["symbol"].unique()

    for sym in symbols:
        df_sym = df[df["symbol"] == sym].copy()
        if df_sym.empty:
            continue

        # Compute features
        # engine.compute_features returns df with new cols
        try:
            df_enriched = engine.compute_features(df_sym)
            enriched_chunks.append(df_enriched)
        except Exception as e:
            print(f"Error computing features for {sym}: {e}")
            continue

    if enriched_chunks:
        df = pd.concat(enriched_chunks)
        df = df.sort_index()

    return df


# ------------------------------------------------------------
# 2.1) Model/feature saver
# ------------------------------------------------------------

def save_momentum_models_and_features(
    target_name: str,
    version: str,
    model_up,
    model_dn,
    feature_cols: list,
):
    base_dir = Path(__file__).resolve().parent
    models_dir = (base_dir / ".." / ".." / "models" / "momentum").resolve()
    models_dir.mkdir(parents=True, exist_ok=True)

    up_path = models_dir / f"{target_name}__{version}__up.pkl"
    dn_path = models_dir / f"{target_name}__{version}__down.pkl"
    meta_path = models_dir / f"{target_name}__{version}__features.json"

    joblib.dump(model_up, up_path)
    joblib.dump(model_dn, dn_path)

    with open(meta_path, "w") as f:
        json.dump({"features": list(feature_cols)}, f, indent=2)

    print(f"[MOM_ML] Saved UP model   → {up_path}")
    print(f"[MOM_ML] Saved DOWN model → {dn_path}")
    print(f"[MOM_ML] Saved features   → {meta_path}")


# ------------------------------------------------------------
# 3) Label builder
# ------------------------------------------------------------

def build_up_label_direction_threshold(
    df: pd.DataFrame,
    target: MomentumMLTarget,
) -> pd.Series:
    if "future_ret_pct" not in df.columns:
        raise ValueError("DataFrame missing 'future_ret_pct' column")

    y = (df["future_ret_pct"] >= target.threshold_up_pct).astype(int)
    return y


# ------------------------------------------------------------
# 4) Orchestrator
# ------------------------------------------------------------

def train_and_write_all_momentum_models(
    targets_ini: str = ML_TARGETS_INI,
    lookback_days: int = 180,
    only_target: Optional[str] = None,
):
    targets = load_ml_targets_for_momentum(targets_ini)
    if not targets:
        print(f"[MOM_ML] No targets found in {targets_ini}")
        return

    for target in targets:
        if not target.enabled:
            continue
        if only_target and target.name != only_target:
            continue

        print(f"[MOM_ML] Training target={target.name} base_tf={target.base_tf}")

        try:
            df = load_momentum_features_and_returns(target, lookback_days=lookback_days)
        except RuntimeError as e:
            print(f"[MOM_ML] {e}")
            continue

        if target.mode == "direction_threshold":
            y = build_up_label_direction_threshold(df, target)
        else:
            raise NotImplementedError(f"Mode {target.mode} not supported")

        mask = y.notna()
        df = df.loc[mask]
        y = y.loc[mask]

        # Features
        # Drop non-features
        drop_cols = {
            "future_ret_pct", "future_close", "market_type", "tf", "symbol",
            "ts_future", "open", "high", "low", "close", "volume"
        }
        # Also drop cols that are not numeric features used by engine?
        # The engine produces specific columns.
        # We can use engine.get_feature_columns() if available, or just select numeric.

        # NOTE: We keep 'MOM.score' and 'MOM.veto_flag' as features?
        # Yes, Flow used 'FLOW.score' as feature too.

        features_df = df.drop(columns=[c for c in drop_cols if c in df.columns])
        features_df = features_df.select_dtypes(include=["number", "bool"])

        feature_cols = list(features_df.columns)
        X = features_df.copy().replace([np.inf, -np.inf], np.nan).fillna(0.0)

        if X.empty:
            print(f"[MOM_ML] No features for {target.name}")
            continue

        # Train
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

        # Save
        save_momentum_models_and_features(
            target_name=target.name,
            version=target.version,
            model_up=model,
            model_dn=model,
            feature_cols=feature_cols,
        )

        # In-sample predictions
        prob_long = model.predict_proba(X)[:, 1]
        prob_short = 1.0 - prob_long

        rows = []
        for (ts, row), p_long, p_short in zip(df.iterrows(), prob_long, prob_short):
            rows.append(
                (
                    row["symbol"],
                    target.market_type,
                    "momentum",
                    target.base_tf, # or '15m'
                    ts.to_pydatetime(),
                    target.name,
                    target.version,
                    float(p_long),
                    float(p_short),
                    float(row["future_ret_pct"]),
                    json.dumps({"features": feature_cols})
                )
            )

        if not rows:
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
                   context        = EXCLUDED.context
        """

        with get_db_connection() as conn, conn.cursor() as cur:
            execute_values(cur, sql, rows, page_size=1000)
            conn.commit()

        print(f"[MOM_ML] Wrote {len(rows)} predictions.")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets-ini", default=ML_TARGETS_INI)
    parser.add_argument("--lookback-days", type=int, default=180)
    args = parser.parse_args()

    train_and_write_all_momentum_models(
        targets_ini=args.targets_ini,
        lookback_days=args.lookback_days,
    )

if __name__ == "__main__":
    main()
