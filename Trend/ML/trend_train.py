# Trend/ML/trend_train.py

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
from Trend.Pillar.trend_features import TrendFeatureEngine

TREND_TARGETS_INI = str(Path(__file__).parent / "trend_targets.ini")

# ------------------------------------------------------------
# 1) Target config
# ------------------------------------------------------------

@dataclass
class TrendMLTarget:
    """
    Configuration for a single TREND ML target.
    """
    name: str                   # full section name
    enabled: bool
    mode: str                   # 'direction_threshold'
    base_tf: str                # '15m'
    horizon_bars: int
    threshold_up_pct: float
    threshold_down_pct: float
    min_hold_bars: int
    market_type: str = "futures"
    version: str = "xgb_v1"     # model version tag


def load_ml_targets_for_trend(
    ini_path: str = TREND_TARGETS_INI,
) -> List[TrendMLTarget]:
    """
    Read [trend_ml.target.*] sections from trend_targets.ini and return TrendMLTarget objects.
    """
    cp = configparser.ConfigParser(
        inline_comment_prefixes=(";", "#"),
        interpolation=None,
        strict=False,
    )
    cp.read(ini_path)

    targets: List[TrendMLTarget] = []

    for sec in cp.sections():
        if not sec.startswith("trend_ml.target."):
            continue

        enabled = cp.getboolean(sec, "enabled", fallback=True)
        mode = cp.get(sec, "mode", fallback="direction_threshold").strip()
        base_tf = cp.get(sec, "base_tf", fallback="15m").strip()
        horizon_bars = cp.getint(sec, "horizon_bars", fallback=16)

        threshold_up_pct = cp.getfloat(sec, "threshold_up_pct", fallback=2.0)
        threshold_down_pct = cp.getfloat(sec, "threshold_down_pct", fallback=-2.0)
        min_hold_bars = cp.getint(sec, "min_hold_bars", fallback=0)

        market_type = cp.get(sec, "market_type", fallback="futures").strip()
        version = cp.get(sec, "version", fallback="xgb_v1").strip()

        targets.append(
            TrendMLTarget(
                name=sec,
                enabled=enabled,
                mode=mode,
                base_tf=base_tf,
                horizon_bars=horizon_bars,
                threshold_up_pct=threshold_up_pct,
                threshold_down_pct=threshold_down_pct,
                min_hold_bars=min_hold_bars,
                market_type=market_type,
                version=version,
            )
        )

    return targets


# ------------------------------------------------------------
# 2) Load features + future returns
# ------------------------------------------------------------

def load_trend_features_and_returns(
    target: TrendMLTarget,
    lookback_days: int = 180,
) -> pd.DataFrame:
    """
    Build the training dataframe for a given TREND ML target.
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

    # Base Data Query (Price + Volume)
    sql = """
    WITH base AS (
        SELECT
            f.symbol,
            f.ts,
            f.close,
            f.volume,
            f.high,
            f.low,
            f.open,
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
            b.open, b.high, b.low, b.close, b.volume,
            b.future_close,
            CASE
                WHEN b.future_close IS NULL THEN NULL
                ELSE 100.0 * (b.future_close - b.close) / NULLIF(b.close, 0)
            END AS future_ret_pct

        FROM base b
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
            f"No training data returned for TREND ML target={target.name}"
        )

    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df = df.set_index("ts").sort_index()

    # ------------------------------------------------------------------
    # Guardrail: ensure future horizon stays within same day & trading hours
    # ------------------------------------------------------------------
    bar_minutes = 15
    horizon_minutes = bar_minutes * int(target.horizon_bars)
    df["ts_future"] = df.index + pd.Timedelta(minutes=horizon_minutes)

    same_day = df.index.date == df["ts_future"].dt.date
    within_close = df["ts_future"].dt.time <= time(15, 30)

    mask = same_day & within_close
    df.loc[~mask, ["future_close", "future_ret_pct"]] = np.nan
    df = df.dropna(subset=["future_ret_pct"])

    # ------------------------------------------------------------------
    # 3) Enrich with Trend Features
    # ------------------------------------------------------------------
    print(f"[TREND_ML] Enriching with Trend Features for {len(df)} rows...")

    enriched_chunks = []
    symbols = df["symbol"].unique()

    for sym in symbols:
        df_sym = df[df["symbol"] == sym].copy()
        if df_sym.empty:
            continue

        # TrendFeatureEngine needs daily_df and metrics_df mostly for specialized feats
        # We can pass empty or minimal if not strictly required for V1 training
        # But to be safe, let's load what we can if needed.
        # For now, pass empty as V1 features (EMA, RSI) are self-contained.
        daily_df = pd.DataFrame()
        metrics_df = pd.DataFrame()

        engine = TrendFeatureEngine(sym, "futures")
        df_enriched = engine.compute_all_features(df_sym, daily_df, metrics_df)
        enriched_chunks.append(df_enriched)

    if enriched_chunks:
        df = pd.concat(enriched_chunks)
        df = df.sort_index()

    return df


# ------------------------------------------------------------
# Model Saver
# ------------------------------------------------------------

def save_trend_models_and_features(
    target_name: str,
    version: str,
    model_up,
    feature_cols: list,
):
    """
    Save models + feature list.
    """
    base_dir = Path(__file__).resolve().parent
    models_dir = (base_dir / ".." / ".." / "models" / "trend").resolve()
    models_dir.mkdir(parents=True, exist_ok=True)

    up_path = models_dir / f"{target_name}__{version}__up.pkl"
    meta_path = models_dir / f"{target_name}__{version}__features.json"

    joblib.dump(model_up, up_path)

    with open(meta_path, "w") as f:
        json.dump({"features": list(feature_cols)}, f, indent=2)

    print(f"[TREND_ML] Saved UP model   → {up_path}")
    print(f"[TREND_ML] Saved features   → {meta_path}")


# ------------------------------------------------------------
# Label Builder
# ------------------------------------------------------------

def build_up_label_direction_threshold(
    df: pd.DataFrame,
    target: TrendMLTarget,
) -> pd.Series:
    if "future_ret_pct" not in df.columns:
        raise ValueError("DataFrame missing 'future_ret_pct' column")
    y = (df["future_ret_pct"] >= target.threshold_up_pct).astype(int)
    return y


# ------------------------------------------------------------
# Orchestrator
# ------------------------------------------------------------

def train_and_write_all_trend_models(
    targets_ini: str = TREND_TARGETS_INI,
    lookback_days: int = 180,
    only_target: Optional[str] = None,
):
    targets = load_ml_targets_for_trend(targets_ini)
    if not targets:
        print(f"[TREND_ML] No targets found in {targets_ini}")
        return

    for target in targets:
        if not target.enabled:
            continue
        if only_target and target.name != only_target:
            continue

        print(f"[TREND_ML] Training target={target.name}")

        # 1) Load Data
        df = load_trend_features_and_returns(target, lookback_days=lookback_days)

        # 2) Build Labels
        y = build_up_label_direction_threshold(df, target)

        mask = y.notna()
        df = df.loc[mask]
        y = y.loc[mask]

        # 3) Build Features
        drop_cols = {
            "future_ret_pct", "future_close", "market_type", "tf", "symbol",
            "ts", "ts_future", "open", "high", "low", "close", "volume"
        }

        features_df = df.drop(columns=[c for c in drop_cols if c in df.columns])
        features_df = features_df.select_dtypes(include=["number", "bool"])
        feature_cols = list(features_df.columns)

        X = features_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        if X.empty:
            print(f"[TREND_ML] No usable features, skipping.")
            continue

        # 4) Train
        model = XGBClassifier(
            max_depth=4, n_estimators=200, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            objective="binary:logistic", eval_metric="logloss", n_jobs=4
        )
        model.fit(X, y.values)

        # 5) Save
        save_trend_models_and_features(target.name, target.version, model, feature_cols)

        # 6) Predict & Write DB (In-Sample)
        prob_long = model.predict_proba(X)[:, 1]
        prob_short = 1.0 - prob_long

        rows = []
        for (ts, row), p_long, p_short in zip(df.iterrows(), prob_long, prob_short):
            rows.append((
                row["symbol"], target.market_type, "trend", row["tf"], ts.to_pydatetime(),
                target.name, target.version, float(p_long), float(p_short), float(row["future_ret_pct"]),
                json.dumps({"features": feature_cols})
            ))

        sql = """
            INSERT INTO indicators.ml_pillars
                (symbol, market_type, pillar, tf, ts, target_name, version, prob_up, prob_down, future_ret_pct, context)
            VALUES %s
            ON CONFLICT (symbol, market_type, pillar, tf, ts, target_name, version)
            DO UPDATE SET
                prob_up=EXCLUDED.prob_up, prob_down=EXCLUDED.prob_down,
                future_ret_pct=EXCLUDED.future_ret_pct, context=EXCLUDED.context
        """

        with get_db_connection() as conn, conn.cursor() as cur:
            execute_values(cur, sql, rows, page_size=1000)
            conn.commit()

        print(f"[TREND_ML] Wrote {len(rows)} rows for {target.name}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets-ini", default=TREND_TARGETS_INI)
    parser.add_argument("--lookback-days", type=int, default=180)
    parser.add_argument("--only-target", default=None)
    args = parser.parse_args()

    train_and_write_all_trend_models(args.targets_ini, args.lookback_days, args.only_target)
