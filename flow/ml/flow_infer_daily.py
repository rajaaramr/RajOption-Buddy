# flow/ml/flow_infer_daily.py

from __future__ import annotations
import argparse
import datetime as dt
import json
import os
from typing import List, Tuple

import numpy as np
import pandas as pd
import joblib

from utils.db import get_db_connection
from flow.runtime.utils import update_flow_ml_jobs_status

PILLAR = "flow"
DEFAULT_TARGET = "flow_ml.target.bull_2pct_4h"
DEFAULT_VERSION = "xgb_v1"


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Flow ML daily/weekly inference job.")
    ap.add_argument("--target_name", default=DEFAULT_TARGET)
    ap.add_argument("--version", default=DEFAULT_VERSION)
    ap.add_argument("--tf", default="15m")
    ap.add_argument("--market-type", default="futures")  # or 'spot'
    ap.add_argument(
        "--lookback-days",
        type=int,
        default=5,
        help="How many days of recent feature rows to scan for missing ML preds.",
    )
    ap.add_argument(
        "--max-rows",
        type=int,
        default=200_000,
        help="Safety cap on number of feature rows to score in one run.",
    )
    return ap.parse_args()


def _load_models_and_features(
    target_name: str,
    version: str,
) -> Tuple[object, object, List[str]]:
    """
    Load UP & DOWN XGB models + feature list that were saved by flow_train.py.
    """
    base_dir = os.path.dirname(__file__)  # .../flow/ml
    models_dir = os.path.join(base_dir, "..", "..", "models", "flow")
    models_dir = os.path.normpath(models_dir)

    up_path = os.path.join(models_dir, f"{target_name}__{version}__up.pkl")
    dn_path = os.path.join(models_dir, f"{target_name}__{version}__down.pkl")
    meta_path = os.path.join(models_dir, f"{target_name}__{version}__features.json")

    if not (os.path.exists(up_path) and os.path.exists(dn_path) and os.path.exists(meta_path)):
        raise FileNotFoundError(
            f"[FLOW_INFER] Model files not found for target={target_name}, "
            f"version={version} under {models_dir}"
        )

    model_up = joblib.load(up_path)
    model_dn = joblib.load(dn_path)
    with open(meta_path, "r") as f:
        meta = json.load(f)
    feature_cols = meta.get("features", [])
    if not feature_cols:
        raise RuntimeError("[FLOW_INFER] No feature list found in meta JSON.")

    print(f"[FLOW_INFER] Loaded models from {models_dir}")
    print(f"[FLOW_INFER] Feature count: {len(feature_cols)}")

    return model_up, model_dn, feature_cols


def _load_missing_feature_rows(args: argparse.Namespace) -> pd.DataFrame:
    """
    Load recent feature rows from indicators.flow_ml_feature_view
    for which we DO NOT yet have entries in indicators.ml_pillars
    for the given (pillar, target_name, version).
    """
    cutoff = dt.datetime.utcnow() - dt.timedelta(days=args.lookback_days)

    sql = """
        WITH feat AS (
            SELECT *
              FROM indicators.flow_ml_feature_view
             WHERE market_type = %(market_type)s
               AND tf = %(tf)s
               AND ts >= %(cutoff)s
        )
        SELECT f.*
          FROM feat f
          LEFT JOIN indicators.ml_pillars p
            ON  p.symbol      = f.symbol
            AND p.pillar      = %(pillar)s
            AND p.market_type = f.market_type
            AND p.tf          = f.tf
            AND p.ts          = f.ts
            AND p.target_name = %(target_name)s
            AND p.version     = %(version)s
         WHERE p.symbol IS NULL
         ORDER BY f.ts, f.symbol
         LIMIT %(max_rows)s
    """

    with get_db_connection() as conn:
        df = pd.read_sql(
            sql,
            conn,
            params={
                "market_type": args.market_type,
                "tf": args.tf,
                "cutoff": cutoff,
                "pillar": PILLAR,
                "target_name": args.target_name,
                "version": args.version,
                "max_rows": args.max_rows,
            },
        )

    if df.empty:
        print("[FLOW_INFER] No new feature rows needing inference.")
    else:
        print(f"[FLOW_INFER] Loaded {len(df)} feature rows needing ML predictions.")
    return df


def _prep_X(df: pd.DataFrame, feature_cols: List[str]) -> np.ndarray:
    """
    Align feature frame to the trained feature list and fill missing cols with 0.
    """
    X = df.copy()

    # Ensure all required columns exist
    for col in feature_cols:
        if col not in X.columns:
            X[col] = 0.0

    # Keep only feature columns, in correct order
    X = X[feature_cols]

    # Fill NaNs / infinities
    X = X.replace([np.inf, -np.inf], 0.0).fillna(0.0)

    return X.values


def main():
    args = _parse_args()

    print(
        f"[FLOW_INFER] Starting inference for target={args.target_name}, "
        f"version={args.version}, tf={args.tf}, market_type={args.market_type}, "
        f"lookback_days={args.lookback_days}"
    )

    # 1) Load trained models + feature list
    model_up, model_dn, feature_cols = _load_models_and_features(
        target_name=args.target_name,
        version=args.version,
    )

    # 2) Load delta feature rows needing predictions
    df = _load_missing_feature_rows(args)
    if df.empty:
        return

    # 3) Build X matrix
    X = _prep_X(df, feature_cols)

    # 4) Predict probabilities
    print("[FLOW_INFER] Running model inference...")
    prob_up = model_up.predict_proba(X)[:, 1]
    prob_dn = model_dn.predict_proba(X)[:, 1]

    # 5) Prepare rows for ml_pillars upsert
    rows = []
    for (_, row), p_up, p_dn in zip(df.iterrows(), prob_up, prob_dn):
        rows.append(
            (
                row["symbol"],
                PILLAR,
                row["market_type"],  # should match args.market_type
                row["tf"],
                row["ts"],
                args.target_name,
                args.version,
                float(p_up),
                float(p_dn),
                None,  # future_ret_pct unknown at inference time
            )
        )

    if not rows:
        print("[FLOW_INFER] No rows to insert after inference.")
        return

    # 6) Write to indicators.ml_pillars
    with get_db_connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS indicators.ml_pillars (
                symbol         text,
                pillar         text,
                market_type    text,
                tf             text,
                ts             timestamptz,
                target_name    text,
                version        text,
                prob_up        double precision,
                prob_down      double precision,
                future_ret_pct double precision,
                PRIMARY KEY (symbol, pillar, market_type, tf, ts, target_name, version)
            );
            """
        )
        cur.executemany(
            """
            INSERT INTO indicators.ml_pillars (
                symbol, pillar, market_type, tf, ts,
                target_name, version,
                prob_up, prob_down, future_ret_pct
            )
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            ON CONFLICT (symbol, pillar, market_type, tf, ts, target_name, version)
            DO UPDATE SET
                prob_up        = EXCLUDED.prob_up,
                prob_down      = EXCLUDED.prob_down,
                future_ret_pct = EXCLUDED.future_ret_pct;
            """,
            rows,
        )
        conn.commit()

        # Mark infer timestamp globally for FLOW pillar
        update_flow_ml_jobs_status(conn, infer_at=dt.datetime.utcnow())

    print(f"[FLOW_INFER] Wrote/updated {len(rows)} rows into indicators.ml_pillars.")


if __name__ == "__main__":
    main()
