# momentum/ml/momentum_train.py

from __future__ import annotations
import argparse, json, os, datetime as dt
from typing import List, Tuple

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from utils.db import get_db_engine, get_db_connection

PILLAR = "momentum"
DEFAULT_TARGET = "momentum_ml.target.bull_2pct_4h"
DEFAULT_VERSION = "xgb_v1"


def _parse_args():
    ap = argparse.ArgumentParser(description="Momentum ML trainer: ±2% in 4h (15m futures).")
    ap.add_argument("--target-name", default=DEFAULT_TARGET)
    ap.add_argument("--version", default=DEFAULT_VERSION)
    ap.add_argument("--lookback-days", type=int, default=180)
    ap.add_argument("--tf", default="15m")
    ap.add_argument("--market-type", default="futures")
    ap.add_argument("--run-id", default=None)
    return ap.parse_args()


def _load_training_frame(args) -> pd.DataFrame:
    """
    TODO: Adjust SQL to your actual feature view.

    Expect columns at least:
      symbol, ts, tf, market_type,
      future_ret_4h (as fraction, e.g. 0.021 = +2.1%),
      plus a bunch of numeric feature columns.
    """
    engine = get_db_engine()
    cutoff = dt.datetime.utcnow() - dt.timedelta(days=args.lookback_days)

    sql = """
        SELECT *
          FROM indicators.momentum_ml_feature_view
         WHERE market_type = %(market_type)s
           AND tf = %(tf)s
           AND ts >= %(cutoff)s
    """
    df = pd.read_sql(sql, engine, params={
        "market_type": args.market_type,
        "tf": args.tf,
        "cutoff": cutoff,
    })
    if df.empty:
        raise RuntimeError("No rows in momentum_ml_feature_view for training.")

    return df


def _make_targets(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    """Define UP and DOWN targets from future_ret_4h."""
    if "future_ret_4h" not in df.columns:
        raise RuntimeError("future_ret_4h column missing in feature frame.")

    y_up = (df["future_ret_4h"] >= 0.02).astype(int)
    y_dn = (df["future_ret_4h"] <= -0.02).astype(int)
    return y_up, y_dn


def _select_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    drop_cols = [
        "symbol", "ts", "tf", "market_type",
        "future_ret_4h",
    ]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    num_cols = [c for c in X.columns if np.issubdtype(X[c].dtype, np.number)]
    X = X[num_cols].fillna(0.0)
    return X, num_cols


def _train_xgb(X: pd.DataFrame, y: pd.Series) -> XGBClassifier:
    model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="binary:logistic",
        eval_metric="logloss",
        n_jobs=4,
        tree_method="hist",
    )
    model.fit(X, y)
    return model


def main():
    args = _parse_args()
    if args.run_id is None:
        args.run_id = f"MOM_ML_{dt.datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}"

    df = _load_training_frame(args)
    y_up, y_dn = _make_targets(df)
    X, feature_cols = _select_features(df)

    print(f"[MOM_ML] Training UP model on {len(X)} rows, {len(feature_cols)} features.")
    model_up = _train_xgb(X, y_up)

    # Optional: also train DOWN model if you want symmetric signals
    print(f"[MOM_ML] Training DOWN model on {len(X)} rows.")
    model_dn = _train_xgb(X, y_dn)

    # Save models + feature list
    models_dir = os.path.join(os.path.dirname(__file__), "..", "..", "models", "momentum")
    os.makedirs(models_dir, exist_ok=True)

    import joblib
    up_path = os.path.join(models_dir, f"{args.target_name}__{args.version}__up.pkl")
    dn_path = os.path.join(models_dir, f"{args.target_name}__{args.version}__down.pkl")
    meta_path = os.path.join(models_dir, f"{args.target_name}__{args.version}__features.json")

    joblib.dump(model_up, up_path)
    joblib.dump(model_dn, dn_path)
    with open(meta_path, "w") as f:
        json.dump({"features": feature_cols}, f, indent=2)

    print(f"[MOM_ML] Saved models to:\n  {up_path}\n  {dn_path}\n  {meta_path}")

    # Write predictions back into indicators.ml_pillars (train frame itself as "backtest" / signals)
    probs_up = model_up.predict_proba(X)[:, 1]
    probs_dn = model_dn.predict_proba(X)[:, 1]

    rows = []
    for (_, row), p_up, p_dn in zip(df.iterrows(), probs_up, probs_dn):
        rows.append((
            row["symbol"],
            PILLAR,
            args.market_type,
            row["tf"],
            row["ts"],
            args.target_name,
            args.version,
            float(p_up),
            float(p_dn),
            float(row["future_ret_4h"]),
        ))

    with get_db_connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS indicators.ml_pillars (
                symbol        text,
                pillar        text,
                market_type   text,
                tf            text,
                ts            timestamptz,
                target_name   text,
                version       text,
                prob_up       double precision,
                prob_down     double precision,
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
                prob_up = EXCLUDED.prob_up,
                prob_down = EXCLUDED.prob_down,
                future_ret_pct = EXCLUDED.future_ret_pct;
            """,
            rows,
        )
        conn.commit()
        # --- Update ML train freshness on symbol_universe ---
    from utils.db import get_db_connection  # already imported at top, so this is optional

    with get_db_connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            UPDATE reference.symbol_universe
               SET mom_ml_train_last_at = now()
             WHERE enabled = TRUE;
            """
        )
        conn.commit()

    print("[MOM_ML] Updated mom_ml_train_last_at on reference.symbol_universe")

    print(f"[MOM_ML] Wrote {len(rows)} rows into indicators.ml_pillars.")



if __name__ == "__main__":
    main()
