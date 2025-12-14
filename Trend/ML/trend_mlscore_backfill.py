# Trend/ML/trend_mlscore_backfill.py

import argparse
import json
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from psycopg2.extras import execute_values

from utils.db import get_db_connection
from pillars.common import TZ
from Trend.Pillar.trend_features import TrendFeatureEngine
from Trend.ML.trend_train import load_ml_targets_for_trend, TrendMLTarget

def load_trend_model(target_name: str, version: str):
    base_dir = Path(__file__).resolve().parent
    models_dir = (base_dir / ".." / ".." / "models" / "trend").resolve()

    model_path = models_dir / f"{target_name}__{version}__up.pkl"
    meta_path = models_dir / f"{target_name}__{version}__features.json"

    if not model_path.exists() or not meta_path.exists():
        raise FileNotFoundError(f"Model or features not found for {target_name} {version}")

    model = joblib.load(model_path)
    with open(meta_path, 'r') as f:
        meta = json.load(f)

    return model, meta['features']

def run_backfill(symbol: str = None, days: int = 365, target_name: str = None):
    targets = load_ml_targets_for_trend()

    for target in targets:
        if not target.enabled: continue
        if target_name and target.name != target_name: continue

        print(f"[TREND_ML_BACKFILL] Processing {target.name}...")

        try:
            model, feature_cols = load_trend_model(target.name, target.version)
        except Exception as e:
            print(f"Skipping {target.name}: {e}")
            continue

        # Load Historical Data
        cutoff = datetime.now(TZ) - timedelta(days=days)

        # Determine universe
        if symbol:
            symbols = [symbol]
        else:
            with get_db_connection() as conn, conn.cursor() as cur:
                cur.execute("SELECT symbol FROM reference.symbol_universe WHERE is_active=true")
                symbols = [r[0] for r in cur.fetchall()]

        for sym in symbols:
            # Load Candles
            sql = """
                SELECT ts, open, high, low, close, volume
                FROM market.futures_candles
                WHERE symbol=%s AND interval=%s AND ts >= %s
                ORDER BY ts ASC
            """
            with get_db_connection() as conn:
                df = pd.read_sql(sql, conn, params=(sym, target.base_tf, cutoff), parse_dates=['ts'])

            if df.empty: continue

            df = df.set_index('ts')
            df['symbol'] = sym # Required for feature engine if it used it (it stores it in self)

            # Compute Features
            engine = TrendFeatureEngine(sym, "futures")
            df_enriched = engine.compute_all_features(df, pd.DataFrame(), pd.DataFrame())

            # Align Features
            X = df_enriched.reindex(columns=feature_cols).fillna(0.0).replace([np.inf, -np.inf], 0.0)

            # Predict
            probs = model.predict_proba(X)[:, 1]

            # Write
            rows = []
            for ts, p_up in zip(df_enriched.index, probs):
                rows.append((
                    sym, target.market_type, "trend", target.base_tf, ts.to_pydatetime(),
                    target.name, target.version, float(p_up), float(1-p_up), 0.0, "{}"
                ))

            sql_ins = """
                INSERT INTO indicators.ml_pillars
                    (symbol, market_type, pillar, tf, ts, target_name, version, prob_up, prob_down, future_ret_pct, context)
                VALUES %s
                ON CONFLICT (symbol, market_type, pillar, tf, ts, target_name, version)
                DO UPDATE SET prob_up=EXCLUDED.prob_up, prob_down=EXCLUDED.prob_down
            """
            with get_db_connection() as conn, conn.cursor() as cur:
                execute_values(cur, sql_ins, rows, page_size=2000)
                conn.commit()

            print(f"  {sym}: Backfilled {len(rows)} rows.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=365)
    parser.add_argument("--symbol", type=str, default=None)
    parser.add_argument("--target", type=str, default=None)
    args = parser.parse_args()

    run_backfill(args.symbol, args.days, args.target)
