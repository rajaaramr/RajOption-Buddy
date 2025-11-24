# scheduler/nonlinear_features.py
from __future__ import annotations

import ast
import math
import os
import json
import configparser
import functools
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
from datetime import timezone

import numpy as np
import pandas as pd
import psycopg2.extras as pgx
import joblib  # Required for model loading

from utils.db import get_db_connection

DEFAULT_INI = os.getenv("NONLINEAR_INI", "nonlinear.ini")
TZ = timezone.utc

# ====================== Config ======================

@dataclass
class NLConfig:
    tfs: List[str]
    lookback_n: int
    score_method: str
    score_k: float
    interactions: Dict[str, str]
    backfill_bars: int

@dataclass
class MLCfg:
    enabled: bool
    model_path: str
    provider: str
    calibration: str
    blend_weight: float
    featureset: str
    ab_tag: str

def _load_ini(path: str = DEFAULT_INI) -> NLConfig:
    cfg = configparser.ConfigParser(inline_comment_prefixes=(";", "#"), strict=False)
    cfg.read(path)

    tfs = [x.strip() for x in cfg.get("nonlinear", "tfs", fallback="15m,30m,60m").split(",") if x.strip()]

    interactions = {}
    if cfg.has_section("interactions"):
        for k, v in cfg.items("interactions"):
            interactions[k.upper()] = v.strip() # Normalize keys

    return NLConfig(
        tfs=tfs,
        lookback_n=cfg.getint("nonlinear", "lookback_n", fallback=200),
        score_method=cfg.get("nonlinear", "score_method", fallback="logistic"),
        score_k=cfg.getfloat("nonlinear", "score_k", fallback=0.9),
        interactions=interactions,
        backfill_bars=cfg.getint("nonlinear", "backfill_bars", fallback=2000)
    )

def _load_ml_cfg(path: str = DEFAULT_INI) -> MLCfg:
    cfg = configparser.ConfigParser(inline_comment_prefixes=(";", "#"), strict=False)
    cfg.read(path)
    return MLCfg(
        enabled=cfg.getboolean("ml", "enabled", fallback=False),
        model_path=cfg.get("ml", "model_path", fallback="models/confidence_v1.lgb.pkl"),
        provider=cfg.get("ml", "provider", fallback="lightgbm"),
        calibration=cfg.get("ml", "calibration", fallback="isotonic"),
        blend_weight=cfg.getfloat("ml", "blend_weight", fallback=0.5),
        featureset=cfg.get("ml", "featureset", fallback="v1"),
        ab_tag=cfg.get("ml", "ab_tag", fallback="OFF")
    )

# ====================== DB & Schema Helpers ======================

def _frames_table(kind: str) -> str:
    return "indicators.futures_frames" if kind == "futures" else "indicators.spot_frames"

# Map metric names in INI expressions to DB columns
_COL_MAP = {
    "RSI": "rsi", "ADX": "adx", "ROC": "roc", "ATR": "atr_14", "MFI": "mfi_14",
    "DI+": "plus_di", "DI-": "minus_di",
    "MACD.HIST": "macd_hist", "MACD.LINE": "macd", "MACD.SIGNAL": "macd_signal",
    "BB.SCORE": "bb_score", "VP.POC": "vp_poc", "VP.VAL": "vp_val", "VP.VAH": "vp_vah",
    # Add Pillars if they exist in frames
    "STRUCT.SCORE": "struct_score", "QUAL.SCORE": "qual_score", "RISK.SCORE": "risk_score"
}

def _bulk_write_conf_ml(pred_df: pd.DataFrame, tf: str, model_ver: str, ab_tag: str):
    """
    Vectorized upsert into indicators.conf_ml.
    pred_df index: ts (datetime, UTC)
    columns: ["symbol", "prob_long"]  (prob_short is 1 - prob_long)
    """
    if pred_df is None or pred_df.empty:
        return 0

    from psycopg2.extras import execute_values

    rows = []
    for ts, row in pred_df.iterrows():
        prob_long = float(row["prob_long"])
        prob_short = float(1.0 - prob_long)
        rows.append(
            (
                ts.to_pydatetime(),
                row["symbol"],
                tf,
                prob_long,
                prob_short,
                model_ver,
                ab_tag or "OFF",
            )
        )

    sql = """
        INSERT INTO indicators.conf_ml
            (ts, symbol, tf, prob_long, prob_short, model_ver, ab_tag)
        VALUES %s
        ON CONFLICT (ts, symbol, tf, model_ver)
        DO UPDATE SET
            prob_long = EXCLUDED.prob_long,
            prob_short = EXCLUDED.prob_short,
            ab_tag     = EXCLUDED.ab_tag
    """

    with get_db_connection() as conn, conn.cursor() as cur:
        pgx.execute_values(cur, sql, rows, page_size=2000)
        conn.commit()
    return len(rows)



@functools.lru_cache(maxsize=4)
def _get_existing_columns(kind: str) -> set[str]:
    """Cache table schema to avoid querying information_schema repeatedly."""
    tbl = _frames_table(kind)
    schema, table = tbl.split(".")
    with get_db_connection() as conn, conn.cursor() as cur:
        cur.execute("""
            SELECT column_name FROM information_schema.columns
            WHERE table_schema=%s AND table_name=%s
        """, (schema, table))
        return {r[0] for r in cur.fetchall()}

def _fetch_data_vectorized(symbol: str, kind: str, tf: str, lookback: int, required_metrics: List[str]) -> pd.DataFrame:
    """Fetch all required columns in ONE query."""
    existing_cols = _get_existing_columns(kind)

    # Map requested metrics to actual DB columns
    db_cols = {"ts"}
    for m in required_metrics:
        # Handle cases like "RSI" or raw column names
        col = _COL_MAP.get(m.upper()) or m.lower()
        if col in existing_cols:
            db_cols.add(col)

    if len(db_cols) < 2: return pd.DataFrame() # Only TS found

    query_cols = ", ".join(sorted(list(db_cols)))
    tbl = _frames_table(kind)

    with get_db_connection() as conn, conn.cursor() as cur:
        cur.execute(f"""
            SELECT {query_cols} FROM {tbl}
            WHERE symbol=%s AND interval=%s
            ORDER BY ts DESC LIMIT %s
        """, (symbol, tf, lookback))
        rows = cur.fetchall()

    if not rows: return pd.DataFrame()

    df = pd.DataFrame(rows, columns=sorted(list(db_cols)))
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df.set_index("ts").sort_index()

def _bulk_upsert(kind: str, df: pd.DataFrame, symbol: str, tf: str, run_id: str):
    """Bulk write results."""
    if df.empty: return 0

    tbl = _frames_table(kind)

    # Prepare data for execute_values
    # We write: nli_*, nl_prob, nl_score, nl_prob_final, nl_score_final
    cols_to_write = [c for c in df.columns if c.startswith("nli_") or c.startswith("nl_")]
    if not cols_to_write: return 0

    # Build column list for INSERT
    db_cols = ["symbol", "interval", "ts", "run_id", "source"] + cols_to_write

    # Prepare values
    values = []
    for ts, row in df.iterrows():
        vals = [symbol, tf, ts.to_pydatetime(), run_id, "nonlinear"]
        vals.extend([None if pd.isna(row[c]) else float(row[c]) for c in cols_to_write])
        values.append(tuple(vals))

    # Construct ON CONFLICT update
    updates = [f"{c}=EXCLUDED.{c}" for c in cols_to_write]
    updates.append("updated_at=NOW()")

    sql = f"""
        INSERT INTO {tbl} ({', '.join(db_cols)})
        VALUES %s
        ON CONFLICT (symbol, interval, ts)
        DO UPDATE SET {', '.join(updates)}
    """

    with get_db_connection() as conn, conn.cursor() as cur:
        pgx.execute_values(cur, sql, values, page_size=2000)
        conn.commit()

    return len(values)

# ====================== Vectorized Calculation ======================

def _calculate_zscores(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """Vectorized Rolling Z-Score."""
    # Only calculate for numeric feature columns (exclude ts, metadata)
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    roll = df[numeric_cols].rolling(window=window, min_periods=max(1, window//4))
    mean = roll.mean()
    std = roll.std(ddof=1).replace(0, np.nan)

    return (df[numeric_cols] - mean) / std

def _evaluate_interactions_vectorized(df_z: pd.DataFrame, interactions: Dict[str, str]) -> pd.DataFrame:
    """
    Evaluates expressions like "RSI * ADX" on the entire dataframe at once.
    """
    results = pd.DataFrame(index=df_z.index)

    # Create a namespace mapping keys to Series
    # If expression uses "MACD.hist", we need to map it to column "macd_hist"
    # We create a local dictionary where keys are the config names
    local_dict = {}
    for key in _COL_MAP:
        col = _COL_MAP[key]
        if col in df_z.columns:
            local_dict[key] = df_z[col] # Direct Series reference
            local_dict[key.replace(".", "_")] = df_z[col] # Safe alias

    for name, expr in interactions.items():
        try:
            # Sanitize expr for pd.eval (simple math only)
            # If safe, use pd.eval or numexpr.
            # For now, python eval with pandas Series overrides operators efficiently.
            # We replace known metrics with 'local_dict["METRIC"]'

            # Simple parsing: Replace keywords in Expr with local_dict lookups
            # Token-based replacement to handle case sensitivity (e.g. MACD.hist vs MACD.HIST)
            # We iterate over the tokens we extracted earlier
            for t in tokens:
                # Ignore numeric literals
                if t.isnumeric():
                    continue

                # Try to find mapping
                col = _COL_MAP.get(t.upper()) or t.lower()

                # If we have this column in our dataframe, replace the token in the expression
                if col in df_z.columns:
                    # Simple replace might be risky if tokens are substrings of each other (e.g. A and AA)
                    # But tokens come from split(), so likely distinct.
                    # Ideally use regex or strict replacement, but .replace() is decent for these simple math expressions.
                    mapped_expr = mapped_expr.replace(t, col)
            tokens = [t for t in expr.replace("*"," ").replace("+"," ").replace("-"," ").replace("/"," ").split() if t.strip()]

            # 2. Check data availability
            if not all((t in local_dict or t.replace(".","_") in local_dict) for t in tokens if not t.isnumeric()):
                results[f"nli_{name.lower()}"] = 0.0
                continue

            # 3. Evaluate using pandas engine (safest for math strings)
            # Map col names in expression to dataframe columns
            # e.g. "MACD.hist * ATR" -> "macd_hist * atr_14"
            mapped_expr = expr

            # Token-based replacement to handle case sensitivity (e.g. MACD.hist vs MACD.HIST)
            # We iterate over the tokens we extracted earlier
            for t in tokens:
                # Ignore numeric literals
                if t.isnumeric():
                    continue

                # Try to find mapping
                col = _COL_MAP.get(t.upper()) or t.lower()

                # If we have this column in our dataframe, replace the token in the expression
                if col in df_z.columns:
                    # Simple replace might be risky if tokens are substrings of each other (e.g. A and AA)
                    # But tokens come from split(), so likely distinct.
                    # Ideally use regex or strict replacement, but .replace() is decent for these simple math expressions.
                    mapped_expr = mapped_expr.replace(t, col)

            # Evaluate on the Z-Score DataFrame
            res = df_z.eval(mapped_expr)
            results[f"nli_{name.lower()}"] = res

        except Exception as e:
            # print(f"Vector eval error {name}: {e}")
            results[f"nli_{name.lower()}"] = 0.0

    return results.fillna(0.0)

def _score_vectorized(series: pd.Series, method: str, k: float) -> pd.Series:
    if method == "tanh":
        return 0.5 * (np.tanh(k * series) + 1.0)
    # Logistic
    return 1.0 / (1.0 + np.exp(-k * series))

# ====================== ML Integration ======================

_MODEL_CACHE = {}

def _get_model(ml_cfg: MLCfg):
    if not ml_cfg.enabled: return None
    if ml_cfg.model_path in _MODEL_CACHE: return _MODEL_CACHE[ml_cfg.model_path]

    try:
        artifact = joblib.load(ml_cfg.model_path)
        _MODEL_CACHE[ml_cfg.model_path] = artifact
        return artifact
    except Exception as e:
        print(f"[NL] ML Load Error: {e}")
        return None

def _predict_vectorized(model_artifact, df_features: pd.DataFrame) -> pd.Series:
    """Runs prediction on the whole dataframe."""
    model = model_artifact.get("model")
    features = model_artifact.get("features", [])

    # Align columns
    X = pd.DataFrame(index=df_features.index)
    for f in features:
        # Map feature name to DB column if possible
        col = _COL_MAP.get(f.upper()) or f
        if col in df_features.columns:
            X[f] = df_features[col]
        else:
            X[f] = 0.0 # Missing feature fill

    X = X.fillna(0.0)

    try:
        # Predict
        probs = model.predict(X)
        # Handle calibrator if present
        calib = model_artifact.get("calib")
        if calib:
            probs = calib.predict_proba(probs.reshape(-1, 1))[:, 1]
        return pd.Series(probs, index=df_features.index)
    except Exception as e:
        print(f"[NL] Predict Error: {e}")
        return pd.Series(0.5, index=df_features.index)

# ====================== Main Pipeline ======================

def process_symbol(symbol: str, kind: str = "futures", cfg: Optional[NLConfig] = None, ini_path: str = DEFAULT_INI) -> int:
    if cfg is None: cfg = _load_ini(ini_path)
    ml_cfg = _load_ml_cfg(ini_path)

    total_rows = 0

    # 1. Identify Required Columns
    # Extract tokens from interaction expressions
    required_metrics = set()
    for expr in cfg.interactions.values():
        # simplistic extraction of words
        tokens = [t.strip() for t in expr.replace("*"," ").replace("+"," ").split()]
        required_metrics.update(tokens)

    # Add ML features if enabled
    model_artifact = _get_model(ml_cfg)
    if model_artifact:
        required_metrics.update(model_artifact.get("features", []))

    required_metrics = list(required_metrics)

    for tf in cfg.tfs:
        # 2. Fetch Data (Bulk)
        # We need lookback_n (for z-score) + backfill_bars (for processing)
        fetch_rows = cfg.lookback_n + cfg.backfill_bars
        df = _fetch_data_vectorized(symbol, kind, tf, fetch_rows, required_metrics)

        if len(df) < cfg.lookback_n: continue

        # 3. Calculate Z-Scores (Vectorized)
        df_z = _calculate_zscores(df, cfg.lookback_n)

        # 4. Calculate Interactions (Vectorized)
        df_res = _evaluate_interactions_vectorized(df_z, cfg.interactions)

        # 5. Calculate Baseline Score
        # Sum of interactions
        z_sum = df_res.sum(axis=1)
        df_res["nl_prob"] = _score_vectorized(z_sum, cfg.score_method, cfg.score_k)
        df_res["nl_score"] = (df_res["nl_prob"] * 100.0).round(2)

        # 6. ML Prediction (Vectorized)
                # 6. ML Prediction (Vectorized)
        probs_ml = None
        if model_artifact:
            # Note: here you're passing RAW df; if your model was trained on z-scores,
            # switch this to df_z instead of df.
            probs_ml = _predict_vectorized(model_artifact, df)

            w = ml_cfg.blend_weight
            df_res["nl_prob_final"] = (w * probs_ml) + ((1 - w) * df_res["nl_prob"])
            df_res["nl_score_final"] = (df_res["nl_prob_final"] * 100.0).round(2)
        else:
            # fall back: no ML, final = base
            df_res["nl_prob_final"] = df_res["nl_prob"]
            df_res["nl_score_final"] = df_res["nl_score"]

        # 7. Upsert Batch (frames) – only tail
        to_write = df_res.iloc[-cfg.backfill_bars:]
        n = _bulk_upsert(kind, to_write, symbol, tf, "nl_run")
        total_rows += n

        # 8. Write conf_ml in bulk (if ML enabled)
        if probs_ml is not None:
            # align ML probs to same index as df_res
            probs_ml = probs_ml.reindex(df_res.index)
            ml_tail = probs_ml.iloc[-cfg.backfill_bars:]

            conf_df = pd.DataFrame(
                {
                    "symbol": symbol,
                    "prob_long": ml_tail,
                },
                index=ml_tail.index,
            )
            n_conf = _bulk_write_conf_ml(
                conf_df,
                tf=tf,
                model_ver="confidence_v1",
                ab_tag=ml_cfg.ab_tag,
            )
            print(f"   [NL-ML] {symbol} {kind} {tf}: wrote {n_conf} conf_ml rows")

        print(f"   [NL] {symbol} {kind} {tf}: Processed {n} frame rows (Vectorized)")

        print(f"   [NL] {symbol} {tf}: Processed {n} rows (Vectorized)")

    return total_rows

def run(symbols: Optional[List[str]] = None, kinds: Tuple[str,...] = ("futures","spot"), ini_path: str = DEFAULT_INI) -> int:
    cfg = _load_ini(ini_path)
    if not symbols: return 0

    count = 0
    for s in symbols:
        for k in kinds:
            try:
                count += process_symbol(s, kind=k, cfg=cfg, ini_path=ini_path)
            except Exception as e:
                print(f"❌ [NL] Error {s} {k}: {e}")
                # traceback.print_exc()
    return count

if __name__ == "__main__":
    import sys
    # CLI args
    run(None) # Caller usually supplies symbols