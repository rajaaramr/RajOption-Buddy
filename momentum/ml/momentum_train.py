from __future__ import annotations

import os
import json
import configparser
from dataclasses import dataclass
from datetime import datetime, timedelta, time
from pathlib import Path
from typing import List, Optional, Tuple, Iterable

import numpy as np
import pandas as pd
import joblib
from xgboost import XGBClassifier
from psycopg2.extras import execute_values

from utils.db import get_db_connection
from pillars.common import TZ
from momentum.pillar.momentum_features_optimized import MomentumFeatureEngine

ML_TARGETS_INI = "momentum/ml/ml_targets.ini"
PILLAR = "momentum"
IST_TZ = "Asia/Kolkata"


# ------------------------------------------------------------
# 1) Target config
# ------------------------------------------------------------
@dataclass
class MomentumMLTarget:
    name: str
    enabled: bool
    mode: str
    base_tf: str
    horizon_bars: int
    threshold_up_pct: float
    threshold_down_pct: float
    min_hold_bars: int
    market_type: str = "futures"
    version: str = "xgb_v1"


def load_ml_targets_for_momentum(ini_path: str = ML_TARGETS_INI) -> List[MomentumMLTarget]:
    if not os.path.isabs(ini_path) and not os.path.exists(ini_path):
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
        targets.append(
            MomentumMLTarget(
                name=sec,
                enabled=cp.getboolean(sec, "enabled", fallback=True),
                mode=cp.get(sec, "mode", fallback="direction_threshold").strip(),
                base_tf=cp.get(sec, "base_tf", fallback="15m").strip(),
                horizon_bars=cp.getint(sec, "horizon_bars", fallback=16),
                threshold_up_pct=cp.getfloat(sec, "threshold_up_pct", fallback=1.0),
                threshold_down_pct=cp.getfloat(sec, "threshold_down_pct", fallback=-0.5),
                min_hold_bars=cp.getint(sec, "min_hold_bars", fallback=0),
                market_type=cp.get(sec, "market_type", fallback="futures").strip(),
                version=cp.get(sec, "version", fallback="xgb_v1").strip(),
            )
        )
    return targets


# ------------------------------------------------------------
# 2) Raw load
# ------------------------------------------------------------
def _read_symbols_file(path: str) -> List[str]:
    out: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            out.append(s)
    return out


def _load_raw_candles(
    base_tf: str,
    lookback_days: int,
    symbol: Optional[str] = None,
    symbols: Optional[List[str]] = None,
) -> pd.DataFrame:
    cutoff = datetime.now(TZ) - timedelta(days=int(lookback_days))

    where_sym = ""
    params = {"base_tf": base_tf, "cutoff": cutoff}

    if symbol:
        where_sym = " AND symbol = %(symbol)s"
        params["symbol"] = symbol
    elif symbols:
        where_sym = " AND symbol = ANY(%(symbols)s)"
        params["symbols"] = symbols

    sql = f"""
    SELECT
        symbol,
        ts,
        open, high, low, close, volume,
        oi
    FROM market.futures_candles
    WHERE interval = %(base_tf)s
      AND ts >= %(cutoff)s
      AND EXTRACT(ISODOW FROM (ts AT TIME ZONE '{IST_TZ}')) BETWEEN 1 AND 5
      AND (ts AT TIME ZONE '{IST_TZ}')::time >= TIME '09:15:00'
      AND (ts AT TIME ZONE '{IST_TZ}')::time <= TIME '15:30:00'
      {where_sym}
    ORDER BY symbol, ts
    """

    with get_db_connection() as conn:
        df = pd.read_sql(sql, conn, params=params)

    if df.empty:
        return df

    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    df = df.dropna(subset=["ts"]).copy()
    return df


def _load_daily_futures(
    lookback_days: int,
    symbol: Optional[str] = None,
    symbols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Load data from raw_ingest.daily_futures
    """
    cutoff = datetime.now(TZ) - timedelta(days=int(lookback_days))

    where_sym = ""
    params = {"cutoff": cutoff}

    if symbol:
        where_sym = " AND symbol = %(symbol)s"
        params["symbol"] = symbol
    elif symbols:
        where_sym = " AND symbol = ANY(%(symbols)s)"
        params["symbols"] = symbols

    sql = f"""
    SELECT
        trade_date,
        symbol,
        price,
        open_price,
        high_price,
        low_price,
        prev_close
    FROM raw_ingest.daily_futures
    WHERE trade_date >= %(cutoff)s
      {where_sym}
    ORDER BY symbol, trade_date
    """

    with get_db_connection() as conn:
        df = pd.read_sql(sql, conn, params=params)

    if df.empty:
        return pd.DataFrame()

    df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date
    return df


# ------------------------------------------------------------
# 3) Feature enrichment + Labeling
# ------------------------------------------------------------
def _compute_path_dependent_labels(
    df: pd.DataFrame,
    horizon_bars: int,
    target_mult: float,
    stop_mult: float,
    atr_col: str = "day_atr",
    mode: str = "bull"
) -> pd.Series:
    """
    Vectorized Path-Dependent Labeler.
    """
    if atr_col not in df.columns:
        return pd.Series(0, index=df.index)

    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    atr = df[atr_col].values

    N = len(df)

    # Pre-calculate levels
    target_price = close + (target_mult * atr)
    stop_price = close + (stop_mult * atr)

    # Initialize "Earliest Bar Index"
    first_target_idx = np.full(N, 99999, dtype=int)
    first_stop_idx = np.full(N, 99999, dtype=int)

    # Iterate strides
    for i in range(1, horizon_bars + 1):
        fut_high = np.roll(high, -i)
        fut_low = np.roll(low, -i)
        valid_mask = np.ones(N, dtype=bool)
        valid_mask[-i:] = False

        if mode == "bull":
            # Bull Target: High >= Target
            hit_tgt = (fut_high >= target_price) & valid_mask
            # Bull Stop: Low <= Stop
            hit_stp = (fut_low <= stop_price) & valid_mask
        else: # bear
            # Bear Target: Low <= Target (target is lower)
            hit_tgt = (fut_low <= target_price) & valid_mask
            # Bear Stop: High >= Stop (stop is higher)
            hit_stp = (fut_high >= stop_price) & valid_mask

        first_target_idx = np.where((first_target_idx == 99999) & hit_tgt, i, first_target_idx)
        first_stop_idx = np.where((first_stop_idx == 99999) & hit_stp, i, first_stop_idx)

    # Win if Target hit BEFORE Stop
    win_mask = (first_target_idx != 99999) & (first_target_idx < first_stop_idx)

    return pd.Series(win_mask.astype(int), index=df.index)


def enrich_and_label(
    df_raw: pd.DataFrame,
    daily_raw: pd.DataFrame,
    target: MomentumMLTarget,
    ini_path: Optional[str] = None
) -> pd.DataFrame:
    engine = MomentumFeatureEngine(ini_path=ini_path) if ini_path else MomentumFeatureEngine()

    out_chunks: List[pd.DataFrame] = []
    df_raw = df_raw.copy()
    df_raw["ts"] = pd.to_datetime(df_raw["ts"], utc=True, errors="coerce")
    df_raw = df_raw.dropna(subset=["ts"]).copy()

    for sym, g in df_raw.groupby("symbol", sort=False):
        g = g.sort_values("ts")

        # Filter Daily for this symbol
        d_sym = daily_raw[daily_raw["symbol"] == sym] if not daily_raw.empty else None

        try:
            # 1. Compute Features (includes Daily Context & OI)
            feats = engine.compute_features(g, daily_df=d_sym)
        except Exception as e:
            print(f"[MOM_ML] feature compute failed for {sym}: {e}")
            continue

        feats = feats.reset_index(drop=True)

        # 2. Compute Labels (Path Dependent)
        feats["y_bull"] = _compute_path_dependent_labels(
            feats,
            horizon_bars=target.horizon_bars,
            target_mult=target.threshold_up_pct, # e.g. 1.0
            stop_mult=target.threshold_down_pct, # e.g. -0.5
            atr_col="day_atr",
            mode="bull"
        )

        feats["y_bear"] = _compute_path_dependent_labels(
            feats,
            horizon_bars=target.horizon_bars,
            target_mult=-1.0 * target.threshold_up_pct,
            stop_mult=-1.0 * target.threshold_down_pct, # -(-0.5) = +0.5
            atr_col="day_atr",
            mode="bear"
        )

        # Compute "Future Return" (simple horizon return) for reference
        horizon_close = feats["close"].shift(-target.horizon_bars)
        feats["future_ret_pct"] = (horizon_close - feats["close"]) / feats["close"] * 100.0

        # Drop end-of-series where label is unknown
        feats = feats.iloc[:-target.horizon_bars].copy()

        out_chunks.append(feats)

    if not out_chunks:
        return pd.DataFrame()

    df = pd.concat(out_chunks, ignore_index=True)
    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    df = df.sort_values(["ts", "symbol"]).reset_index(drop=True)
    return df


# ------------------------------------------------------------
# 4) Walk-forward splits
# ------------------------------------------------------------
def iter_walkforward_splits(
    df: pd.DataFrame,
    train_days: int,
    test_days: int,
    step_days: int,
    min_train_rows: int,
) -> Iterable[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    ts = pd.to_datetime(df["ts"], utc=True, errors="coerce").dropna()
    if ts.empty:
        return

    ts_min = ts.min()
    ts_max = ts.max()

    cur_train_end = (ts_min + pd.Timedelta(days=train_days))

    while True:
        test_start = cur_train_end
        test_end = test_start + pd.Timedelta(days=test_days)

        if test_end > ts_max:
            break

        n_train = int((ts < cur_train_end).sum())

        if n_train >= int(min_train_rows):
            yield (cur_train_end, test_start, test_end)

        cur_train_end = cur_train_end + pd.Timedelta(days=int(step_days))


# ------------------------------------------------------------
# 5) Features + model
# ------------------------------------------------------------
def _select_feature_matrix(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    drop_cols = {
        "symbol", "ts", "d_ist",
        "future_ret_pct", "future_close", "ts_future",
        "open", "high", "low", "close", "volume", "oi",
        "y_bull", "y_bear",
        "day_open", "prev_close", "prev_day_atr", "_day_high_sofar", "_day_low_sofar",
        "_merge_date", "trade_date"
    }
    X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
    X = X.select_dtypes(include=["number", "bool"]).copy()
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return X, list(X.columns)


def _make_model(seed: int = 42) -> XGBClassifier:
    return XGBClassifier(
        max_depth=4,
        n_estimators=400,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        n_jobs=4,
        random_state=seed,
    )


def save_model_and_features(target_name: str, version: str, model_up, model_dn, feature_cols: List[str]) -> None:
    base_dir = Path(__file__).resolve().parent
    models_dir = (base_dir / ".." / ".." / "models" / "momentum").resolve()
    models_dir.mkdir(parents=True, exist_ok=True)

    up_path = models_dir / f"{target_name}__{version}__up.pkl"
    dn_path = models_dir / f"{target_name}__{version}__down.pkl"
    meta_path = models_dir / f"{target_name}__{version}__features.json"

    joblib.dump(model_up, up_path)
    joblib.dump(model_dn, dn_path)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({"features": feature_cols}, f, indent=2)

    print(f"[MOM_ML] Saved UP model   → {up_path}")
    print(f"[MOM_ML] Saved DOWN model → {dn_path}")
    print(f"[MOM_ML] Saved meta       → {meta_path}")


# ------------------------------------------------------------
# 6) DB writer
# ------------------------------------------------------------
def write_oof_predictions(rows: List[tuple]) -> int:
    if not rows:
        return 0

    sql = """
        INSERT INTO indicators.ml_pillars
            (symbol, market_type, pillar, tf, ts,
             target_name, version,
             prob_up, prob_down, future_ret_pct, context)
        VALUES %s
        ON CONFLICT (symbol, market_type, pillar, tf, ts, target_name, version)
        DO UPDATE
           SET prob_up        = EXCLUDED.prob_up,
               prob_down      = EXCLUDED.prob_down,
               future_ret_pct = EXCLUDED.future_ret_pct,
               context        = EXCLUDED.context
    """

    with get_db_connection() as conn, conn.cursor() as cur:
        execute_values(cur, sql, rows, page_size=2000)
        conn.commit()

    return len(rows)


# ------------------------------------------------------------
# 7) Orchestrator
# ------------------------------------------------------------
def train_walkforward_oof(
    targets_ini: str = ML_TARGETS_INI,
    lookback_days: int = 180,
    train_days: int = 90,
    test_days: int = 14,
    step_days: int = 7,
    only_target: Optional[str] = None,
    momentum_ini: Optional[str] = None,
    save_live_model: bool = True,
    min_train_rows: int = 2000,
    symbol: Optional[str] = None,
    symbols: Optional[List[str]] = None,
    same_day_only: bool = False,
) -> None:
    targets = load_ml_targets_for_momentum(targets_ini)
    if not targets:
        print(f"[MOM_ML] No targets found in {targets_ini}")
        return

    for target in targets:
        if not target.enabled:
            continue
        if only_target and target.name != only_target:
            continue

        # New mode check
        if target.mode != "vol_adj_path_dependent":
             print(f"[MOM_ML] skipping {target.name}: mode={target.mode} mismatch (expected vol_adj_path_dependent)")
             continue

        print(f"\n[MOM_ML] Target={target.name} version={target.version} base_tf={target.base_tf} [DUAL MODEL]")

        # Load 15m
        raw = _load_raw_candles(
            base_tf=target.base_tf,
            lookback_days=lookback_days,
            symbol=symbol,
            symbols=symbols,
        )
        if raw.empty:
            print(f"[MOM_ML] No raw data for {target.name}")
            continue

        # Load Daily
        # Need slightly more lookback to ensure 'prev' values exist for start of 15m
        daily_raw = _load_daily_futures(
            lookback_days=lookback_days + 30,
            symbol=symbol,
            symbols=symbols
        )
        if daily_raw.empty:
            print(f"[MOM_ML] WARNING: No daily data found. Features will be approximate.")

        print(f"[MOM_ML] raw rows={len(raw)} daily rows={len(daily_raw)} symbols={raw['symbol'].nunique()}")

        # Enrich + Label
        feats = enrich_and_label(raw, daily_raw, target, ini_path=momentum_ini)
        if feats.empty:
            print(f"[MOM_ML] No labeled features produced")
            continue

        print(f"[MOM_ML] feats rows={len(feats)}")
        print(f"   y_bull dist: {feats['y_bull'].value_counts(dropna=False).to_dict()}")
        print(f"   y_bear dist: {feats['y_bear'].value_counts(dropna=False).to_dict()}")

        # Day key in IST for slicing
        feats["d_ist"] = feats["ts"].dt.tz_convert(IST_TZ).dt.floor("D")

        total_written = 0
        split_no = 0

        for train_end, test_start, test_end in iter_walkforward_splits(
            feats,
            train_days=train_days,
            test_days=test_days,
            step_days=step_days,
            min_train_rows=min_train_rows,
        ):
            split_no += 1

            train_end_d  = pd.Timestamp(train_end).tz_convert(IST_TZ).floor("D")
            test_start_d = pd.Timestamp(test_start).tz_convert(IST_TZ).floor("D")
            test_end_d   = pd.Timestamp(test_end).tz_convert(IST_TZ).floor("D")

            df_train = feats[feats["d_ist"] < train_end_d].copy()
            df_test  = feats[(feats["d_ist"] >= test_start_d) & (feats["d_ist"] < test_end_d)].copy()

            if df_train.empty or df_test.empty:
                continue

            X_train, feature_cols = _select_feature_matrix(df_train)
            X_test, _ = _select_feature_matrix(df_test)

            y_bull_train = df_train["y_bull"].values
            y_bear_train = df_train["y_bear"].values

            if X_train.empty or X_test.empty:
                continue

            # --- Train Bull Model ---
            model_bull = _make_model(seed=42)
            model_bull.fit(X_train, y_bull_train)
            prob_up = model_bull.predict_proba(X_test)[:, 1]

            # --- Train Bear Model ---
            model_bear = _make_model(seed=43) # diff seed
            model_bear.fit(X_train, y_bear_train)
            prob_dn = model_bear.predict_proba(X_test)[:, 1]

            # --- Metrics (Precision @ Top 10%) ---
            def calc_top_k_prec(prob, y_true, k=0.1):
                n_top = int(len(prob) * k)
                if n_top < 1: return 0.0
                idx = np.argsort(prob)[-n_top:]
                return y_true.iloc[idx].mean()

            prec_bull = calc_top_k_prec(prob_up, df_test["y_bull"], 0.1)
            prec_bear = calc_top_k_prec(prob_dn, df_test["y_bear"], 0.1)

            print(f"[WF] split#{split_no} train={len(df_train)} test={len(df_test)} | "
                  f"Prec@10% Bull={prec_bull:.2f} Bear={prec_bear:.2f}")

            # Write OOF
            ctx_base = {
                "base_tf": target.base_tf,
                "mode": target.mode,
                "horizon_bars": int(target.horizon_bars),
                "threshold_up": float(target.threshold_up_pct),
                "threshold_dn": float(target.threshold_down_pct),
                "is_oof": True,
                "split_id": split_no,
                "features": feature_cols,
            }

            rows: List[tuple] = []
            df_test2 = df_test.reset_index(drop=True)
            for i, r in df_test2.iterrows():
                rows.append(
                    (
                        r["symbol"],
                        target.market_type,
                        PILLAR,
                        target.base_tf,
                        pd.Timestamp(r["ts"]).to_pydatetime(),
                        target.name,
                        target.version,
                        float(prob_up[i]),
                        float(prob_dn[i]),
                        float(r.get("future_ret_pct", 0.0)),
                        json.dumps(ctx_base),
                    )
                )

            written = write_oof_predictions(rows)
            total_written += written

        print(f"[MOM_ML] Total OOF rows written for {target.name}: {total_written}")

        if save_live_model:
            # Train on ALL data
            X_all, feature_cols = _select_feature_matrix(feats)
            if not X_all.empty:
                y_bull_all = feats["y_bull"].values
                y_bear_all = feats["y_bear"].values

                model_bull = _make_model(seed=42)
                model_bull.fit(X_all, y_bull_all)

                model_bear = _make_model(seed=43)
                model_bear.fit(X_all, y_bear_all)

                save_model_and_features(target.name, target.version, model_bull, model_bear, feature_cols)

        print(f"[MOM_ML] Done target={target.name}\n")


def main():
    import argparse

    p = argparse.ArgumentParser("Momentum ML walk-forward OOF trainer (Dual Model)")
    p.add_argument("--targets-ini", default=ML_TARGETS_INI)
    p.add_argument("--momentum-ini", default=None)

    p.add_argument("--lookback-days", type=int, default=180)
    p.add_argument("--train-days", type=int, default=90)
    p.add_argument("--test-days", type=int, default=14)
    p.add_argument("--step-days", type=int, default=7)
    p.add_argument("--min-train-rows", type=int, default=2000)

    p.add_argument("--only-target", default=None)
    p.add_argument("--no-save-live-model", action="store_true")

    p.add_argument("--symbol", default=None)
    p.add_argument("--symbols-file", default=None)

    p.add_argument("--same-day-only", action="store_true")

    args = p.parse_args()

    symbols = None
    if args.symbols_file:
        symbols = _read_symbols_file(args.symbols_file)

    train_walkforward_oof(
        targets_ini=args.targets_ini,
        lookback_days=args.lookback_days,
        train_days=args.train_days,
        test_days=args.test_days,
        step_days=args.step_days,
        only_target=args.only_target,
        momentum_ini=args.momentum_ini,
        save_live_model=not args.no_save_live_model,
        min_train_rows=args.min_train_rows,
        symbol=args.symbol,
        symbols=symbols,
        same_day_only=args.same_day_only,
    )


if __name__ == "__main__":
    main()
