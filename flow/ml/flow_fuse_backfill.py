# flow/ml/flow_fuse_backfill.py

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, Dict, Any, List

import numpy as np
import pandas as pd
import configparser

from utils.db import get_db_connection
from pillars.common import write_values, TZ
from flow.runtime.utils import update_flow_ml_backfill_status

FLOW_DIR = Path(__file__).resolve().parents[1] / "pillar"
DEFAULT_FLOW_INI = FLOW_DIR / "flow_scenarios.ini"


# ---------- Config + calibration helpers (reuse from flow_pillar idea) ----------

def _load_flow_ml_cfg(flow_ini_path: str | Path) -> Dict[str, Any]:
    cp = configparser.ConfigParser(
        inline_comment_prefixes=(";", "#"),
        interpolation=None,
        strict=False,
    )
    cp.read(flow_ini_path)

    cfg = {
        "blend_weight": cp.getfloat("flow_ml", "blend_weight", fallback=0.35),
        "veto_if_prob_lt": cp.get("flow_ml", "veto_if_prob_lt", fallback="").strip(),
        "cp": cp,
    }
    return cfg


def _load_flow_calibration() -> List[Dict[str, float]]:
    """
    Read calibration buckets from indicators.flow_calibration_4h.
    """
    rows: List[Dict[str, float]] = []
    sql = """
        SELECT
            p_min,
            p_max,
            realized_up_rate
        FROM indicators.flow_calibration_4h
        ORDER BY p_min
    """
    try:
        with get_db_connection() as conn, conn.cursor() as cur:
            cur.execute(sql)
            for p_min, p_max, realized_up_rate in cur.fetchall():
                if realized_up_rate is None:
                    continue
                rows.append(
                    {
                        "p_min": float(p_min),
                        "p_max": float(p_max),
                        "realized_up_rate": float(realized_up_rate),
                    }
                )
    except Exception as e:
        print(f"[FLOW_FUSE] WARNING: failed to load calibration → {e}")
        rows = []

    return rows


_CALIB_BUCKETS: Optional[List[Dict[str, float]]] = None


def _calibrate_prob(p: Optional[float]) -> float:
    """
    Map raw prob -> calibrated prob using bucket stats.
    If calibration not available, return p unchanged (clamped).
    """
    global _CALIB_BUCKETS

    if p is None:
        return 0.0

    try:
        if not np.isfinite(p):
            p = 0.0
    except Exception:
        p = 0.0

    p = max(0.0, min(1.0, float(p)))

    if _CALIB_BUCKETS is None:
        _CALIB_BUCKETS = _load_flow_calibration()

    buckets = _CALIB_BUCKETS or []
    if not buckets:
        return p

    for row in buckets:
        p_min = row.get("p_min", 0.0)
        p_max = row.get("p_max", 1.0)
        if p_min <= p < p_max:
            cal = row.get("realized_up_rate", p)
            if cal is None or not np.isfinite(cal):
                return p
            return max(0.0, min(1.0, float(cal)))

    last = buckets[-1]
    cal = last.get("realized_up_rate", p)
    if cal is None or not np.isfinite(cal):
        return p
    return max(0.0, min(1.0, float(cal)))


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        if not np.isfinite(v):
            return default
        return v
    except Exception:
        return default

def sigmoid(x: float, center: float = 0.5, steepness: float = 12.0) -> float:
    """
    Sigmoid normalization function.
    Maps input x (typically 0-1) to an S-curve.

    Args:
        x: Input value
        center: The x-value where sigmoid is 0.5
        steepness: Controls how sharp the transition is
    """
    try:
        val = 1.0 / (1.0 + np.exp(-steepness * (x - center)))
        return float(val)
    except Exception:
        return 0.5


# ---------- DB loaders ----------
def _load_rules_and_ml(
    market_type: str,
    tf: str,
    cutoff,
    target_name: str,
    version: str,
) -> pd.DataFrame:
    """
    Join existing FLOW.rules + veto from indicator.values
    with ml_pillars.prob_up for matching ts/symbol.

    - indicators.values: uses `interval`, `val`
    - indicators.ml_pillars: uses `tf`, `prob_up`
    """
    sql = """
        WITH rules AS (
            SELECT
                symbol,
                market_type,
                interval,
                ts,
                MAX(CASE WHEN metric = 'FLOW.score'
                         THEN val END) AS rules_score,
                MAX(CASE WHEN metric = 'FLOW.veto_flag'
                         THEN val END) AS rules_veto
            FROM indicators.values
            WHERE ts >= %(cutoff)s
              AND market_type = %(market_type)s
              AND interval    = %(interval)s
              AND metric IN ('FLOW.score', 'FLOW.veto_flag')
            GROUP BY symbol, market_type, interval, ts
        ),
        ml AS (
            SELECT
                symbol,
                market_type,
                tf,
                ts,
                prob_up
            FROM indicators.ml_pillars
            WHERE ts >= %(cutoff)s
              AND market_type = %(market_type)s
              AND tf          = %(interval)s
              AND pillar      = 'flow'
              AND target_name = %(target_name)s
              AND version     = %(version)s
        )
        SELECT
            r.symbol,
            r.market_type,
            r.interval AS tf,   -- 👈 alias so downstream sees 'tf'
            r.ts,
            r.rules_score,
            r.rules_veto,
            m.prob_up
        FROM rules r
        LEFT JOIN ml m
          ON  m.symbol      = r.symbol
          AND m.market_type = r.market_type
          AND m.tf          = r.interval
          AND m.ts          = r.ts
        WHERE m.prob_up IS NOT NULL
        ORDER BY r.symbol, r.ts
    """

    with get_db_connection() as conn:
        df = pd.read_sql(
            sql,
            conn,
            params={
                "cutoff": cutoff,
                "market_type": market_type,
                "interval": tf,      # we pass "15m", maps to both interval & tf
                "target_name": target_name,
                "version": version,
            },
        )

    if df.empty:
        print(f"[FLOW_FUSE] No joined RULES+ML rows for tf={tf}, market_type={market_type}")
        return df

    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df


# ---------- Fusion core ----------

def _fuse_row(
    row: pd.Series,
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Take one joined row → compute ml_score, fused_score, fused_veto.
    """
    rules_score = _safe_float(row["rules_score"], 0.0)
    rules_veto_raw = _safe_float(row["rules_veto"], 0.0)
    rules_veto = rules_veto_raw >= 0.5

    ml_prob_raw = _safe_float(row["prob_up"], 0.0)
    ml_prob_raw = max(0.0, min(1.0, ml_prob_raw))

    ml_prob_cal = _calibrate_prob(ml_prob_raw)
    ml_score = round(100.0 * ml_prob_cal, 2)

    # Calculate ML Bucket (1: Bearish to 5: Bullish)
    # 0-20, 20-40, 40-60, 60-80, 80-100
    ml_bucket = min(5, max(1, int(ml_score / 20.0) + 1))

    base_prob = rules_score / 100.0
    w = cfg["blend_weight"]
    w = max(0.0, min(1.0, float(w)))

    fused_prob = (1.0 - w) * base_prob + w * ml_prob_cal
    fused_prob = max(0.0, min(1.0, fused_prob))
    fused_score = round(100.0 * fused_prob, 2)

    # Sigmoid Normalization for Final Score
    # Centers the fused probability (0-1) around 0.5 with steepness
    fused_prob_sigmoid = sigmoid(fused_prob, center=0.5, steepness=10.0)
    score_final = round(100.0 * fused_prob_sigmoid, 2)

    fused_veto = rules_veto

    veto_thr_str = cfg.get("veto_if_prob_lt") or ""
    if veto_thr_str:
        try:
            thr = float(veto_thr_str)
            if thr > 1.0:
                thr = thr / 100.0
            if fused_prob < thr:
                fused_veto = True
        except Exception:
            pass

    return {
        "rules_score": rules_score,
        "rules_veto": rules_veto,
        "ml_prob_raw": ml_prob_raw,
        "ml_prob_cal": ml_prob_cal,
        "ml_score": ml_score,
        "ml_bucket": ml_bucket,
        "fused_prob": fused_prob,
        "fused_score": fused_score,
        "score_final": score_final,
        "fused_veto": fused_veto,
    }


def _run_backfill(
    *,
    market_type: str,
    tf: str,
    lookback_days: int,
    from_date: Optional[str],
    target_name: str,
    version: str,
    flow_ini: str,
    run_id: str,
    source: str,
):
    now_utc = datetime.now(timezone.utc)

    if from_date:
        cutoff = datetime.fromisoformat(from_date).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        cutoff = cutoff.replace(tzinfo=timezone.utc)
    else:
        cutoff = now_utc - timedelta(days=lookback_days)

    print(
        f"[FLOW_FUSE] Starting ML-only fusion for "
        f"market_type={market_type} tf={tf} cutoff={cutoff} "
        f"target={target_name} version={version}"
    )

    cfg = _load_flow_ml_cfg(flow_ini)
    df = _load_rules_and_ml(
        market_type=market_type,
        tf=tf,
        cutoff=cutoff,
        target_name=target_name,
        version=version,
    )

    if df.empty:
        print("[FLOW_FUSE] Nothing to do.")
        return

    rows: List[tuple] = []
    last_ts_per_symbol: Dict[str, datetime] = {}

    for _, row in df.iterrows():
        sym = row["symbol"]
        mt = row["market_type"]
        tf_row = row["tf"]
        ts = row["ts"].to_pydatetime().replace(tzinfo=TZ)

        fused = _fuse_row(row, cfg)

        # Save last ts per symbol for status update
        prev = last_ts_per_symbol.get(sym)
        if prev is None or ts > prev:
            last_ts_per_symbol[sym] = ts

        # ML metrics
        rows.extend(
            [
                (
                    sym,
                    mt,
                    tf_row,
                    ts,
                    "FLOW.ml_score",
                    float(fused["ml_score"]),
                    "{}",
                    run_id,
                    source,
                ),
                (
                    sym,
                    mt,
                    tf_row,
                    ts,
                    "FLOW.ml_bucket",
                    float(fused["ml_bucket"]),
                    "{}",
                    run_id,
                    source,
                ),
                (
                    sym,
                    mt,
                    tf_row,
                    ts,
                    "FLOW.ml_p_up_cal",
                    float(fused["ml_prob_cal"]),
                    "{}",
                    run_id,
                    source,
                ),
                (
                    sym,
                    mt,
                    tf_row,
                    ts,
                    "FLOW.ml_p_down_cal",
                    float(1.0 - fused["ml_prob_cal"]),
                    "{}",
                    run_id,
                    source,
                ),
                (
                    sym,
                    mt,
                    tf_row,
                    ts,
                    "FLOW.fused_score",
                    float(fused["fused_score"]),
                    "{}",
                    run_id,
                    source,
                ),
                (
                    sym,
                    mt,
                    tf_row,
                    ts,
                    "FLOW.score_final",
                    float(fused["score_final"]),
                    "{}",
                    run_id,
                    source,
                ),
                (
                    sym,
                    mt,
                    tf_row,
                    ts,
                    "FLOW.fused_veto",
                    1.0 if fused["fused_veto"] else 0.0,
                    "{}",
                    run_id,
                    source,
                ),
            ]
        )

    print(f"[FLOW_FUSE] Prepared {len(rows)} metric rows to write.")

    if rows:
        write_values(rows)
        print("[FLOW_FUSE] Wrote ML + fused metrics into indicators.values")

    # Update symbol_universe ML status
    with get_db_connection() as conn:
        for sym, last_ts in last_ts_per_symbol.items():
            update_flow_ml_backfill_status(
                conn,
                symbol=sym,
                kind=market_type,
                last_ts=last_ts,
                run_id=run_id,
            )
        print(
            f"[FLOW_FUSE] Updated flow_ml_last_*_ts for {len(last_ts_per_symbol)} symbols"
        )


# ---------- CLI ----------

def main():
    p = argparse.ArgumentParser("Flow ML-only fusion backfill")

    p.add_argument("--market-type", choices=["futures", "spot"], required=True)
    p.add_argument("--tf", default="15m", help="Timeframe (Flow ML is 15m-based)")
    p.add_argument("--lookback-days", type=int, default=200)
    p.add_argument("--from-date", help="YYYY-MM-DD (overrides lookback-days)", default=None)

    p.add_argument(
        "--target-name",
        default="flow_ml.target.bull_2pct_4h",
        help="ML target_name (must match ml_pillars)",
    )
    p.add_argument(
        "--version",
        default="xgb_v1",
        help="ML model version (must match ml_pillars)",
    )

    p.add_argument(
        "--flow-ini",
        default=str(DEFAULT_FLOW_INI),
        help="Path to flow_scenarios.ini (for [flow_ml] config)",
    )
    p.add_argument(
        "--run-id",
        default="FLOW_ML_FUSE_BACKFILL",
        help="Run id to stamp in indicators.values",
    )
    p.add_argument(
        "--source",
        default="flow_ml_fuse_backfill",
        help="Source to stamp in indicators.values",
    )

    args = p.parse_args()

    _run_backfill(
        market_type=args.market_type,
        tf=args.tf,
        lookback_days=args.lookback_days,
        from_date=args.from_date,
        target_name=args.target_name,
        version=args.version,
        flow_ini=args.flow_ini,
        run_id=args.run_id,
        source=args.source,
    )


if __name__ == "__main__":
    main()
