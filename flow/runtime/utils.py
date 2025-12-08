# flow/runtime/utils.py
from __future__ import annotations

from typing import Optional, Tuple
from datetime import datetime
from psycopg2.extensions import connection as PgConn


def update_flow_status(
    conn: PgConn,
    symbol: str,
    run_id: str,
    *,
    rules_ts_fut: Optional[datetime] = None,
    rules_ts_spot: Optional[datetime] = None,
    ml_ts_fut: Optional[datetime] = None,
    ml_ts_spot: Optional[datetime] = None,
    mode: str = "rules",
) -> None:
    """
    Update Flow status columns in reference.symbol_universe.

    Uses COALESCE so that passing None does NOT wipe existing values.

    mode:
      - 'rules' → updates flow_rules_* + flow_rules_last_run_*
      - 'ml'    → updates flow_ml_*    + flow_ml_last_run_*
      - 'both'  → updates both groups

    NOTE: For ML backfill we prefer update_flow_ml_backfill_status()
    because it uses GREATEST() to avoid moving timestamps backwards.
    """
    mode = (mode or "rules").lower()

    with conn.cursor() as cur:
        if mode == "rules":
            cur.execute(
                """
                UPDATE reference.symbol_universe
                   SET flow_rules_last_fut_ts  = COALESCE(%s, flow_rules_last_fut_ts),
                       flow_rules_last_spot_ts = COALESCE(%s, flow_rules_last_spot_ts),
                       flow_rules_last_run_at  = NOW(),
                       flow_rules_last_run_id  = %s
                 WHERE symbol = %s
                """,
                (rules_ts_fut, rules_ts_spot, run_id, symbol),
            )

        elif mode == "ml":
            cur.execute(
                """
                UPDATE reference.symbol_universe
                   SET flow_ml_last_fut_ts  = COALESCE(%s, flow_ml_last_fut_ts),
                       flow_ml_last_spot_ts = COALESCE(%s, flow_ml_last_spot_ts),
                       flow_ml_last_run_at  = NOW(),
                       flow_ml_last_run_id  = %s
                 WHERE symbol = %s
                """,
                (ml_ts_fut, ml_ts_spot, run_id, symbol),
            )

        elif mode == "both":
            cur.execute(
                """
                UPDATE reference.symbol_universe
                   SET flow_rules_last_fut_ts  = COALESCE(%s, flow_rules_last_fut_ts),
                       flow_rules_last_spot_ts = COALESCE(%s, flow_rules_last_spot_ts),
                       flow_rules_last_run_at  = NOW(),
                       flow_rules_last_run_id  = %s,
                       flow_ml_last_fut_ts     = COALESCE(%s, flow_ml_last_fut_ts),
                       flow_ml_last_spot_ts    = COALESCE(%s, flow_ml_last_spot_ts),
                       flow_ml_last_run_at     = NOW(),
                       flow_ml_last_run_id     = %s
                 WHERE symbol = %s
                """,
                (
                    rules_ts_fut,
                    rules_ts_spot,
                    run_id,
                    ml_ts_fut,
                    ml_ts_spot,
                    run_id,
                    symbol,
                ),
            )
        else:
            # invalid mode → no-op
            return

    conn.commit()


def get_flow_last_ts(
    conn: PgConn,
    symbol: str,
    kind: str,
    mode: str = "rules",
) -> Tuple[Optional[datetime], Optional[datetime]]:
    """
    Read last Flow timestamps for a symbol.

    Returns (rules_ts, ml_ts) depending on mode/kind.

      kind='futures' → *_fut_ts
      kind='spot'    → *_spot_ts
    """
    mode = (mode or "rules").lower()
    kind = kind.lower()

    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT flow_rules_last_fut_ts,
                   flow_rules_last_spot_ts,
                   flow_ml_last_fut_ts,
                   flow_ml_last_spot_ts
              FROM reference.symbol_universe
             WHERE symbol = %s
            """,
            (symbol,),
        )
        row = cur.fetchone()

    if not row:
        return (None, None)

    rules_fut, rules_spot, ml_fut, ml_spot = row

    if kind == "futures":
        rules_ts = rules_fut
        ml_ts = ml_fut
    else:
        rules_ts = rules_spot
        ml_ts = ml_spot

    if mode == "rules":
        return (rules_ts, None)
    elif mode == "ml":
        return (None, ml_ts)
    else:  # "both"
        return (rules_ts, ml_ts)


def update_flow_ml_jobs_status(
    conn: PgConn,
    *,
    train_at: Optional[datetime] = None,
    calib_at: Optional[datetime] = None,
    infer_at: Optional[datetime] = None,
) -> None:
    """
    Update global Flow ML job timestamps in reference.symbol_universe.

    Expected columns:
      - flow_ml_train_last_at  (timestamptz, nullable)
      - flow_ml_calib_last_at  (timestamptz, nullable)
      - flow_ml_infer_last_at  (timestamptz, nullable)

    This is GLOBAL (per universe), so we update all rows.

    Safe if some args are None (COALESCE keeps old values).
    """
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE reference.symbol_universe
                   SET flow_ml_train_last_at  = COALESCE(%s, flow_ml_train_last_at),
                       flow_ml_calib_last_at  = COALESCE(%s, flow_ml_calib_last_at),
                       flow_ml_infer_last_at  = COALESCE(%s, flow_ml_infer_last_at)
                """,
                (train_at, calib_at, infer_at),
            )
        conn.commit()
    except Exception as e:
        print(f"[FLOW_ML_STATUS] WARNING: failed to update ML job status → {e}")
        conn.rollback()


def update_flow_ml_backfill_status(
    conn: PgConn,
    symbol: str,
    kind: str,
    last_ts: Optional[datetime],
    run_id: str,
) -> None:
    """
    Update per-symbol FLOW ML backfill markers in reference.symbol_universe.

      kind='futures' -> flow_ml_last_fut_ts
      kind='spot'    -> flow_ml_last_spot_ts

    We:
      - bump the relevant ts column to GREATEST(existing, last_ts)
      - always refresh flow_ml_last_run_at / flow_ml_last_run_id
    """
    if last_ts is None:
        return

    if kind == "futures":
        col_ts = "flow_ml_last_fut_ts"
    elif kind == "spot":
        col_ts = "flow_ml_last_spot_ts"
    else:
        print(f"[FLOW_ML] update_flow_ml_backfill_status: unknown kind={kind!r}")
        return

    try:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                UPDATE reference.symbol_universe
                   SET {col_ts}            = GREATEST(COALESCE({col_ts}, %s), %s),
                       flow_ml_last_run_at  = NOW(),
                       flow_ml_last_run_id  = %s
                 WHERE symbol = %s
                   AND enabled = TRUE
                """,
                (last_ts, last_ts, run_id, symbol),
            )
        conn.commit()
    except Exception as e:
        print(f"[FLOW_ML] WARNING: failed to update ML backfill status for {symbol} → {e}")
        conn.rollback()
