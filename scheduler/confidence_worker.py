# scheduler/confidence_worker.py
from __future__ import annotations

from typing import List, Tuple, Optional
from contextlib import contextmanager

from utils.db import get_db_connection

# Calls your existing modules:
from scheduler import update_confidence, update_confidence_oi, composite_writer  # noqa: F401

BATCH_SIZE = 50
CONF_THRESHOLD = 65.0  # composite final_score threshold

# ───────────────── DB helpers ─────────────────
@contextmanager
def _conn_cur():
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        yield conn, cur
        conn.commit()
    finally:
        conn.close()

def _exec(sql: str, params: tuple = ()) -> None:
    with _conn_cur() as (_, cur):
        cur.execute(sql, params)

def _fetchall(sql: str, params: tuple = ()) -> list:
    with _conn_cur() as (_, cur):
        cur.execute(sql, params)
        return cur.fetchall()

# ───────────────── Status helpers ─────────────────
def set_sub(unique_id: str, sub_status: str) -> None:
    _exec(
        "UPDATE webhooks.webhook_alerts SET sub_status=%s, last_checked_at=NOW() WHERE unique_id=%s",
        (sub_status, unique_id),
    )

def set_error(unique_id: str, msg: str) -> None:
    _exec(
        """UPDATE webhooks.webhook_alerts
              SET status='ERROR', sub_status='SIG_ERROR', last_error=%s, last_checked_at=NOW()
            WHERE unique_id=%s""",
        (msg[:500], unique_id),
    )

def set_rejected(unique_id: str, reason: str) -> None:
    _exec(
        """UPDATE webhooks.webhook_alerts
              SET status='REJECTED', sub_status='SIG_REJECTED',
                  rejection_reason=%s, last_checked_at=NOW()
            WHERE unique_id=%s""",
        (reason[:500], unique_id),
    )

def set_ready(unique_id: str) -> None:
    _exec(
        "UPDATE webhooks.webhook_alerts SET sub_status='SIG_READY', last_checked_at=NOW() WHERE unique_id=%s",
        (unique_id,),
    )

# ───────────────── Atomic claim ─────────────────
def _claim_batch() -> List[Tuple[str, str]]:
    with _conn_cur() as (_, cur):
        cur.execute(
            """
            UPDATE webhooks.webhook_alerts AS w
               SET sub_status='SIG_EVALUATING', last_checked_at=NOW()
             WHERE w.unique_id IN (
                    SELECT unique_id
                      FROM webhooks.webhook_alerts
                     WHERE status='SIGNAL_PROCESS'
                       AND sub_status IN ('SIG_PENDING','SIG_OK')
                     ORDER BY received_at ASC
                     LIMIT %s
                     FOR UPDATE SKIP LOCKED
             )
            RETURNING unique_id, symbol
            """,
            (BATCH_SIZE,),
        )
        rows = cur.fetchall()
    return [(r[0], r[1]) for r in rows or []]

# ───────────────── Module invocations ─────────────────
def _run_indicator_confidence(symbols: List[str]) -> None:
    try:
        if hasattr(update_confidence, "run_for_symbols"):
            update_confidence.run_for_symbols(symbols=symbols)
        elif hasattr(update_confidence, "run"):
            update_confidence.run(symbols=symbols)
        elif hasattr(update_confidence, "main"):
            update_confidence.main()  # type: ignore
    except Exception as e:
        raise RuntimeError(f"update_confidence failed: {e}")

def _run_oi_confidence(symbols: List[str]) -> None:
    try:
        if hasattr(update_confidence_oi, "run_for_symbols"):
            update_confidence_oi.run_for_symbols(symbols=symbols)
        elif hasattr(update_confidence_oi, "run"):
            update_confidence_oi.run(symbols=symbols)
        elif hasattr(update_confidence_oi, "main"):
            update_confidence_oi.main()  # type: ignore
    except Exception as e:
        raise RuntimeError(f"update_confidence_oi failed: {e}")

def _run_composite(symbols: List[str]) -> None:
    try:
        # both supported now
        if hasattr(composite_writer, "run_for_symbols"):
            composite_writer.run_for_symbols(symbols=symbols)
        else:
            composite_writer.run(symbols=symbols)  # run() now accepts optional symbols
    except Exception as e:
        raise RuntimeError(f"composite_writer failed: {e}")

# ───────────────── Final score read ─────────────────
def _get_final_score(symbol: str) -> Optional[float]:
    rows = _fetchall(
        """
        SELECT final_score
          FROM signals.confidence_composite
         WHERE symbol=%s
         ORDER BY ts DESC
         LIMIT 1
        """,
        (symbol,),
    )
    return float(rows[0][0]) if rows else None

# ───────────────── Orchestration ─────────────────
def process_one(unique_id: str, symbol: str) -> None:
    try:
        set_sub(unique_id, "SIG_EVALUATING")

        _run_indicator_confidence([symbol])   # 1) indicator subscores
        _run_oi_confidence([symbol])          # 2) OI pillar/subscores
        _run_composite([symbol])              # 3) composite (writes final_score)

        score = _get_final_score(symbol) or 0.0
        if score >= CONF_THRESHOLD:
            set_ready(unique_id)
            print(f"✅ {symbol} → SIG_READY (final_score={score:.2f})")
        else:
            set_rejected(unique_id, f"Composite below threshold ({score:.2f} < {CONF_THRESHOLD})")
            print(f"🚫 {symbol} → SIG_REJECTED (final_score={score:.2f})")
    except Exception as e:
        set_error(unique_id, str(e))
        print(f"❌ {symbol} → ERROR: {e}")

def run():
    batch = _claim_batch()
    if not batch:
        print("🔍 No rows to evaluate.")
        return
    print(f"🧠 Confidence batch: {len(batch)}")
    for uid, sym in batch:
        process_one(uid, sym)

if __name__ == "__main__":
    run()
