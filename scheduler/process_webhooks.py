# scheduler/process_webhooks.py — SIGNAL_PROCESS stage (routes to ingestion only)
from __future__ import annotations

import configparser
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Optional, Tuple

from utils.db import get_db_connection

# ---------------- Status constants (inline; move to utils/statuses.py later if you wish) ----------------
class SignalStatus:
    SIGNAL_PROCESS   = "SIGNAL_PROCESS"
    DATA_PROCESSING  = "DATA_PROCESSING"   # ingestion queue
    REJECTED         = "REJECTED"
    ERROR            = "ERROR"

class SubStatus:
    SIG_PENDING         = "SIG_PENDING"
    SIG_EVALUATING      = "SIG_EVALUATING"
    SIG_OK              = "SIG_OK"
    SIG_RETRY_WAIT      = "SIG_RETRY_WAIT"
    SIG_RETRY_DUE       = "SIG_RETRY_DUE"
    SIG_REJECTED        = "SIG_REJECTED"
    SIG_ERROR           = "SIG_ERROR"
    SIG_WAIT_MANUAL     = "SIG_WAIT_MANUAL"
    IND_OK              = "IND_OK"
    ZON_OK              = "ZON_OK"
    OI_OK               = "OI_OK"
    SIG_READY           = "SIG_READY"
    INGESTION_PENDING   = "INGESTION_PENDING"   # fetch_prices picks this up

# ---------------- Config / constants ----------------
BATCH_SIZE = 100
MAX_SIG_RETRIES = 3
# Exponential-ish backoff (minutes). Index by current retry_count (capped).
RETRY_BACKOFF_MIN = [5, 15, 30]

TZ = timezone.utc

# ---------------- Types ----------------
@dataclass
class Alert:
    unique_id: str
    symbol: str
    strategy: str
    payload: Optional[dict]
    received_at: datetime
    signal_type: Optional[str]

def _utcnow() -> datetime:
    return datetime.now(tz=TZ)

def _side_from_signal(signal_type: Optional[str]) -> Optional[str]:
    if not signal_type:
        return None
    s = signal_type.strip().upper()
    if s in ("BUY", "LONG"):
        return "LONG"
    if s in ("SELL", "SHORT"):
        return "SHORT"
    # CLOSE/EXIT handled by a different worker
    return None

# ---------------- DB helpers ----------------
def _claim_signal_alerts(cur) -> List[Alert]:
    """
    Claim up to BATCH_SIZE rows for SIGNAL_PROCESS.
    Picks fresh items + due retries. Respects MAX_SIG_RETRIES.
    Uses SKIP LOCKED for safe parallelism.
    """
    sql = """
    UPDATE webhooks.webhook_alerts AS w
       SET sub_status = %s,
           last_checked_at = now()
     WHERE w.unique_id IN (
           SELECT unique_id
             FROM webhooks.webhook_alerts
            WHERE status = %s
              AND (
                    sub_status IN (%s,%s,%s,%s,%s)
                 OR (sub_status = %s AND now() >= COALESCE(next_retry_at, now()))
              )
              AND COALESCE(retry_count, 0) < %s
            ORDER BY COALESCE(next_retry_at, received_at) ASC, received_at ASC
            LIMIT %s
            FOR UPDATE SKIP LOCKED
     )
    RETURNING w.unique_id, w.symbol, COALESCE(w.strategy,'UNKNOWN') AS strategy,
              w.payload, w.received_at, w.signal_type;
    """
    cur.execute(
        sql,
        (
            SubStatus.SIG_EVALUATING,
            SignalStatus.SIGNAL_PROCESS,
            SubStatus.SIG_PENDING, SubStatus.IND_OK, SubStatus.ZON_OK,
            SubStatus.OI_OK, SubStatus.SIG_READY,
            SubStatus.SIG_RETRY_WAIT,
            MAX_SIG_RETRIES, BATCH_SIZE,
        )
    )
    rows = cur.fetchall()
    out: List[Alert] = []
    for uid, sym, strat, payload, rcvd, st in rows:
        out.append(Alert(str(uid), sym, strat, payload if isinstance(payload, dict) else None, rcvd, st))
    return out

def _get_status_snapshot(cur, unique_id: str) -> Tuple[str, str]:
    cur.execute("SELECT status, sub_status FROM webhooks.webhook_alerts WHERE unique_id=%s", (unique_id,))
    row = cur.fetchone()
    return (row[0], row[1]) if row else ("?", "?")

def _next_attempt_number(cur, unique_id: str) -> int:
    cur.execute("SELECT COALESCE(MAX(attempt_number),0) FROM journal.rejections_log WHERE unique_id=%s;", (unique_id,))
    return int(cur.fetchone()[0] or 0) + 1

def _log_attempt(cur, unique_id: str, attempt_number: int, status: str,
                 rejection_reason: Optional[str] = None, trigger: Optional[str] = None, notes: Optional[str] = None):
    # Append current status/sub_status snapshot to notes for forensics
    st, sub = _get_status_snapshot(cur, unique_id)
    snap = f"@{st}/{sub}"
    combined_notes = (notes or "").strip()
    combined_notes = (combined_notes + " | " if combined_notes else "") + snap
    cur.execute(
        """
        INSERT INTO journal.rejections_log
            (unique_id, attempt_number, processed_at, status, rejection_reason, re_run_trigger, notes)
        VALUES (%s,%s, now(), %s, %s, %s, %s);
        """,
        (unique_id, attempt_number, status, rejection_reason, trigger, combined_notes)
    )

def _finalize_alert(cur, unique_id: str, status: str, rejection_reason: Optional[str] = None):
    cur.execute(
        """
        UPDATE webhooks.webhook_alerts
           SET status = %s,
               last_checked_at = now(),
               rejection_reason = COALESCE(%s, rejection_reason)
         WHERE unique_id = %s;
        """,
        (status, rejection_reason, unique_id)
    )

def _set_sub(cur, unique_id: str, sub: str):
    cur.execute("UPDATE webhooks.webhook_alerts SET sub_status=%s WHERE unique_id=%s", (sub, unique_id))

def _clear_retry_fields_on_success(cur, unique_id: str):
    cur.execute(
        """
        UPDATE webhooks.webhook_alerts
           SET retry_count = 0,
               next_retry_at = NULL
         WHERE unique_id=%s
        """,
        (unique_id,)
    )

def _has_manual_override(cur, symbol: str, strategy: str) -> bool:
    cur.execute(
        "SELECT 1 FROM manual_alerts WHERE symbol=%s AND strategy=%s LIMIT 1",
        (symbol, strategy)
    )
    return cur.fetchone() is not None

# ---------------- Retry scheduler ----------------
def _schedule_retry(cur, unique_id: str, attempt_number: int, base_reason: str):
    # Figure current retry_count
    cur.execute("SELECT COALESCE(retry_count,0) FROM webhooks.webhook_alerts WHERE unique_id=%s", (unique_id,))
    rc = int(cur.fetchone()[0] or 0)

    if rc >= MAX_SIG_RETRIES - 1:
        # Hard stop: max retries reached
        _log_attempt(cur, unique_id, attempt_number, status="rejected",
                     rejection_reason=f"{base_reason} (max retries reached)")
        _finalize_alert(cur, a_unique_id := unique_id, status=SignalStatus.REJECTED, rejection_reason=base_reason)
        _set_sub(cur, a_unique_id, SubStatus.SIG_REJECTED)
        return

    mins = RETRY_BACKOFF_MIN[min(rc, len(RETRY_BACKOFF_MIN)-1)]
    cur.execute(
        """
        UPDATE webhooks.webhook_alerts
           SET sub_status   = %s,
               next_retry_at= now() + make_interval(mins := %s),
               retry_count  = COALESCE(retry_count,0) + 1
         WHERE unique_id = %s
        """,
        (SubStatus.SIG_RETRY_WAIT, mins, unique_id)
    )
    _log_attempt(cur, unique_id, attempt_number, status="rejected",
                 rejection_reason=f"{base_reason} (retry in {mins}m)")

# ---------------- Main runner ----------------
def process_signal_stage(single_run: bool = True) -> None:
    """
    SIGNAL_PROCESS stage (no rule evaluation, no trade here):
      - Claim → SIG_EVALUATING
      - Validate side + Check manual_alerts gate (symbol+strategy)
      - If ok → route to ingestion: status=DATA_PROCESSING, sub_status=INGESTION_PENDING
      - If manual missing → retry with backoff
      - If invalid side → REJECTED
      - On error → ERROR
    """
    with get_db_connection() as conn, conn.cursor() as cur:
        routed = rejected = errs = 0
        alerts = _claim_signal_alerts(cur)
        if not alerts:
            print("🔍 No signal alerts ready.")
            return

        print(f"🚦 Processing {len(alerts)} signal alert(s)...")
        for a in alerts:
            try:
                side = _side_from_signal(a.signal_type)
                if side not in ("LONG", "SHORT"):
                    _finalize_alert(cur, a.unique_id, status=SignalStatus.REJECTED, rejection_reason="Unsupported/missing side")
                    _set_sub(cur, a.unique_id, SubStatus.SIG_REJECTED)
                    _log_attempt(cur, a.unique_id, _next_attempt_number(cur, a.unique_id), status="rejected",
                                 rejection_reason="Unsupported/missing side")
                    rejected += 1
                    continue

                # --- Manual gate (required) ---
                if not _has_manual_override(cur, a.symbol, a.strategy):
                    attempt = _next_attempt_number(cur, a.unique_id)
                    _set_sub(cur, a.unique_id, SubStatus.SIG_WAIT_MANUAL)
                    _schedule_retry(cur, a.unique_id, attempt, "No manual override found")
                    rejected += 1
                    continue

                # --- Route straight to ingestion ---
                attempt = _next_attempt_number(cur, a.unique_id)
                _log_attempt(cur, a.unique_id, attempt, status="accepted", notes="routed_to_ingestion")
                _finalize_alert(cur, a.unique_id, status=SignalStatus.DATA_PROCESSING)
                _set_sub(cur, a.unique_id, SubStatus.INGESTION_PENDING)
                _clear_retry_fields_on_success(cur, a.unique_id)
                routed += 1

            except Exception as e:
                attempt = _next_attempt_number(cur, a.unique_id)
                _log_attempt(cur, a.unique_id, attempt, status="error", rejection_reason=str(e)[:500])
                _finalize_alert(cur, a.unique_id, status=SignalStatus.ERROR, rejection_reason=str(e)[:500])
                _set_sub(cur, a.unique_id, SubStatus.SIG_ERROR)
                errs += 1

        print(f"📝 Done. Routed to ingestion={routed}, Rejected={rejected}, Errors={errs}")

if __name__ == "__main__":
    process_signal_stage(single_run=True)
