# File: scheduler/recheck_rejected.py
# Purpose: Periodically re-evaluate previously REJECTED alerts. Safely claim a batch
#          using SKIP LOCKED, run the rule engine, and if now valid, open a trade
#          and mark ACCEPTED. Idempotent & race-safe for PostgreSQL/TimescaleDB.

from __future__ import annotations

import configparser
from datetime import datetime, timezone
from typing import Iterable, Optional, Sequence

from utils.db import get_db_connection
from utils.rule_evaluator import evaluate_alert

TZ = timezone.utc

# --- Adjust these if your tables live in different schemas ---
ALERTS_TABLE = "journal.webhook_alerts"
TRADES_TABLE = "journal.trading_journal"
# -------------------------------------------------------------

def _utcnow_str() -> str:
    return datetime.now(tz=TZ).strftime("%Y-%m-%d %H:%M:%S")

def load_config_flag() -> bool:
    """Checks .ini config for retry toggle."""
    config = configparser.ConfigParser()
    config.read("zerodha.ini")
    return config.get("settings", "retry_rejected_alerts", fallback="no").strip().lower() == "yes"

def _claim_rejected(conn, batch_size: int = 200) -> Sequence[tuple]:
    """
    Atomically claim up to `batch_size` REJECTED alerts into RECHECKING state,
    returning the claimed rows. Uses SKIP LOCKED to allow safe parallel runners.
    """
    with conn.cursor() as cur:
        # Use alert_time when available, else fallback to timestamp
        cur.execute(
            f"""
            WITH cte AS (
              SELECT id
              FROM {ALERTS_TABLE}
              WHERE status = 'REJECTED'
              ORDER BY COALESCE(alert_time, timestamp) ASC
              LIMIT %s
              FOR UPDATE SKIP LOCKED
            )
            UPDATE {ALERTS_TABLE} wa
            SET status = 'RECHECKING',
                last_checked_at = %s,
                last_message = %s
            FROM cte
            WHERE wa.id = cte.id
            RETURNING wa.id, wa.symbol, wa.strategy_name, wa.strategy_version, wa.signal_type
            """,
            (batch_size, _utcnow_str(), "retry"),
        )
        rows = cur.fetchall()
        return rows

def _insert_trading_journal(cursor,
                            alert_id: int,
                            symbol: str,
                            strategy: str,
                            version: str,
                            signal_type: str,
                            rule_matched: str,
                            score: float,
                            decision_tags: Optional[Iterable[str]] = None) -> None:
    cursor.execute(
        f"""
        INSERT INTO {TRADES_TABLE} (
            alert_id, timestamp, symbol, strategy_name, strategy_version,
            signal_type, rule_matched, confidence_score, rule_engine_version,
            decision_tags, status
        ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,'OPEN')
        """,
        (
            alert_id,
            _utcnow_str(),
            symbol,
            strategy or "UNKNOWN",
            version or "v1.0",
            (signal_type or "UNKNOWN").upper(),
            rule_matched,
            float(score or 0),
            "retry",
            ",".join(list(decision_tags or [])),
        ),
    )

def reprocess_rejected_alerts(batch_size: int = 200) -> None:
    """Re-evaluates REJECTED alerts. If valid, reclassify as ACCEPTED and
    create a new OPEN trade linked by alert_id. Otherwise set back to REJECTED.
    Safe for concurrent schedulers (SKIP LOCKED)."""
    with get_db_connection() as conn:
        claimed = _claim_rejected(conn, batch_size=batch_size)
        if not claimed:
            print("🔁 No rejected alerts to reprocess")
            return

        print(f"🔁 Reprocessing {len(claimed)} rejected alerts…")

        with conn.cursor() as cursor:
            for alert_id, symbol, strategy, version, signal in claimed:
                try:
                    # evaluate_alert may return (4) or (5) tuple
                    result = evaluate_alert(symbol)
                    if len(result) == 5:
                        is_valid, rule_matched, failed_reasons, score, decision_tags = result
                    else:
                        is_valid, rule_matched, failed_reasons, score = result
                        decision_tags = []

                    if is_valid:
                        _insert_trading_journal(
                            cursor,
                            alert_id=alert_id,
                            symbol=symbol,
                            strategy=strategy,
                            version=version or "v1.0",
                            signal_type=signal or "LONG",
                            rule_matched=rule_matched,
                            score=score,
                            decision_tags=decision_tags,
                        )
                        cursor.execute(
                            f"""
                            UPDATE {ALERTS_TABLE}
                            SET status = 'ACCEPTED',
                                last_checked_at = %s,
                                last_message = %s
                            WHERE id = %s
                            """,
                            (_utcnow_str(), (rule_matched or "")[:255], alert_id),
                        )
                        print(f"✅ Re-Accepted: {symbol} [{rule_matched}] | score={score}")
                    else:
                        msg = "; ".join(f"{r}:{rsn}" for r, rsn in (failed_reasons or []))[:255]
                        cursor.execute(
                            f"""
                            UPDATE {ALERTS_TABLE}
                            SET status = 'REJECTED',
                                last_checked_at = %s,
                                last_message = %s
                            WHERE id = %s
                            """,
                            (_utcnow_str(), msg, alert_id),
                        )
                        print(f"❌ Still Rejected: {symbol} | score={score} | failed={len(failed_reasons or [])}")

                except Exception as e:
                    cursor.execute(
                        f"""
                        UPDATE {ALERTS_TABLE}
                        SET status = 'ERROR',
                            last_checked_at = %s,
                            last_message = %s
                        WHERE id = %s
                        """,
                        (_utcnow_str(), str(e)[:255], alert_id),
                    )
                    print(f"⚠️ Error reprocessing alert ID={alert_id} ({symbol}): {e}")

        conn.commit()
        print("✅ Recheck complete.")

if __name__ == "__main__":
    if load_config_flag():
        reprocess_rejected_alerts()
    else:
        print("⚠️ Retry of rejected alerts is disabled in zerodha.ini. Exiting.")
