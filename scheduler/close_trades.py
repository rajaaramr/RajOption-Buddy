# File: scheduler/close_trades.py
# Purpose: Close OPEN trades when an exit alert is present. Schema-aligned with
#          Contd 6 (uses trading_journal.future_price as entry, timestamp as
#          entry_time, and signal_type as entry side). Idempotent & UTC-safe.

from __future__ import annotations

from datetime import datetime, timezone
from typing import Tuple

from utils.kite_utils import fetch_futures_data, load_kite
from utils.db import get_db_connection

TZ = timezone.utc

# --- Adjust if your schema names differ ---
ALERTS_TABLE = "journal.webhook_alerts"
TRADES_TABLE = "journal.trading_journal"
# -----------------------------------------

def _utcnow_str() -> str:
    return datetime.now(tz=TZ).strftime("%Y-%m-%d %H:%M:%S")

ALERTS_TABLE = "webhooks.webhook_alerts"

def _claim_exit_alerts(conn, limit: int = 200):
    """
    Claim exit alerts into PROCESSING (SKIP LOCKED).
    Exit is detected from payload JSON:
      - payload->>'signal' IN ('close','exit')
      - OR payload->>'action' IN ('close','exit')
    """
    with conn.cursor() as cur:
        cur.execute(
            f"""
            WITH cte AS (
              SELECT unique_id
                FROM {ALERTS_TABLE}
               WHERE status <> 'PROCESSING'
                 AND (
                      LOWER(COALESCE(payload->>'signal','')) IN ('close','exit')
                   OR LOWER(COALESCE(payload->>'action','')) IN ('close','exit')
                 )
               ORDER BY COALESCE(received_at, now()) ASC
               LIMIT %s
               FOR UPDATE SKIP LOCKED
            )
            UPDATE {ALERTS_TABLE} a
               SET status = 'PROCESSING',
                   last_checked_at = now()
              FROM cte
             WHERE a.unique_id = cte.unique_id
          RETURNING a.unique_id,
                    a.symbol,
                    a.strategy,
                    a.timeframe,
                    COALESCE(a.payload->>'signal', a.payload->>'action') AS signal;
            """,
            (limit,)
        )
        return cur.fetchall()  # [(uid, symbol, strategy, timeframe, signal), ...]



def _fetch_latest_open_trade(cursor, symbol: str, strategy: str):
    """Use future_price as entry price, timestamp as entry time, signal_type as entry side."""
    cursor.execute(
        f"""
        SELECT id, future_price, timestamp, signal_type
        FROM {TRADES_TABLE}
        WHERE symbol = %s
          AND strategy_name = %s
          AND status = 'OPEN'
        ORDER BY id DESC
        LIMIT 1
        """,
        (symbol, strategy),
    )
    return cursor.fetchone()  # (trade_id, entry_price, entry_time, entry_signal)

def _compute_pnl(entry_side: str, entry_price: float, exit_price: float) -> Tuple[float, float]:
    side = (entry_side or "LONG").upper()
    pnl = (exit_price - entry_price) if side == "LONG" else (entry_price - exit_price)
    pnl_pct = (pnl / entry_price * 100.0) if entry_price else 0.0
    return pnl, pnl_pct

def close_trades():
    with get_db_connection() as conn:
        kite = load_kite()
        alerts = _claim_exit_alerts(conn)
        if not alerts:
            print("ℹ️ No exit alerts to process.")
            return

        print(f"🚦 Processing {len(alerts)} exit alert(s)…")

        with conn.cursor() as cursor:
            for alert_id, symbol, strategy, signal_type in alerts:
                try:
                    row = _fetch_latest_open_trade(cursor, symbol, strategy)
                    if not row:
                        cursor.execute(
                            f"""
                            UPDATE {ALERTS_TABLE}
                            SET status = 'REJECTED', last_checked_at = %s, last_message = %s
                            WHERE id = %s
                            """,
                            (_utcnow_str(), "No OPEN trade to close", alert_id),
                        )
                        print(f"⚠️ No OPEN trade for {symbol} | {strategy}")
                        continue

                    trade_id, entry_price, entry_time, entry_signal = row

                    # Current futures price
                    fut_info = fetch_futures_data(symbol, kite) or {}
                    exit_price = float(fut_info.get("last_price") or 0)
                    if not exit_price:
                        raise ValueError("Missing futures LTP")

                    # Duration
                    if isinstance(entry_time, str):
                        try:
                            entry_dt = datetime.strptime(entry_time, "%Y-%m-%d %H:%M:%S")
                        except Exception:
                            entry_dt = datetime.now(tz=TZ).replace(tzinfo=None)
                    else:
                        entry_dt = entry_time  # assume naive UTC from DB
                    exit_dt = datetime.now(tz=TZ).replace(tzinfo=None)
                    duration_minutes = int((exit_dt - entry_dt).total_seconds() / 60)

                    # PnL
                    pnl, pnl_pct = _compute_pnl(str(entry_signal or "LONG"), float(entry_price or 0), exit_price)

                    # Update journal
                    cursor.execute(
                        f"""
                        UPDATE {TRADES_TABLE} SET
                            exit_price = %s,
                            exit_time = %s,
                            pnl = %s,
                            pnl_percent = %s,
                            duration_minutes = %s,
                            status = 'CLOSED',
                            exit_reason = %s,
                            exit_direction = %s
                        WHERE id = %s
                        """,
                        (
                            exit_price,
                            _utcnow_str(),
                            pnl,
                            pnl_pct,
                            duration_minutes,
                            "Signal-based exit",
                            (signal_type or "CLOSE").upper(),
                            trade_id,
                        ),
                    )

                    # Update alert
                    cursor.execute(
                        f"""
                        UPDATE {ALERTS_TABLE}
                        SET status = 'CLOSED', last_checked_at = %s, last_message = %s
                        WHERE id = %s
                        """,
                        (_utcnow_str(), f"Closed trade #{trade_id}", alert_id),
                    )

                    print(f"✅ Closed {symbol} | {strategy} | PnL ₹{pnl:.2f} ({pnl_pct:.2f}%) | {duration_minutes}m")

                except Exception as e:
                    cursor.execute(
                        f"""
                        UPDATE {ALERTS_TABLE}
                        SET status = 'ERROR', last_checked_at = %s, last_message = %s
                        WHERE id = %s
                        """,
                        (_utcnow_str(), str(e)[:255], alert_id),
                    )
                    print(f"❌ Close failed for {symbol} | {strategy}: {e}")

        conn.commit()
        print("📝 Close run complete.")

if __name__ == "__main__":
    close_trades()
