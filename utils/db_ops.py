# utils/db_ops.py
# Centralized DB helpers for Postgres/Timescale.
# - Safe UPSERTs with ON CONFLICT / row-exists guard
# - UTC-aware timestamps
# - Explicit commits on all writers
# - Helpers for options + dashboards

from __future__ import annotations

from typing import Any, Dict, Optional, Iterable, Tuple, Literal
from datetime import datetime, timezone
from utils.db import get_db_connection
import psycopg2.extras as pgx  # used by snapshot_universe

# ---------------------------------
# Constants
# ---------------------------------
TZ = timezone.utc
DEFAULT_INTERVAL = "5m"

# ---------------------------------
# Helpers
# ---------------------------------
def _as_aware_utc(ts_in) -> datetime:
    try:
        if isinstance(ts_in, datetime):
            dt = ts_in
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=TZ)
            return dt.astimezone(TZ)
        if isinstance(ts_in, (int, float)):
            return datetime.fromtimestamp(float(ts_in), tz=TZ)
        if isinstance(ts_in, str):
            s = ts_in.strip().replace("Z", "+00:00")
            try:
                dt = datetime.fromisoformat(s)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=TZ)
                return dt.astimezone(TZ)
            except Exception:
                return datetime.fromtimestamp(float(s), tz=TZ)
        return datetime.now(TZ)
    except Exception:
        return datetime.now(TZ)

def log_run_status(*, run_id: str, job: str, symbol: str | None,
                   phase: str, status: str, error_code: str | None = None,
                   info: dict | None = None) -> None:
    from utils.db import get_db_connection
    with get_db_connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO journal.run_status
              (run_id, job, symbol, phase, status, error_code, info)
            VALUES (%s,%s,%s,%s,%s,%s,%s::jsonb)
            """,
            (run_id, job, symbol, phase, status, error_code, json_dumps(info or {}))
        )
        conn.commit()

def _row_exists(cur, table: str, symbol: str, interval: str, ts: datetime) -> bool:
    cur.execute(
        f"SELECT 1 FROM {table} WHERE symbol=%s AND interval=%s AND ts=%s LIMIT 1",
        (symbol, interval, ts),
    )
    return cur.fetchone() is not None

def json_dumps(obj) -> str:
    import json as _json
    try:
        return _json.dumps(obj, separators=(",", ":"), ensure_ascii=False)
    except Exception:
        return "{}"

# ---------------------------------
# INSERTS / UPSERTS
# ---------------------------------
def insert_futures_price(
    *,
    ts: datetime | str,
    symbol: str,
    open: float,
    high: float,
    low: float,
    close: float,
    volume: float,
    oi: int = 0,
    interval: str = DEFAULT_INTERVAL,
    source: Optional[str] = None,
) -> None:
    ts = _as_aware_utc(ts)
    volume = float(volume or 0.0)
    oi = int(oi or 0)

    with get_db_connection() as conn, conn.cursor() as cur:
        if _row_exists(cur, "market.futures_candles", symbol, interval, ts):
            cur.execute(
                """
                UPDATE market.futures_candles
                   SET open=%s, high=%s, low=%s, close=%s,
                       volume=GREATEST(%s, volume),
                       oi=GREATEST(%s, oi),
                       source=COALESCE(%s, source)
                 WHERE symbol=%s AND interval=%s AND ts=%s
                """,
                (open, high, low, close, volume, oi, source, symbol, interval, ts),
            )
        else:
            cur.execute(
                """
                INSERT INTO market.futures_candles
                  (symbol, interval, ts, open, high, low, close, volume, oi, source)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                """,
                (symbol, interval, ts, open, high, low, close, volume, oi, source),
            )
        conn.commit()

def insert_spot_price(
    *,
    ts: datetime | str,
    symbol: str,
    open: float,
    high: float,
    low: float,
    close: float,
    volume: float,
    interval: str = DEFAULT_INTERVAL,
    source: Optional[str] = None,
) -> None:
    ts = _as_aware_utc(ts)
    volume = float(volume or 0.0)

    with get_db_connection() as conn, conn.cursor() as cur:
        if _row_exists(cur, "market.spot_candles", symbol, interval, ts):
            cur.execute(
                """
                UPDATE market.spot_candles
                   SET open=%s, high=%s, low=%s, close=%s,
                       volume=GREATEST(%s, volume),
                       source=COALESCE(%s, source)
                 WHERE symbol=%s AND interval=%s AND ts=%s
                """,
                (open, high, low, close, volume, source, symbol, interval, ts),
            )
        else:
            cur.execute(
                """
                INSERT INTO market.spot_candles
                  (symbol, interval, ts, open, high, low, close, volume, source)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
                """,
                (symbol, interval, ts, open, high, low, close, volume, source),
            )
        conn.commit()

# (… keep the rest of your file exactly as before: option_chain, snapshots,
#  upsert_volume_zone, fetch_latest_snapshots, fetch_latest_zone_data, dashboard, adapters …)

# ---------------------------------
# Adapters (compat with older code)
# ---------------------------------
def insert_webhook_alert(
    *, symbol: str, strategy_name: str, payload_json: dict | str,
    timeframe: Optional[str] = None, source: str = "TradingView",
    status: str = "PENDING", signal_type: Optional[str] = None,
    strategy_version: Optional[str] = None, rule_version: Optional[str] = None,
) -> str:
    # keep this helper here so the function is importable
    def _json_dumps(obj) -> str:
        import json as _json
        try:
            return _json.dumps(obj, separators=(",", ":"), ensure_ascii=False)
        except Exception:
            return "{}"

    # embed extras into payload for audit
    if isinstance(payload_json, dict):
        pj = dict(payload_json)
        if signal_type is not None:        pj.setdefault("signal_type", signal_type)
        if strategy_version is not None:   pj.setdefault("strategy_version", strategy_version)
        if rule_version is not None:       pj.setdefault("rule_version", rule_version)
        payload = _json_dumps(pj)
    else:
        payload = str(payload_json)

    with get_db_connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO webhooks.webhook_alerts
                (received_at, source, strategy, symbol, timeframe, payload, status, last_checked_at)
            VALUES (NOW(), %s, %s, %s, %s, %s::jsonb, %s, NOW())
            RETURNING unique_id;
            """,
            (source, strategy_name, symbol, timeframe, payload, status),
        )
        uid = cur.fetchone()[0]
        conn.commit()
        return str(uid)

def get_dashboard_rows(limit_alerts: int = 20, limit_trades: int = 20) -> Dict[str, Any]:
    """
    Returns:
      - counts: open trades, pending alerts, rejected today
      - recent_alerts: latest alerts (symbol, status, ts)
      - recent_trades: latest trades (symbol, side, status, entry_ts, exit_ts, score)
    Uses: journal.trading_journal, webhooks.webhook_alerts
    """
    out: Dict[str, Any] = {
        "counts": {"open_trades": 0, "pending_alerts": 0, "rejected_today": 0},
        "recent_alerts": [],
        "recent_trades": []
    }

    with get_db_connection() as conn, conn.cursor() as cur:
        try:
            cur.execute("SELECT COUNT(*) FROM journal.trading_journal WHERE status='OPEN';")
            out["counts"]["open_trades"] = int(cur.fetchone()[0])
        except Exception:
            pass

        try:
            cur.execute("SELECT COUNT(*) FROM webhooks.webhook_alerts WHERE status='PENDING';")
            out["counts"]["pending_alerts"] = int(cur.fetchone()[0])
        except Exception:
            pass

        try:
            cur.execute("""
                SELECT COUNT(*)
                  FROM webhooks.webhook_alerts
                 WHERE status='REJECTED'
                   AND DATE(received_at AT TIME ZONE 'UTC') = DATE(now() AT TIME ZONE 'UTC');
            """)
            out["counts"]["rejected_today"] = int(cur.fetchone()[0])
        except Exception:
            pass

        try:
            cur.execute("""
                SELECT symbol, status, COALESCE(received_at, last_checked_at) AS ts
                  FROM webhooks.webhook_alerts
                 ORDER BY ts DESC
                 LIMIT %s;
            """, (limit_alerts,))
            out["recent_alerts"] = [
                {"symbol": r[0], "status": r[1], "ts": r[2].isoformat() if r[2] else None}
                for r in cur.fetchall()
            ]
        except Exception:
            pass

        try:
            cur.execute("""
                SELECT symbol, side, status, entry_ts, exit_ts, decision_score
                  FROM journal.trading_journal
                 ORDER BY COALESCE(exit_ts, entry_ts) DESC
                 LIMIT %s;
            """, (limit_trades,))
            out["recent_trades"] = [
                {
                    "symbol": r[0], "side": r[1], "status": r[2],
                    "entry_ts": r[3].isoformat() if r[3] else None,
                    "exit_ts":  r[4].isoformat() if r[4] else None,
                    "score": float(r[5]) if r[5] is not None else None
                }
                for r in cur.fetchall()
            ]
        except Exception:
            pass

    return out


def insert_futures_bar(*, symbol, interval, ts, open_price, high_price, low_price, close_price, volume, oi):
    insert_futures_price(
        ts=ts, symbol=symbol,
        open=open_price, high=high_price, low=low_price, close=close_price,
        volume=volume, oi=oi, interval=interval
    )

def insert_spot_bar(*, symbol, interval, ts, open_price, high_price, low_price, close_price, volume):
    insert_spot_price(
        ts=ts, symbol=symbol,
        open=open_price, high=high_price, low=low_price, close=close_price,
        volume=volume, interval=interval
    )
