# File: utils/fetch_and_process_alerts.py
# Purpose: Claim pending webhooks (Postgres/Timescale), build context (futures/zones),
#          evaluate entries/exits, write journal rows (futures-first) and log attempts.
# Notes:
#   - ENTRY: stores futures entry price, rule & score, and the chosen nearest option (CE/PE)
#   - EXIT : computes PnL on futures only
#   - Webhooks: expects signal in payload JSON (buy/long/sell/short/close)

from __future__ import annotations

import configparser
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional, Tuple

from kiteconnect import KiteConnect

from utils.db import get_db_connection
from utils.kite_utils import fetch_futures_data
from utils.rule_evaluator import evaluate_alert
from utils.exit_rule_evaluator import evaluate_exit
from utils.option_strikes import pick_nearest_option
from utils.db_ops import fetch_latest_zone_data  # your existing impl; keep it for zone context

# NEW – config-driven, multi-TF scores
from utils.indicators import compute_weighted_scores, load_indicator_config

# NEW – futures & option-chain buildup (kept separate)
from utils.buildups import compute_futures_buildup, compute_optionchain_buildup

# OPTIONAL – single call that merges indicators + buildups + recompute final score
from utils.compose_signals import compose_blob

# OPTIONAL – if you persist JSONB indicator snapshots for the dashboard
from utils.db_ops import insert_indicator_snapshot, json_dumps


# ---------------- Config ----------------
CONFIG_PATH = "zerodha.ini"
RULE_ENGINE_VERSION = "v1.2.4"
BATCH_SIZE = 100
TZ = timezone.utc


# ------------- Types --------------------
@dataclass
class Alert:
    unique_id: str
    symbol: str
    strategy: str
    payload: Optional[dict]
    received_at: datetime


# ------------- Utils --------------------
def _utcnow() -> datetime:
    return datetime.now(tz=TZ)


def _side_from_signal(signal_type: Optional[str]) -> Optional[str]:
    if not signal_type:
        return None
    s = signal_type.strip().lower()
    if s in ("buy", "long"):
        return "LONG"
    if s in ("sell", "short"):
        return "SHORT"
    if s in ("close", "exit"):
        return "CLOSE"
    return None


def load_kite_session() -> KiteConnect:
    cfg = configparser.ConfigParser()
    cfg.read(CONFIG_PATH)
    api_key = cfg["kite"]["api_key"]
    access_token = cfg["kite"]["access_token"]
    kite = KiteConnect(api_key=api_key)
    kite.set_access_token(access_token)
    return kite


# --------- DB helpers (Timescale) --------
def _claim_pending_alerts(cur) -> List[Alert]:
    """
    Atomically claim up to BATCH_SIZE pending alerts by flipping to PROCESSING
    and returning the claimed rows (avoids races).
    """
    sql = """
    UPDATE webhooks.webhook_alerts AS w
       SET status = 'PROCESSING',
           last_checked_at = now()
     WHERE w.unique_id IN (
           SELECT unique_id
             FROM webhooks.webhook_alerts
            WHERE status = 'PENDING'
            ORDER BY received_at ASC
            LIMIT %s
            FOR UPDATE SKIP LOCKED
     )
    RETURNING w.unique_id, w.symbol, w.strategy, w.payload, w.received_at;
    """
    cur.execute(sql, (BATCH_SIZE,))
    rows = cur.fetchall()
    alerts: List[Alert] = []
    for uid, sym, strat, payload, rcvd in rows:
        payload_dict = payload if isinstance(payload, dict) else None
        alerts.append(Alert(str(uid), sym, strat or "UNKNOWN", payload_dict, rcvd))
    return alerts


def _next_attempt_number(cur, unique_id: str) -> int:
    cur.execute(
        "SELECT COALESCE(MAX(attempt_number),0) FROM journal.rejections_log WHERE unique_id=%s;",
        (unique_id,)
    )
    return int(cur.fetchone()[0] or 0) + 1


def _log_attempt(cur, unique_id: str, attempt_number: int, status: str,
                 rejection_reason: Optional[str] = None, trigger: Optional[str] = None, notes: Optional[str] = None):
    cur.execute(
        """
        INSERT INTO journal.rejections_log
            (unique_id, attempt_number, processed_at, status, rejection_reason, re_run_trigger, notes)
        VALUES (%s,%s, now(), %s, %s, %s, %s);
        """,
        (unique_id, attempt_number, status, rejection_reason, trigger, notes)
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


# -------- Journal helpers (futures-led) -------
def _insert_entry(cur,
                  unique_id: str,
                  symbol: str,
                  side: Optional[str],
                  entry_price_fut: Optional[float],
                  rule_matched: Optional[str],
                  decision_score: Optional[float],
                  option_type: Optional[str],
                  option_strike: Optional[float],
                  option_symbol: Optional[str]):
    """
    Minimal OPEN trade row on accept, storing futures price and chosen option meta.
    """
    cur.execute(
        """
        INSERT INTO journal.trading_journal
            (unique_id, symbol, side, entry_ts,
             entry_price_fut, rule_matched, decision_score,
             option_type, option_strike, option_symbol,
             status)
        VALUES (%s,%s,%s, now(),
                %s,%s,%s,
                %s,%s,%s,
                'OPEN');
        """,
        (unique_id, symbol, side,
         entry_price_fut, rule_matched, float(decision_score or 0),
         option_type, option_strike, option_symbol)
    )


def _fetch_latest_open_trade(cur, symbol: str):
    """
    Returns (trade_id, side, entry_price_fut) for latest OPEN trade of symbol (or None).
    """
    cur.execute(
        """
        SELECT trade_id, side, entry_price_fut
          FROM journal.trading_journal
         WHERE symbol=%s AND status='OPEN'
         ORDER BY entry_ts DESC
         LIMIT 1;
        """,
        (symbol,)
    )
    return cur.fetchone()


def _close_trade(cur, trade_id: int, exit_price_fut: float, reason: str,
                 score: float, exit_direction: str, entry_side: Optional[str], entry_price_fut: Optional[float]):
    side = (entry_side or "LONG").upper()
    ep = float(entry_price_fut or 0.0)
    pnl_raw = (exit_price_fut - ep) if side == "LONG" else (ep - exit_price_fut)
    pnl_pct = (pnl_raw / ep * 100.0) if ep else 0.0

    cur.execute(
        """
        UPDATE journal.trading_journal
           SET exit_ts       = now(),
               exit_price_fut= %s,
               exit_reason   = %s,
               decision_score= %s,
               status        = 'CLOSED',
               pnl           = %s,
               pnl_pct       = %s,
               exit_direction= %s
         WHERE trade_id = %s;
        """,
        (exit_price_fut, reason, float(score or 0), pnl_raw, pnl_pct, exit_direction, trade_id)
    )


# ------------- Context build ----------------
def _preprocess_alert(symbol: str, kite: KiteConnect):
    """
    Build minimal context for evaluation:
      - futures LTP + volume
      - latest zone info (VAL/VAH, flags) via your existing db_ops.fetch_latest_zone_data
    """
    fut = fetch_futures_data(symbol, kite)
    if not fut:
        return None, "Futures data unavailable"

    zone = fetch_latest_zone_data(symbol) or {}

    return {
        "future_price": float(fut.get("last_price") or 0),
        "volume": int(fut.get("volume") or 0),
        "support_zone": zone.get("val"),
        "resistance_zone": zone.get("vah"),
        "zone_break_type": zone.get("zone_break_type"),
        "zone_conf_score": zone.get("zone_confidence_score"),
    }, None


# ------------- Main runner ------------------
def process_webhook_alerts() -> None:
    kite = load_kite_session()

    with get_db_connection() as conn, conn.cursor() as cur:
        alerts = _claim_pending_alerts(cur)
        if not alerts:
            print("🔍 No pending alerts.")
            return

        print(f"🚦 Processing {len(alerts)} alert(s)...")
        symbol_ctx_cache: Dict[str, Tuple[Optional[Dict], Optional[str]]] = {}

        for a in alerts:
            try:
                payload = a.payload or {}
                signal_type = payload.get("signal_type") or payload.get("signal")
                side = _side_from_signal(signal_type)

                # Build/cache context for the symbol
                if a.symbol in symbol_ctx_cache:
                    ctx, ctx_err = symbol_ctx_cache[a.symbol]
                else:
                    ctx, ctx_err = _preprocess_alert(a.symbol, kite)
                    symbol_ctx_cache[a.symbol] = (ctx, ctx_err)

                attempt = _next_attempt_number(cur, a.unique_id)

                if ctx_err or not ctx:
                    reason = ctx_err or "context missing"
                    _log_attempt(cur, a.unique_id, attempt, status="rejected", rejection_reason=reason)
                    _finalize_alert(cur, a.unique_id, "REJECTED", rejection_reason=reason)
                    print(f"⚠️ Skipping {a.symbol} — {reason}")
                    continue

                # ---------------- EXIT flow ----------------
                if side == "CLOSE":
                    should_close, reason, score = evaluate_exit(a.symbol)
                    if should_close:
                        row = _fetch_latest_open_trade(cur, a.symbol)
                        if row:
                            trade_id, entry_side, entry_price_fut = row
                            _close_trade(
                                cur,
                                trade_id=trade_id,
                                exit_price_fut=float(ctx["future_price"]),
                                reason=reason,
                                score=float(score or 0),
                                exit_direction="CLOSE",
                                entry_side=entry_side,
                                entry_price_fut=entry_price_fut,
                            )
                            _log_attempt(cur, a.unique_id, attempt, status="accepted", notes=f"exit: {reason}")
                            _finalize_alert(cur, a.unique_id, "ACCEPTED")
                            print(f"🔚 EXITED: {a.symbol} | {reason}")
                        else:
                            msg = "No OPEN trade found"
                            _log_attempt(cur, a.unique_id, attempt, status="rejected", rejection_reason=msg)
                            _finalize_alert(cur, a.unique_id, "REJECTED", rejection_reason=msg)
                            print(f"⚠️ EXIT REJECTED: {a.symbol} | {msg}")
                    else:
                        _log_attempt(cur, a.unique_id, attempt, status="rejected", rejection_reason=reason)
                        _finalize_alert(cur, a.unique_id, "REJECTED", rejection_reason=reason)
                        print(f"⚠️ EXIT REJECTED: {a.symbol} | {reason}")
                    continue

                # ---------------- ENTRY flow ----------------
                is_valid, rule_matched, failed_rules, score, decision_tags = evaluate_alert(a.symbol)

                if is_valid:
                    entry_side = side or "LONG"
                    fut_price = float(ctx["future_price"])

                    # CE for LONG, PE for SHORT — pick nearest strike at nearest expiry
                    opt_type = "CE" if entry_side == "LONG" else "PE"
                    expiry, strike, tsym = pick_nearest_option(
                        a.symbol, kite, target_price=fut_price, option_type=opt_type
                    )

                    _insert_entry(
                        cur,
                        unique_id=a.unique_id,
                        symbol=a.symbol,
                        side=entry_side,
                        entry_price_fut=fut_price,
                        rule_matched=rule_matched,
                        decision_score=float(score or 0),
                        option_type=opt_type if tsym else None,
                        option_strike=float(strike) if strike else None,
                        option_symbol=tsym
                    )

                    _log_attempt(cur, a.unique_id, attempt, status="accepted",
                                 notes=f"rule={rule_matched}, score={score}, opt={opt_type}@{strike} {tsym or ''}".strip())
                    _finalize_alert(cur, a.unique_id, "ACCEPTED")
                    print(f"✅ ENTRY: {a.symbol} | {a.strategy} | {entry_side} | {rule_matched} | {opt_type} {strike} {tsym}")
                else:
                    reason_text = "; ".join(f"{r}: {rsn}" for (r, rsn) in (failed_rules or [])) or "Rules not met"
                    _log_attempt(cur, a.unique_id, attempt, status="rejected", rejection_reason=reason_text)
                    _finalize_alert(cur, a.unique_id, "REJECTED", rejection_reason=reason_text)
                    print(f"⛔ ENTRY REJECTED: {a.symbol} | score={score} | {reason_text}")
            except Exception as row_err:
                print(f"⚠️ [{a.symbol}] Insert fail for {a.symbol}: {row_err}")

        print("📝 Processing complete.")


if __name__ == "__main__":
    process_webhook_alerts()
