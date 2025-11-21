# routes.py (refactored)
from __future__ import annotations

import json, os, traceback
from typing import Any, Dict
from datetime import datetime, timezone
from flask import Flask, request, jsonify, render_template

from utils.kite_utils import exchange_and_store_token
from utils.db_ops import insert_webhook_alert, get_dashboard_rows  # insert should RETURN unique_id

app = Flask(__name__)

LOGDIR = "logs"
os.makedirs(LOGDIR, exist_ok=True)
LOGFILE = os.path.join(LOGDIR, "webhook.log")

def _log(line: str):
    ts = datetime.now(timezone.utc).isoformat()
    with open(LOGFILE, "a", encoding="utf-8") as f:
        f.write(f"{ts} {line}\n")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/login")
def login():
    from utils.kite_utils import load_config, generate_login_url
    cfg = load_config()
    url = generate_login_url(cfg["kite"]["api_key"])
    return f'<meta http-equiv="refresh" content="0;url={url}">'

@app.route("/auth/callback")
def auth_callback():
    request_token = request.args.get("request_token")
    if not request_token:
        return jsonify({"error":"missing request_token"}), 400
    try:
        access_token = exchange_and_store_token(request_token)
        return render_template("dashboard.html", journal=[], metrics={"access_token": access_token[:8]+"…"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def _normalize_webhook(data: Dict[str, Any]) -> Dict[str, Any]:
    """Map flexible TradingView payloads into canonical fields."""
    return {
        "symbol": (data.get("symbol") or data.get("ticker") or "UNKNOWN").upper(),
        "strategy_name": data.get("strategy_name") or data.get("strategy") or "unknown_strategy",
        "signal_type": (data.get("signal_type") or data.get("side") or "unknown").upper(),
        "strategy_version": data.get("strategy_version") or data.get("version") or "v1.0",
        "rule_version": data.get("rule_version") or "v1.0",
        "source": data.get("source") or "tradingview",
        "timeframe": data.get("timeframe") or data.get("tf"),
        "client_event_id": data.get("event_id") or data.get("id"),   # optional idempotency
        "payload_json": data,  # full dict; db_ops should store as JSONB
    }

@app.route("/webhook", methods=["POST"])
def webhook():
    raw = request.get_data(as_text=True) or ""
    _log(f"RAW: {raw!r}")
    try:
        data = request.get_json(force=True, silent=False) or {}
        _log(f"JSON: {json.dumps(data, separators=(',',':'))}")
    except Exception as e:
        _log(f"JSON_ERR: {repr(e)}\n{traceback.format_exc()}")
        return jsonify({"status": "error", "message": "invalid_json"}), 200  # keep 200 to stop TV retries

    norm = _normalize_webhook(data)

    try:
        unique_id = insert_webhook_alert(
            symbol=norm["symbol"],
            strategy_name=norm["strategy_name"],
            payload_json=norm["payload_json"],
            timeframe=norm["timeframe"],
            source=norm["source"],
            signal_type=norm["signal_type"],
            strategy_version=norm["strategy_version"],
            rule_version=norm["rule_version"],
            # NEW defaults for pipeline handoff:
            status="DATA_PROCESSING",
            sub_status="INGESTION_PENDING",
            client_event_id=norm.get("client_event_id"),  # optional; use for ON CONFLICT
        )
        _log(f"DB_OK: {unique_id} {norm['symbol']} {norm['strategy_name']} {norm.get('timeframe')}")
        return jsonify({"status": "success", "unique_id": unique_id}), 200
    except Exception as e:
        _log(f"DB_ERR: {repr(e)}\n{traceback.format_exc()}")
        return jsonify({"status": "error", "message": "db_error"}), 200

@app.route("/get_token", methods=["GET"])
def get_token():
    request_token = request.args.get("request_token")
    if not request_token:
        return jsonify({"status": "error", "message": "request_token is required"}), 400
    try:
        access_token = exchange_and_store_token(request_token)
        return jsonify({"status": "success", "access_token": access_token[:8] + "…",
                        "message": "✅ Access token saved to zerodha.ini"})
    except Exception as e:
        return jsonify({"status": "error", "message": f"❌ Token exchange failed: {e}"}), 500

@app.route("/dashboard")
def dashboard():
    data = get_dashboard_rows(limit_alerts=100, limit_trades=100)
    recent_trades = data.get("recent_trades", [])
    metrics = {
        "total_trades": sum(1 for r in recent_trades if r.get("entry_ts")),
        "open_positions": sum(1 for r in recent_trades if r.get("entry_ts") and not r.get("exit_ts")),
        "win_rate": None,
        "avg_confidence": None,
    }
    return render_template("dashboard.html", journal=recent_trades,
                           metrics=metrics, counts=data.get("counts", {}), alerts=data.get("recent_alerts", []))

@app.route("/healthz")
def healthz():
    return jsonify({"status": "ok"}), 200

if __name__ == "__main__":
    app.run(debug=True, port=5000)
