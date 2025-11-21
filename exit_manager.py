# exit_manager.py
from __future__ import annotations
import os, psycopg2, psycopg2.extras
from datetime import timezone

NEUTRAL_PCR_LO = float(os.getenv("NEUTRAL_PCR_LO", "0.95"))
NEUTRAL_PCR_HI = float(os.getenv("NEUTRAL_PCR_HI", "1.05"))

def db():
    return psycopg2.connect(os.getenv("DB_URL"))

def fetch_open_trades(cur):
    cur.execute("""
      SELECT t.trade_id, t.symbol, t.side, t.instrument, t.entry_px, t.stop_px, t.target_px, t.size_units,
             d.ts AS decision_ts
      FROM analytics.paper_trades t
      JOIN analytics.decisions_live d
        ON d.symbol=t.symbol AND d.status IN ('OPEN','OPEN_SETUP')   -- allow early closes if needed
      WHERE t.status='OPEN'
    """)
    return cur.fetchall()

def latest_ctx(cur, symbol):
    cur.execute("""
      WITH lb AS (
        SELECT DISTINCT ON (symbol) symbol, ts, close
        FROM market.futures_candles
        WHERE interval='5m' AND symbol=%s
        ORDER BY symbol, ts DESC
      )
      SELECT lb.ts, lb.close, ind.ema5, lp.pcr
      FROM lb
      JOIN indicators.futures_5m ind ON ind.symbol=lb.symbol AND ind.ts=lb.ts
      JOIN analytics.latest_pcr lp ON lp.symbol=lb.symbol
    """, (symbol,))
    row = cur.fetchone()
    if not row: return None
    return {"ts": row[0], "fut_close": float(row[1]), "ema5": float(row[2]) if row[2] is not None else None, "pcr": float(row[3]) if row[3] is not None else None}

def latest_option_ltp(cur, symbol, trade_side):
    cur.execute("""
      SELECT ltp FROM market.options_chain_snapshots
      WHERE symbol=%s
      ORDER BY snapshot_ts DESC
      LIMIT 1
    """, (symbol,))
    r = cur.fetchone()
    return float(r[0]) if r and r[0] is not None else None

def close_trade(cur, trade_id, exit_px, reason):
    cur.execute("""
      UPDATE analytics.paper_trades
      SET closed_at=NOW(), exit_px=%s,
          pnl_amount = (CASE WHEN side='LONG' THEN (exit_px - entry_px) ELSE (entry_px - exit_px) END) * size_units,
          pnl_r = ((CASE WHEN side='LONG' THEN (exit_px - entry_px) ELSE (entry_px - exit_px) END) * size_units) / risk_per_trade,
          status='CLOSED'
      WHERE trade_id=%s
    """, (exit_px, trade_id))
    cur.execute("""
      INSERT INTO analytics.trade_journal (trade_id, ts, event, note, px)
      VALUES (%s, NOW(), 'EXIT', %s, %s)
    """, (trade_id, reason, exit_px))

def run():
    with db() as conn, conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
      trades = fetch_open_trades(cur)
      for t in trades:
          ctx = latest_ctx(cur, t["symbol"])
          if not ctx: continue
          fut_px = ctx["fut_close"]
          ema5   = ctx["ema5"]
          pcr    = ctx["pcr"]
          side   = t["side"]

          # Futures exits
          if t["instrument"] == 'FUTURES':
              stop_hit = (side=='LONG' and fut_px <= t["stop_px"]) or (side=='SHORT' and fut_px >= t["stop_px"])
              tgt_hit  = (side=='LONG' and fut_px >= t["target_px"]) or (side=='SHORT' and fut_px <= t["target_px"])
              ema_flip = (side=='LONG' and fut_px <  ema5) or (side=='SHORT' and fut_px > ema5)
              neutral  = (pcr is not None and NEUTRAL_PCR_LO <= pcr <= NEUTRAL_PCR_HI)

              if stop_hit:
                  close_trade(cur, t["trade_id"], fut_px, "STOP_HIT")
              elif tgt_hit:
                  close_trade(cur, t["trade_id"], fut_px, "TARGET_1R")
              elif ema5 is not None and ema_flip:
                  close_trade(cur, t["trade_id"], fut_px, "EMA5_FLIP")
              elif neutral:
                  close_trade(cur, t["trade_id"], fut_px, "PCR_NEUTRAL")

          # Options exits (use premium stops/targets; EMA5/PCR as context)
          elif t["instrument"] in ('OPTION_CE','OPTION_PE'):
              opt_ltp = latest_option_ltp(cur, t["symbol"], side)
              if opt_ltp is None or opt_ltp <= 0: continue
              stop_hit = (side=='LONG' and opt_ltp <= t["stop_px"]) or (side=='SHORT' and opt_ltp >= t["stop_px"])
              tgt_hit  = (side=='LONG' and opt_ltp >= t["target_px"]) or (side=='SHORT' and opt_ltp <= t["target_px"])
              ema_flip = (side=='LONG' and fut_px < ema5) or (side=='SHORT' and fut_px > ema5)
              neutral  = (pcr is not None and NEUTRAL_PCR_LO <= pcr <= NEUTRAL_PCR_HI)

              if stop_hit:
                  close_trade(cur, t["trade_id"], opt_ltp, "OPT_STOP_HIT")
              elif tgt_hit:
                  close_trade(cur, t["trade_id"], opt_ltp, "OPT_TARGET_HIT")
              elif ema5 is not None and ema_flip:
                  close_trade(cur, t["trade_id"], opt_ltp, "EMA5_FLIP")
              elif neutral:
                  close_trade(cur, t["trade_id"], opt_ltp, "PCR_NEUTRAL")

      conn.commit()

if __name__ == "__main__":
    run()
