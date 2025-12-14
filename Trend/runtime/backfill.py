import argparse
import logging
import pandas as pd
from datetime import datetime

from pillars.common import BaseCfg, TZ
from Trend.Pillar.trend_pillar import process_symbol_vectorized
from Trend.runtime.trend_utils import load_metrics_for_symbol
from utils.db import get_db_connection

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TrendBackfill")

def run_backfill(symbol: str = None, days: int = 30):
    """
    Backfills Trend Pillar for all timeframes.
    """
    db = get_db_connection()
    base_cfg = BaseCfg(run_id=f"backfill_trend_{int(datetime.now().timestamp())}", source="backfill")

    # 1. Get Universe
    if symbol:
        universe = [{'symbol': symbol, 'kind': 'fut'}] # Default to fut for test
        # Need to check actual kind or loop both? usually based on 'reference.symbol_universe'
    else:
        # Load from DB
        q = "SELECT symbol, market_type as kind FROM reference.symbol_universe WHERE is_active=true"
        universe = pd.read_sql(q, db).to_dict('records')

    logger.info(f"Starting backfill for {len(universe)} symbols, lookback {days} days.")

    tfs = ["15m", "30m", "60m", "120m", "240m"]

    for item in universe:
        sym = item['symbol']
        kind = item['kind']

        # 2. Load Data (15m Candles)
        # We load 15m and resample to others inside the processor
        table = "market.futures_candles" if kind == 'fut' else "market.spot_candles"
        q_candles = f"""
            SELECT ts, open, high, low, close, volume
            FROM {table}
            WHERE symbol = '{sym}'
            AND ts >= NOW() - INTERVAL '{days} days'
            ORDER BY ts ASC
        """
        df15 = pd.read_sql(q_candles, db, parse_dates=['ts'], index_col='ts')

        if df15.empty:
            logger.warning(f"No data for {sym}")
            continue

        # Load Metrics (if needed for POC etc)
        # metrics_df = load_metrics_for_symbol(sym, days)
        metrics_df = pd.DataFrame() # Placeholder

        # 3. Process Each TF
        for tf in tfs:
            try:
                rows = process_symbol_vectorized(sym, kind, tf, df15, pd.DataFrame(), metrics_df, base_cfg)
                if rows:
                    # Write to DB (Batched)
                    # For speed, we might want a bulk insert function
                    # 'write_values' does execute_values
                    from pillars.common import write_values
                    write_values(rows)
                    logger.info(f"Processed {sym} {tf}: {len(rows)} rows.")
            except Exception as e:
                logger.error(f"Error processing {sym} {tf}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", type=str, help="Single symbol to run")
    parser.add_argument("--days", type=int, default=30, help="Days to backfill")
    args = parser.parse_args()

    run_backfill(args.symbol, args.days)
