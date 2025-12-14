import argparse
import pandas as pd
from pillars.common import BaseCfg
from Trend.Pillar.trend_pillar import process_symbol_vectorized
from utils.db import get_db_connection

def run_cli(symbol: str, kind: str, tf: str, days: int = 10):
    print(f"--- Trend Pillar Debug Run: {symbol} {kind} {tf} ---")

    db = get_db_connection()

    # Load Data
    table = "market.futures_candles" if kind == 'fut' else "market.spot_candles"
    q = f"""
        SELECT ts, open, high, low, close, volume
        FROM {table}
        WHERE symbol = '{symbol}'
        AND ts >= NOW() - INTERVAL '{days} days'
        ORDER BY ts ASC
    """
    df = pd.read_sql(q, db, parse_dates=['ts'], index_col='ts')

    if df.empty:
        print("No data found.")
        return

    base = BaseCfg(run_id="cli_debug", source="cli")

    # Process
    rows = process_symbol_vectorized(symbol, kind, tf, df, pd.DataFrame(), pd.DataFrame(), base)

    # Convert results to DataFrame for viewing
    results = []
    for r in rows:
        # (symbol, kind, tf, ts, metric, val, ctx, run_id, source)
        results.append({
            'ts': r[3],
            'metric': r[4],
            'val': r[5],
            'ctx': r[6]
        })

    res_df = pd.DataFrame(results)
    if not res_df.empty:
        # Pivot to see metrics side-by-side
        pivot = res_df.pivot(index='ts', columns='metric', values='val')
        print(pivot.tail(20))

        # Check Debug Context
        last_ctx = res_df[res_df['metric'] == 'TREND.debug_ctx'].iloc[-1]['ctx']
        print("\nLast Debug Context:")
        print(last_ctx)
    else:
        print("No results generated.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("symbol", type=str)
    parser.add_argument("--kind", type=str, default="fut")
    parser.add_argument("--tf", type=str, default="15m")
    parser.add_argument("--days", type=int, default=10)
    args = parser.parse_args()

    run_cli(args.symbol, args.kind, args.tf, args.days)
