import argparse
import os
import pandas as pd

from utils.db import get_db_connection
from flow.pillar.flow_pillar import score_flow
from pillars.common import BaseCfg

# Default to your separate INI file


def load_df5_from_db(symbol: str, kind: str = "futures") -> pd.DataFrame:
    """
    Load 5m OHLCV(+OI) from Postgres for a single symbol.

    kind:
      - "futures" → market.futures_candles (with real OI)
      - "spot"    → market.spot_candles   (fake OI = 0, just to keep columns consistent)
    """
    if kind == "futures":
        table = "market.futures_candles"
        sql = f"""
            SELECT ts, open, high, low, close, volume, oi
            FROM {table}
            WHERE symbol = %s
            ORDER BY ts
        """
    else:
        table = "market.spot_candles"
        sql = f"""
            SELECT ts, open, high, low, close, volume,
                   0::bigint AS oi
            FROM {table}
            WHERE symbol = %s
            ORDER BY ts
        """

    with get_db_connection() as conn:
        df = pd.read_sql(sql, conn, params=(symbol,))

    if df.empty:
        raise RuntimeError(f"No data found in {table} for symbol={symbol!r}")

    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df = df.set_index("ts").sort_index()
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--kind", default="futures", choices=["futures", "spot"])
    # Use 15m / 30m / 60m etc.
    parser.add_argument("--tf", default="30m")
    args = parser.parse_args()

    df5 = load_df5_from_db(args.symbol, args.kind)

    base = BaseCfg(
        run_id="CLI_FLOW_TEST",
        source="cli",
        tfs=[args.tf],       # minimal valid list
        lookback_days=5,     # dummy but required
    )


    res = score_flow(args.symbol, args.kind, args.tf, df5, base)
    if res is None:
        print(f"[FLOW] {args.symbol} {args.kind} {args.tf}: not enough bars → skipped")
        return

    # Flexible unpack – works even if we later add more return fields
    ts         = res[0]
    rules_score = res[1] if len(res) > 1 else None
    rules_veto  = bool(res[2]) if len(res) > 2 else False
    ml_score    = res[3] if len(res) > 3 else None
    fused_score = res[4] if len(res) > 4 else None
    fused_veto  = bool(res[5]) if len(res) > 5 else False

    print(
        f"[FLOW] {args.symbol} {args.kind} {args.tf} @ {ts} → "
        f"rules={rules_score}, rules_veto={rules_veto}, "
        f"ml={ml_score}, fused={fused_score}, fused_veto={fused_veto}"
    )



if __name__ == "__main__":
    main()
