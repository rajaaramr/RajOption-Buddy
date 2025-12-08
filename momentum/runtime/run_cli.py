# momentum/runtime/run_cli.py
from __future__ import annotations

import argparse
import datetime as dt

import pandas as pd

from utils.db import get_db_connection
from momentum.pillar.momentum_pillar import score_momentum, BaseCfg  # adjust if path differs

TABLE_FUTURES_5M = "market.futures_ohlcv_5m"
TABLE_SPOT_5M    = "market.spot_ohlcv_5m"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Momentum pillar CLI tester")
    p.add_argument("--symbol", required=True)
    p.add_argument("--kind", choices=["futures", "spot"], required=True)
    p.add_argument("--tf", default="15m")
    p.add_argument("--lookback-days", type=int, default=10)
    p.add_argument("--momentum-ini", default=None)
    return p.parse_args()


def _load_df5(symbol: str, kind: str, lookback_days: int) -> pd.DataFrame:
    table = TABLE_FUTURES_5M if kind == "futures" else TABLE_SPOT_5M
    now = dt.datetime.now(dt.timezone.utc)
    from_ts = now - dt.timedelta(days=lookback_days)

    with get_db_connection() as conn:
        sql = f"""
            SELECT ts, open, high, low, close, volume
              FROM {table}
             WHERE symbol = %s
               AND ts >= %s
             ORDER BY ts
        """
        df = pd.read_sql(sql, conn, params=(symbol, from_ts))
        if not df.empty:
            df.set_index("ts", inplace=True)
            df.index = pd.to_datetime(df.index)
    return df


def main():
    args = _parse_args()
    df5 = _load_df5(args.symbol, args.kind, args.lookback_days)

    if df5.empty:
        print(f"[MOM_CLI] No data for {args.symbol} {args.kind}")
        return

    base = BaseCfg(
        run_id="MOM_CLI_TEST",
        source="cli",
    )

    res = score_momentum(
        symbol=args.symbol,
        kind=args.kind,
        tf=args.tf,
        df5=df5,
        base=base,
        ini_path=args.momentum_ini,
    )

    if res is None:
        print(f"[MOM_CLI] No score produced (min bars / config gate).")
    else:
        ts, final_score, final_veto = res
        print(f"[MOM_CLI] {args.symbol} {args.kind} {args.tf} @ {ts} → "
              f"MOM.score_final={final_score}, MOM.veto_final={final_veto}")


if __name__ == "__main__":
    main()
