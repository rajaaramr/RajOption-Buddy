# momentum/runtime/utils.py
from __future__ import annotations
import configparser
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import psycopg2
from psycopg2.extras import DictCursor

BASE_DIR = Path(__file__).resolve().parents[1]  # .../momentum
ROOT_DIR = BASE_DIR.parent                      # project root

# -------- DB helpers --------

def get_pg_conn():
    cfg_path = ROOT_DIR / "config.ini"
    cp = configparser.ConfigParser()
    cp.read(cfg_path)
    pg = cp["postgres"]
    conn = psycopg2.connect(
        host=pg["host"],
        port=pg.getint("port"),
        user=pg["user"],
        password=pg["password"],
        dbname=pg["dbname"],
    )
    return conn


def load_universe_symbols(kind: str = "futures") -> List[str]:
    """
    Returns list of enabled symbols from reference.symbol_universe.
    `kind` included for future spot/futures branching if needed.
    """
    sql = """
        SELECT symbol
        FROM reference.symbol_universe
        WHERE enabled = TRUE
        ORDER BY symbol
    """
    with get_pg_conn() as conn, conn.cursor() as cur:
        cur.execute(sql)
        rows = cur.fetchall()
    return [r[0] for r in rows]


def get_momentum_resume_ts(symbol: str, kind: str, mode: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Returns (last_rules_ts, last_ml_ts) for this symbol/kind from symbol_universe.
    Plain ISO strings, you can parse to pandas later.

    mode = rules / ml / both → you can ignore the non-relevant one if you want.
    """
    if kind not in ("futures", "spot"):
        raise ValueError(f"Unsupported kind: {kind}")

    if kind == "futures":
        cols = """
            mom_rules_last_fut_ts,
            mom_ml_last_fut_ts
        """
    else:
        cols = """
            mom_rules_last_spot_ts,
            mom_ml_last_spot_ts
        """

    sql = f"""
        SELECT {cols}
        FROM reference.symbol_universe
        WHERE symbol = %s
    """

    with get_pg_conn() as conn, conn.cursor() as cur:
        cur.execute(sql, (symbol,))
        row = cur.fetchone()

    if not row:
        return (None, None)

    last_rules_ts, last_ml_ts = row
    return (
        last_rules_ts.isoformat() if last_rules_ts else None,
        last_ml_ts.isoformat() if last_ml_ts else None,
    )


def update_momentum_status(
    symbol: str,
    kind: str,
    *,
    mode: str,
    last_rules_ts=None,
    last_ml_ts=None,
    run_id: str,
    ml_train_at=None,
    ml_calib_at=None,
):
    """
    One-stop update of momentum-related columns on reference.symbol_universe.

    - For mode='rules': update mom_rules_last_* + mom_rules_last_run_at + mom_rules_last_run_id
    - For mode='ml':    update mom_ml_last_*    + mom_ml_last_run_at    + mom_ml_last_run_id
    - ml_train_at/ml_calib_at: for trainer & bucket_stats jobs
    """
    if kind not in ("futures", "spot"):
        raise ValueError(f"Unsupported kind: {kind}")

    fut = (kind == "futures")

    sets = []
    args: List = []

    from datetime import datetime, timezone
    now = datetime.now(timezone.utc)

    if mode in ("rules", "both") and last_rules_ts is not None:
        if fut:
            sets.append("mom_rules_last_fut_ts = %s")
        else:
            sets.append("mom_rules_last_spot_ts = %s")
        args.append(last_rules_ts)
        sets.append("mom_rules_last_run_at = %s")
        args.append(now)
        sets.append("mom_rules_last_run_id = %s")
        args.append(run_id)

    if mode in ("ml", "both") and last_ml_ts is not None:
        if fut:
            sets.append("mom_ml_last_fut_ts = %s")
        else:
            sets.append("mom_ml_last_spot_ts = %s")
        args.append(last_ml_ts)
        sets.append("mom_ml_last_run_at = %s")
        args.append(now)
        sets.append("mom_ml_last_run_id = %s")
        args.append(run_id)

    if ml_train_at is not None:
        sets.append("mom_ml_train_last_at = %s")
        args.append(ml_train_at)

    if ml_calib_at is not None:
        sets.append("mom_ml_calib_last_at = %s")
        args.append(ml_calib_at)

    if not sets:
        return

    sql = f"""
        UPDATE reference.symbol_universe
        SET {", ".join(sets)}
        WHERE symbol = %s
    """
    args.append(symbol)

    with get_pg_conn() as conn, conn.cursor() as cur:
        cur.execute(sql, args)
        conn.commit()


@dataclass
class SimpleBaseCfg:
    """
    Minimal 'BaseCfg' duck-type for score_momentum.
    """
    run_id: str
    source: str = "momentum_runtime"
