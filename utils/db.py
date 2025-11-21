# File: utils/db.py
# Purpose: Centralized PostgreSQL/TimescaleDB connection helper (UTC, context-manager friendly)

from __future__ import annotations

import os
import configparser
import psycopg2
from psycopg2 import sql
from typing import Optional

# Default locations:
# - Env var DATABASE_URL (e.g., postgres://user:pass@host:5432/dbname)
# - INI file (default: zerodha.ini), section: [postgres]
#     host=localhost
#     port=5432
#     dbname=your_db
#     user=your_user
#     password=your_pass
#     search_path=public,journal,market   (optional)
#     sslmode=require                     (optional)

DEFAULT_INI = os.environ.get("DB_CONFIG_INI", "zerodha.ini")
DEFAULT_SECTION = "postgres"

def _load_ini(path: str = DEFAULT_INI, section: str = DEFAULT_SECTION) -> dict:
    cfg = configparser.ConfigParser()
    read = cfg.read(path)
    if not read or section not in cfg:
        return {}
    s = cfg[section]
    return {
        "host": s.get("host"),
        "port": s.get("port", "5432"),
        "dbname": s.get("dbname") or s.get("database"),
        "user": s.get("user"),
        "password": s.get("password"),
        "sslmode": s.get("sslmode", None),
        "search_path": s.get("search_path", None),
    }

def get_db_connection(*, autocommit: bool = False):
    """
    Return a psycopg2 connection to PostgreSQL/TimescaleDB.
    - Prefers env DATABASE_URL if set.
    - Otherwise uses [postgres] from INI (zerodha.ini by default).
    - Sets session timezone to UTC.
    - Applies search_path if provided in INI.
    - Works with `with get_db_connection() as conn: ...`
    """
    dsn = os.environ.get("DATABASE_URL")
    params = {}
    if not dsn:
        params = _load_ini()
        required = ("host", "port", "dbname", "user", "password")
        if not all(params.get(k) for k in required):
            raise RuntimeError(
                "PostgreSQL config missing. Set DATABASE_URL or provide [postgres] in zerodha.ini"
            )

    try:
        if dsn:
            conn = psycopg2.connect(dsn)
        else:
            conn = psycopg2.connect(
                host=params["host"],
                port=params["port"],
                dbname=params["dbname"],
                user=params["user"],
                password=params["password"],
                sslmode=params.get("sslmode") or None,
            )

        conn.autocommit = autocommit

        # Ensure UTC timestamps across the session
        with conn.cursor() as cur:
            cur.execute("SET TIME ZONE 'UTC';")
            sp = params.get("search_path") if params else None
            if sp:
                # e.g., "public,journal,market"
                cur.execute(sql.SQL("SET search_path TO {};").format(sql.SQL(sp)))

        return conn

    except Exception as e:
        # Raise (don’t return None) so `with get_db_connection()` doesn’t blow up silently
        raise RuntimeError(f"Failed to connect to PostgreSQL: {e}") from e
