# utils/feature_io.py
from __future__ import annotations
import json
from typing import Dict
from utils.db import get_db_connection

def upsert_features_snapshot(*, ts, symbol: str, tf: str, features: Dict, featureset: str, run_id: str, source: str = "engine") -> int:
    sql = """
      INSERT INTO indicators.feature_snapshots (ts, symbol, tf, featureset, features, run_id, source)
      VALUES (%s, %s, %s, %s, %s::jsonb, %s, %s)
      ON CONFLICT (ts, symbol, tf, featureset)
      DO UPDATE SET
         features = EXCLUDED.features,
         run_id   = EXCLUDED.run_id,
         source   = EXCLUDED.source
    """
    with get_db_connection() as conn, conn.cursor() as cur:
        cur.execute(sql, (ts, symbol, tf, featureset, json.dumps(features), run_id, source))
        conn.commit()
    return 1

def merge_features_snapshot(*, ts, symbol: str, tf: str, features: Dict, featureset: str, run_id: str, source: str = "engine") -> int:
    # JSONB deep-merge (existing || new); if row absent, insert first then merge (one-statement UPSERT)
    sql = """
      INSERT INTO indicators.feature_snapshots (ts, symbol, tf, featureset, features, run_id, source)
      VALUES (%s, %s, %s, %s, %s::jsonb, %s, %s)
      ON CONFLICT (ts, symbol, tf, featureset)
      DO UPDATE SET
         features = indicators.feature_snapshots.features || EXCLUDED.features,
         run_id   = EXCLUDED.run_id,
         source   = EXCLUDED.source
    """
    with get_db_connection() as conn, conn.cursor() as cur:
        cur.execute(sql, (ts, symbol, tf, featureset, json.dumps(features), run_id, source))
        conn.commit()
    return 1
