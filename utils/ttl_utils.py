# File: utils/ttl_utils.py
from __future__ import annotations
from datetime import datetime, timedelta
from typing import Union

def is_data_stale(last_timestamp: Union[str, datetime], ttl_minutes: int = 5) -> bool:
    """
    Returns True if last_timestamp is older than ttl_minutes.
    Accepts:
      - datetime object
      - timestamp string in "%Y-%m-%d %H:%M:%S" format
    """
    try:
        if isinstance(last_timestamp, str):
            last_time = datetime.strptime(last_timestamp, "%Y-%m-%d %H:%M:%S")
        elif isinstance(last_timestamp, datetime):
            last_time = last_timestamp
        else:
            raise ValueError("Unsupported timestamp type")

        return datetime.now() - last_time > timedelta(minutes=ttl_minutes)
    except Exception as e:
        print(f"⚠️ TTL check error: {e}")
        return True
# File: utils/ttl_utils.py

from datetime import datetime, timedelta, timezone

def is_fresh(ts: datetime, ttl_seconds: int) -> bool:
    """
    Check if a given timestamp is still fresh within a TTL window.

    Args:
        ts (datetime): The timestamp to check. Should be timezone-aware.
        ttl_seconds (int): Time-to-live in seconds.

    Returns:
        bool: True if ts is within ttl_seconds from now (UTC), else False.
    """
    if ts is None:
        return False

    if ts.tzinfo is None:
        # Assume UTC if tz is missing
        ts = ts.replace(tzinfo=timezone.utc)

    now = datetime.now(timezone.utc)
    return (now - ts) <= timedelta(seconds=ttl_seconds)
