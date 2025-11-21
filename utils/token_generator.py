# File: utils/token_generator.py
# Purpose: Exchange Kite request_token -> access_token and persist to zerodha.ini

from __future__ import annotations

import sys
import argparse
import configparser
from kiteconnect import KiteConnect

INI_PATH = "zerodha.ini"

def main():
    parser = argparse.ArgumentParser(description="Exchange Kite request_token for access_token")
    parser.add_argument("--request-token", required=True, help="One-time request_token from login redirect")
    parser.add_argument("--ini", default=INI_PATH, help="Path to zerodha.ini (default: zerodha.ini)")
    args = parser.parse_args()

    cfg = configparser.ConfigParser()
    if not cfg.read(args.ini):
        sys.exit(f"❌ Could not read INI: {args.ini}")

    try:
        api_key = cfg["kite"]["api_key"]
        api_secret = cfg["kite"]["api_secret"]
    except KeyError:
        sys.exit("❌ Missing [kite] api_key/api_secret in INI")

    try:
        kite = KiteConnect(api_key=api_key)
        session = kite.generate_session(args.request_token, api_secret=api_secret)
        access_token = session["access_token"]
    except Exception as e:
        sys.exit(f"❌ Token exchange failed: {e}")

    cfg["kite"]["access_token"] = access_token
    with open(args.ini, "w") as f:
        cfg.write(f)

    print("✅ Access token updated:", access_token[:10], "...")

if __name__ == "__main__":
    main()
