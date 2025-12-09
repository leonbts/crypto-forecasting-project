import os
import requests
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import List, Optional

# ---- CONFIG ----
API_URL = "https://min-api.cryptocompare.com/data/v2/histoday"

# You can set this as an env var later: export CRYPTOCOMPARE_API_KEY="your_key"
API_KEY = os.getenv("CRYPTOCOMPARE_API_KEY", None)

COINS = ["BTC", "ETH", "BNB", "XRP", "SOL"]
FIAT = "USD"
LIMIT = 2000  # ~2000 days of history per coin (~5.5 years)

# Paths (relative to project root)
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # adjust if needed
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def fetch_coin_history(
    symbol: str,
    tsym: str = FIAT,
    limit: int = LIMIT,
    api_key: Optional[str] = API_KEY,
) -> pd.DataFrame:
    """
    Fetch daily OHLCV historical data for a single coin from CryptoCompare.

    :param symbol: e.g. "BTC"
    :param tsym: target currency, e.g. "USD"
    :param limit: number of data points (days - 1)
    :param api_key: optional API key
    :return: DataFrame with columns [date, symbol, open, high, low, close, volumefrom, volumeto]
    """
    params = {
        "fsym": symbol,
        "tsym": tsym,
        "limit": limit,
    }
    if api_key:
        params["api_key"] = api_key

    print(f"[INFO] Requesting {symbol} {tsym} daily data...")
    resp = requests.get(API_URL, params=params)
    resp.raise_for_status()

    data = resp.json()

    if data.get("Response") != "Success":
        raise RuntimeError(f"Error from API for {symbol}: {data.get('Message')}")

    # The nested "Data" → "Data" is how CryptoCompare structures it
    records = data["Data"]["Data"]

    # Save raw JSON per coin (optional but nice for debugging / audit)
    raw_path = DATA_RAW_DIR / f"{symbol}_histoday_raw.json"
    with open(raw_path, "w", encoding="utf-8") as f:
        import json
        json.dump(data, f, indent=2)
    print(f"[INFO] Saved raw JSON for {symbol} to {raw_path}")

    df = pd.DataFrame(records)

    # CryptoCompare uses Unix timestamps in 'time'
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df = df.rename(columns={"time": "date"})
    df["symbol"] = symbol

    # Reorder/keep relevant columns
    cols = ["date", "symbol", "open", "high", "low", "close", "volumefrom", "volumeto"]
    df = df[cols].sort_values("date").reset_index(drop=True)

    return df


def fetch_all_coins(coins: List[str] = COINS) -> pd.DataFrame:
    """
    Fetch daily OHLCV for all specified coins and concatenate into a single DataFrame.
    """
    all_dfs = []
    for c in coins:
        try:
            df_c = fetch_coin_history(c)
            all_dfs.append(df_c)
        except Exception as e:
            print(f"[ERROR] Failed to fetch {c}: {e}")

    if not all_dfs:
        raise RuntimeError("No data fetched for any coins.")

    df_all = pd.concat(all_dfs, ignore_index=True)
    df_all = df_all.sort_values(["symbol", "date"]).reset_index(drop=True)
    return df_all


def main():
    df_all = fetch_all_coins()

    # Info/debug
    print("[INFO] Combined DataFrame head:")
    print(df_all.head())
    print("\n[INFO] Data summary:")
    print(df_all.groupby("symbol")["date"].agg(["min", "max", "count"]))

    # Save processed CSV
    out_path = DATA_PROCESSED_DIR / "crypto_ohlcv_daily.csv"
    df_all.to_csv(out_path, index=False)
    print(f"\n[INFO] Saved combined CSV to {out_path}")


if __name__ == "__main__":
    main()