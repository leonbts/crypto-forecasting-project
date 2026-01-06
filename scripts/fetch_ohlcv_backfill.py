import os
import time
from datetime import datetime, timezone
from pathlib import Path

import boto3
import pandas as pd

# pip install ccxt
import ccxt


SYMBOLS = ["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "XRP/USDT"]
TIMEFRAME = "1d"
DAYS = 365  # adjust (e.g. 180, 730)
OUTFILE = "backfill_daily_ohlcv.csv"

S3_BUCKET = "crypto-forecast-bucket"
S3_KEY = "processed/daily/backfill_daily_ohlcv.csv"
AWS_REGION = "eu-central-1"


def ms_since_epoch(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)


def fetch_ohlcv_full(exchange, symbol: str, timeframe: str, since_ms: int, limit: int = 1000):
    """Fetch OHLCV in chunks from `since_ms` up to now."""
    all_rows = []
    while True:
        rows = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since_ms, limit=limit)
        if not rows:
            break
        all_rows.extend(rows)

        last_ts = rows[-1][0]
        # advance by 1 ms to avoid duplicates
        since_ms = last_ts + 1

        # stop if the last candle is very recent (already reached "now")
        if last_ts >= ms_since_epoch(datetime.now(timezone.utc)) - 24 * 3600 * 1000:
            break

        # be polite with rate limits
        time.sleep(exchange.rateLimit / 1000)

    return all_rows


def main():
    # Pick an exchange. Binance is common; OKX also works.
    exchange = ccxt.binance(
        {
            "enableRateLimit": True,
        }
    )

    # Compute "since" for backfill
    since = datetime.now(timezone.utc) - pd.Timedelta(days=DAYS)
    since_ms = ms_since_epoch(since)

    frames = []
    for sym in SYMBOLS:
        print(f"[INFO] Fetching {sym} {TIMEFRAME} since {since.isoformat()} ...")
        rows = fetch_ohlcv_full(exchange, sym, TIMEFRAME, since_ms)
        if not rows:
            print(f"[WARN] No rows for {sym}, skipping")
            continue

        df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["date"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True).dt.date.astype(str)
        df["symbol"] = sym.split("/")[0]  # "BTC/USDT" -> "BTC"

        df = df[["date", "symbol", "open", "high", "low", "close", "volume"]]
        frames.append(df)

    if not frames:
        raise RuntimeError("No data fetched for any symbol.")

    out = pd.concat(frames, ignore_index=True)

    # De-dup and sort just in case
    out = out.drop_duplicates(subset=["date", "symbol"]).sort_values(["symbol", "date"]).reset_index(drop=True)

    # Write locally
    Path("data").mkdir(exist_ok=True)
    local_path = Path("data") / OUTFILE
    out.to_csv(local_path, index=False)
    print(f"[INFO] Wrote {len(out)} rows to {local_path}")

    # Upload to S3
    s3 = boto3.client("s3", region_name=AWS_REGION)
    s3.upload_file(str(local_path), S3_BUCKET, S3_KEY)
    print(f"[INFO] Uploaded to s3://{S3_BUCKET}/{S3_KEY}")


if __name__ == "__main__":
    main()