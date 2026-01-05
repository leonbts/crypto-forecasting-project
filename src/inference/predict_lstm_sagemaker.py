# src/inference/predict_lstm_sagemaker.py
import argparse
import json
import os
from datetime import timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests
import tensorflow as tf


API_URL = "https://min-api.cryptocompare.com/data/v2/histoday"


def fetch_coin_history(symbol: str, tsym: str, limit: int, api_key: Optional[str]) -> pd.DataFrame:
    params = {"fsym": symbol, "tsym": tsym, "limit": limit}
    if api_key:
        params["api_key"] = api_key

    resp = requests.get(API_URL, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    if data.get("Response") != "Success":
        raise RuntimeError(f"CryptoCompare error: {data.get('Message')}")

    records = data["Data"]["Data"]
    df = pd.DataFrame(records)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df = df.rename(columns={"time": "date"})
    df["symbol"] = symbol
    cols = ["date", "symbol", "open", "high", "low", "close", "volumefrom", "volumeto"]
    return df[cols].sort_values("date").reset_index(drop=True)


def add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    # Matches src/sagemaker/train_lstm_sagemaker.py :contentReference[oaicite:7]{index=7}
    df = df.sort_values(["symbol", "date"]).copy()
    out = []
    for symbol, g in df.groupby("symbol"):
        g = g.sort_values("date").copy()
        g["return_close_pct"] = g["close"].pct_change()
        g["return_close_log"] = np.log1p(g["return_close_pct"])
        g["sma_7"] = g["close"].rolling(7).mean()
        g["sma_14"] = g["close"].rolling(14).mean()
        g["ema_7"] = g["close"].ewm(span=7, adjust=False).mean()
        g["volatility_7"] = g["return_close_log"].rolling(7).std()
        out.append(g)
    return pd.concat(out, ignore_index=True).sort_values(["symbol", "date"]).reset_index(drop=True)


def build_last_window(df_feat: pd.DataFrame, feature_cols: list[str], window_size: int) -> tuple[np.ndarray, pd.Timestamp]:
    df_sym = df_feat.sort_values("date").copy()

    # Drop rows where any needed feature is nan
    df_sym = df_sym.dropna(subset=feature_cols + ["close", "date"])
    if len(df_sym) < window_size:
        raise RuntimeError(f"Not enough rows after feature/NaN drop: have {len(df_sym)}, need {window_size}")

    last_date = pd.to_datetime(df_sym["date"].iloc[-1])
    window = df_sym[feature_cols].iloc[-window_size:].to_numpy(dtype=np.float32)  # [T, F]
    return window, last_date


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", default="BTC")
    p.add_argument("--fiat", default="USD")
    p.add_argument("--limit", type=int, default=400)  # enough for rolling features + window
    p.add_argument("--model-dir", default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    p.add_argument("--output", default=os.environ.get("SM_OUTPUT_DATA_DIR", "/opt/ml/output"))
    args = p.parse_args()

    model_dir = Path(args.model_dir)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    api_key = os.getenv("CRYPTOCOMPARE_API_KEY", None)

    # 1) Load scaler + metadata saved by SageMaker training
    scaler_path = model_dir / "scaler_stats.npz"
    if not scaler_path.exists():
        raise FileNotFoundError(f"Missing scaler_stats.npz at {scaler_path}")

    scaler = np.load(scaler_path, allow_pickle=True)
    mean = scaler["mean"]
    std = scaler["std"]
    feature_cols = scaler["feature_cols"].tolist()
    window_size = int(scaler["window_size"][0])
    horizon = int(scaler["horizon"][0])

    # 2) Load model
    saved_model_dir = model_dir / "saved_model"
    if not saved_model_dir.exists():
        raise FileNotFoundError(f"Missing saved_model/ at {saved_model_dir}")

    model = tf.keras.models.load_model(saved_model_dir)

    # 3) Fetch latest data
    df = fetch_coin_history(args.symbol, args.fiat, args.limit, api_key)

    # 4) Feature engineering (must match training)
    df_feat = add_basic_features(df)

    # 5) Build last window in correct feature order
    X_win, last_date = build_last_window(df_feat, feature_cols, window_size)

    # 6) Standardize using train stats
    X_std = (X_win - mean) / (std + 1e-8)
    X_std = X_std[np.newaxis, :, :]  # [1, T, F]

    # 7) Predict
    yhat = model.predict(X_std, verbose=0)[0].tolist()

    # 8) Build forecast dates
    forecast_dates = [(last_date + timedelta(days=i)).date().isoformat() for i in range(1, horizon + 1)]

    payload = {
        "symbol": args.symbol,
        "fiat": args.fiat,
        "run_time_utc": pd.Timestamp.utcnow().isoformat(),
        "last_observed_date": last_date.date().isoformat(),
        "window_size": window_size,
        "horizon": horizon,
        "feature_cols": feature_cols,
        "yhat": [{"date": d, "pred_close": float(v)} for d, v in zip(forecast_dates, yhat)],
    }

    out_path = output_dir / f"forecast_{args.symbol}.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(json.dumps(payload, indent=2))
    print(f"[INFO] Wrote forecast to {out_path}")


if __name__ == "__main__":
    main()