import argparse
import json
import tarfile
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf


def add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    # Matches src/sagemaker/train_lstm_sagemaker.py
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


def maybe_extract_model_tar(model_dir: Path) -> None:
    tar_path = model_dir / "model.tar.gz"
    if tar_path.exists():
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(path=model_dir)


def load_model_and_scaler(model_dir: Path):
    maybe_extract_model_tar(model_dir)

    scaler_path = model_dir / "scaler_stats.npz"
    keras_model_path = model_dir / "model.keras"

    if not scaler_path.exists():
        raise FileNotFoundError(f"Missing scaler_stats.npz in {model_dir}")
    if not keras_model_path.exists():
        raise FileNotFoundError(f"Missing model.keras in {model_dir}")

    s = np.load(scaler_path, allow_pickle=True)
    mean = s["mean"]
    std = s["std"]
    feature_cols = s["feature_cols"].tolist()
    window_size = int(s["window_size"][0])
    horizon = int(s["horizon"][0])

    model = tf.keras.models.load_model(keras_model_path)
    return model, mean, std, feature_cols, window_size, horizon


def make_forecast_for_symbol(df: pd.DataFrame, symbol: str, model, mean, std, feature_cols, window_size, horizon):
    df_sym = df[df["symbol"] == symbol].copy()
    if df_sym.empty:
        raise RuntimeError(f"No rows for symbol {symbol} in input CSV")

    df_sym = df_sym.sort_values("date").copy()
    df_feat = add_basic_features(df_sym)

    df_feat = df_feat.dropna(subset=feature_cols + ["close", "date"])
    if len(df_feat) < window_size:
        raise RuntimeError(f"Not enough rows for {symbol}: have {len(df_feat)}, need {window_size}")

    last_date = pd.to_datetime(df_feat["date"].iloc[-1])
    X_win = df_feat[feature_cols].iloc[-window_size:].to_numpy(dtype=np.float32)
    X_std = (X_win - mean) / (std + 1e-8)
    X_std = X_std[np.newaxis, :, :]

    yhat = model.predict(X_std, verbose=0)[0].tolist()
    forecast_dates = [(last_date + timedelta(days=i)).date().isoformat() for i in range(1, horizon + 1)]

    return {
        "symbol": symbol,
        "last_observed_date": last_date.date().isoformat(),
        "yhat": [{"date": d, "pred_close": float(v)} for d, v in zip(forecast_dates, yhat)],
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input-csv", required=True)
    p.add_argument("--model-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--symbols", default="BTC,ETH,BNB,XRP,SOL")
    args = p.parse_args()

    input_csv = Path(args.input_csv)
    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv, parse_dates=["date"])
    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]

    model, mean, std, feature_cols, window_size, horizon = load_model_and_scaler(model_dir)

    run_time_utc = pd.Timestamp.utcnow().isoformat()
    all_payloads = []

    for sym in symbols:
        payload = make_forecast_for_symbol(
            df=df,
            symbol=sym,
            model=model,
            mean=mean,
            std=std,
            feature_cols=feature_cols,
            window_size=window_size,
            horizon=horizon,
        )
        payload.update({
            "fiat": "USD",
            "run_time_utc": run_time_utc,
            "window_size": window_size,
            "horizon": horizon,
            "feature_cols": feature_cols,
        })

        (output_dir / sym).mkdir(parents=True, exist_ok=True)
        (output_dir / sym / "latest.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        all_payloads.append(payload)

    (output_dir / "latest_all.json").write_text(json.dumps(all_payloads, indent=2), encoding="utf-8")
    print(json.dumps(all_payloads, indent=2))


if __name__ == "__main__":
    main()
