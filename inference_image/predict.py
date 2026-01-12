from keras import initializers
import os
import argparse
import json
import tarfile
import csv
from datetime import timedelta
from pathlib import Path
from io import StringIO

import numpy as np
import pandas as pd
import tensorflow as tf
import keras
print("TF:", tf.__version__)
print("Keras:", keras.__version__)


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


# def maybe_extract_model_tar(model_dir: Path) -> None:
#     tar_path = model_dir / "model.tar.gz"
#     if tar_path.exists():
#         with tarfile.open(tar_path, "r:gz") as tar:
#             tar.extractall(path=model_dir)

def maybe_extract_model_tar(model_dir: Path) -> None:
    # find the tarball anywhere under model_dir
    candidates = list(model_dir.rglob("model.tar.gz"))
    if not candidates:
        candidates = list(model_dir.rglob("*.tar.gz"))
    if not candidates:
        print(f"[WARN] No .tar.gz found under {model_dir}")
        return

    tar_path = candidates[0]
    print(f"[INFO] Extracting {tar_path} into {model_dir}")

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
    window_size = int(s["window_size"][0]) if "window_size" in s.files else int(os.environ.get("WINDOW_SIZE", "60"))
    horizon = int(s["horizon"][0]) if "horizon" in s.files else int(os.environ.get("HORIZON", "7"))

    model = tf.keras.models.load_model(
        keras_model_path,
        safe_mode=False,
        custom_objects={"Orthogonal": initializers.Orthogonal},
    )
    return model, mean, std, feature_cols, window_size, horizon


def make_forecast_for_symbol(df: pd.DataFrame, symbol: str, model, mean, std, feature_cols, window_size, horizon):
    df_sym = df[df["symbol"] == symbol].copy()
    if df_sym.empty:
        raise RuntimeError(f"No rows for symbol {symbol} in input CSV")

    # Feature-compat: some datasets use volumefrom/volumeto instead of volume
    if "volume" not in df_sym.columns and "volumefrom" in df_sym.columns:
        df_sym["volume"] = df_sym["volumefrom"]

    df_sym = df_sym.sort_values("date").copy()
    df_feat = add_basic_features(df_sym)

    df_feat = df_feat.dropna(subset=feature_cols + ["close", "date"])
    if len(df_feat) < window_size:
        raise RuntimeError(f"Not enough rows for {symbol}: have {len(df_feat)}, need {window_size}")

    last_date = pd.to_datetime(df_feat["date"].iloc[-1])
    last_close = float(df_feat["close"].iloc[-1])

    X_win = df_feat[feature_cols].iloc[-window_size:].to_numpy(dtype=np.float32)
    X_std = (X_win - mean) / (std + 1e-8)
    X_std = X_std[np.newaxis, :, :]

    # Model now predicts returns, not prices
    rhat = model.predict(X_std, verbose=0)[0].astype(float).tolist()

    forecast_dates = [(last_date + timedelta(days=i)).date().isoformat() for i in range(1, horizon + 1)]

    # Convert step-by-step returns -> absolute prices (compounded)
    pred_prices = []
    p = last_close
    for r in rhat:
        p = p * (1.0 + float(r))
        pred_prices.append(p)

    return {
        "symbol": symbol,
        "last_observed_date": last_date.date().isoformat(),
        "last_observed_close": last_close,
        "yhat": [
            {"date": d, "pred_return": float(r), "pred_close": float(pc)}
            for d, r, pc in zip(forecast_dates, rhat, pred_prices)
        ],
    }


def payload_to_csv_rows(payload: dict) -> list[dict]:
    """
    Flatten your payload into row-wise records for analytics tools.

    Output columns are stable and dashboard-friendly.
    """
    base = {
        "run_time_utc": payload.get("run_time_utc"),
        "symbol": payload.get("symbol"),
        "fiat": payload.get("fiat"),
        "last_observed_date": payload.get("last_observed_date"),
        "window_size": payload.get("window_size"),
        "horizon": payload.get("horizon"),
    }

    rows = []
    for i, pt in enumerate(payload.get("yhat", []), start=1):
        rows.append({
            **base,
            "target_date": pt.get("date"),
            "horizon_day": i,
            "pred_return": pt.get("pred_return"),
            "pred_close": pt.get("pred_close"),
        })
    return rows


def write_csv(path, rows: list[dict]):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)

    # Fixed column order (helps Athena/QuickSight)
    fieldnames = [
        "run_time_utc", "symbol", "fiat",
        "last_observed_date", "target_date",
        "horizon", "horizon_day",
        "window_size", "pred_return", "pred_close",
    ]

    buf = StringIO()
    w = csv.DictWriter(buf, fieldnames=fieldnames)
    w.writeheader()
    for r in rows:
        w.writerow({k: r.get(k) for k in fieldnames})

    path.write_text(buf.getvalue(), encoding="utf-8")


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
    missing = []

    for sym in symbols:
        try:
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
        except RuntimeError as e:
            msg = str(e)
            if "No rows for symbol" in msg:
                print(f"[WARN] {msg} -> skipping {sym}")
                missing.append(sym)
                continue
            raise  # anything else should still fail loudly

        payload.update({
            "fiat": "USD",
            "run_time_utc": run_time_utc,
            "window_size": window_size,
            "horizon": horizon,
            "feature_cols": feature_cols,
        })

        (output_dir / sym).mkdir(parents=True, exist_ok=True)
        (output_dir / sym / "latest.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        # NEW: also write a flat CSV for Athena/QuickSight
        write_csv(output_dir / sym / "latest.csv", payload_to_csv_rows(payload))
        all_payloads.append(payload)

    # Write global outputs
    (output_dir / "latest_all.json").write_text(json.dumps(all_payloads, indent=2), encoding="utf-8")
    (output_dir / "missing_symbols.json").write_text(json.dumps(missing, indent=2), encoding="utf-8")

    # NEW: global flat file across symbols (nice for Athena/QuickSight)
    all_rows = []
    for pld in all_payloads:
        all_rows.extend(payload_to_csv_rows(pld))
    write_csv(output_dir / "latest_all.csv", all_rows)

    print(json.dumps(all_payloads, indent=2))
    if missing:
        print(f"[WARN] Missing symbols skipped: {missing}")


if __name__ == "__main__":
    main()