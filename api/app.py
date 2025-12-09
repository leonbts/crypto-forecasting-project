from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
import tensorflow as tf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

ENRICHED_CSV = DATA_PROCESSED_DIR / "crypto_ohlcv_daily_enriched.csv"
SCALER_PATH = MODELS_DIR / "lstm_scaler_stats.npz"
MODEL_PATH = MODELS_DIR / "lstm_best_model.keras"

WINDOW_SIZE = 60
HORIZON = 7

app = FastAPI(title="Crypto LSTM Forecast API", version="0.1.0")

# ---- Load artifacts at startup ----
print("[INFO] Loading model and scaler...")

if not MODEL_PATH.exists():
    raise RuntimeError(f"Model file not found at {MODEL_PATH}. Train the model first.")

if not SCALER_PATH.exists():
    raise RuntimeError(f"Scaler stats not found at {SCALER_PATH}. Run training first.")

model = tf.keras.models.load_model(MODEL_PATH)

scaler_data = np.load(SCALER_PATH, allow_pickle=True)
FEATURE_COLS: List[str] = scaler_data["feature_cols"].tolist()
MEAN = scaler_data["mean"]
STD = scaler_data["std"] + 1e-8


# ---- Data loading ----
if not ENRICHED_CSV.exists():
    raise RuntimeError(f"Enriched CSV not found at {ENRICHED_CSV}. Run preprocessing first.")

df_enriched = pd.read_csv(ENRICHED_CSV, parse_dates=["date"])
df_enriched = df_enriched.sort_values(["symbol", "date"]).reset_index(drop=True)


class ForecastResponse(BaseModel):
    symbol: str
    horizon_days: int
    forecasts: List[float]


def build_latest_window(symbol: str) -> np.ndarray:
    """
    Build the latest WINDOW_SIZE sequence for the given symbol.
    Returns standardized numpy array of shape (1, WINDOW_SIZE, num_features).
    """
    df_sym = df_enriched[df_enriched["symbol"] == symbol.upper()].copy()
    if df_sym.shape[0] < WINDOW_SIZE:
        raise ValueError(f"Not enough data for symbol {symbol}, need at least {WINDOW_SIZE} rows.")

    df_sym = df_sym.sort_values("date").reset_index(drop=True)
    df_last = df_sym.iloc[-WINDOW_SIZE:]  # last 60 days

    # Ensure we have all feature columns
    missing = [c for c in FEATURE_COLS if c not in df_last.columns]
    if missing:
        raise ValueError(f"Missing feature columns for inference: {missing}")

    X = df_last[FEATURE_COLS].to_numpy()  # shape (WINDOW_SIZE, num_features)

    # Standardize using training stats (same as training)
    X = (X - MEAN) / STD

    # Add batch dimension
    X = np.expand_dims(X, axis=0)  # (1, WINDOW_SIZE, num_features)
    return X


@app.get("/health")
def health_check():
    return {"status": "ok", "message": "Crypto forecast API is running."}


@app.get("/symbols")
def list_symbols():
    syms = sorted(df_enriched["symbol"].unique().tolist())
    return {"symbols": syms}


@app.get("/predict", response_model=ForecastResponse)
def predict(symbol: str = Query(..., description="Crypto symbol, e.g. BTC, ETH, BNB, XRP, SOL")):
    symbol = symbol.upper()
    available_symbols = df_enriched["symbol"].unique().tolist()
    if symbol not in available_symbols:
        raise HTTPException(status_code=400, detail=f"Symbol {symbol} not found. Available: {available_symbols}")

    try:
        X = build_latest_window(symbol)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    preds = model.predict(X)
    preds = preds[0].tolist()  # shape (HORIZON,)

    return ForecastResponse(symbol=symbol, horizon_days=HORIZON, forecasts=preds)