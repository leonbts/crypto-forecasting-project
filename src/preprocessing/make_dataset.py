import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple


# ------------ CONFIG ------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # adjust if needed
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

INPUT_CSV = DATA_PROCESSED_DIR / "crypto_ohlcv_daily.csv"
ENRICHED_CSV = DATA_PROCESSED_DIR / "crypto_ohlcv_daily_enriched.csv"
SEQ_DATA_PATH = DATA_PROCESSED_DIR / "crypto_seq_window60_horizon7.npz"

WINDOW_SIZE = 60   # past 60 days
HORIZON = 7        # predict next 7 days


# ------------ TECHNICAL INDICATORS ------------

def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()

    rs = avg_gain / (avg_loss + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add basic technical indicators per symbol.
    Assumes df has columns: date, symbol, open, high, low, close, volumefrom, volumeto
    """
    df = df.sort_values(["symbol", "date"]).copy()

    all_frames = []
    for symbol, group in df.groupby("symbol"):
        g = group.copy().sort_values("date")

        # Returns
        g["return_close_pct"] = g["close"].pct_change()
        g["return_close_log"] = np.log1p(g["return_close_pct"])

        # Simple moving averages
        g["sma_7"] = g["close"].rolling(7).mean()
        g["sma_14"] = g["close"].rolling(14).mean()
        g["sma_30"] = g["close"].rolling(30).mean()

        # Exponential moving averages
        g["ema_7"] = g["close"].ewm(span=7, adjust=False).mean()
        g["ema_14"] = g["close"].ewm(span=14, adjust=False).mean()

        # Volatility (rolling std of log returns)
        g["volatility_7"] = g["return_close_log"].rolling(7).std()
        g["volatility_30"] = g["return_close_log"].rolling(30).std()

        # Bollinger Bands (20-day)
        sma_20 = g["close"].rolling(20).mean()
        std_20 = g["close"].rolling(20).std()
        g["bb_upper_20"] = sma_20 + 2 * std_20
        g["bb_lower_20"] = sma_20 - 2 * std_20

        # RSI
        g["rsi_14"] = compute_rsi(g["close"], period=14)

        all_frames.append(g)

    df_out = pd.concat(all_frames, ignore_index=True)
    df_out = df_out.sort_values(["symbol", "date"]).reset_index(drop=True)
    return df_out


# ------------ SEQUENCE CREATION ------------

def build_sequences_for_symbol(
    df_symbol: pd.DataFrame,
    feature_cols: List[str],
    window_size: int = WINDOW_SIZE,
    horizon: int = HORIZON,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build (X, y) sequences for one symbol.
    X: [num_samples, window_size, num_features]
    y: [num_samples, horizon]  (future close prices)
    """
    df_symbol = df_symbol.sort_values("date").reset_index(drop=True).copy()

    # Drop rows with any NaNs in feature columns
    df_symbol = df_symbol.dropna(subset=feature_cols + ["close"])

    values = df_symbol[feature_cols].to_numpy()
    close_prices = df_symbol["close"].to_numpy()

    X_list = []
    y_list = []

    total_len = len(df_symbol)
    max_start = total_len - window_size - horizon + 1

    for start in range(max_start):
        end_window = start + window_size
        end_horizon = end_window + horizon

        x_seq = values[start:end_window, :]             # shape (window_size, n_features)
        y_seq = close_prices[end_window:end_horizon]    # shape (horizon,)

        X_list.append(x_seq)
        y_list.append(y_seq)

    if not X_list:
        return np.empty((0, window_size, len(feature_cols))), np.empty((0, horizon))

    X = np.stack(X_list, axis=0)
    y = np.stack(y_list, axis=0)
    return X, y


def build_sequences_all_symbols(
    df: pd.DataFrame,
    window_size: int = WINDOW_SIZE,
    horizon: int = HORIZON,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], List[str]]:
    """
    Build sequences across all symbols.

    Returns:
    X: [N, window_size, num_features]
    y: [N, horizon]
    symbol_idx: [N] (index of symbol for each sample)
    symbols: list of symbol names
    feature_cols: list of feature column names
    """
    # Define which columns are features:
    exclude = {"date", "symbol"}
    feature_cols = [c for c in df.columns if c not in exclude]

    all_X = []
    all_y = []
    all_symbol_idx = []
    symbols = sorted(df["symbol"].unique())

    for idx, sym in enumerate(symbols):
        df_sym = df[df["symbol"] == sym].copy()
        X_sym, y_sym = build_sequences_for_symbol(df_sym, feature_cols, window_size, horizon)

        if X_sym.shape[0] == 0:
            continue

        all_X.append(X_sym)
        all_y.append(y_sym)
        all_symbol_idx.append(np.full(X_sym.shape[0], idx, dtype=np.int32))

    if not all_X:
        raise RuntimeError("No sequences built for any symbol.")

    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)
    symbol_idx = np.concatenate(all_symbol_idx, axis=0)

    return X, y, symbol_idx, symbols, feature_cols


# ------------ MAIN ------------

def main():
    print(f"[INFO] Loading input CSV from {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV, parse_dates=["date"])

    # 1) Add indicators
    print("[INFO] Adding technical indicators...")
    df_enriched = add_technical_indicators(df)

    # Save enriched CSV (good for EDA, Prophet, etc.)
    df_enriched.to_csv(ENRICHED_CSV, index=False)
    print(f"[INFO] Saved enriched CSV to {ENRICHED_CSV}")

    # 2) Build windowed sequences for deep learning models
    print("[INFO] Building sequences for all symbols...")
    X, y, symbol_idx, symbols, feature_cols = build_sequences_all_symbols(df_enriched)

    print(f"[INFO] X shape: {X.shape}  (samples, window, features)")
    print(f"[INFO] y shape: {y.shape}  (samples, horizon)")
    print(f"[INFO] Symbols: {symbols}")
    print(f"[INFO] Num features: {len(feature_cols)}")

    # Save as .npz (compressed numpy, easy to load later)
    np.savez_compressed(
        SEQ_DATA_PATH,
        X=X,
        y=y,
        symbol_idx=symbol_idx,
        symbols=np.array(symbols),
        feature_cols=np.array(feature_cols),
        window_size=np.array([WINDOW_SIZE]),
        horizon=np.array([HORIZON]),
    )
    print(f"[INFO] Saved sequence data to {SEQ_DATA_PATH}")


if __name__ == "__main__":
    main()