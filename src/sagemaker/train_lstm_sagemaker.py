import argparse
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.utils import shuffle as sk_shuffle


# =============== Helper functions ===============

def load_training_data(input_dir: str) -> pd.DataFrame:
    """
    Load one or more CSV files from the SageMaker training channel.
    Expected structure: /opt/ml/input/data/train/*.csv

    Each CSV should have at least:
    date, symbol, open, high, low, close, volumefrom, volumeto
    (Plus any extra features you decide to include later.)
    """
    input_path = Path(input_dir)
    csv_files = list(input_path.rglob("*.csv"))

    print("[INFO] Found CSV files:")
    for f in csv_files:
        print(" -", f)

    if not csv_files:
        raise RuntimeError(f"No CSV files found in training channel: {input_dir}")

    dfs = []
    for f in csv_files:
        print(f"[INFO] Loading training file: {f}")
        df = pd.read_csv(f, parse_dates=["date"])
        dfs.append(df)

    df_all = pd.concat(dfs, ignore_index=True)
    df_all = df_all.sort_values(["symbol", "date"]).reset_index(drop=True)

    print(f"[INFO] Combined training data shape: {df_all.shape}")
    print(f"[INFO] Symbols in training data: {sorted(df_all['symbol'].unique().tolist())}")
    return df_all


def add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add some simple features per symbol.
    You can later expand this with the same indicators you used locally.
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

        # Exponential moving averages
        g["ema_7"] = g["close"].ewm(span=7, adjust=False).mean()

        # Volatility
        g["volatility_7"] = g["return_close_log"].rolling(7).std()

        all_frames.append(g)

    df_feat = pd.concat(all_frames, ignore_index=True)
    df_feat = df_feat.sort_values(["symbol", "date"]).reset_index(drop=True)
    return df_feat


def build_sequences_for_symbol(
    df_symbol: pd.DataFrame,
    feature_cols: List[str],
    window_size: int,
    horizon: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build (X, y) for one symbol.
    X: [num_samples, window_size, num_features]
    y: [num_samples, horizon]  (future close prices)
    """
    df_symbol = df_symbol.sort_values("date").reset_index(drop=True).copy()

    # Drop rows with any NaNs in feature columns or close
    df_symbol = df_symbol.dropna(subset=feature_cols + ["close"])

    values = df_symbol[feature_cols].to_numpy()
    close_prices = df_symbol["close"].to_numpy()

    X_list, Y_list = [], []

    total_len = len(df_symbol)
    max_start = total_len - window_size - horizon + 1

    for start in range(max_start):
        end_window = start + window_size
        end_horizon = end_window + horizon

        x_seq = values[start:end_window, :]
        y_seq = close_prices[end_window:end_horizon]

        X_list.append(x_seq)
        Y_list.append(y_seq)

    if not X_list:
        return np.empty((0, window_size, len(feature_cols))), np.empty((0, horizon))

    X = np.stack(X_list, axis=0)
    Y = np.stack(Y_list, axis=0)
    return X, Y


def build_sequences_all_symbols(
    df: pd.DataFrame,
    window_size: int,
    horizon: int,
) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    """
    Build sequences across all symbols.
    Returns:
      X: [N, window_size, num_features]
      y: [N, horizon]
      symbols: list of symbols
      feature_cols: list of feature names used
    """
    exclude = {"date", "symbol"}
    feature_cols = [c for c in df.columns if c not in exclude]

    all_X, all_y = [], []
    symbols = sorted(df["symbol"].unique())

    for sym in symbols:
        df_sym = df[df["symbol"] == sym].copy()
        X_sym, y_sym = build_sequences_for_symbol(df_sym, feature_cols, window_size, horizon)

        if X_sym.shape[0] == 0:
            print(f"[WARN] No sequences built for symbol {sym}, skipping.")
            continue

        all_X.append(X_sym)
        all_y.append(y_sym)

    if not all_X:
        raise RuntimeError("No sequences built for any symbol.")

    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)

    print(f"[INFO] Built sequences: X shape = {X.shape}, y shape = {y.shape}")
    print(f"[INFO] Feature columns: {feature_cols}")
    print(f"[INFO] Symbols used: {symbols}")
    return X, y, symbols, feature_cols


def train_val_test_split(
    X: np.ndarray,
    y: np.ndarray,
    train_frac: float,
    val_frac: float,
    random_seed: int,
):
    X, y = sk_shuffle(X, y, random_state=random_seed)

    n = X.shape[0]
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    n_test = n - n_train - n_val

    X_train = X[:n_train]
    y_train = y[:n_train]

    X_val = X[n_train:n_train + n_val]
    y_val = y[n_train:n_train + n_val]

    X_test = X[n_train + n_val:]
    y_test = y[n_train + n_val:]

    print(f"[INFO] Dataset split: train={X_train.shape[0]}, val={X_val.shape[0]}, test={X_test.shape[0]}")
    return X_train, y_train, X_val, y_val, X_test, y_test


def standardize_features(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
):
    """
    Standardize features (per feature across time+batch), using train statistics.
    """
    n_train, t_train, f_train = X_train.shape
    train_flat = X_train.reshape(-1, f_train)

    mean = train_flat.mean(axis=0)
    std = train_flat.std(axis=0) + 1e-8

    def transform(X):
        n, t, f = X.shape
        flat = X.reshape(-1, f)
        flat = (flat - mean) / std
        return flat.reshape(n, t, f)

    X_train_std = transform(X_train)
    X_val_std = transform(X_val)
    X_test_std = transform(X_test)

    scaler = {"mean": mean, "std": std}
    print("[INFO] Standardized features using train mean/std.")
    return X_train_std, X_val_std, X_test_std, scaler


def build_lstm_model(
    window_size: int,
    num_features: int,
    horizon: int,
    learning_rate: float,
) -> tf.keras.Model:
    """
    Build a simple LSTM model.
    """
    inputs = tf.keras.Input(shape=(window_size, num_features))

    x = tf.keras.layers.LSTM(64, return_sequences=True)(inputs)
    x = tf.keras.layers.LSTM(32)(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    outputs = tf.keras.layers.Dense(horizon, name="forecast")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="crypto_lstm_forecaster")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=["mae"],
    )

    model.summary()
    return model


# =============== Main training entry point ===============

def parse_args():
    parser = argparse.ArgumentParser()

    # SageMaker-specific args (input/output dirs)
    parser.add_argument(
        "--train-data-dir",
        type=str,
        default=os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train"),
    )
    parser.add_argument(
        "--model-dir",
        "--model_dir",
        type=str,
        default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"),
    )

    # Hyperparameters
    parser.add_argument("--window-size", type=int, default=60)
    parser.add_argument("--horizon", type=int, default=7)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--train-fraction", type=float, default=0.7)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--random-seed", type=int, default=42)

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    print("[INFO] Training arguments:", args)

    # Set seeds for reproducibility
    tf.random.set_seed(args.random_seed)
    np.random.seed(args.random_seed)

    # 1) Load data
    df = load_training_data(args.train_data_dir)

    # 2) Add features
    df_feat = add_basic_features(df)

    # 3) Build sequences
    X, y, symbols, feature_cols = build_sequences_all_symbols(
        df_feat,
        window_size=args.window_size,
        horizon=args.horizon,
    )

    print(f"[INFO] Final X shape: {X.shape} | y shape: {y.shape}")
    print(f"[INFO] Using features: {feature_cols}")

    # 4) Split
    X_train, y_train, X_val, y_val, X_test, y_test = train_val_test_split(
        X,
        y,
        train_frac=args.train_fraction,
        val_frac=args.val_fraction,
        random_seed=args.random_seed,
    )

    # 5) Standardize
    X_train, X_val, X_test, scaler = standardize_features(X_train, X_val, X_test)

    # 6) Build model
    num_features = X_train.shape[-1]
    model = build_lstm_model(
        window_size=args.window_size,
        num_features=num_features,
        horizon=args.horizon,
        learning_rate=args.learning_rate,
    )

    # 7) Train
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
        )
    ]

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    # 8) Evaluate
    test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
    print(f"[INFO] Test Loss (MSE): {test_loss:.6f}")
    print(f"[INFO] Test MAE: {test_mae:.6f}")

    # 9) Save model + scaler into model_dir (SageMaker will upload this to S3)
    model_dir = Path(os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    model_dir.mkdir(parents=True, exist_ok=True)
    print("[DEBUG] SM_MODEL_DIR =", os.environ.get("SM_MODEL_DIR"))

    # Save as TensorFlow SavedModel directory (most compatible with SageMaker packaging)
    saved_model_dir = model_dir / "saved_model"
    model.save(saved_model_dir)  # creates a folder with saved_model.pb + variables/
    print(f"[INFO] Saved SavedModel to {saved_model_dir}")

    # Save scaler + metadata
    scaler_path = model_dir / "scaler_stats.npz"
    np.savez_compressed(
        scaler_path,
        mean=scaler["mean"],
        std=scaler["std"],
        feature_cols=np.array(feature_cols),
        symbols=np.array(symbols),
        window_size=np.array([args.window_size]),
        horizon=np.array([args.horizon]),
    )
    print(f"[INFO] Saved scaler stats to {scaler_path}")

    # Extra sanity: list what ended up in SM_MODEL_DIR
    print("[INFO] Contents of model_dir:")
    for p in sorted(model_dir.rglob("*")):
        print(" -", p)


if __name__ == "__main__":
    main()