import os
from pathlib import Path
from typing import Tuple

import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle as sk_shuffle


# ---------- CONFIG ----------
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # adjust if needed
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

SEQ_DATA_PATH = DATA_PROCESSED_DIR / "crypto_seq_window60_horizon7.npz"

MODELS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
TRAIN_FRACTION = 0.7
VAL_FRACTION = 0.15  # test will be 1 - train - val = 0.15

EPOCHS = 30
BATCH_SIZE = 64


# ---------- HELPERS ----------

def load_sequence_data(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, list, list, int, int]:
    """
    Load sequence data from .npz file.

    Returns:
        X: [N, window_size, num_features]
        y: [N, horizon]
        symbol_idx: [N]
        symbols: list of symbols (str)
        feature_cols: list of feature names (str)
        window_size: int
        horizon: int
    """
    data = np.load(path, allow_pickle=True)

    X = data["X"]
    y = data["y"]
    symbol_idx = data["symbol_idx"]
    symbols = data["symbols"].tolist()
    feature_cols = data["feature_cols"].tolist()
    window_size = int(data["window_size"][0])
    horizon = int(data["horizon"][0])

    return X, y, symbol_idx, symbols, feature_cols, window_size, horizon


def train_val_test_split(
    X: np.ndarray,
    y: np.ndarray,
    train_frac: float = TRAIN_FRACTION,
    val_frac: float = VAL_FRACTION,
    random_seed: int = RANDOM_SEED,
):
    """
    Shuffle and split X, y into train/val/test sets.
    """
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
    Standardize features (per feature, across all time steps) using train set stats.
    X shape: [N, window_size, num_features]
    """
    # Flatten time dimension and batch: [N * T, F]
    n_train, t_train, f_train = X_train.shape
    train_flat = X_train.reshape(-1, f_train)

    mean = train_flat.mean(axis=0)
    std = train_flat.std(axis=0) + 1e-8  # avoid div by zero

    def transform(X):
        n, t, f = X.shape
        flat = X.reshape(-1, f)
        flat = (flat - mean) / std
        return flat.reshape(n, t, f)

    X_train_std = transform(X_train)
    X_val_std = transform(X_val)
    X_test_std = transform(X_test)

    scaler = {"mean": mean, "std": std}
    return X_train_std, X_val_std, X_test_std, scaler


def build_lstm_model(window_size: int, num_features: int, horizon: int) -> tf.keras.Model:
    """
    Build a simple LSTM model for multi-step forecasting.
    """
    inputs = tf.keras.Input(shape=(window_size, num_features))

    x = tf.keras.layers.LSTM(64, return_sequences=True)(inputs)
    x = tf.keras.layers.LSTM(32)(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    outputs = tf.keras.layers.Dense(horizon, name="forecast")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="crypto_lstm_forecaster")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="mse",
        metrics=["mae"],
    )

    model.summary()
    return model


# ---------- MAIN TRAINING ----------

def main():
    print(f"[INFO] Loading sequence data from {SEQ_DATA_PATH}")
    X, y, symbol_idx, symbols, feature_cols, window_size, horizon = load_sequence_data(SEQ_DATA_PATH)

    print(f"[INFO] X shape: {X.shape}")
    print(f"[INFO] y shape: {y.shape}")
    print(f"[INFO] Symbols: {symbols}")
    print(f"[INFO] Num features: {len(feature_cols)} | Window size: {window_size} | Horizon: {horizon}")

    # Split
    X_train, y_train, X_val, y_val, X_test, y_test = train_val_test_split(X, y)

    # Standardize features (based only on train)
    X_train, X_val, X_test, scaler = standardize_features(X_train, X_val, X_test)

    # Save scaler params (for inference later)
    scaler_path = MODELS_DIR / "lstm_scaler_stats.npz"
    np.savez_compressed(scaler_path, mean=scaler["mean"], std=scaler["std"], feature_cols=np.array(feature_cols))
    print(f"[INFO] Saved scaler stats to {scaler_path}")

    # Build model
    num_features = X_train.shape[-1]
    model = build_lstm_model(window_size, num_features, horizon)

    # Callbacks: early stopping + best model checkpoint
    checkpoint_path = MODELS_DIR / "lstm_best_model.keras"
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_loss",
            save_best_only=True,
        ),
    ]

    # Train
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1,
    )

    # Evaluate on test set
    test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
    print(f"[INFO] Test Loss (MSE): {test_loss:.6f}")
    print(f"[INFO] Test MAE: {test_mae:.6f}")

    # Save final model (even if checkpoint also saved)
    final_model_path = MODELS_DIR / "lstm_final_model.keras"
    model.save(final_model_path)
    print(f"[INFO] Saved final model to {final_model_path}")

    # You can also print a quick example prediction
    sample_input = X_test[:1]
    preds = model.predict(sample_input)
    print(f"[INFO] Example prediction (7-day forecast): {preds[0]}")
    print(f"[INFO] Corresponding true values:        {y_test[0]}")


if __name__ == "__main__":
    # Fix random seeds for reproducibility
    tf.random.set_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # suppress verbose TF logs
    main()