import os
from pathlib import Path
from typing import Tuple

import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle as sk_shuffle
import mlflow
import mlflow.tensorflow


# ---------- CONFIG ----------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

SEQ_DATA_PATH = DATA_PROCESSED_DIR / "crypto_seq_window60_horizon7.npz"

MODELS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
TRAIN_FRACTION = 0.7
VAL_FRACTION = 0.15  # test will be 1 - train - val = 0.15

EPOCHS = 30
BATCH_SIZE = 64

EXPERIMENT_NAME = "crypto_lstm_forecasting"


# ---------- HELPERS ----------
def load_sequence_data(path: Path):
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


def standardize_features(X_train, X_val, X_test):
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
    return X_train_std, X_val_std, X_test_std, scaler


def build_lstm_model(window_size: int, num_features: int, horizon: int) -> tf.keras.Model:
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
    return model


# ---------- MAIN TRAINING WITH MLFLOW ----------
def main():
    print(f"[INFO] Loading sequence data from {SEQ_DATA_PATH}")
    X, y, symbol_idx, symbols, feature_cols, window_size, horizon = load_sequence_data(SEQ_DATA_PATH)

    print(f"[INFO] X shape: {X.shape}")
    print(f"[INFO] y shape: {y.shape}")
    print(f"[INFO] Symbols: {symbols}")
    print(f"[INFO] Num features: {len(feature_cols)} | Window size: {window_size} | Horizon: {horizon}")

    X_train, y_train, X_val, y_val, X_test, y_test = train_val_test_split(X, y)
    X_train, X_val, X_test, scaler = standardize_features(X_train, X_val, X_test)

    scaler_path = MODELS_DIR / "lstm_scaler_stats.npz"
    np.savez_compressed(scaler_path, mean=scaler["mean"], std=scaler["std"], feature_cols=np.array(feature_cols))
    print(f"[INFO] Saved scaler stats to {scaler_path}")

    num_features = X_train.shape[-1]
    model = build_lstm_model(window_size, num_features, horizon)

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

    # ---- MLFLOW SETUP ----
    mlflow.set_experiment(EXPERIMENT_NAME)
    mlflow.tensorflow.autolog()  # automatically logs params, metrics, and model

    with mlflow.start_run():
        # Log a few custom params as well
        mlflow.log_param("window_size", window_size)
        mlflow.log_param("horizon", horizon)
        mlflow.log_param("epochs", EPOCHS)
        mlflow.log_param("batch_size", BATCH_SIZE)
        mlflow.log_param("train_fraction", TRAIN_FRACTION)
        mlflow.log_param("val_fraction", VAL_FRACTION)
        mlflow.log_param("num_features", num_features)
        mlflow.log_param("symbols", ",".join(symbols))

        history = model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=callbacks,
            verbose=1,
        )

        test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
        print(f"[INFO] Test Loss (MSE): {test_loss:.6f}")
        print(f"[INFO] Test MAE: {test_mae:.6f}")

        mlflow.log_metric("test_mse", float(test_loss))
        mlflow.log_metric("test_mae", float(test_mae))

        # Save final model explicitly
        final_model_path = MODELS_DIR / "lstm_final_model.keras"
        model.save(final_model_path)
        print(f"[INFO] Saved final model to {final_model_path}")

        # Log artifacts (model + scaler)
        mlflow.log_artifact(str(final_model_path), artifact_path="models")
        mlflow.log_artifact(str(checkpoint_path), artifact_path="models")
        mlflow.log_artifact(str(scaler_path), artifact_path="preprocessing")

        # Example prediction
        sample_input = X_test[:1]
        preds = model.predict(sample_input)
        print(f"[INFO] Example prediction (7-day forecast): {preds[0]}")
        print(f"[INFO] True values:                     {y_test[0]}")


if __name__ == "__main__":
    tf.random.set_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    main()