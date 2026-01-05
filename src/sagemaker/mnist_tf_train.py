import os
import tensorflow as tf


def main():
    # Loads MNIST from the internet inside the container (no S3 input needed)
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(10),
        ]
    )

    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    model.fit(x_train, y_train, epochs=1, batch_size=128, validation_split=0.1)
    loss, acc = model.evaluate(x_test, y_test, verbose=2)
    print(f"[RESULT] test_loss={loss:.4f} test_acc={acc:.4f}")

    # Save to the SageMaker model directory
    model_dir = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
    tf.saved_model.save(model, model_dir)
    print(f"[INFO] Saved model to {model_dir}")


if __name__ == "__main__":
    main()