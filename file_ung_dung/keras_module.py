import numpy as np
import tensorflow as tf
from keras import Input, Sequential, layers, utils


NUM_CLASSES = 10
INPUT_SHAPE = (28, 28, 1)
BATCH_SIZE = 128
EPOCHS = 500
VALIDATION_SPLIT = 0.1
EXPORT_PATH = "mnist_keras_model.keras"


def configure_gpu() -> bool:
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        return False
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU detected. Enabled memory growth.")
        return True
    except RuntimeError as exc:
        print(f"Không thể cấu hình GPU: {exc}")
        return False


def load_data():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = np.expand_dims(x_train.astype("float32") / 255, -1)
    x_test = np.expand_dims(x_test.astype("float32") / 255, -1)
    y_train = utils.to_categorical(y_train, NUM_CLASSES)
    y_test = utils.to_categorical(y_test, NUM_CLASSES)
    print(f"Train: {x_train.shape[0]} mẫu, Test: {x_test.shape[0]} mẫu")
    return (x_train, y_train), (x_test, y_test)


def build_model() -> Sequential:
    return Sequential(
        [
            Input(shape=INPUT_SHAPE),
            layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(NUM_CLASSES, activation="softmax"),
        ]
    )


def train_and_evaluate():
    configure_gpu()
    (x_train, y_train), (x_test, y_test) = load_data()

    model = build_model()
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    model.summary()

    model.fit(
        x_train,
        y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_split=VALIDATION_SPLIT,
        verbose=2,
    )

    score = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test loss: {score[0]:.4f}")
    print(f"Test accuracy: {score[1]:.4f}")

    model.save(EXPORT_PATH)
    print(f"Mô hình đã được lưu tại '{EXPORT_PATH}'")


if __name__ == "__main__":
    train_and_evaluate()
