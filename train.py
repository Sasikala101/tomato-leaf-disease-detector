"""Train a transfer-learning classifier from data/train and data/validation."""
from pathlib import Path

import tensorflow as tf

from app.diseases import CLASS_NAMES

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 12


def main():
    train = tf.keras.utils.image_dataset_from_directory(
        "data/train", class_names=CLASS_NAMES, image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE, label_mode="categorical", seed=42,
    )
    validation = tf.keras.utils.image_dataset_from_directory(
        "data/validation", class_names=CLASS_NAMES, image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE, label_mode="categorical", shuffle=False,
    )
    augment = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.08),
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.RandomContrast(0.1),
    ])
    base = tf.keras.applications.MobileNetV2(
        input_shape=IMAGE_SIZE + (3,), include_top=False, weights="imagenet"
    )
    base.trainable = False
    inputs = tf.keras.Input(shape=IMAGE_SIZE + (3,))
    x = augment(inputs)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    outputs = tf.keras.layers.Dense(len(CLASS_NAMES), activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(patience=2),
    ]
    model.fit(train, validation_data=validation, epochs=EPOCHS, callbacks=callbacks)
    Path("models").mkdir(exist_ok=True)
    model.save("models/tomato_leaf_model.keras")
    loss, accuracy = model.evaluate(validation)
    print({"validation_loss": loss, "validation_accuracy": accuracy})


if __name__ == "__main__":
    main()

