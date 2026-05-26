from __future__ import annotations

import tensorflow as tf


def build_small_cnn(
    image_size: tuple[int, int],
    num_classes: int,
    dropout: float = 0.3,
    learning_rate: float = 3e-4,
    label_smoothing: float = 0.0,
) -> tf.keras.Model:
    h, w = image_size
    inputs = tf.keras.Input(shape=(h, w, 3))

    # Augmentation is part of the model so it's applied consistently in training.
    x = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.08),
            tf.keras.layers.RandomZoom(0.10),
            tf.keras.layers.RandomContrast(0.10),
        ],
        name="augment",
    )(inputs)

    x = tf.keras.layers.Rescaling(1.0 / 255.0)(x)

    for filters in (32, 64, 128):
        x = tf.keras.layers.Conv2D(filters, 3, padding="same")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation("relu")(x)
        x = tf.keras.layers.MaxPooling2D()(x)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="small_cnn")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing),
        metrics=["accuracy"],
    )
    return model

