from __future__ import annotations

from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight


def load_image_datasets(
    train_dir: str | Path,
    val_dir: str | Path,
    image_size: tuple[int, int],
    batch_size: int,
    seed: int,
) -> tuple[tf.data.Dataset, tf.data.Dataset, list[str]]:
    train_dir = Path(train_dir)
    val_dir = Path(val_dir)

    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        labels="inferred",
        label_mode="categorical",
        image_size=image_size,
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
    )

    class_names = list(train_ds.class_names)

    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir,
        labels="inferred",
        label_mode="categorical",
        image_size=image_size,
        batch_size=batch_size,
        shuffle=False,
    )

    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.cache().prefetch(autotune)
    val_ds = val_ds.cache().prefetch(autotune)
    return train_ds, val_ds, class_names


def compute_class_weights_from_directory(train_dir: str | Path, class_names: list[str]) -> dict[int, float]:
    """
    Compute sklearn-style class weights from class folder counts.
    Returns a dict that Keras accepts as `class_weight={class_index: weight}`.
    """
    train_dir = Path(train_dir)
    counts = []
    for name in class_names:
        p = train_dir / name
        if not p.exists():
            counts.append(0)
        else:
            counts.append(len([x for x in p.iterdir() if x.is_file()]))

    y = []
    for idx, c in enumerate(counts):
        y.extend([idx] * c)

    if len(y) == 0:
        raise ValueError(f"No training images found under {train_dir}.")

    classes = np.arange(len(class_names))
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=np.array(y))
    return {int(i): float(w) for i, w in zip(classes, weights)}

