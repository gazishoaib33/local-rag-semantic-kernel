from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf

from xbd_damage.data.dataset import compute_class_weights_from_directory, load_image_datasets
from xbd_damage.models.cnn import build_small_cnn
from xbd_damage.utils.config import load_config
from xbd_damage.utils.metrics import save_confusion_matrix, save_training_curves, write_classification_report
from xbd_damage.utils.seed import set_seed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train damage classifier on xBD-style directory dataset.")
    parser.add_argument("--config", type=str, default="configs/train.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.seed)

    cfg.artifacts.dir.mkdir(parents=True, exist_ok=True)

    train_ds, val_ds, class_names = load_image_datasets(
        train_dir=cfg.data.train_dir,
        val_dir=cfg.data.val_dir,
        image_size=cfg.data.image_size,
        batch_size=cfg.data.batch_size,
        seed=cfg.seed,
    )

    class_weight = compute_class_weights_from_directory(cfg.data.train_dir, class_names)

    model = build_small_cnn(
        image_size=cfg.data.image_size,
        num_classes=len(class_names),
        dropout=cfg.model.dropout,
        learning_rate=cfg.train.learning_rate,
        label_smoothing=cfg.train.label_smoothing,
    )

    callbacks: list[tf.keras.callbacks.Callback] = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=cfg.train.early_stopping_patience,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(cfg.artifacts.model_path),
            monitor="val_loss",
            save_best_only=True,
        ),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=cfg.train.epochs,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1,
    )

    # Save curves
    save_training_curves(history, cfg.artifacts.dir / "training_curves.png")

    # Evaluate on val and save confusion matrix + report
    y_true = []
    y_pred = []
    for batch_x, batch_y in val_ds:
        probs = model.predict(batch_x, verbose=0)
        y_pred.extend(np.argmax(probs, axis=1).tolist())
        y_true.extend(np.argmax(batch_y.numpy(), axis=1).tolist())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    save_confusion_matrix(y_true, y_pred, class_names, cfg.artifacts.dir / "confusion_matrix_val.png")
    write_classification_report(y_true, y_pred, class_names, cfg.artifacts.dir / "classification_report_val.txt")

    print(f"Saved best model to: {cfg.artifacts.model_path}")
    print(f"Artifacts directory: {cfg.artifacts.dir}")

