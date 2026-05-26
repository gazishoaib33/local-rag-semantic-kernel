from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf

from xbd_damage.utils.metrics import save_confusion_matrix, write_classification_report


def load_test_dataset(data_dir: str | Path, image_size: tuple[int, int], batch_size: int) -> tuple[tf.data.Dataset, list[str]]:
    ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        labels="inferred",
        label_mode="categorical",
        image_size=image_size,
        batch_size=batch_size,
        shuffle=False,
    )
    return ds.cache().prefetch(tf.data.AUTOTUNE), list(ds.class_names)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a saved model on a directory dataset.")
    parser.add_argument("--model", required=True, type=str, help="Path to .keras model file")
    parser.add_argument("--data-dir", required=True, type=str, help="Directory with class subfolders")
    parser.add_argument("--image-size", default="128,128", type=str, help="HxW, e.g. 128,128")
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--out-dir", default="artifacts", type=str)
    args = parser.parse_args()

    h, w = [int(x.strip()) for x in args.image_size.split(",")]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = tf.keras.models.load_model(args.model)
    ds, class_names = load_test_dataset(args.data_dir, (h, w), args.batch_size)

    y_true = []
    y_pred = []
    for batch_x, batch_y in ds:
        probs = model.predict(batch_x, verbose=0)
        y_pred.extend(np.argmax(probs, axis=1).tolist())
        y_true.extend(np.argmax(batch_y.numpy(), axis=1).tolist())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    save_confusion_matrix(y_true, y_pred, class_names, out_dir / "confusion_matrix.png")
    report = write_classification_report(y_true, y_pred, class_names, out_dir / "classification_report.txt")

    print(report)
    print(f"Saved: {out_dir / 'confusion_matrix.png'}")
    print(f"Saved: {out_dir / 'classification_report.txt'}")

