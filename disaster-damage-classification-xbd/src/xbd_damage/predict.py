from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image


def load_image(path: str | Path, image_size: tuple[int, int]) -> np.ndarray:
    img = Image.open(path).convert("RGB").resize((image_size[1], image_size[0]))
    arr = np.asarray(img, dtype=np.float32)
    return arr


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict damage class for a single image.")
    parser.add_argument("--model", required=True, type=str, help="Path to .keras model file")
    parser.add_argument("--image", required=True, type=str, help="Path to image")
    parser.add_argument("--class-names", default="Minor,Major,Destroyed", type=str, help="Comma-separated names")
    parser.add_argument("--image-size", default="128,128", type=str, help="HxW, e.g. 128,128")
    args = parser.parse_args()

    h, w = [int(x.strip()) for x in args.image_size.split(",")]
    class_names = [x.strip() for x in args.class_names.split(",") if x.strip()]

    model = tf.keras.models.load_model(args.model)
    x = load_image(args.image, (h, w))
    x = np.expand_dims(x, axis=0)

    probs = model.predict(x, verbose=0)[0]
    idx = int(np.argmax(probs))
    label = class_names[idx] if idx < len(class_names) else str(idx)

    print(f"pred_class={label}")
    print(f"prob={float(probs[idx]):.6f}")
    print("probs=" + ", ".join([f"{p:.6f}" for p in probs.tolist()]))

