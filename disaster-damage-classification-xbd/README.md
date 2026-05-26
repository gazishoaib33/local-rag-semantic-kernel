# Disaster Damage Classification (xBD) — BSc Thesis

CNN · TensorFlow/Keras · OpenCV · Python

This repository recreates my BSc thesis project: **classifying building damage severity** from satellite imagery to support faster post-disaster resource allocation.

## What it does

- **Task**: multi-class classification of damage severity (**Minor / Major / Destroyed**)
- **Input**: post-disaster building image chips (or crops) derived from xBD/xView2 imagery
- **Output**: predicted class + probability

## Dataset: xBD (xView2)

xBD is a large-scale dataset for building damage assessment from satellite imagery. The accompanying paper describes:

- Pre- and post-event satellite imagery
- Building polygons with **ordinal damage labels**
- Large-scale annotations (hundreds of thousands of buildings)

If you use this repo, please cite the original dataset/paper and follow xBD/xView2 terms.

### Label mapping (this thesis repo)

This repo focuses on **3 classes** for severity:

- `Minor`
- `Major`
- `Destroyed`

If your xBD export includes additional categories (e.g., `No-damage`), either remove them for this setup or extend the code/config to include them.

## Repo structure

```
disaster-damage-classification-xbd/
├─ configs/
│  └─ train.yaml
├─ src/
│  └─ xbd_damage/
│     ├─ data/
│     │  ├─ dataset.py
│     │  └─ splits.py
│     ├─ models/
│     │  └─ cnn.py
│     ├─ utils/
│     │  ├─ config.py
│     │  ├─ metrics.py
│     │  └─ seed.py
│     ├─ evaluate.py
│     ├─ predict.py
│     └─ train.py
├─ assets/
├─ reports/
└─ requirements.txt
```

## Setup

```bash
cd disaster-damage-classification-xbd
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

## Data layout (expected)

This repo **does not** commit xBD data. Prepare your dataset into this simple folder layout:

```
data/
├─ train/
│  ├─ Minor/
│  ├─ Major/
│  └─ Destroyed/
├─ val/
│  ├─ Minor/
│  ├─ Major/
│  └─ Destroyed/
└─ test/
   ├─ Minor/
   ├─ Major/
   └─ Destroyed/
```

Each class folder contains images (e.g. `.png`, `.jpg`). Images can be chips/crops of buildings from the post-disaster imagery.

### How to build chips/crops from xBD (high level)

xBD provides building polygons and pre/post imagery. A typical workflow is:

- Select **post-disaster** image for each tile
- For each building polygon, crop an image chip (optionally with padding)
- Assign the chip the building damage label
- Export into `train/val/test` directories per class

This repo keeps that preprocessing step out-of-scope so you can plug in your preferred pipeline.

## Train (augmentation + class weights)

```bash
python -m xbd_damage.train --config configs/train.yaml
```

Outputs:
- Saved model in `artifacts/`
- Training curves and confusion matrix in `artifacts/`

## Evaluate

```bash
python -m xbd_damage.evaluate --model artifacts/model.keras --data-dir data/test
```

## Predict on one image

```bash
python -m xbd_damage.predict --model artifacts/model.keras --image path\to\image.png
```

## Notes (portfolio)

- **Imbalance handling**: class-weight balancing (computed from training set) + augmentation
- **Recommended metric**: macro-F1 / per-class recall (accuracy alone can be misleading on imbalanced data)

## Suggested thesis bullet (CV)

Built a TensorFlow CNN pipeline to classify building damage severity (**Minor/Major/Destroyed**) from xBD satellite imagery, handling severe class imbalance with augmentation + class-weighted training.

## License

MIT (same as the parent portfolio repo). If you need a different license for thesis artifacts, adjust accordingly.

