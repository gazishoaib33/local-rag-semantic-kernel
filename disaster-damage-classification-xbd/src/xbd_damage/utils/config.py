from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class DataConfig:
    train_dir: Path
    val_dir: Path
    image_size: tuple[int, int]
    batch_size: int


@dataclass(frozen=True)
class ModelConfig:
    name: str
    dropout: float


@dataclass(frozen=True)
class TrainConfig:
    epochs: int
    learning_rate: float
    label_smoothing: float
    early_stopping_patience: int


@dataclass(frozen=True)
class ArtifactsConfig:
    dir: Path
    model_path: Path


@dataclass(frozen=True)
class Config:
    seed: int
    data: DataConfig
    model: ModelConfig
    train: TrainConfig
    artifacts: ArtifactsConfig


def _as_path(v: Any) -> Path:
    return Path(v).expanduser().resolve()


def load_config(path: str | Path) -> Config:
    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))

    data = raw["data"]
    model = raw["model"]
    train = raw["train"]
    artifacts = raw["artifacts"]

    return Config(
        seed=int(raw.get("seed", 42)),
        data=DataConfig(
            train_dir=_as_path(data["train_dir"]),
            val_dir=_as_path(data["val_dir"]),
            image_size=(int(data["image_size"][0]), int(data["image_size"][1])),
            batch_size=int(data["batch_size"]),
        ),
        model=ModelConfig(name=str(model["name"]), dropout=float(model["dropout"])),
        train=TrainConfig(
            epochs=int(train["epochs"]),
            learning_rate=float(train["learning_rate"]),
            label_smoothing=float(train.get("label_smoothing", 0.0)),
            early_stopping_patience=int(train.get("early_stopping_patience", 5)),
        ),
        artifacts=ArtifactsConfig(
            dir=_as_path(artifacts["dir"]),
            model_path=_as_path(artifacts["model_path"]),
        ),
    )

