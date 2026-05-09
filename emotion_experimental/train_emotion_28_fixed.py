#!/usr/bin/env python3
"""Train 28-class emotion classifier."""

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, Input, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator


EMOTION_LABELS = [
    "neutral", "happy", "sad", "surprise", "fear", "anger", "disgust",
    "happily_surprised", "happily_disgusted", "contentment", "pride", "admiration", "awe",
    "sadly_fearful", "sadly_angry", "sadly_surprised", "sadly_disgusted",
    "fearfully_angry", "fearfully_surprised", "fearfully_disgusted",
    "angrily_surprised", "angrily_disgusted", "disgustedly_surprised",
    "contempt", "embarrassment", "boredom", "confusion", "pain"
]


@dataclass
class TrainConfig:
    train_dir: Path
    test_dir: Path
    output_model: Path
    metrics_path: Path
    confusion_matrix_csv: Path
    epochs: int
    batch_size: int
    image_size: int
    learning_rate: float
    patience: int
    use_augmentation: bool
    use_class_weights: bool
    seed: int


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def _match_class_dirs(root: Path, canonical_labels: List[str]) -> List[str]:
    if not root.exists():
        raise FileNotFoundError(f"Data directory not found: {root}")
    subdirs = [p.name for p in root.iterdir() if p.is_dir()]
    lower_to_actual: Dict[str, List[str]] = {}
    for name in subdirs:
        lower_to_actual.setdefault(name.lower(), []).append(name)

    ordered: List[str] = []
    for label in canonical_labels:
        matches = lower_to_actual.get(label, [])
        if len(matches) != 1:
            raise ValueError(
                f"Expected exactly one subdirectory for class '{label}' under {root}, found {matches or 'none'}"
            )
        ordered.append(matches[0])
    return ordered


def _build_model(image_size: int, num_classes: int, learning_rate: float) -> tf.keras.Model:
    model = Sequential(
        [
            Input(shape=(image_size, image_size, 1)),
            Conv2D(32, (3, 3), activation="relu", padding="same"),
            Conv2D(64, (3, 3), activation="relu", padding="same"),
            MaxPooling2D((2, 2)),
            Dropout(0.25),
            Conv2D(128, (3, 3), activation="relu", padding="same"),
            MaxPooling2D((2, 2)),
            Dropout(0.25),
            Conv2D(256, (3, 3), activation="relu", padding="same"),
            MaxPooling2D((2, 2)),
            Dropout(0.25),
            Flatten(),
            Dense(512, activation="relu"),
            Dropout(0.5),
            Dense(256, activation="relu"),
            Dropout(0.3),
            Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def _steps_for(generator) -> int:
    return int(math.ceil(generator.samples / float(generator.batch_size)))


def _save_confusion_matrix_csv(path: Path, cm: np.ndarray, labels: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("," + ",".join(labels) + "\n")
        for i, row in enumerate(cm):
            f.write(labels[i] + "," + ",".join(str(int(x)) for x in row) + "\n")


def train_and_evaluate(config: TrainConfig) -> Dict[str, object]:
    _seed_everything(config.seed)
    config.output_model.parent.mkdir(parents=True, exist_ok=True)
    config.metrics_path.parent.mkdir(parents=True, exist_ok=True)

    train_classes = _match_class_dirs(config.train_dir, EMOTION_LABELS)
    test_classes = _match_class_dirs(config.test_dir, EMOTION_LABELS)

    if config.use_augmentation:
        train_datagen = ImageDataGenerator(
            rescale=1.0 / 255.0,
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            zoom_range=0.1,
        )
    else:
        train_datagen = ImageDataGenerator(rescale=1.0 / 255.0)
    eval_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

    train_gen = train_datagen.flow_from_directory(
        str(config.train_dir),
        classes=train_classes,
        target_size=(config.image_size, config.image_size),
        color_mode="grayscale",
        class_mode="categorical",
        batch_size=config.batch_size,
        shuffle=True,
        seed=config.seed,
    )
    val_gen = eval_datagen.flow_from_directory(
        str(config.test_dir),
        classes=test_classes,
        target_size=(config.image_size, config.image_size),
        color_mode="grayscale",
        class_mode="categorical",
        batch_size=config.batch_size,
        shuffle=False,
    )

    if train_gen.samples <= 0 or val_gen.samples <= 0:
        raise ValueError(
            "No images found in train/test directories. "
            f"Found train={train_gen.samples}, test={val_gen.samples}. "
            "Place image files under each class folder."
        )

    model = _build_model(
        image_size=config.image_size,
        num_classes=len(EMOTION_LABELS),
        learning_rate=config.learning_rate,
    )
    checkpoint_path = config.output_model.with_suffix(".best.keras")
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=config.patience, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=max(2, config.patience // 3), min_lr=1e-6),
        ModelCheckpoint(filepath=str(checkpoint_path), monitor="val_loss", save_best_only=True),
    ]

    class_weight = None
    if config.use_class_weights:
        y_train = train_gen.classes
        unique = np.unique(y_train).astype(np.int64)
        if unique.size > 0:
            weights = compute_class_weight(class_weight="balanced", classes=unique, y=y_train)
            class_weight = {int(k): float(v) for k, v in zip(unique, weights)}

    history = model.fit(
        train_gen,
        epochs=config.epochs,
        validation_data=val_gen,
        callbacks=callbacks,
        class_weight=class_weight,
        steps_per_epoch=_steps_for(train_gen),
        validation_steps=_steps_for(val_gen),
        verbose=1,
    )

    val_gen.reset()
    y_prob = model.predict(val_gen, steps=_steps_for(val_gen), verbose=1)
    y_pred = np.argmax(y_prob, axis=1)[: len(val_gen.classes)]
    y_true = val_gen.classes

    cm = confusion_matrix(y_true, y_pred)
    report_dict = classification_report(y_true, y_pred, target_names=EMOTION_LABELS, output_dict=True)
    report_text = classification_report(y_true, y_pred, target_names=EMOTION_LABELS, digits=4)

    model.save(str(config.output_model))
    _save_confusion_matrix_csv(config.confusion_matrix_csv, cm, EMOTION_LABELS)

    metrics = {
        "model_path": str(config.output_model),
        "checkpoint_path": str(checkpoint_path),
        "train_dir": str(config.train_dir),
        "test_dir": str(config.test_dir),
        "class_order": EMOTION_LABELS,
        "samples_train": int(train_gen.samples),
        "samples_test": int(val_gen.samples),
        "epochs_ran": int(len(history.history.get("loss", []))),
        "best_val_loss": float(min(history.history.get("val_loss", [0.0]))),
        "best_val_accuracy": float(max(history.history.get("val_accuracy", [0.0]))),
        "classification_report": report_dict,
    }
    with config.metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("\n=== Evaluation (test, shuffle=False) ===")
    print(report_text)
    print(f"\nSaved model: {config.output_model}")
    print(f"Saved metrics JSON: {config.metrics_path}")
    print(f"Saved confusion matrix CSV: {config.confusion_matrix_csv}")
    return metrics


def _repo_root() -> Path:
    return Path(__file__).resolve().parent


def _parse_args() -> argparse.Namespace:
    root = _repo_root()
    parser = argparse.ArgumentParser(description="Train 28-class emotion classifier")
    parser.add_argument("--train-dir", type=Path, default=root / "data" / "emotions_28" / "train")
    parser.add_argument("--test-dir", type=Path, default=root / "data" / "emotions_28" / "test")
    parser>
</xai:function_call name="execute_command">
<parameter name="command">python3 train_emotion_28_fixed.py
