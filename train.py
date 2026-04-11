"""
AI Smart Farming Assistant — Plant Disease Detection Training Script
====================================================================
Crops    : Tomato, Banana
Model    : MobileNetV2 (transfer learning)
Framework: TensorFlow / Keras

Usage
-----
  python train.py
  python train.py --data_dir dataset --epochs 15 --batch_size 16
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

# ─────────────────────────── Early dependency check ───────────────────────────

try:
    import numpy as np
    import tensorflow as tf
    from tensorflow.keras import layers, models, optimizers
    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras.callbacks import (
        EarlyStopping,
        ModelCheckpoint,
        ReduceLROnPlateau,
        TensorBoard,
    )
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
except ImportError as e:
    print(f"\nERROR — missing package: {e}")
    print("Fix:  pip install -r requirements.txt\n")
    sys.exit(1)

# ─────────────────────────── Constants ────────────────────────────────────────

IMAGE_SIZE     = (224, 224)
DEFAULT_BATCH  = 32
DEFAULT_EPOCHS = 10

BASE_DIR      = Path(__file__).resolve().parent
MODEL_DIR     = BASE_DIR / "models"
LOG_DIR       = BASE_DIR / "logs"
FLAT_DIR      = BASE_DIR / "dataset_flat"   # auto-created working copy

MODEL_PATH    = MODEL_DIR / "plant_model.h5"
METADATA_PATH = MODEL_DIR / "model_metadata.json"

# Mapping:  flat class name  →  (crop_folder, class_subfolder)
CLASS_MAPPING = {
    "tomato_early_blight":   ("tomato", "early_blight"),
    "tomato_late_blight":    ("tomato", "late_blight"),
    "tomato_leaf_mold":      ("tomato", "leaf_mold"),
    "tomato_healthy":        ("tomato", "healthy"),
    "banana_sigatoka":       ("banana", "sigatoka"),
    "banana_panama_disease": ("banana", "panama_disease"),
    "banana_healthy":        ("banana", "healthy"),
}

# ─────────────────────────── Dataset helpers ──────────────────────────────────

def verify_dataset(data_dir: Path) -> None:
    """
    Check all required subfolders exist and contain images.

    Expected layout
    ---------------
    dataset/
      tomato/  early_blight/  late_blight/  leaf_mold/  healthy/
      banana/  sigatoka/      panama_disease/            healthy/
    """
    missing = []
    empty   = []

    for class_name, (crop, cls) in CLASS_MAPPING.items():
        folder = data_dir / crop / cls
        if not folder.is_dir():
            missing.append(f"{crop}/{cls}")
        else:
            imgs = [
                p for p in folder.iterdir()
                if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
            ]
            if len(imgs) == 0:
                empty.append(f"{crop}/{cls}")

    if missing:
        print("\nERROR — missing dataset folders:")
        for m in missing:
            print(f"  ✗  dataset/{m}")
        print("\nSee DATASET_PREPARATION.md for download instructions.")
        sys.exit(1)

    if empty:
        print("\nERROR — these folders contain no images:")
        for e in empty:
            print(f"  ✗  dataset/{e}")
        sys.exit(1)

    total = sum(
        1 for p in data_dir.rglob("*")
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    )
    print(f"✓  Dataset verified — {total:,} images found in {data_dir}")


def _link_or_copy(src: Path, dst: Path) -> None:
    """
    Try hard-link first (fast, zero extra disk space).
    Fall back to regular file copy if hard-links are not supported
    (e.g. across drives on Windows, or Python < 3.10).
    """
    try:
        # Python 3.10+
        dst.hardlink_to(src)
    except AttributeError:
        # Python 3.8 / 3.9
        try:
            os.link(str(src), str(dst))
        except OSError:
            shutil.copy2(src, dst)
    except OSError:
        # Different drives or filesystem limitation
        shutil.copy2(src, dst)


def build_flat_dir(data_dir: Path) -> Path:
    """
    Build a flat copy of the dataset so Keras flow_from_directory works.

    dataset/tomato/early_blight/img.jpg
        → dataset_flat/tomato_early_blight/img.jpg

    This avoids symlinks entirely (symlinks break on Windows without
    Developer Mode enabled).
    """
    FLAT_DIR.mkdir(exist_ok=True)

    for class_name, (crop, cls) in CLASS_MAPPING.items():
        src_dir  = data_dir / crop / cls
        dest_dir = FLAT_DIR / class_name

        # Count source images
        src_images = [
            p for p in src_dir.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
        ]

        # Skip rebuild if destination is already up to date
        if dest_dir.is_dir():
            existing = sum(1 for _ in dest_dir.glob("*") if _.is_file())
            if existing >= len(src_images):
                print(f"  ✓  {class_name:<30} {existing} images  (cached)")
                continue
            shutil.rmtree(dest_dir)   # stale — full rebuild

        dest_dir.mkdir(exist_ok=True)
        copied = 0

        for img_path in src_images:
            dest_file = dest_dir / img_path.name
            _link_or_copy(img_path, dest_file)
            copied += 1

        print(f"  ✓  {class_name:<30} {copied} images")

    return FLAT_DIR


def build_data_generators(data_dir: Path, batch_size: int):
    """Return (train_gen, val_gen, class_indices)."""
    print("\nPreparing flat dataset view …")
    flat_dir = build_flat_dir(data_dir)

    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=0.2,
        rotation_range=20,
        width_shift_range=0.10,
        height_shift_range=0.10,
        shear_range=0.10,
        zoom_range=0.15,
        horizontal_flip=True,
        brightness_range=[0.85, 1.15],
        fill_mode="nearest",
    )

    val_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=0.2,
    )

    train_gen = train_datagen.flow_from_directory(
        flat_dir,
        target_size=IMAGE_SIZE,
        batch_size=batch_size,
        class_mode="categorical",
        subset="training",
        shuffle=True,
        seed=42,
    )

    val_gen = val_datagen.flow_from_directory(
        flat_dir,
        target_size=IMAGE_SIZE,
        batch_size=batch_size,
        class_mode="categorical",
        subset="validation",
        shuffle=False,
        seed=42,
    )

    return train_gen, val_gen, train_gen.class_indices


# ─────────────────────────── Model ────────────────────────────────────────────

def build_model(num_classes: int):
    """
    MobileNetV2 backbone + custom head.
    Returns (model, base_model).
    base_model is frozen for Phase 1, partially unfrozen for Phase 2.
    """
    base_model = MobileNetV2(
        input_shape=(*IMAGE_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False   # frozen for phase 1

    inputs  = layers.Input(shape=(*IMAGE_SIZE, 3))
    x       = base_model(inputs, training=False)
    x       = layers.GlobalAveragePooling2D()(x)
    x       = layers.BatchNormalization()(x)
    x       = layers.Dense(256, activation="relu")(x)
    x       = layers.Dropout(0.35)(x)
    x       = layers.Dense(128, activation="relu")(x)
    x       = layers.Dropout(0.25)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs, name="plant_disease_classifier")
    return model, base_model


def compile_model(model, learning_rate: float = 1e-3) -> None:
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.TopKCategoricalAccuracy(k=2, name="top2_acc"),
        ],
    )


# ─────────────────────────── Callbacks ────────────────────────────────────────

def get_callbacks(phase: str) -> list:
    return [
        EarlyStopping(
            monitor="val_accuracy",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        ModelCheckpoint(
            filepath=str(MODEL_DIR / f"checkpoint_{phase}.h5"),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.4,
            patience=3,
            min_lr=1e-7,
            verbose=1,
        ),
        TensorBoard(
            log_dir=str(LOG_DIR / phase),
            histogram_freq=0,
        ),
    ]


# ─────────────────────────── Training ─────────────────────────────────────────

def train(data_dir: Path, epochs_frozen: int, epochs_finetune: int, batch_size: int):
    """Full two-phase training pipeline."""

    # Create output directories
    MODEL_DIR.mkdir(exist_ok=True)
    LOG_DIR.mkdir(exist_ok=True)

    # Validate + load data
    verify_dataset(data_dir)
    train_gen, val_gen, class_indices = build_data_generators(data_dir, batch_size)
    num_classes = len(class_indices)

    print(f"\n{'─' * 55}")
    print(f"  Classes    : {num_classes}")
    print(f"  Training   : {train_gen.samples:,} images")
    print(f"  Validation : {val_gen.samples:,} images")
    print(f"  Image size : {IMAGE_SIZE}")
    print(f"  Batch size : {batch_size}")
    print(f"{'─' * 55}\n")

    # Build model
    model, base_model = build_model(num_classes)
    compile_model(model, learning_rate=1e-3)
    model.summary()

    # ── Phase 1: train head only (backbone frozen) ────────────────────────────
    print("\n[Phase 1] Training classification head — backbone frozen …\n")
    model.fit(
        train_gen,
        epochs=epochs_frozen,
        validation_data=val_gen,
        callbacks=get_callbacks("phase1"),
        verbose=1,
    )

    # ── Phase 2: fine-tune top 30 backbone layers ─────────────────────────────
    print("\n[Phase 2] Fine-tuning top 30 MobileNetV2 layers …\n")
    base_model.trainable = True
    for layer in base_model.layers[:-30]:
        layer.trainable = False

    compile_model(model, learning_rate=1e-4)    # 10× lower LR for fine-tuning

    history2 = model.fit(
        train_gen,
        epochs=epochs_finetune,
        validation_data=val_gen,
        callbacks=get_callbacks("phase2"),
        verbose=1,
    )

    # ── Save model ────────────────────────────────────────────────────────────
    model.save(str(MODEL_PATH))
    print(f"\n✓  Model saved  →  {MODEL_PATH}")

    # ── Save metadata ─────────────────────────────────────────────────────────
    index_to_class = {str(v): k for k, v in class_indices.items()}
    best_val_acc   = max(history2.history["val_accuracy"])

    metadata = {
        "class_names":       index_to_class,
        "num_classes":       num_classes,
        "image_size":        list(IMAGE_SIZE),
        "best_val_accuracy": round(float(best_val_acc), 4),
        "model_path":        str(MODEL_PATH),
    }

    with METADATA_PATH.open("w") as f:
        json.dump(metadata, f, indent=2)

    print(f"✓  Metadata saved  →  {METADATA_PATH}")
    print(f"\n🌿  Best validation accuracy : {best_val_acc:.2%}")
    print("\n✅  Training complete!  Start the backend:")
    print("      uvicorn backend.main:app --reload\n")

    return model, metadata


# ─────────────────────────── CLI ──────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the AI Smart Farming plant disease model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data_dir",
        default="dataset",
        help="Root dataset directory (must contain tomato/ and banana/ subfolders)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help="Epochs for EACH training phase (phase1 + phase2)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=DEFAULT_BATCH,
        help="Batch size — use 16 on CPU, 32 on GPU",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args      = parse_args()
    data_path = Path(args.data_dir)

    if not data_path.is_dir():
        print(f"\nERROR — dataset directory not found: {data_path.resolve()}")
        print("Run:  python prepare_dataset.py   to download and organise images.\n")
        sys.exit(1)

    train(
        data_dir=data_path,
        epochs_frozen=args.epochs,
        epochs_finetune=args.epochs,
        batch_size=args.batch_size,
    )