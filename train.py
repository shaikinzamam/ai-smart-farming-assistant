"""
AI Smart Farming Assistant — Plant Disease Detection Training Script
====================================================================
Crops    : Tomato, Banana
Model    : MobileNetV2 (transfer learning)
Framework: TensorFlow / Keras
Usage    : python train.py
           python train.py --data_dir dataset --epochs 20 --batch_size 32
"""

import argparse
import json
import sys
from pathlib import Path

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

# ─────────────────────────── Constants ────────────────────────────────────────

IMAGE_SIZE   = (224, 224)
BATCH_SIZE   = 32
EPOCHS_FROZEN  = 10   # phase 1: train only custom head
EPOCHS_FINETUNE = 10  # phase 2: fine-tune top layers of base model

CLASS_NAMES = [
    "banana_healthy",
    "banana_panama_disease",
    "banana_sigatoka",
    "tomato_early_blight",
    "tomato_healthy",
    "tomato_late_blight",
    "tomato_leaf_mold",
]

BASE_DIR   = Path(__file__).resolve().parent
MODEL_DIR  = BASE_DIR / "models"
LOG_DIR    = BASE_DIR / "logs"

MODEL_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)

MODEL_PATH     = MODEL_DIR / "plant_model.h5"
METADATA_PATH  = MODEL_DIR / "model_metadata.json"


# ─────────────────────────── Dataset helpers ──────────────────────────────────

def verify_dataset(data_dir: Path) -> None:
    """
    Verify the expected folder structure exists.

    Expected layout
    ---------------
    dataset/
      banana/
        healthy/
        panama_disease/
        sigatoka/
      tomato/
        early_blight/
        healthy/
        late_blight/
        leaf_mold/
    """
    required = {
        "banana/healthy",
        "banana/panama_disease",
        "banana/sigatoka",
        "tomato/early_blight",
        "tomato/healthy",
        "tomato/late_blight",
        "tomato/leaf_mold",
    }
    missing = [p for p in required if not (data_dir / p).is_dir()]
    if missing:
        print("ERROR — missing dataset folders:")
        for m in missing:
            print(f"  ✗  {data_dir / m}")
        print(
            "\nSee DATASET_PREPARATION.md for download instructions."
        )
        sys.exit(1)

    total_images = sum(
        1
        for p in data_dir.rglob("*")
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    )
    print(f"✓  Dataset verified — {total_images:,} images found in {data_dir}")


def build_data_generators(data_dir: Path, batch_size: int):
    """Return (train_gen, val_gen, class_indices) from the flat class folders."""

    # Keras flow_from_directory requires a single root with one subfolder per class.
    # We create a merged flat view: dataset_flat/banana_healthy/, dataset_flat/tomato_early_blight/ …
    # by symlinking (or copying). We use a real ImageDataGenerator with class_mode='categorical'.

    # Training augmentation — deliberately modest to avoid over-augmenting.
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=0.2,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.15,
        horizontal_flip=True,
        brightness_range=[0.85, 1.15],
        fill_mode="nearest",
    )

    val_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=0.2,
    )

    # Build a flat directory view so flow_from_directory produces the right labels.
    flat_dir = _build_flat_dir(data_dir)

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


def _build_flat_dir(data_dir: Path) -> Path:
    """
    Produce a 'flat view' directory with symlinks.

    dataset/tomato/early_blight/  → dataset_flat/tomato_early_blight/
    dataset/banana/sigatoka/      → dataset_flat/banana_sigatoka/
    """
    flat_dir = data_dir.parent / "dataset_flat"
    flat_dir.mkdir(exist_ok=True)

    mapping = {
        "tomato_early_blight":   data_dir / "tomato" / "early_blight",
        "tomato_late_blight":    data_dir / "tomato" / "late_blight",
        "tomato_leaf_mold":      data_dir / "tomato" / "leaf_mold",
        "tomato_healthy":        data_dir / "tomato" / "healthy",
        "banana_sigatoka":       data_dir / "banana" / "sigatoka",
        "banana_panama_disease": data_dir / "banana" / "panama_disease",
        "banana_healthy":        data_dir / "banana" / "healthy",
    }

    for class_name, source_dir in mapping.items():
        link = flat_dir / class_name
        if not link.exists():
            link.symlink_to(source_dir.resolve())

    return flat_dir


# ─────────────────────────── Model ────────────────────────────────────────────

def build_model(num_classes: int) -> tf.keras.Model:
    """
    MobileNetV2 backbone + custom classification head.

    Phase 1: backbone frozen → train only the head.
    Phase 2: unfreeze top 30 layers → fine-tune.
    """
    base_model = MobileNetV2(
        input_shape=(*IMAGE_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False  # frozen for phase 1

    inputs = layers.Input(shape=(*IMAGE_SIZE, 3))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.35)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.25)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs, name="plant_disease_classifier")
    return model, base_model


def compile_model(model: tf.keras.Model, learning_rate: float = 1e-3):
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy", tf.keras.metrics.TopKCategoricalAccuracy(k=2, name="top2_acc")],
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
        TensorBoard(log_dir=str(LOG_DIR / phase), histogram_freq=0),
    ]


# ─────────────────────────── Training ─────────────────────────────────────────

def train(data_dir: Path, epochs_frozen: int, epochs_finetune: int, batch_size: int):
    verify_dataset(data_dir)

    train_gen, val_gen, class_indices = build_data_generators(data_dir, batch_size)
    num_classes = len(class_indices)

    print(f"\n{'─'*55}")
    print(f"  Classes  : {num_classes}")
    print(f"  Training : {train_gen.samples:,} images")
    print(f"  Validation: {val_gen.samples:,} images")
    print(f"  Image size: {IMAGE_SIZE}")
    print(f"  Batch size: {batch_size}")
    print(f"{'─'*55}\n")

    model, base_model = build_model(num_classes)
    compile_model(model, learning_rate=1e-3)
    model.summary()

    # ── Phase 1: Train only the custom head ──────────────────────────────────
    print("\n[Phase 1] Training classification head (backbone frozen) …\n")
    history_phase1 = model.fit(
        train_gen,
        epochs=epochs_frozen,
        validation_data=val_gen,
        callbacks=get_callbacks("phase1"),
        verbose=1,
    )

    # ── Phase 2: Fine-tune top layers of MobileNetV2 ─────────────────────────
    print("\n[Phase 2] Fine-tuning top 30 backbone layers …\n")
    base_model.trainable = True
    for layer in base_model.layers[:-30]:
        layer.trainable = False

    compile_model(model, learning_rate=1e-4)  # lower LR for fine-tuning

    history_phase2 = model.fit(
        train_gen,
        epochs=epochs_finetune,
        validation_data=val_gen,
        callbacks=get_callbacks("phase2"),
        verbose=1,
    )

    # ── Save final model ──────────────────────────────────────────────────────
    model.save(str(MODEL_PATH))
    print(f"\n✓  Model saved → {MODEL_PATH}")

    # ── Save metadata (class map + training summary) ──────────────────────────
    # Invert class_indices so index → class name
    index_to_class = {str(v): k for k, v in class_indices.items()}

    final_val_acc = max(history_phase2.history["val_accuracy"])
    metadata = {
        "class_names": index_to_class,
        "num_classes": num_classes,
        "image_size": list(IMAGE_SIZE),
        "best_val_accuracy": round(float(final_val_acc), 4),
        "model_path": str(MODEL_PATH),
    }

    with METADATA_PATH.open("w") as f:
        json.dump(metadata, f, indent=2)

    print(f"✓  Metadata saved → {METADATA_PATH}")
    print(f"\n🌿  Best validation accuracy: {final_val_acc:.2%}")

    return model, metadata


# ─────────────────────────── Entry point ──────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Train plant disease detection model")
    parser.add_argument("--data_dir",      default="dataset",    help="Root dataset directory")
    parser.add_argument("--epochs",        type=int, default=10, help="Epochs per phase (frozen + finetune)")
    parser.add_argument("--batch_size",    type=int, default=32, help="Batch size")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    data_path = Path(args.data_dir)

    if not data_path.is_dir():
        print(f"ERROR — dataset directory not found: {data_path}")
        print("Run:  python prepare_dataset.py  first.")
        sys.exit(1)

    train(
        data_dir=data_path,
        epochs_frozen=args.epochs,
        epochs_finetune=args.epochs,
        batch_size=args.batch_size,
    )