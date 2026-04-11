"""
backend/model.py
================
Production plant disease classifier.

Loads the trained MobileNetV2 model and exposes a clean
predict() interface that the FastAPI backend consumes.

Falls back gracefully to a keyword-based heuristic when the
model file is absent (e.g., during CI or first-run before training).
"""

from __future__ import annotations

import json
import logging
from io import BytesIO
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, UnidentifiedImageError

logger = logging.getLogger(__name__)

# ─── Paths ────────────────────────────────────────────────────────────────────
_BASE_DIR      = Path(__file__).resolve().parent.parent
_MODEL_PATH    = _BASE_DIR / "models" / "plant_model.h5"
_METADATA_PATH = _BASE_DIR / "models" / "model_metadata.json"

# ─── Constants ────────────────────────────────────────────────────────────────
IMAGE_SIZE = (224, 224)

# Human-readable display names (key = folder / class name used during training)
DISPLAY_NAMES: dict[str, str] = {
    "tomato_early_blight":    "Tomato Early Blight",
    "tomato_late_blight":     "Tomato Late Blight",
    "tomato_leaf_mold":       "Tomato Leaf Mold",
    "tomato_healthy":         "Healthy Tomato",
    "banana_sigatoka":        "Banana Sigatoka",
    "banana_panama_disease":  "Banana Panama Disease",
    "banana_healthy":         "Healthy Banana",
}

# Minimum softmax probability to trust a prediction.
CONFIDENCE_THRESHOLD = 0.50


# ─── Image preprocessing ──────────────────────────────────────────────────────

def preprocess_image(image_bytes: bytes) -> Optional[np.ndarray]:
    """
    Decode bytes → PIL Image → normalised NumPy array (1, 224, 224, 3).
    Returns None on failure.
    """
    try:
        with Image.open(BytesIO(image_bytes)) as img:
            img = img.convert("RGB").resize(IMAGE_SIZE, Image.LANCZOS)
            arr = np.array(img, dtype=np.float32) / 255.0
            return np.expand_dims(arr, axis=0)   # shape: (1, 224, 224, 3)
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        logger.warning("Image decode failed: %s", exc)
        return None


# ─── Main classifier ──────────────────────────────────────────────────────────

class PlantDiseaseModel:
    """
    Wraps the trained Keras model with safe loading and a simple predict API.

    Usage
    -----
    model = PlantDiseaseModel()
    result = model.predict(image_bytes=b"...", filename="leaf.jpg")
    # result → {"disease": "Tomato Early Blight", "confidence": "87%",
    #            "class_key": "tomato_early_blight", "model_used": "cnn"}
    """

    def __init__(
        self,
        model_path: Path = _MODEL_PATH,
        metadata_path: Path = _METADATA_PATH,
    ) -> None:
        self._model      = None
        self._index_map: dict[str, str] = {}  # "0" → "tomato_early_blight"
        self._model_path = model_path
        self._load(model_path, metadata_path)

    # ── Loading ───────────────────────────────────────────────────────────────

    def _load(self, model_path: Path, metadata_path: Path) -> None:
        if not model_path.exists():
            logger.warning(
                "Model not found at %s — running in fallback mode. "
                "Train the model first:  python train.py",
                model_path,
            )
            return

        try:
            import tensorflow as tf  # lazy import: not required for fallback
            self._model = tf.keras.models.load_model(str(model_path))
            logger.info("✓ Model loaded from %s", model_path)
        except Exception as exc:
            logger.error("Failed to load model: %s", exc)
            self._model = None
            return

        if metadata_path.exists():
            with metadata_path.open() as f:
                meta = json.load(f)
            self._index_map = meta.get("class_names", {})
            logger.info("✓ Metadata loaded — %d classes", len(self._index_map))
        else:
            # Build a default index map (alphabetical, matching Keras behaviour)
            sorted_keys = sorted(DISPLAY_NAMES.keys())
            self._index_map = {str(i): k for i, k in enumerate(sorted_keys)}
            logger.warning("Metadata file missing — using default class order.")

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict(self, image_bytes: bytes, filename: str = "image") -> dict:
        """
        Parameters
        ----------
        image_bytes : raw bytes of the uploaded image file.
        filename    : original filename (used for logging only).

        Returns
        -------
        dict with keys: disease, confidence, class_key, model_used
        """
        if self._model is None:
            return self._fallback_prediction(filename)

        arr = preprocess_image(image_bytes)
        if arr is None:
            return _unknown_result("Could not decode image.")

        try:
            probs = self._model.predict(arr, verbose=0)[0]   # shape: (num_classes,)
        except Exception as exc:
            logger.error("Inference error: %s", exc)
            return _unknown_result("Inference failed.")

        top_idx     = int(np.argmax(probs))
        top_prob    = float(probs[top_idx])
        class_key   = self._index_map.get(str(top_idx), "unknown")

        if top_prob < CONFIDENCE_THRESHOLD:
            return _low_confidence_result(top_prob)

        display = DISPLAY_NAMES.get(class_key, class_key.replace("_", " ").title())

        return {
            "disease":    display,
            "confidence": f"{top_prob:.0%}",
            "class_key":  class_key,
            "model_used": "cnn",
        }

    # ── Fallback (no model file) ───────────────────────────────────────────────

    def _fallback_prediction(self, filename: str) -> dict:
        """
        Simple pixel-checksum heuristic — keeps the demo usable before training.
        This is visibly labelled as a demo estimate.
        """
        name = filename.lower()
        if "banana" in name:
            keys = ["banana_healthy", "banana_sigatoka", "banana_panama_disease"]
        elif "tomato" in name:
            keys = ["tomato_healthy", "tomato_early_blight", "tomato_late_blight", "tomato_leaf_mold"]
        else:
            keys = list(DISPLAY_NAMES.keys())

        class_key = keys[hash(filename) % len(keys)]
        display   = DISPLAY_NAMES.get(class_key, "Unknown")

        return {
            "disease":    display,
            "confidence": "Demo (model not trained)",
            "class_key":  class_key,
            "model_used": "fallback",
        }


# ─── Helper result builders ───────────────────────────────────────────────────

def _unknown_result(reason: str = "") -> dict:
    return {
        "disease":    "Unknown",
        "confidence": "0%",
        "class_key":  "unknown",
        "model_used": "cnn",
        "note":       reason,
    }


def _low_confidence_result(prob: float) -> dict:
    return {
        "disease":    "Uncertain — please retake photo in better lighting",
        "confidence": f"{prob:.0%}",
        "class_key":  "uncertain",
        "model_used": "cnn",
    }


# ─── Legacy shim (keeps existing utils.py working) ────────────────────────────

class DummyPlantDiseaseModel(PlantDiseaseModel):
    """
    Alias kept so that existing imports in main.py do not break.
    PlantDiseaseModel is the real implementation now.
    """
    def __init__(self, model_path: Path = _MODEL_PATH, solutions: dict = None):
        super().__init__(model_path=model_path)
        self._solutions = solutions or {}

    def predict(self, image_bytes: bytes, filename: str = "image") -> str:  # type: ignore[override]
        """Returns just the disease string for backward compatibility."""
        result = super().predict(image_bytes=image_bytes, filename=filename)
        return result["disease"]


# ─── Standalone test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m backend.model <image_path>")
        sys.exit(1)

    img_path = Path(sys.argv[1])
    if not img_path.exists():
        print(f"File not found: {img_path}")
        sys.exit(1)

    model  = PlantDiseaseModel()
    result = model.predict(image_bytes=img_path.read_bytes(), filename=img_path.name)

    print("\n" + "─" * 40)
    print(f"  Disease    : {result['disease']}")
    print(f"  Confidence : {result['confidence']}")
    print(f"  Class key  : {result['class_key']}")
    print(f"  Model used : {result['model_used']}")
    print("─" * 40)