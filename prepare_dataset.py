"""
prepare_dataset.py
==================
Downloads and organises images for the AI Smart Farming Assistant.

Sources
-------
  Tomato  — PlantVillage dataset (via Kaggle)
  Banana  — Banana Disease Recognition dataset (via Kaggle)

Prerequisites
-------------
  pip install kaggle Pillow tqdm
  Place kaggle.json in ~/.kaggle/  (download from kaggle.com → Account → API)
"""

import os
import shutil
import zipfile
from pathlib import Path

from tqdm import tqdm

BASE_DIR   = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"

# ─────────────────────────── Folder map ───────────────────────────────────────
# Maps PlantVillage class folder names → our target folder paths.

PLANTVILLAGE_TOMATO_MAP = {
    # PlantVillage folder name                     → our path
    "Tomato___Early_blight":                        "tomato/early_blight",
    "Tomato___Late_blight":                         "tomato/late_blight",
    "Tomato___Leaf_Mold":                           "tomato/leaf_mold",
    "Tomato___healthy":                             "tomato/healthy",
}

# For banana we use the "Banana Disease Recognition Dataset" on Kaggle.
# Dataset slug: mdwaquarazam/banana-disease-recognition-dataset
BANANA_MAP = {
    "healthy":         "banana/healthy",
    "sigatoka":        "banana/sigatoka",
    "panama_disease":  "banana/panama_disease",
}


def create_dirs():
    for path in [*PLANTVILLAGE_TOMATO_MAP.values(), *BANANA_MAP.values()]:
        (DATASET_DIR / path).mkdir(parents=True, exist_ok=True)
    print("✓  Target directories created.")


def download_kaggle(slug: str, dest: Path) -> Path:
    """Download a Kaggle dataset and return the extracted folder."""
    import subprocess
    dest.mkdir(parents=True, exist_ok=True)
    print(f"\nDownloading  {slug}  …")
    subprocess.run(
        ["kaggle", "datasets", "download", "-d", slug, "-p", str(dest), "--unzip"],
        check=True,
    )
    print(f"✓  Extracted to {dest}")
    return dest


def copy_images(src_dir: Path, dst_dir: Path, max_per_class: int = 1000):
    """Copy up to max_per_class images from src → dst."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    images = [
        p for p in src_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ][:max_per_class]

    for img in tqdm(images, desc=f"  → {dst_dir.name}", unit="img"):
        shutil.copy2(img, dst_dir / img.name)


def prepare_tomato(raw_dir: Path, max_per_class: int = 1000):
    print("\n[Tomato] Organising PlantVillage images …")
    for pv_name, target_rel in PLANTVILLAGE_TOMATO_MAP.items():
        src = raw_dir / pv_name
        if not src.is_dir():
            # Try case-insensitive search
            matches = [d for d in raw_dir.iterdir() if d.name.lower() == pv_name.lower()]
            if matches:
                src = matches[0]
            else:
                print(f"  ⚠  Skipping '{pv_name}' — folder not found in {raw_dir}")
                continue
        copy_images(src, DATASET_DIR / target_rel, max_per_class)
    print("✓  Tomato images ready.")


def prepare_banana(raw_dir: Path, max_per_class: int = 500):
    print("\n[Banana] Organising banana disease images …")
    for src_name, target_rel in BANANA_MAP.items():
        # Look for a matching subfolder (case-insensitive)
        matches = [
            d for d in raw_dir.rglob("*")
            if d.is_dir() and d.name.lower().replace(" ", "_") == src_name.lower()
        ]
        if not matches:
            print(f"  ⚠  Skipping '{src_name}' — folder not found under {raw_dir}")
            continue
        copy_images(matches[0], DATASET_DIR / target_rel, max_per_class)
    print("✓  Banana images ready.")


def print_summary():
    print("\n" + "─" * 55)
    print("  Dataset summary")
    print("─" * 55)
    total = 0
    for folder in sorted(DATASET_DIR.rglob("*")):
        if folder.is_dir() and folder.parent != DATASET_DIR:
            count = sum(1 for _ in folder.glob("*") if _.is_file())
            total += count
            print(f"  {folder.relative_to(DATASET_DIR):<40} {count:>5} images")
    print("─" * 55)
    print(f"  TOTAL{' ' * 39}{total:>5} images")
    print("─" * 55)


def main():
    print("=" * 55)
    print("  AI Smart Farming — Dataset Preparation")
    print("=" * 55)

    create_dirs()
    raw_root = BASE_DIR / "raw_downloads"

    # ── Tomato (PlantVillage) ─────────────────────────────────────────────────
    tomato_raw = download_kaggle(
        slug="abdallahalidev/plantvillage-dataset",
        dest=raw_root / "plantvillage",
    )
    # PlantVillage extracts to a 'plantvillage/color' sub-folder in most cases.
    pv_color = tomato_raw / "color"
    if not pv_color.is_dir():
        pv_color = tomato_raw  # flat extract fallback
    prepare_tomato(raw_dir=pv_color, max_per_class=1000)

    # ── Banana ───────────────────────────────────────────────────────────────
    banana_raw = download_kaggle(
        slug="mdwaquarazam/banana-disease-recognition-dataset",
        dest=raw_root / "banana",
    )
    prepare_banana(raw_dir=banana_raw, max_per_class=500)

    print_summary()
    print("\n✅  Dataset ready. Run:  python train.py")


if __name__ == "__main__":
    main()