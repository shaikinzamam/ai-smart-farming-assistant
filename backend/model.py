from pathlib import Path
from io import BytesIO

from PIL import Image, UnidentifiedImageError


class DummyPlantDiseaseModel:
    """A lightweight stand-in for a real ML model."""

    def __init__(self, model_path: Path, solutions: dict):
        self.model_path = model_path
        self.diseases = [name for name in solutions.keys() if name != "Unknown"]

    def predict(self, image_bytes: bytes, filename: str) -> str:
        if not self.diseases:
            return "Unknown"

        try:
            with Image.open(BytesIO(image_bytes)) as image:
                rgb_image = image.convert("RGB")
                width, height = rgb_image.size
                sample_image = rgb_image.resize((16, 16))
                pixels = list(sample_image.getdata())
        except (UnidentifiedImageError, OSError, ValueError):
            return "Unknown"

        # Use basic image statistics to create a stable dummy prediction.
        pixel_checksum = sum(sum(pixel) for pixel in pixels)
        checksum = pixel_checksum + width + height + len(filename)
        disease_index = checksum % len(self.diseases)
        return self.diseases[disease_index]
