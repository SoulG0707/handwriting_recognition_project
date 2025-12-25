import os
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = BASE_DIR / "data.yaml"
YOLO_WEIGHTS = BASE_DIR / "best.pt"
MNIST_WEIGHTS = BASE_DIR / "mnist_keras_model.keras"

CONFIDENCE_THRESHOLD = 0.6
TESSERACT_CMD = os.getenv(
    "TESSERACT_CMD", r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
)


def ensure_exists(path: Path, kind: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"{kind} not found: {path}")
    return path


def resolve_paths():
    return {
        "yolo": ensure_exists(YOLO_WEIGHTS, "YOLO weights"),
        "mnist": ensure_exists(MNIST_WEIGHTS, "MNIST weights"),
        "data": ensure_exists(DEFAULT_DATASET, "data.yaml"),
    }
