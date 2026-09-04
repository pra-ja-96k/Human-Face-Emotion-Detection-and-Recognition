from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "emotion-ferplus-8.onnx"
STATIC_DIR = BASE_DIR / "static"
MAX_IMAGE_BYTES = 5 * 1024 * 1024
MAX_FRAME_DIMENSION = 1920

