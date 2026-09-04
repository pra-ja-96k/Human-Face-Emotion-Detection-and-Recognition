from __future__ import annotations

import os
import sys
import urllib.request
from pathlib import Path

MODEL_URL = os.environ.get(
    "MODEL_URL",
    "https://github.com/onnx/models/raw/main/validated/vision/body_analysis/emotion_ferplus/model/emotion-ferplus-8.onnx",
)
TARGET = Path(os.environ.get("MODEL_PATH", "models/emotion-ferplus-8.onnx"))


def main() -> None:
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    if TARGET.is_file() and TARGET.stat().st_size > 1_000_000:
        print(f"Model already present: {TARGET}")
        return
    partial = TARGET.with_suffix(".download")
    print(f"Downloading FER+ ONNX model from {MODEL_URL}")
    try:
        request = urllib.request.Request(
            MODEL_URL,
            headers={"User-Agent": "human-face-emotion-detection-and-recognition/1.0"},
        )
        with urllib.request.urlopen(request, timeout=120) as response, partial.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
        if partial.stat().st_size < 1_000_000:
            raise RuntimeError("Downloaded file is unexpectedly small")
        partial.replace(TARGET)
        print(f"Saved {TARGET} ({TARGET.stat().st_size:,} bytes)")
    except Exception:
        partial.unlink(missing_ok=True)
        raise


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Model download failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
