# Emotion Lens

Emotion Lens is a fully computer-based facial-expression recognition project. It uses the laptop webcam through the browser, detects faces with OpenCV, and performs CPU inference with the FER+ ONNX model. Raspberry Pi, GPIO, sensors, and telecommunication hardware are intentionally excluded.

## Features

- Laptop webcam capture using the browser `getUserMedia` API
- Image upload and multi-face detection
- Eight FER+ expression categories with confidence scores
- Live WebSocket inference with backpressure and 4 FPS sampling
- Local processing with no database or face-image storage
- CPU-optimized ONNX Runtime inference
- Docker and Docker Compose support
- Health endpoint, OpenAPI documentation, and automated API tests

> Facial-expression classification is probabilistic. It must not be used to diagnose health conditions, determine truthfulness, or infer a person's intent.

## Run with Docker

Docker downloads the public FER+ ONNX model during the image build.

```bash
docker compose up --build
```

Open <http://localhost:8000>, then allow camera access. API documentation is available at <http://localhost:8000/docs>.

Stop the application with:

```bash
docker compose down
```

## Run locally

Use Python 3.11:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements-dev.txt
python scripts/download_model.py
uvicorn app.main:app --reload
```

On Linux or macOS, activate the environment with `source .venv/bin/activate`.

## Test

```bash
pytest -q
```

## Architecture

1. The browser captures a frame from the laptop webcam and compresses it as JPEG.
2. A WebSocket sends one frame at a time to the FastAPI container.
3. OpenCV detects all frontal faces.
4. Each crop is converted to grayscale and resized to 64 × 64.
5. ONNX Runtime executes the FER+ convolutional neural network on the CPU.
6. The browser draws bounding boxes and displays smoothed class probabilities.

The browser owns camera access, so the Docker container does not need privileged access to a host camera device. This makes the same design portable across Windows, macOS, and Linux.

## Project structure

```text
app/                    FastAPI API and inference service
static/                 Responsive webcam interface
scripts/download_model.py  Model acquisition during build
tests/                  API and image validation tests
Dockerfile              Multi-stage, non-root production image
compose.yaml            One-command local deployment
```

## Model notes

The application uses the ONNX Model Zoo FER+ model. Its labels are neutral, happiness, surprise, sadness, anger, disgust, fear, and contempt. For a final academic evaluation, add a reproducible FER+ or FER-2013 test split and report macro-F1, per-class precision/recall, a confusion matrix, and end-to-end latency on the target laptop.

