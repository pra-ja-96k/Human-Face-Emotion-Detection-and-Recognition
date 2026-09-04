from __future__ import annotations

import base64
import binascii
from contextlib import asynccontextmanager
from time import perf_counter

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.config import MAX_FRAME_DIMENSION, MAX_IMAGE_BYTES, MODEL_PATH, STATIC_DIR
from app.emotion_service import EmotionService


def decode_image(payload: bytes) -> np.ndarray:
    if not payload or len(payload) > MAX_IMAGE_BYTES:
        raise ValueError("Image is empty or exceeds the 5 MB limit")
    image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("The supplied data is not a supported image")
    height, width = image.shape[:2]
    if max(height, width) > MAX_FRAME_DIMENSION:
        scale = MAX_FRAME_DIMENSION / max(height, width)
        image = cv2.resize(image, (round(width * scale), round(height * scale)))
    return image


def analyze(app: FastAPI, payload: bytes) -> dict:
    if app.state.service is None:
        raise RuntimeError(app.state.model_error or "Emotion model is unavailable")
    image = decode_image(payload)
    started = perf_counter()
    predictions = app.state.service.predict(image)
    return {
        "faces": [prediction.as_dict() for prediction in predictions],
        "image": {"width": image.shape[1], "height": image.shape[0]},
        "latency_ms": round((perf_counter() - started) * 1000, 1),
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.service = None
    app.state.model_error = None
    try:
        app.state.service = EmotionService(MODEL_PATH)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        app.state.model_error = str(exc)
    yield


app = FastAPI(
    title="Human Face Emotion Detection and Recognition",
    version="1.0.0",
    description="Privacy-first facial-expression inference for webcam, images, and video frames.",
    lifespan=lifespan,
)
app.mount("/assets", StaticFiles(directory=STATIC_DIR), name="assets")


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
async def health(request: Request) -> dict:
    ready = request.app.state.service is not None
    return {
        "status": "ready" if ready else "degraded",
        "model_loaded": ready,
        "model": MODEL_PATH.name,
        "error": request.app.state.model_error,
    }


@app.post("/api/analyze")
async def analyze_image(request: Request) -> dict:
    content_type = request.headers.get("content-type", "")
    if not content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="Send an image as the request body")
    payload = await request.body()
    try:
        return analyze(request.app, payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.websocket("/api/live")
async def live(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        while True:
            message = await websocket.receive_json()
            encoded = message.get("image", "")
            if "," in encoded:
                encoded = encoded.split(",", 1)[1]
            try:
                payload = base64.b64decode(encoded, validate=True)
                result = analyze(websocket.app, payload)
                result["frame_id"] = message.get("frame_id")
                await websocket.send_json(result)
            except (ValueError, RuntimeError, binascii.Error) as exc:
                await websocket.send_json({"error": str(exc), "frame_id": message.get("frame_id")})
    except WebSocketDisconnect:
        return
