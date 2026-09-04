from contextlib import nullcontext

import cv2
import numpy as np
from fastapi.testclient import TestClient

from app.main import app, decode_image
from app.emotion_service import EmotionService


class FakePrediction:
    def as_dict(self):
        return {
            "box": {"x": 2, "y": 3, "width": 20, "height": 20},
            "emotion": "happiness",
            "confidence": 0.9,
            "probabilities": {"happiness": 0.9, "neutral": 0.1},
        }


class FakeService:
    def predict(self, image):
        assert image.shape == (32, 32, 3)
        return [FakePrediction()]


def encoded_image() -> bytes:
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    ok, buffer = cv2.imencode(".jpg", image)
    assert ok
    return buffer.tobytes()


def test_decode_image_rejects_invalid_bytes():
    try:
        decode_image(b"not-an-image")
    except ValueError as exc:
        assert "supported image" in str(exc)
    else:
        raise AssertionError("invalid bytes should fail")


def test_health_and_analyze(monkeypatch):
    monkeypatch.setattr(app.router, "lifespan_context", lambda _: nullcontext())
    app.state.service = FakeService()
    app.state.model_error = None
    with TestClient(app) as client:
        health = client.get("/api/health")
        assert health.status_code == 200
        assert health.json()["model_loaded"] is True
        response = client.post("/api/analyze", content=encoded_image(), headers={"content-type": "image/jpeg"})
        assert response.status_code == 200
        assert response.json()["faces"][0]["emotion"] == "happiness"


def test_analyze_requires_image_content_type(monkeypatch):
    monkeypatch.setattr(app.router, "lifespan_context", lambda _: nullcontext())
    app.state.service = FakeService()
    app.state.model_error = None
    with TestClient(app) as client:
        response = client.post("/api/analyze", content=b"text", headers={"content-type": "text/plain"})
        assert response.status_code == 415


def test_overlapping_face_boxes_are_deduplicated():
    boxes = np.array([[10, 10, 100, 100], [15, 15, 90, 90], [200, 20, 80, 80]])
    selected = EmotionService._deduplicate_boxes(boxes)
    assert selected == [(10, 10, 100, 100), (200, 20, 80, 80)]
