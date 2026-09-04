from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from threading import Lock

import cv2
import numpy as np
import onnxruntime as ort

EMOTIONS = (
    "neutral",
    "happiness",
    "surprise",
    "sadness",
    "anger",
    "disgust",
    "fear",
    "contempt",
)


@dataclass(frozen=True)
class Prediction:
    box: tuple[int, int, int, int]
    emotion: str
    confidence: float
    probabilities: dict[str, float]

    def as_dict(self) -> dict:
        x, y, width, height = self.box
        return {
            "box": {"x": x, "y": y, "width": width, "height": height},
            "emotion": self.emotion,
            "confidence": self.confidence,
            "probabilities": self.probabilities,
        }


class EmotionService:
    """CPU-optimized face detection and FER+ ONNX inference."""

    def __init__(self, model_path: Path) -> None:
        if not model_path.is_file():
            raise FileNotFoundError(f"Emotion model not found: {model_path}")

        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(model_path),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        self.detector = cv2.CascadeClassifier(str(cascade_path))
        if self.detector.empty():
            raise RuntimeError("OpenCV face detector could not be loaded")
        self._lock = Lock()

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        logits = logits.astype(np.float32).reshape(-1)
        exp = np.exp(logits - np.max(logits))
        return exp / exp.sum()

    @staticmethod
    def _prepare_face(gray: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
        x, y, width, height = box
        face = gray[y : y + height, x : x + width]
        face = cv2.resize(face, (64, 64), interpolation=cv2.INTER_AREA)
        face = face.astype(np.float32)
        return face[np.newaxis, np.newaxis, :, :]

    @staticmethod
    def _deduplicate_boxes(boxes: np.ndarray, iou_threshold: float = 0.35) -> list[tuple[int, int, int, int]]:
        candidates = sorted(
            (tuple(int(value) for value in box) for box in boxes),
            key=lambda box: box[2] * box[3],
            reverse=True,
        )
        selected: list[tuple[int, int, int, int]] = []
        for candidate in candidates:
            x1, y1, width1, height1 = candidate
            area1 = width1 * height1
            overlaps = False
            for existing in selected:
                x2, y2, width2, height2 = existing
                intersection_width = max(0, min(x1 + width1, x2 + width2) - max(x1, x2))
                intersection_height = max(0, min(y1 + height1, y2 + height2) - max(y1, y2))
                intersection = intersection_width * intersection_height
                union = area1 + width2 * height2 - intersection
                containment = intersection / min(area1, width2 * height2)
                if union and (intersection / union >= iou_threshold or containment >= 0.65):
                    overlaps = True
                    break
            if not overlaps:
                selected.append(candidate)
        return selected

    def predict(self, image: np.ndarray) -> list[Prediction]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        min_face = max(40, min(image.shape[:2]) // 12)
        raw_boxes = self.detector.detectMultiScale(
            gray,
            scaleFactor=1.12,
            minNeighbors=5,
            minSize=(min_face, min_face),
        )

        boxes = self._deduplicate_boxes(raw_boxes)
        predictions: list[Prediction] = []
        for box in boxes:
            tensor = self._prepare_face(gray, box)
            with self._lock:
                logits = self.session.run([self.output_name], {self.input_name: tensor})[0]
            probabilities = self._softmax(logits)
            best = int(np.argmax(probabilities))
            predictions.append(
                Prediction(
                    box=box,
                    emotion=EMOTIONS[best],
                    confidence=round(float(probabilities[best]), 4),
                    probabilities={
                        emotion: round(float(score), 4)
                        for emotion, score in zip(EMOTIONS, probabilities, strict=True)
                    },
                )
            )
        return predictions
