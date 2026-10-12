"""Furniture and room-element detection using a COCO-pretrained YOLO model (Ultralytics).

The model is loaded lazily on first use. If Ultralytics/PyTorch is not installed the
service degrades gracefully: detection returns no objects and the rest of the pipeline
(colour analysis, recommendations, visualisation) still works.
"""
import logging
import threading
from collections import defaultdict

from PIL import Image

from ..config import BASE_DIR, get_settings
from ..domain import COCO_TO_CATEGORY, ROOM_HINTS

log = logging.getLogger(__name__)

_model = None
_model_error: str | None = None
_lock = threading.Lock()

RELEVANT_CLASSES = set(COCO_TO_CATEGORY) | set(ROOM_HINTS) | {"person"}


def _load_model():
    global _model, _model_error
    if _model is not None or _model_error is not None:
        return _model
    with _lock:
        if _model is None and _model_error is None:
            try:
                from ultralytics import YOLO

                weights_dir = BASE_DIR / "models"
                weights_dir.mkdir(exist_ok=True)
                name = get_settings().yolo_model
                local = weights_dir / name
                _model = YOLO(str(local) if local.exists() else name)
                log.info("Loaded YOLO model %s", name)
            except Exception as exc:  # ImportError, download failure, etc.
                _model_error = f"{type(exc).__name__}: {exc}"
                log.warning("Object detection unavailable: %s", _model_error)
    return _model


def detector_status() -> dict:
    try:
        import ultralytics  # noqa: F401

        installed = True
    except ImportError:
        installed = False
    return {"installed": installed, "loaded": _model is not None, "error": _model_error}


def detect_objects(img: Image.Image) -> tuple[list[dict], bool]:
    """Returns (detections, detector_available). Boxes are normalised to 0-1."""
    model = _load_model()
    if model is None:
        return [], False
    conf = get_settings().detection_confidence
    result = model.predict(img, conf=conf, verbose=False, imgsz=640)[0]
    w, h = img.size
    detections = []
    for box in result.boxes:
        label = result.names[int(box.cls)]
        if label not in RELEVANT_CLASSES:
            continue
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        detections.append({
            "label": label,
            "confidence": round(float(box.conf), 3),
            "box": [round(x1 / w, 4), round(y1 / h, 4), round(x2 / w, 4), round(y2 / h, 4)],
        })
    detections.sort(key=lambda d: -d["confidence"])
    return detections, True


def infer_room_type(detections: list[dict]) -> tuple[str | None, dict]:
    """Weighted evidence vote over detected objects. Returns (room_type or None, scores)."""
    scores: dict[str, float] = defaultdict(float)
    for d in detections:
        for room, weight in ROOM_HINTS.get(d["label"], {}).items():
            scores[room] += weight * d["confidence"]
    if not scores:
        return None, {}
    best = max(scores, key=scores.get)
    return (best if scores[best] >= 0.5 else None), {k: round(v, 2) for k, v in scores.items()}


def map_to_categories(detections: list[dict], room_type: str) -> list[dict]:
    """Attach a furniture category to each detection (where one applies) for this room type."""
    mapped = []
    for d in detections:
        rule = COCO_TO_CATEGORY.get(d["label"])
        category = (rule.get(room_type) or rule.get("*")) if rule else None
        mapped.append({**d, "category": category})
    return mapped
