"""End-to-end room analysis pipeline:
image -> furniture detection (YOLO) + room segmentation (SegFormer) -> palette -> room type -> recommendations.
"""
import time
from functools import lru_cache

import numpy as np
from PIL import Image

from ..db import get_db
from ..domain import DISPLAY_NAMES, ROOM_TYPES
from .colours import describe_lighting, extract_palette
from .detection import detect_objects, infer_room_type, map_to_categories
from .imaging import blur_regions
from .recommender import Preferences, Recommender
from .segmentation import describe, segment


@lru_cache
def get_recommender() -> Recommender:
    return Recommender(list(get_db().products.find({}, {"_id": 0})))


def analyse_image(img: Image.Image) -> tuple[Image.Image, np.ndarray | None, dict]:
    """Runs vision analysis. Returns the (privacy-blurred) image, the segmentation label map and the analysis."""
    timings = {}
    t = time.perf_counter()
    detections, available = detect_objects(img)
    timings["detection_ms"] = round((time.perf_counter() - t) * 1000)

    people = [d for d in detections if d["label"] == "person"]
    if people:
        img = blur_regions(img, [d["box"] for d in people])
    detections = [{**d, "name": DISPLAY_NAMES.get(d["label"], d["label"].title())}
                  for d in detections if d["label"] != "person"]

    t = time.perf_counter()
    label_map = segment(img)
    room = describe(img, label_map) if label_map is not None else {"elements": [], "layout": {}}
    timings["segmentation_ms"] = round((time.perf_counter() - t) * 1000)

    t = time.perf_counter()
    palette = extract_palette(img)
    lighting = describe_lighting(img)
    timings["colour_ms"] = round((time.perf_counter() - t) * 1000)

    inferred, scores = infer_room_type(detections)
    return img, label_map, {
        "detector_available": available,
        "segmenter_available": label_map is not None,
        "detections": detections,
        "elements": room["elements"],
        "layout": room["layout"],
        "people_blurred": len(people),
        "palette": palette,
        "lighting": lighting,
        "inferred_room_type": inferred,
        "room_type_scores": scores,
        "image_size": list(img.size),
        "timings": timings,
    }


def resolve_room_type(requested: str, analysis: dict) -> tuple[str, str]:
    """Returns (room_type, source) where source is 'user', 'detected' or 'default'."""
    if requested != "auto":
        return requested, "user"
    if analysis.get("inferred_room_type") in ROOM_TYPES:
        return analysis["inferred_room_type"], "detected"
    return "living_room", "default"


def prefs_for(prefs_in: dict, room_type: str) -> Preferences:
    return Preferences(
        room_type=room_type, style=prefs_in["style"], scheme=prefs_in["scheme"],
        length_m=prefs_in["length_m"], width_m=prefs_in["width_m"], height_m=prefs_in.get("height_m", 2.4),
        budget=prefs_in["budget"], keep_existing=prefs_in.get("keep_existing", False),
    )


def recommend_for(prefs_in: dict, analysis: dict) -> tuple[str, str, list[dict], int]:
    room_type, source = resolve_room_type(prefs_in["room_type"], analysis)
    mapped = map_to_categories(analysis["detections"], room_type)
    t = time.perf_counter()
    options = get_recommender().recommend(prefs_for(prefs_in, room_type), mapped, analysis.get("palette"))
    return room_type, source, options, round((time.perf_counter() - t) * 1000)
