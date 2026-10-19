"""Room-element segmentation with SegFormer (ADE20K, 150 indoor/outdoor classes).

YOLO (COCO) finds furniture *objects* but knows nothing about walls, floors or windows.
SegFormer labels every pixel, which gives us:
  * labelled room elements (walls, floor, ceiling, windows, doors) for the analysis view,
  * layout measurements (visible floor/wall share, number of windows, natural light),
  * the actual wall and floor colours, and
  * pixel masks the offline renderer uses to repaint walls, floors and furniture realistically.
Like detection, it loads lazily and degrades gracefully if `transformers` is unavailable.
"""
import logging
import threading

import numpy as np
from PIL import Image

from ..config import BASE_DIR
from .colours import colour_name, lab_to_rgb, rgb_to_hex, rgb_to_lab

log = logging.getLogger(__name__)

# Mask2Former (Swin-Small, ADE20K). Compared on real rooms with SegFormer B0/B2/B5, OneFormer and the larger
# Mask2Former models, it was the only small model that kept posters, pictures and the shadow under a desk out
# of the wall, with tight edges, and it needs no more memory than SegFormer-B2 on a CPU.
MODEL_ID = "facebook/mask2former-swin-small-ade-semantic"
LOCAL_DIR = BASE_DIR / "models" / "mask2former-small"
# The earlier SegFormer-B0 bundle, used if the new weights have not been downloaded onto a server yet, so the
# app keeps working (with the smaller model) instead of losing room analysis.
FALLBACK_DIR = BASE_DIR / "models" / "segformer"
INPUT_SIZE = 768  # longer side fed to the model; larger keeps more detail at wall edges

# Simplified class ids stored in the label map.
CLASSES = {
    0: "other", 1: "wall", 2: "floor", 3: "ceiling", 4: "window", 5: "door",
    6: "sofa", 7: "bed", 8: "chair", 9: "table", 10: "cabinet", 11: "rug", 12: "curtain",
    13: "plant", 14: "lamp", 15: "painting",
}
# ADE20K label names -> simplified class id.
ADE_TO_CLASS = {
    "wall": 1, "floor": 2, "ceiling": 3, "windowpane": 4, "door": 5, "screen door": 5,
    "sofa": 6, "bed": 7, "chair": 8, "armchair": 8, "swivel chair": 8, "seat": 8, "stool": 8,
    "table": 9, "desk": 9, "coffee table": 9, "kitchen island": 9,
    "cabinet": 10, "chest of drawers": 10, "wardrobe": 10, "bookcase": 10, "shelf": 10, "buffet": 10,
    "rug": 11, "curtain": 12, "blind": 12, "plant": 13, "flower": 13, "pot": 13,
    "lamp": 14, "light": 14, "chandelier": 14, "sconce": 14, "painting": 15, "poster": 15, "mirror": 15,
}
STRUCTURE = {1: "Walls", 2: "Floor", 3: "Ceiling"}
OPENINGS = {4: "Window", 5: "Door"}
OVERLAY_COLOURS = {
    1: (110, 139, 107), 2: (196, 140, 92), 3: (170, 170, 190), 4: (90, 160, 220), 5: (150, 100, 200),
    6: (230, 140, 60), 7: (230, 140, 60), 8: (230, 140, 60), 9: (230, 140, 60), 10: (230, 140, 60),
    11: (220, 90, 90), 12: (90, 200, 200),
}

_model = None
_processor = None
_lut = None
_error: str | None = None
_lock = threading.Lock()


def _load():
    global _model, _processor, _lut, _error
    if _model is not None or _error is not None:
        return _model
    with _lock:
        if _model is None and _error is None:
            try:
                import torch  # noqa: F401
                from transformers import (AutoConfig, AutoImageProcessor, Mask2FormerForUniversalSegmentation,
                                          SegformerForSemanticSegmentation)

                # Prefer a bundled copy (scripts/download_models.py) so servers need no internet access.
                bundled = [d for d in (LOCAL_DIR, FALLBACK_DIR) if (d / "config.json").exists()]
                source = str(bundled[0]) if bundled else MODEL_ID
                is_mask2former = AutoConfig.from_pretrained(source).model_type == "mask2former"
                _processor = AutoImageProcessor.from_pretrained(source)
                model_class = Mask2FormerForUniversalSegmentation if is_mask2former else SegformerForSemanticSegmentation
                model = model_class.from_pretrained(source)
                model.eval()
                lut = np.zeros(len(model.config.id2label), dtype=np.uint8)
                for idx, name in model.config.id2label.items():
                    for part in name.split(";"):
                        cls = ADE_TO_CLASS.get(part.strip().lower())
                        if cls:
                            lut[int(idx)] = cls
                            break
                _lut, _model = lut, model
                log.info("Loaded segmentation model from %s", source)
            except Exception as exc:
                _error = f"{type(exc).__name__}: {exc}"
                log.warning("Room segmentation unavailable: %s", _error)
    return _model


def segmenter_status() -> dict:
    return {"loaded": _model is not None, "error": _error}


def segment(img: Image.Image) -> np.ndarray | None:
    """Returns an (H, W) uint8 label map using the simplified CLASSES, or None if unavailable."""
    model = _load()
    if model is None:
        return None
    import torch

    small = img.convert("RGB").copy()
    small.thumbnail((INPUT_SIZE, INPUT_SIZE))
    inputs = _processor(images=small, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    # Decide each pixel's class at the model's working size, then enlarge only the final label map. Enlarging
    # the per-class scores to full size first needs about 700 MB more memory, too much for shared hosting.
    if model.config.model_type == "mask2former":
        ade = _processor.post_process_semantic_segmentation(outputs, target_sizes=[small.size[::-1]])[0].numpy()
    else:
        up = torch.nn.functional.interpolate(outputs.logits, size=small.size[::-1], mode="bilinear", align_corners=False)
        ade = up.argmax(1)[0].numpy()
    ade = np.asarray(Image.fromarray(ade.astype(np.uint8)).resize(img.size, Image.NEAREST))
    return _lut[ade]


def _component_boxes(mask: np.ndarray, min_frac: float) -> list[list[float]]:
    from scipy import ndimage

    h, w = mask.shape
    labels, n = ndimage.label(mask)
    boxes = []
    for sl in ndimage.find_objects(labels):
        if sl is None:
            continue
        ys, xs = sl
        if (ys.stop - ys.start) * (xs.stop - xs.start) < min_frac * h * w:
            continue
        boxes.append([round(xs.start / w, 4), round(ys.start / h, 4), round(xs.stop / w, 4), round(ys.stop / h, 4)])
    return boxes


def _region_colour(lab: np.ndarray, mask: np.ndarray) -> str:
    return rgb_to_hex(lab_to_rgb(np.median(lab[mask], axis=0)))


def describe(img: Image.Image, label_map: np.ndarray) -> dict:
    """Room elements and layout measurements derived from the label map."""
    small_lab = rgb_to_lab(np.asarray(img.convert("RGB"), dtype=np.float64)[::4, ::4])
    small_map = label_map[::4, ::4]
    total = small_map.size
    elements, layout = [], {}
    for cls, name in STRUCTURE.items():
        mask = small_map == cls
        share = mask.sum() / total
        layout[f"{CLASSES[cls]}_pct"] = round(float(share) * 100, 1)
        if share < 0.03:
            continue
        ys, xs = np.nonzero(mask)
        colour = _region_colour(small_lab, mask)
        elements.append({
            "label": name, "kind": "surface", "coverage": round(float(share), 3),
            "colour": colour, "colour_name": colour_name(colour),
            "anchor": [round(float(np.median(xs)) / small_map.shape[1], 3), round(float(np.median(ys)) / small_map.shape[0], 3)],
        })
    for cls, name in OPENINGS.items():
        boxes = _component_boxes(small_map == cls, 0.006)
        layout[f"{CLASSES[cls]}s"] = len(boxes)
        for b in boxes:
            elements.append({"label": name, "kind": "opening", "box": b})
    window_share = float((small_map == 4).sum() / total)
    layout["natural_light"] = "good" if window_share > 0.08 else "some" if window_share > 0.02 else "limited"
    layout["wall_colour"] = next((e["colour"] for e in elements if e["label"] == "Walls"), None)
    layout["floor_colour"] = next((e["colour"] for e in elements if e["label"] == "Floor"), None)
    return {"elements": elements, "layout": layout}


def overlay(label_map: np.ndarray, alpha: int = 96) -> Image.Image:
    """Translucent colour overlay of the segmentation for display."""
    h, w = label_map.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    for cls, rgb in OVERLAY_COLOURS.items():
        m = label_map == cls
        rgba[m, :3] = rgb
        rgba[m, 3] = alpha
    return Image.fromarray(rgba, "RGBA")
