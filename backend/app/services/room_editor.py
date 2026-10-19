"""Interactive room editing.

The photo is split into editable regions - walls, floor, ceiling and each separate piece of furniture -
using the segmentation produced during analysis. The user picks a region (by tapping the photo or a list)
and a colour or floor finish; edits are applied to the original photo in CIELAB so the room's own
lighting, shadows and texture are kept. Every preview is recomputed from the original photo and the full
list of edits, so undo and reset are simply a shorter list.
"""
import numpy as np
from PIL import Image
from scipy import ndimage

from ..domain import STYLE_FLOORS
from .colours import lab_to_rgb, rgb_to_lab
from .imaging import to_jpeg_bytes
from .visualiser import _floor_mask, _grow, _recolour, _wall_mask

FURNITURE = {6: "Sofa", 7: "Bed", 8: "Chair", 9: "Table", 10: "Cabinet", 11: "Rug", 12: "Curtains"}
MIN_AREA = 0.004          # regions smaller than 0.4% of the photo are too small to tap reliably
MAX_FURNITURE = 40
# (texture, chroma, strength): how much of the original shading and colour variation is kept.
PAINT = {
    "walls": (0.85, 0.15, 1.0), "ceiling": (0.8, 0.1, 0.9), "floor": (1.0, 0.35, 0.92),
    "furniture": (0.8, 0.3, 0.95),
}
FLOOR_FINISHES = {
    "Light oak": STYLE_FLOORS["scandinavian"], "Honey oak": STYLE_FLOORS["bohemian"],
    "Walnut": STYLE_FLOORS["traditional"], "Smoked oak": STYLE_FLOORS["industrial"],
    "Grey wash": STYLE_FLOORS["modern"], "Limed ash": STYLE_FLOORS["minimalist"],
    "Whitewash": STYLE_FLOORS["coastal"],
}


class EditError(ValueError):
    pass


def _box(mask: np.ndarray) -> list[float]:
    ys, xs = np.nonzero(mask)
    h, w = mask.shape
    return [round(xs.min() / w, 4), round(ys.min() / h, 4), round((xs.max() + 1) / w, 4), round((ys.max() + 1) / h, 4)]


def build_regions(image: Image.Image, label_map: np.ndarray) -> tuple[np.ndarray, list[dict]]:
    """Returns (index_map, regions): index_map holds each pixel's region number (0 = not editable)."""
    lab = rgb_to_lab(np.asarray(image.convert("RGB"), dtype=np.float64))
    total = label_map.size
    walls = _wall_mask(lab, label_map)
    floor = _floor_mask(lab, label_map) & ~walls
    ceiling = _grow(lab, label_map, 3) & ~walls & ~floor
    surfaces = walls | floor | ceiling

    index = np.zeros(label_map.shape, dtype=np.uint8)
    regions: list[dict] = []

    def add(key: str, label: str, kind: str, category: str, mask: np.ndarray) -> None:
        number = len(regions) + 1
        index[mask] = number
        regions.append({"index": number, "key": key, "label": label, "kind": kind, "category": category,
                        "area": round(float(mask.sum()) / total, 4), "box": _box(mask)})

    for key, label, mask in (("walls", "Walls", walls), ("floor", "Floor", floor), ("ceiling", "Ceiling", ceiling)):
        if mask.sum() >= total * MIN_AREA:
            add(key, label, "surface", key, mask)

    pieces = []
    for cls, name in FURNITURE.items():
        parts, n = ndimage.label((label_map == cls) & ~surfaces)
        for i in range(1, n + 1):
            mask = parts == i
            if mask.sum() >= total * MIN_AREA:
                pieces.append((name, mask))
    pieces.sort(key=lambda p: -p[1].sum())
    pieces = pieces[:MAX_FURNITURE]
    counts: dict[str, int] = {}
    for name, _ in pieces:
        counts[name] = counts.get(name, 0) + 1
    seen: dict[str, int] = {}
    for name, mask in pieces:
        seen[name] = seen.get(name, 0) + 1
        label = name if counts[name] == 1 else f"{name} {seen[name]}"
        add(f"{name.lower()}-{seen[name]}", label, "furniture", name.lower(), mask)
    return index, regions


def apply_edits(image: Image.Image, index: np.ndarray, regions: list[dict], edits: list[dict]) -> bytes:
    """Applies [{region, colour}] in order (later edits of the same region win) and returns a JPEG."""
    by_key = {r["key"]: r for r in regions}
    final: dict[str, str] = {}
    for edit in edits:
        if edit["region"] not in by_key:
            raise EditError(f"Unknown region: {edit['region']}")
        final[edit["region"]] = edit["colour"]
    lab = rgb_to_lab(np.asarray(image.convert("RGB"), dtype=np.float64))
    for key, colour in final.items():
        region = by_key[key]
        texture, chroma, strength = PAINT.get(key, PAINT["furniture"]) if region["kind"] == "surface" else PAINT["furniture"]
        lab = _recolour(lab, index == region["index"], colour, texture=texture, chroma=chroma, strength=strength)
    return to_jpeg_bytes(Image.fromarray(lab_to_rgb(lab).astype(np.uint8), "RGB"), quality=92)


# --- Replace or remove a piece of furniture with generative AI ------------------------------------
NEGATIVE = ("people, person, text, watermark, logo, blurry, distorted, deformed, warped perspective, "
            "low quality, cartoon, illustration, duplicate furniture")


def replace_prompt(description: str, style_label: str, room_label: str) -> str:
    return (f"{description}, {style_label} style, in a {room_label.lower()}, photorealistic interior photograph, "
            f"natural light, matching the room's perspective, lighting and shadows, high detail")


def _ai_mask(mask: np.ndarray) -> np.ndarray:
    """The region plus a small margin, so the old piece's outline does not survive at the edges."""
    return ndimage.binary_dilation(mask, iterations=3)


def _stability_edit(service: str, image: Image.Image, mask: np.ndarray, data: dict) -> bytes:
    import io

    import httpx

    from ..config import get_settings
    from .visualiser import VisualisationError

    key = get_settings().stability_api_key
    if not key:
        raise VisualisationError("The AI image service is not configured")
    buf = io.BytesIO()
    Image.fromarray((mask * 255).astype(np.uint8), "L").save(buf, format="PNG")
    resp = httpx.post(
        f"https://api.stability.ai/v2beta/stable-image/edit/{service}",
        headers={"Authorization": f"Bearer {key}", "Accept": "image/*"},
        files={"image": ("room.jpg", to_jpeg_bytes(image), "image/jpeg"), "mask": ("mask.png", buf.getvalue(), "image/png")},
        data={**data, "output_format": "jpeg"},
        timeout=120,
    )
    if resp.status_code != 200:
        raise VisualisationError(f"Stability API {resp.status_code}: {resp.text[:200]}")
    return resp.content


def ai_edit(image: Image.Image, mask: np.ndarray, action: str, prompt: str | None = None) -> Image.Image:
    """Redraws only the masked region: 'replace' paints new furniture from the prompt, 'remove' erases it.
    The result is returned at the original size so region maps and later edits stay aligned."""
    import io

    if action == "replace":
        data = {"prompt": prompt, "negative_prompt": NEGATIVE, "grow_mask": "8"}
        raw = _stability_edit("inpaint", image, _ai_mask(mask), data)
    elif action == "remove":
        raw = _stability_edit("erase", image, _ai_mask(mask), {"grow_mask": "8"})
    else:
        raise EditError(f"Unknown AI action: {action}")
    result = Image.open(io.BytesIO(raw)).convert("RGB")
    if result.size != image.size:
        result = result.resize(image.size, Image.LANCZOS)
    return result


def index_png(index: np.ndarray) -> bytes:
    """The region map as a lossless greyscale PNG: pixel value = region number (0 = not editable)."""
    import io

    buf = io.BytesIO()
    Image.fromarray(index, "L").save(buf, format="PNG", optimize=True)
    return buf.getvalue()
