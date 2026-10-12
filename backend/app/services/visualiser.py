"""Room visualisation: generates an image of the redesigned room.

Providers (selected by the VISUALISER setting, or automatically in "auto" mode):
  * stability  - Stability AI "Structure" control: diffusion guided by the edges/structure of
                 the original photo, so walls, windows and layout are preserved.
  * replicate  - an interior-design ControlNet model hosted on Replicate.
  * local      - offline realistic restyle: using the segmentation masks, walls are repainted in the
                 scheme's wall colour, the floor in the style's floor tone and furniture surfaces in the
                 recommended finishes, preserving the photo's own shading. Needs no GPU, key or internet.
"""
import base64
import io
import logging
import time

import httpx
import numpy as np
from PIL import Image, ImageFilter

from ..config import get_settings
from ..domain import CATEGORIES, COLOUR_SCHEMES, ROOM_TYPES, STYLE_FLOORS, STYLES
from .colours import hex_to_rgb, lab_to_rgb, rgb_to_lab
from .imaging import to_jpeg_bytes

log = logging.getLogger(__name__)


class VisualisationError(RuntimeError):
    pass


def active_provider() -> str:
    s = get_settings()
    if s.visualiser != "auto":
        return s.visualiser
    if s.stability_api_key:
        return "stability"
    if s.replicate_api_token:
        return "replicate"
    return "local"


def build_prompt(room_type: str, style: str, scheme: str, items: list[dict]) -> tuple[str, str]:
    scheme_label = COLOUR_SCHEMES[scheme]["label"].lower()
    key_items = ", ".join(
        f"{i['colour_name'].lower()} {i['material']} {CATEGORIES[i['category']].lower()}" for i in items[:6]
    )
    prompt = (
        f"A beautifully redesigned {STYLES[style]['label']} {ROOM_TYPES[room_type].lower()} interior, "
        f"{scheme_label} colour palette, featuring {key_items}. Photorealistic interior design photography, "
        f"natural daylight, balanced composition, high detail, magazine quality."
    )
    negative = "people, text, watermark, blurry, distorted furniture, cluttered, low quality, cartoon, deformed"
    return prompt, negative


def render(image: Image.Image, room_type: str, style: str, scheme: str, option: dict,
           label_map: np.ndarray | None = None, detections: list[dict] | None = None) -> tuple[bytes, str, str]:
    """Returns (jpeg_bytes, provider_used, prompt). Falls back to local preview if a provider fails."""
    prompt, negative = build_prompt(room_type, style, scheme, option["items"])
    provider = active_provider()
    try:
        if provider == "stability":
            return _stability(image, prompt, negative), "stability", prompt
        if provider == "replicate":
            return _replicate(image, prompt, negative), "replicate", prompt
    except Exception as exc:
        log.warning("%s visualisation failed, using local preview: %s", provider, exc)
    return realistic_render(image, label_map, style, scheme, option, detections), "local", prompt


# --- Stability AI ------------------------------------------------------------------------
def _stability(image: Image.Image, prompt: str, negative: str) -> bytes:
    s = get_settings()
    resp = httpx.post(
        "https://api.stability.ai/v2beta/stable-image/control/structure",
        headers={"Authorization": f"Bearer {s.stability_api_key}", "Accept": "image/*"},
        files={"image": ("room.jpg", to_jpeg_bytes(image), "image/jpeg")},
        data={"prompt": prompt, "negative_prompt": negative, "control_strength": "0.75", "output_format": "jpeg"},
        timeout=120,
    )
    if resp.status_code != 200:
        raise VisualisationError(f"Stability API {resp.status_code}: {resp.text[:200]}")
    return resp.content


# --- Replicate -----------------------------------------------------------------------------
def _replicate(image: Image.Image, prompt: str, negative: str) -> bytes:
    s = get_settings()
    headers = {"Authorization": f"Bearer {s.replicate_api_token}"}
    with httpx.Client(base_url="https://api.replicate.com/v1", headers=headers, timeout=60) as client:
        model = client.get(f"/models/{s.replicate_model}")
        model.raise_for_status()
        version = model.json()["latest_version"]["id"]
        data_uri = "data:image/jpeg;base64," + base64.b64encode(to_jpeg_bytes(image)).decode()
        pred = client.post("/predictions", headers={"Prefer": "wait"}, json={
            "version": version,
            "input": {"image": data_uri, "prompt": prompt, "negative_prompt": negative,
                      "guidance_scale": 15, "prompt_strength": 0.8, "num_inference_steps": 50},
        })
        pred.raise_for_status()
        body = pred.json()
        deadline = time.time() + 180
        while body["status"] not in ("succeeded", "failed", "canceled"):
            if time.time() > deadline:
                raise VisualisationError("Replicate prediction timed out")
            time.sleep(2)
            body = client.get(body["urls"]["get"]).json()
        if body["status"] != "succeeded":
            raise VisualisationError(f"Replicate prediction {body['status']}: {body.get('error')}")
        output = body["output"]
        url = output[0] if isinstance(output, list) else output
        img = client.get(url, headers={}, timeout=60)
        img.raise_for_status()
        return img.content


# --- Local offline renderer ---------------------------------------------------------------
# Which segmented surface each recommended item category repaints.
CATEGORY_TO_CLASS = {
    "sofa": 6, "bed": 7, "armchair": 8, "dining_chair": 8, "office_chair": 8,
    "coffee_table": 9, "dining_table": 9, "desk": 9,
    "tv_unit": 10, "sideboard": 10, "wardrobe": 10, "dresser": 10, "storage_cabinet": 10, "shelving": 10,
    "rug": 11, "curtains": 12,
}


def colour_transfer(image: Image.Image, palette_hex: list[str], strength: float = 0.65) -> Image.Image:
    """Reinhard-style colour transfer in CIELAB towards a target palette (fallback when no masks)."""
    arr = np.asarray(image.convert("RGB"), dtype=np.float64)
    lab = rgb_to_lab(arr)
    target = rgb_to_lab(np.array([hex_to_rgb(h) for h in palette_hex], dtype=np.float64))
    out = lab.copy()
    for ch, w in ((0, 0.25), (1, 1.0), (2, 1.0)):
        src_mu, src_sd = lab[..., ch].mean(), lab[..., ch].std() + 1e-6
        tgt_mu, tgt_sd = target[:, ch].mean(), target[:, ch].std() + 1e-6
        tgt_sd = max(tgt_sd, src_sd * 0.6) if ch else src_sd  # keep lightness contrast
        transferred = (lab[..., ch] - src_mu) / src_sd * tgt_sd + tgt_mu
        out[..., ch] = lab[..., ch] + (transferred - lab[..., ch]) * strength * w
    return Image.fromarray(lab_to_rgb(out).astype(np.uint8), "RGB")


def wall_colour_for(scheme: str) -> str:
    """The scheme colour that reads best as a wall paint (lightness closest to L*=86)."""
    colours = COLOUR_SCHEMES[scheme]["colours"]
    labs = rgb_to_lab(np.array([hex_to_rgb(c) for c in colours], dtype=np.float64))
    return colours[int(np.argmin(np.abs(labs[:, 0] - 86)))]


def _feather(mask: np.ndarray, radius: float = 2.0) -> np.ndarray:
    img = Image.fromarray((mask * 255).astype(np.uint8), "L").filter(ImageFilter.GaussianBlur(radius))
    return np.asarray(img, dtype=np.float64) / 255


def _grow(lab: np.ndarray, label_map: np.ndarray, cls: int, threshold: float = 18, reach: int = 16) -> np.ndarray:
    """Extend a surface into nearby unlabelled pixels of the same original colour (e.g. window reveals,
    edges the segmenter missed) so repainting leaves no halos of the old colour."""
    from scipy import ndimage

    base = label_map == cls
    if base.sum() < 64:
        return base
    mean = lab[base].mean(0)
    near = ndimage.binary_dilation(base, iterations=reach)
    # Same paint in shadow keeps its hue but loses lightness, so compare hue tightly and lightness loosely.
    similar = (np.linalg.norm(lab[..., 1:] - mean[1:], axis=-1) < threshold * 0.8) & (np.abs(lab[..., 0] - mean[0]) < 40)
    # Window reveals, frames and blinds are often painted like the wall but labelled as windows/curtains.
    return base | (near & similar & np.isin(label_map, (0, 4, 5, 12)))


def _recolour(lab: np.ndarray, mask: np.ndarray, target_hex: str, texture: float, chroma: float, strength: float = 1.0):
    """Repaint a region: move it to the target colour while keeping its own shading and texture."""
    if mask.sum() < 64:
        return lab
    target = rgb_to_lab(np.array(hex_to_rgb(target_hex), dtype=np.float64))
    mean = lab[mask].mean(0)
    new = np.empty_like(lab)
    new[..., 0] = np.clip(target[0] + (lab[..., 0] - mean[0]) * texture, 0, 100)
    new[..., 1:] = target[1:] + (lab[..., 1:] - mean[1:]) * chroma
    alpha = _feather(mask)[..., None] * strength
    return lab * (1 - alpha) + new * alpha


def _furniture_masks(label_map: np.ndarray, option: dict, detections: list[dict]) -> list[tuple[np.ndarray, str]]:
    """Pixel masks to repaint for each recommended piece.

    Segmentation alone can confuse look-alikes (a radiator labelled as a cabinet), so furniture is only
    repainted where YOLO also detected that piece; rugs are the exception as the detector has no rug class.
    """
    h, w = label_map.shape
    masks, done = [], set()
    for item in sorted(option["items"], key=lambda i: -i["line_total"]):
        cls = CATEGORY_TO_CLASS.get(item["category"])
        if cls is None or cls in done:
            continue
        region = label_map == cls
        if cls != 11:
            boxes = np.zeros_like(region)
            for d in detections:
                if d.get("category") == item["category"]:
                    x1, y1, x2, y2 = d["box"]
                    pad = 0.03
                    boxes[int(max(0, y1 - pad) * h):int(min(1, y2 + pad) * h),
                          int(max(0, x1 - pad) * w):int(min(1, x2 + pad) * w)] = True
            region &= boxes
        if region.sum() >= 64:
            done.add(cls)
            masks.append((region, item["colour_hex"]))
    return masks


def realistic_render(image: Image.Image, label_map: np.ndarray | None, style: str, scheme: str, option: dict,
                     detections: list[dict] | None = None) -> bytes:
    """Offline redesign: repaint walls, floor and furniture surfaces with the recommended colours."""
    if label_map is None or label_map.shape != image.size[::-1]:
        return to_jpeg_bytes(colour_transfer(image, COLOUR_SCHEMES[scheme]["colours"], 0.5))
    lab = rgb_to_lab(np.asarray(image.convert("RGB"), dtype=np.float64))
    walls, ceiling, floor = (_grow(lab, label_map, c) for c in (1, 3, 2))
    furniture = _furniture_masks(label_map, option, detections or [])
    painted = walls | ceiling | floor
    for m, _ in furniture:
        painted |= m
    wall_hex = wall_colour_for(scheme)
    if walls.sum() >= 64:
        # Walls bounce coloured light onto everything else (window reveals, blinds, radiators).
        # Shift unpainted areas by part of the wall's colour change so no old colour cast remains.
        # Remove the old wall hue from each pixel in proportion to how much of it is present
        # (neutral or differently coloured pixels are untouched) and add the new wall hue instead.
        target_ab = rgb_to_lab(np.array(hex_to_rgb(wall_hex), dtype=np.float64))[1:]
        old_ab = lab[walls][:, 1:].mean(0)
        strength = np.linalg.norm(old_ab)
        if strength > 4:
            u = old_ab / strength
            c = np.clip(lab[..., 1:] @ u, 0, None)
            frac = np.clip(c / strength, 0, 1)[..., None]
            # Pull the whole chroma (not just the wall-hue component) towards the new wall colour, so no
            # leftover tint (e.g. green under a yellow cast) is exposed.
            new_ab = lab[..., 1:] * (1 - frac * 0.85) + target_ab * frac * 0.85
            keep = 1 - _feather(painted, 3.0)[..., None]
            lab[..., 1:] = lab[..., 1:] * (1 - keep) + new_ab * keep
    lab = _recolour(lab, walls, wall_hex, texture=0.85, chroma=0.15)
    lab = _recolour(lab, ceiling, "#F7F5F0", texture=0.8, chroma=0.1, strength=0.7)
    lab = _recolour(lab, floor, STYLE_FLOORS.get(style, "#C8A165"), texture=1.0, chroma=0.35, strength=0.9)
    for mask, colour in furniture:
        lab = _recolour(lab, mask, colour, texture=0.8, chroma=0.3, strength=0.9)
    return to_jpeg_bytes(Image.fromarray(lab_to_rgb(lab).astype(np.uint8), "RGB"), quality=92)
