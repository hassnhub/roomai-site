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
        f"The same {ROOM_TYPES[room_type].lower()} completely refurnished in {STYLES[style]['label']} style "
        f"with all-new furniture and décor: {key_items}. {scheme_label.capitalize()} colour palette on the walls "
        f"and soft furnishings. Same walls, windows, doors and camera angle. Photorealistic interior design "
        f"photography, natural daylight, balanced composition, high detail, magazine quality."
    )
    negative = ("people, text, watermark, blurry, distorted furniture, cluttered, low quality, cartoon, deformed, "
                "old worn furniture, mess, extra windows, warped walls")
    return prompt, negative


def render(image: Image.Image, room_type: str, style: str, scheme: str, option: dict,
           label_map: np.ndarray | None = None, detections: list[dict] | None = None,
           provider: str | None = None) -> tuple[bytes, str, str]:
    """Returns (jpeg_bytes, provider_used, prompt). Falls back to local preview if a provider fails.
    ``provider="local"`` forces the instant offline preview even when a cloud provider is configured."""
    prompt, negative = build_prompt(room_type, style, scheme, option["items"])
    provider = provider or active_provider()
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
        # Lower strength gives the model freedom to replace the furniture; the room's walls, windows
        # and perspective still follow the photo's structure.
        data={"prompt": prompt, "negative_prompt": negative, "control_strength": str(s.stability_control_strength),
              "output_format": "jpeg"},
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


def _wall_tint(hex_colour: str, room_lightness: float) -> str:
    """Wall paint as a decorator would mix it: the scheme colour softened to a tint (lower chroma) and kept
    about as light as the room's current walls, so bright rooms stay bright and colours read as paint."""
    lab = rgb_to_lab(np.array(hex_to_rgb(hex_colour), dtype=np.float64))
    lightness = np.clip(max(lab[0], room_lightness - 6), 0, 97)
    tint = np.array([lightness, lab[1] * WALL_CHROMA, lab[2] * WALL_CHROMA])
    r, g, b = np.clip(lab_to_rgb(tint), 0, 255).astype(int)
    return f"#{r:02X}{g:02X}{b:02X}"


def _feather(mask: np.ndarray, radius: float = 2.0) -> np.ndarray:
    img = Image.fromarray((mask * 255).astype(np.uint8), "L").filter(ImageFilter.GaussianBlur(radius))
    return np.asarray(img, dtype=np.float64) / 255


CAST_MIN_CHROMA = 12  # walls with less colour than this (near-white) cast no visible tint
WALL_CHROMA = 0.7  # wall paint uses 70% of the scheme colour's saturation (a tint, not the pure colour)
FURNITURE_CLASSES = (6, 7, 8, 9, 10, 11)


def _grow(lab: np.ndarray, label_map: np.ndarray, cls: int, threshold: float = 18, reach: int = 16,
          into: tuple[int, ...] = (0, 4, 5, 12), light_tol: float = 40) -> np.ndarray:
    """Extend a surface into nearby pixels of the given classes that have the same original colour (e.g.
    window reveals painted like the wall, edges the segmenter missed) so repainting leaves no halos."""
    from scipy import ndimage

    base = label_map == cls
    if base.sum() < 64:
        return base
    mean = lab[base].mean(0)
    near = ndimage.binary_dilation(base, iterations=reach)
    # Same paint in shadow keeps its hue but loses lightness, so compare hue tightly and lightness loosely.
    similar = (np.linalg.norm(lab[..., 1:] - mean[1:], axis=-1) < threshold * 0.8) & (np.abs(lab[..., 0] - mean[0]) < light_tol)
    chroma = np.linalg.norm(mean[1:])
    if chroma > CAST_MIN_CHROMA and light_tol >= 40:
        # Strongly coloured paint also loses saturation in shadow: match it by hue angle instead.
        pix_chroma = np.linalg.norm(lab[..., 1:], axis=-1)
        cos = (lab[..., 1:] @ mean[1:]) / np.maximum(pix_chroma * chroma, 1e-6)
        similar |= (cos > np.cos(0.45)) & (pix_chroma > 0.35 * chroma) & (np.abs(lab[..., 0] - mean[0]) < 55)
    return base | (near & similar & np.isin(label_map, into))


def _wall_mask(lab: np.ndarray, label_map: np.ndarray) -> np.ndarray:
    """Wall pixels to repaint.

    Strongly coloured walls (e.g. orange) tint their window reveals and frames, so those are grown into
    pixel by pixel. Light walls are handled per region instead: a small region the segmenter labelled as a
    window, curtain or picture that mostly looks like the wall (typically a sunlit or shaded strip of
    wall) becomes wall as a whole, while real blinds and windows are left alone as a whole - painting
    part of a region produces patches."""
    from scipy import ndimage

    base = label_map == 1
    if base.sum() < 64:
        return base
    mean = lab[base].mean(0)
    if np.linalg.norm(mean[1:]) > CAST_MIN_CHROMA:
        return _grow(lab, label_map, 1)
    walls = _grow(lab, label_map, 1, into=(0,))
    wall_like = (np.linalg.norm(lab[..., 1:] - mean[1:], axis=-1) < 10) & (np.abs(lab[..., 0] - mean[0]) < 15)
    touching = ndimage.binary_dilation(walls, iterations=2)
    # Roughness (mean lightness change between neighbouring pixels) separates a smooth sunlit or shaded
    # strip of wall from striped blinds and framed windows of the same colour.
    light = lab[..., 0]
    rough = np.abs(np.diff(light, axis=0, prepend=light[:1])) + np.abs(np.diff(light, axis=1, prepend=light[:, :1]))
    wall_rough = rough[base].mean()
    regions, n = ndimage.label(np.isin(label_map, (0, 4, 5, 12, 15)) & ~walls)
    small, larger = label_map.size * 0.025, label_map.size * 0.05
    for i in range(1, n + 1):
        region = regions == i
        area = region.sum()
        if not ((region & touching).any() and wall_like[region].mean() >= 0.6):
            continue
        if area <= small or (area <= larger and rough[region].mean() <= wall_rough):
            walls |= region
    return walls


def _floor_mask(lab: np.ndarray, label_map: np.ndarray) -> np.ndarray:
    """Floor pixels to repaint. The floor only grows into unlabelled pixels (never windows or blinds, which
    may share its neutral colour), and it reclaims thin bands along rug and furniture outlines that the
    segmenter over-extended onto the floor, which would otherwise be painted as a bright or dark halo."""
    from scipy import ndimage

    floor = _grow(lab, label_map, 2, reach=8, into=(0,), light_tol=15)
    if floor.sum() < 64:
        return floor
    # Compare with the floor right next to each pixel (sunlight and shadow change the floor's colour
    # across the room), not with the average floor colour.
    weight = ndimage.gaussian_filter(floor.astype(np.float64), 12)
    local = np.stack([ndimage.gaussian_filter(lab[..., i] * floor, 12) for i in range(3)], -1) / np.maximum(weight, 1e-6)[..., None]
    floor_like = (weight > 0.05) & (np.linalg.norm(lab - local, axis=-1) < 9)
    band = ndimage.binary_dilation(floor, iterations=25) & ~floor & np.isin(label_map, FURNITURE_CLASSES)
    candidates = band & floor_like
    # Only reclaim bands connected to the floor, so floor-coloured patches inside a rug stay part of it.
    seed = ndimage.binary_dilation(floor, iterations=1) & candidates
    reclaimed = ndimage.binary_propagation(seed, mask=candidates)
    return floor | reclaimed


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
    walls = _wall_mask(lab, label_map)
    ceiling = _grow(lab, label_map, 3)
    floor = _floor_mask(lab, label_map)
    furniture = [(m & ~floor, colour) for m, colour in _furniture_masks(label_map, option, detections or [])]
    painted = walls | ceiling | floor
    for m, _ in furniture:
        painted |= m
    wall_hex = _wall_tint(wall_colour_for(scheme), lab[walls][:, 0].mean() if walls.any() else 85)
    if walls.sum() >= 64:
        # Strongly coloured walls (e.g. orange) bounce their colour onto everything else - window reveals,
        # blinds, radiators - so after repainting those would keep the old cast. Remove the old wall hue from
        # each pixel in proportion to how much of it is present and add the new wall hue instead. Near-white
        # walls cast no visible colour; correcting them would drain the colour from cushions, plants and wood.
        target_ab = rgb_to_lab(np.array(hex_to_rgb(wall_hex), dtype=np.float64))[1:]
        old_ab = lab[walls][:, 1:].mean(0)
        strength = np.linalg.norm(old_ab)
        if strength > CAST_MIN_CHROMA:
            u = old_ab / strength
            c = np.clip(lab[..., 1:] @ u, 0, None)
            frac = np.clip(c / strength, 0, 1)[..., None]
            new_ab = lab[..., 1:] * (1 - frac * 0.85) + target_ab * frac * 0.85
            keep = 1 - _feather(painted, 3.0)[..., None]
            lab[..., 1:] = lab[..., 1:] * (1 - keep) + new_ab * keep
    lab = _recolour(lab, walls, wall_hex, texture=0.85, chroma=0.15)
    lab = _recolour(lab, ceiling, "#F7F5F0", texture=0.8, chroma=0.1, strength=0.7)
    lab = _recolour(lab, floor, STYLE_FLOORS.get(style, "#C8A165"), texture=1.0, chroma=0.35, strength=0.9)
    for mask, colour in furniture:
        lab = _recolour(lab, mask, colour, texture=0.8, chroma=0.3, strength=0.9)
    return to_jpeg_bytes(Image.fromarray(lab_to_rgb(lab).astype(np.uint8), "RGB"), quality=92)
