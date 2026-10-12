"""Colour analysis: dominant-palette extraction (k-means) and perceptual colour distance (CIELAB)."""
import numpy as np
from PIL import Image

NAMED_COLOURS = {
    "white": "#FFFFFF", "off-white": "#F2EEE6", "cream": "#EFE6D2", "beige": "#D8C3A5", "sand": "#E1D3B8",
    "light grey": "#C8C8C8", "grey": "#8C8C8C", "charcoal": "#3B3B3B", "black": "#141414",
    "taupe": "#8B7D6B", "brown": "#6F4E37", "tan": "#C19A6B", "walnut": "#6B4429", "oak": "#C8A165",
    "terracotta": "#C2643A", "rust": "#A0522D", "orange": "#E07B39", "mustard": "#D4A33B", "yellow": "#F2D24B",
    "olive": "#8A9A5B", "sage": "#A9B69A", "green": "#3F7D4E", "forest green": "#2F4B3A", "teal": "#2F7C7A",
    "sky blue": "#A7C4D8", "blue": "#4F7CAC", "navy": "#2C3E63", "lavender": "#B8A9D9", "pink": "#E8A0A8",
    "blush": "#F6C6C7", "burgundy": "#6D1F2A", "red": "#B83A3A",
}


def hex_to_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def rgb_to_hex(rgb) -> str:
    r, g, b = (int(max(0, min(255, round(float(c))))) for c in rgb)
    return f"#{r:02X}{g:02X}{b:02X}"


# --- sRGB <-> CIELAB (D65) -------------------------------------------------------------
_M = np.array([[0.4124564, 0.3575761, 0.1804375],
               [0.2126729, 0.7151522, 0.0721750],
               [0.0193339, 0.1191920, 0.9503041]])
_M_INV = np.linalg.inv(_M)
_WHITE = np.array([0.95047, 1.0, 1.08883])


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """rgb: (..., 3) array in 0-255 -> (..., 3) Lab."""
    c = np.asarray(rgb, dtype=np.float64) / 255.0
    c = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    xyz = c @ _M.T / _WHITE
    f = np.where(xyz > 216 / 24389, np.cbrt(xyz), (24389 / 27 * xyz + 16) / 116)
    L = 116 * f[..., 1] - 16
    a = 500 * (f[..., 0] - f[..., 1])
    b = 200 * (f[..., 1] - f[..., 2])
    return np.stack([L, a, b], axis=-1)


def lab_to_rgb(lab: np.ndarray) -> np.ndarray:
    lab = np.asarray(lab, dtype=np.float64)
    fy = (lab[..., 0] + 16) / 116
    fx = fy + lab[..., 1] / 500
    fz = fy - lab[..., 2] / 200
    f = np.stack([fx, fy, fz], axis=-1)
    xyz = np.where(f ** 3 > 216 / 24389, f ** 3, (116 * f - 16) / (24389 / 27)) * _WHITE
    c = xyz @ _M_INV.T
    c = np.clip(c, 0, 1)
    c = np.where(c <= 0.0031308, 12.92 * c, 1.055 * np.power(c, 1 / 2.4) - 0.055)
    return np.clip(c * 255, 0, 255)


def delta_e(hex_a: str, hex_b: str) -> float:
    """CIE76 colour difference. <10 ≈ similar, >40 ≈ clearly different."""
    la, lb = rgb_to_lab(np.array([hex_to_rgb(hex_a), hex_to_rgb(hex_b)]))
    return float(np.linalg.norm(la - lb))


_NAMED_LAB = {name: rgb_to_lab(np.array(hex_to_rgb(h))) for name, h in NAMED_COLOURS.items()}


def colour_name(hex_code: str) -> str:
    lab = rgb_to_lab(np.array(hex_to_rgb(hex_code)))
    return min(_NAMED_LAB, key=lambda n: float(np.linalg.norm(_NAMED_LAB[n] - lab)))


# --- Palette extraction ------------------------------------------------------------------
def _kmeans(points: np.ndarray, k: int, iters: int = 20, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """k-means with k-means++ initialisation. Returns (centroids, labels)."""
    rng = np.random.default_rng(seed)
    centroids = [points[rng.integers(len(points))]]
    for _ in range(1, k):
        d2 = np.min(((points[:, None, :] - np.array(centroids)[None]) ** 2).sum(-1), axis=1)
        probs = d2 / d2.sum() if d2.sum() > 0 else np.full(len(points), 1 / len(points))
        centroids.append(points[rng.choice(len(points), p=probs)])
    centroids = np.array(centroids)
    labels = np.zeros(len(points), dtype=int)
    for it in range(iters):
        dists = ((points[:, None, :] - centroids[None]) ** 2).sum(-1)
        new_labels = dists.argmin(1)
        if it > 0 and np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for j in range(k):
            members = points[labels == j]
            if len(members):
                centroids[j] = members.mean(0)
    return centroids, labels


def extract_palette(img: Image.Image, k: int = 5) -> list[dict]:
    """Dominant colours of an image, clustered in Lab space so groups match human perception."""
    small = img.convert("RGB").copy()
    small.thumbnail((120, 120))
    pixels = np.asarray(small, dtype=np.float64).reshape(-1, 3)
    lab = rgb_to_lab(pixels)
    k = min(k, len(np.unique(pixels, axis=0)))
    centroids, labels = _kmeans(lab, k)
    counts = np.bincount(labels, minlength=k)
    order = np.argsort(-counts)
    palette = []
    for j in order:
        if counts[j] == 0:
            continue
        hex_code = rgb_to_hex(lab_to_rgb(centroids[j]))
        palette.append({"hex": hex_code, "name": colour_name(hex_code), "proportion": round(float(counts[j] / counts.sum()), 3)})
    return palette


def describe_lighting(img: Image.Image) -> dict:
    """Simple descriptors of the room's light: brightness (0-100) and colour temperature."""
    small = img.convert("RGB").copy()
    small.thumbnail((200, 200))
    arr = np.asarray(small, dtype=np.float64)
    lab = rgb_to_lab(arr.reshape(-1, 3))
    brightness = float(lab[:, 0].mean())
    warmth = float(lab[:, 2].mean())  # +b = yellow/warm, -b = blue/cool
    return {
        "brightness": round(brightness, 1),
        "brightness_label": "bright" if brightness > 65 else "moderately lit" if brightness > 45 else "dim",
        "warmth": round(warmth, 1),
        "temperature_label": "warm" if warmth > 8 else "cool" if warmth < 0 else "neutral",
    }


def harmony_score(product_hex: str, scheme_hexes: list[str], room_palette: list[dict] | None = None) -> tuple[float, float]:
    """How well a product colour fits the chosen scheme (and existing room). Returns (score 0-1, best ΔE)."""
    best = min(delta_e(product_hex, h) for h in scheme_hexes)
    scheme_score = max(0.0, 1 - best / 60)
    if not room_palette:
        return scheme_score, best
    room_best = min(delta_e(product_hex, c["hex"]) for c in room_palette[:3])
    room_score = max(0.0, 1 - room_best / 80)
    return 0.75 * scheme_score + 0.25 * room_score, best
