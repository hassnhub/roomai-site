"""Secure image intake: validation, EXIF stripping, resizing and privacy blurring."""
import io

from PIL import Image, ImageFilter, ImageOps, UnidentifiedImageError

from ..config import get_settings

ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP"}
Image.MAX_IMAGE_PIXELS = 40_000_000  # guards against decompression-bomb uploads


class ImageValidationError(ValueError):
    pass


def load_upload(data: bytes) -> Image.Image:
    """Validate an uploaded file by decoding it (never trust the extension or MIME type)."""
    s = get_settings()
    if not data:
        raise ImageValidationError("The uploaded file is empty.")
    if len(data) > s.max_upload_mb * 1024 * 1024:
        raise ImageValidationError(f"Image is larger than {s.max_upload_mb} MB.")
    try:
        probe = Image.open(io.BytesIO(data))
        fmt = probe.format
        probe.verify()
        img = Image.open(io.BytesIO(data))
        img.load()
    except (UnidentifiedImageError, OSError, SyntaxError, Image.DecompressionBombError):
        raise ImageValidationError("File is not a valid image.")
    if fmt not in ALLOWED_FORMATS:
        raise ImageValidationError("Only JPEG, PNG and WebP images are accepted.")
    if min(img.size) < 200:
        raise ImageValidationError("Image is too small; please upload a photo at least 200px on each side.")
    # Honour camera orientation, then drop all metadata (EXIF can contain GPS location).
    img = ImageOps.exif_transpose(img).convert("RGB")
    img.thumbnail((s.max_image_side, s.max_image_side), Image.Resampling.LANCZOS)
    clean = Image.new("RGB", img.size)
    clean.paste(img)
    return clean


def to_jpeg_bytes(img: Image.Image, quality: int = 90) -> bytes:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality, optimize=True)
    return buf.getvalue()


def blur_regions(img: Image.Image, boxes: list[list[float]]) -> Image.Image:
    """Heavily blur normalised [x1, y1, x2, y2] regions (used to anonymise people)."""
    out = img.copy()
    w, h = out.size
    for x1, y1, x2, y2 in boxes:
        box = (int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h))
        if box[2] - box[0] < 2 or box[3] - box[1] < 2:
            continue
        region = out.crop(box).filter(ImageFilter.GaussianBlur(radius=max(8, (box[2] - box[0]) // 6)))
        out.paste(region, box)
    return out
