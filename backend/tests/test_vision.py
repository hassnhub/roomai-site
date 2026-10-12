"""Unit tests for colour analysis, image intake security and object detection."""
import io

import pytest
from PIL import Image

from app.services.colours import delta_e, extract_palette, describe_lighting, harmony_score
from app.services.detection import infer_room_type, map_to_categories
from app.services.imaging import ImageValidationError, load_upload
from app.services.visualiser import build_prompt, colour_transfer


def _jpeg(size=(400, 300), colour=(200, 180, 150), **save_kw):
    buf = io.BytesIO()
    Image.new("RGB", size, colour).save(buf, "JPEG", **save_kw)
    return buf.getvalue()


# --- Colour -------------------------------------------------------------------------------
def test_delta_e_identity_and_extremes():
    assert delta_e("#336699", "#336699") == pytest.approx(0, abs=1e-6)
    assert delta_e("#000000", "#FFFFFF") == pytest.approx(100, abs=0.5)


def test_palette_of_two_colour_image():
    img = Image.new("RGB", (100, 100), (255, 255, 255))
    img.paste((20, 40, 120), (0, 0, 100, 30))  # 30% navy, 70% white
    pal = extract_palette(img, k=2)
    assert pal[0]["proportion"] == pytest.approx(0.7, abs=0.02)
    assert delta_e(pal[0]["hex"], "#FFFFFF") < 3
    assert delta_e(pal[1]["hex"], "#142878") < 3


def test_lighting_labels():
    assert describe_lighting(Image.new("RGB", (50, 50), (245, 240, 230)))["brightness_label"] == "bright"
    assert describe_lighting(Image.new("RGB", (50, 50), (30, 30, 35)))["brightness_label"] == "dim"


def test_harmony_score_prefers_close_colours():
    near, _ = harmony_score("#F4F0E8", ["#F5F1EA"])
    far, _ = harmony_score("#C8102E", ["#F5F1EA"])
    assert near > 0.9 > far


def test_colour_transfer_keeps_size_and_shifts_colour():
    img = Image.new("RGB", (64, 48), (128, 128, 128))
    out = colour_transfer(img, ["#4F7CAC", "#2E4A62", "#A7C4D8"])
    assert out.size == img.size
    r, g, b = out.getpixel((10, 10))
    assert b > r  # pulled towards the cool palette


# --- Secure upload handling (security requirements) -----------------------------------------
def test_rejects_non_image():
    with pytest.raises(ImageValidationError):
        load_upload(b"<?php system($_GET['c']); ?>")


def test_rejects_empty_and_tiny():
    with pytest.raises(ImageValidationError):
        load_upload(b"")
    with pytest.raises(ImageValidationError):
        load_upload(_jpeg((100, 100)))


def test_rejects_oversized_file(monkeypatch):
    from app.config import get_settings
    monkeypatch.setattr(get_settings(), "max_upload_mb", 0)
    with pytest.raises(ImageValidationError):
        load_upload(_jpeg())


def test_rejects_unsupported_format():
    buf = io.BytesIO()
    Image.new("RGB", (300, 300)).save(buf, "BMP")
    with pytest.raises(ImageValidationError):
        load_upload(buf.getvalue())


def test_strips_exif_metadata_and_resizes():
    exif = Image.Exif()
    exif[0x010F] = "PhoneMaker"          # Make
    exif[0x8825] = {2: (51.0, 30.0, 0.0)}  # GPS IFD (latitude)
    data = _jpeg((3000, 2000), exif=exif)
    assert b"PhoneMaker" in data
    img = load_upload(data)
    assert max(img.size) == 1280
    assert not img.getexif()


# --- Detection post-processing ---------------------------------------------------------------
def test_room_type_inference():
    dets = [{"label": "bed", "confidence": 0.9}, {"label": "tv", "confidence": 0.8}]
    assert infer_room_type(dets)[0] == "bedroom"
    assert infer_room_type([{"label": "laptop", "confidence": 0.9}])[0] == "home_office"
    assert infer_room_type([])[0] is None


def test_category_mapping_depends_on_room():
    dets = [{"label": "chair", "confidence": 0.9, "box": [0, 0, 1, 1]}]
    assert map_to_categories(dets, "dining_room")[0]["category"] == "dining_chair"
    assert map_to_categories(dets, "home_office")[0]["category"] == "office_chair"
    assert map_to_categories(dets, "living_room")[0]["category"] == "armchair"


def test_prompt_mentions_style_and_items():
    items = [{"colour_name": "Light Oak", "material": "oak", "category": "sofa"}]
    prompt, negative = build_prompt("living_room", "scandinavian", "neutral", items)
    assert "Scandinavian" in prompt and "sofa" in prompt and "people" in negative


@pytest.mark.slow
def test_yolo_detects_furniture_in_sample_room(room_image_bytes):
    from app.services.detection import detect_objects
    dets, available = detect_objects(load_upload(room_image_bytes))
    if not available:
        pytest.skip("Ultralytics not installed")
    labels = {d["label"] for d in dets}
    assert {"tv", "chair"} <= labels


def test_realistic_render_repaints_walls_only_where_labelled():
    import io as _io

    import numpy as np

    from app.services.visualiser import realistic_render, wall_colour_for
    from app.services.colours import delta_e, rgb_to_hex

    img = Image.new("RGB", (200, 100), (210, 180, 60))  # mustard room
    img.paste((40, 40, 40), (0, 70, 200, 100))           # dark floor strip
    label_map = np.zeros((100, 200), dtype=np.uint8)
    label_map[:60] = 1   # walls
    label_map[70:] = 2   # floor
    option = {"items": []}
    out = Image.open(_io.BytesIO(realistic_render(img, label_map, "scandinavian", "neutral", option)))
    wall_px = rgb_to_hex(out.getpixel((100, 20)))
    assert delta_e(wall_px, wall_colour_for("neutral")) < 8
    assert delta_e(rgb_to_hex(out.getpixel((100, 90))), "#282828") > 20  # floor repainted too
