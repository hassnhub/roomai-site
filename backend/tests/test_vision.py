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
    # A script disguised as an image upload. (A PHP web-shell string is avoided here because the
    # hosting provider's malware scanner rewrites files containing one.)
    with pytest.raises(ImageValidationError):
        load_upload(b"#!/bin/sh\necho this is not an image\n")


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


def _render(img, label_map, style, scheme, option=None, detections=None):
    import io as _io

    from app.services.visualiser import realistic_render
    return Image.open(_io.BytesIO(realistic_render(img, label_map, style, scheme, option or {"items": []}, detections)))


def test_white_walls_do_not_drain_colour_from_other_objects():
    """Regression: repainting near-white walls once pulled cushions and plants towards the wall colour."""
    import numpy as np

    from app.services.colours import delta_e, rgb_to_hex

    img = Image.new("RGB", (200, 120), (246, 242, 234))   # near-white wall
    img.paste((200, 200, 196), (0, 80, 200, 120))         # light grey floor
    img.paste((222, 160, 40), (90, 30, 120, 60))          # mustard cushion on the wall area
    label_map = np.zeros((120, 200), dtype=np.uint8)
    label_map[:80] = 1
    label_map[30:60, 90:120] = 0                          # the cushion is not wall
    label_map[80:] = 2
    out = _render(img, label_map, "traditional", "pastel")
    assert delta_e(rgb_to_hex(out.getpixel((105, 45))), "#DEA028") < 6
    assert delta_e(rgb_to_hex(out.getpixel((40, 20))), "#F6F2EA") > 3  # the wall itself was repainted


def test_wall_paint_is_a_soft_tint_of_the_scheme_colour():
    import numpy as np

    from app.services.colours import hex_to_rgb, rgb_to_lab
    from app.services.visualiser import _wall_tint, wall_colour_for

    pure = rgb_to_lab(np.array(hex_to_rgb(wall_colour_for("pastel")), dtype=float))
    tint = rgb_to_lab(np.array(hex_to_rgb(_wall_tint(wall_colour_for("pastel"), 95)), dtype=float))
    assert np.linalg.norm(tint[1:]) < np.linalg.norm(pure[1:]) * 0.8   # less saturated
    assert tint[0] >= 88                                               # bright room stays bright


def test_oversized_rug_outline_is_returned_to_the_floor():
    """Regression: the segmenter's rug outline spilled onto the floor and was painted as a bright halo."""
    import numpy as np

    from app.services.colours import delta_e, rgb_to_hex

    img = Image.new("RGB", (200, 140), (240, 238, 232))
    img.paste((200, 200, 196), (0, 60, 200, 140))         # floor
    img.paste((120, 80, 50), (70, 90, 130, 120))          # real rug (dark)
    label_map = np.zeros((140, 200), dtype=np.uint8)
    label_map[:60] = 1
    label_map[60:] = 2
    label_map[80:130, 55:145] = 11                        # rug label overshoots by ~10 px
    option = {"items": [{"category": "rug", "colour_hex": "#EFE6D2", "line_total": 100}]}
    out = _render(img, label_map, "traditional", "neutral", option)
    halo, floor = rgb_to_hex(out.getpixel((60, 125))), rgb_to_hex(out.getpixel((20, 125)))
    assert delta_e(halo, floor) < 10                       # overshoot painted like the floor
    assert delta_e(rgb_to_hex(out.getpixel((100, 105))), floor) > 15   # the real rug is still a rug


def test_floor_never_spreads_into_windows():
    import numpy as np

    from app.services.colours import rgb_to_lab
    from app.services.visualiser import _floor_mask

    img = Image.new("RGB", (100, 100), (205, 205, 200))   # window blinds and floor look alike
    label_map = np.zeros((100, 100), dtype=np.uint8)
    label_map[:50] = 4                                    # window
    label_map[50:] = 2                                    # floor
    lab = rgb_to_lab(np.asarray(img, dtype=float))
    assert not _floor_mask(lab, label_map)[:50].any()
