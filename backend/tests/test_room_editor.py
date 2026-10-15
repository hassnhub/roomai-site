"""Edit Room: tap a region of the photo and recolour it."""
import io

import numpy as np
from PIL import Image

from app.services import room_editor
from conftest import register
from test_api import create_design, email


def _design(client, room_image_bytes):
    h = register(client, email())
    return h, create_design(client, h, room_image_bytes).json()


def test_regions_cover_surfaces_and_furniture(client, room_image_bytes):
    h, d = _design(client, room_image_bytes)
    r = client.get(f"/api/designs/{d['id']}/regions", headers=h)
    assert r.status_code == 200
    body = r.json()
    keys = [x["key"] for x in body["regions"]]
    assert {"walls", "floor"} <= set(keys)
    assert any(x["kind"] == "furniture" for x in body["regions"])
    assert len(keys) == len(set(keys))                                  # keys are unique
    assert "Walnut" in body["floor_finishes"]
    png = client.get(f"/api/designs/{d['id']}/regions.png", headers=h)
    index = np.asarray(Image.open(io.BytesIO(png.content)))
    assert index.shape == (body["height"], body["width"])
    assert set(np.unique(index)) - {0} <= {x["index"] for x in body["regions"]}   # every pixel maps to a listed region


def test_preview_changes_only_the_chosen_region(client, room_image_bytes):
    h, d = _design(client, room_image_bytes)
    regions = client.get(f"/api/designs/{d['id']}/regions", headers=h).json()["regions"]
    index = np.asarray(Image.open(io.BytesIO(client.get(f"/api/designs/{d['id']}/regions.png", headers=h).content)))
    walls = next(x for x in regions if x["key"] == "walls")
    original = np.asarray(Image.open(io.BytesIO(client.get(f"/api/designs/{d['id']}/images/{d['image_id']}", headers=h).content)), dtype=int)
    r = client.post(f"/api/designs/{d['id']}/edit/preview", headers=h, json={"edits": [{"region": "walls", "colour": "#4F7CAC"}]})
    assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg"
    edited = np.asarray(Image.open(io.BytesIO(r.content)), dtype=int)
    diff = np.abs(edited - original).sum(-1)
    inside = index == walls["index"]
    far = ~inside & (np.abs(np.gradient(inside.astype(float))[0]) == 0)   # away from the soft edge
    assert diff[inside].mean() > 25                                      # the walls changed clearly
    assert diff[far & ~inside].mean() < diff[inside].mean() / 4          # the rest of the room barely did


def test_save_undo_and_delete_edit(client, room_image_bytes):
    h, d = _design(client, room_image_bytes)
    edits = [{"region": "walls", "colour": "#D7D2F0"}, {"region": "floor", "colour": "#6B3E26"}]
    saved = client.post(f"/api/designs/{d['id']}/edit", headers=h, json={"edits": edits}).json()
    assert saved["custom"]["edits"] == edits
    img = client.get(f"/api/designs/{d['id']}/images/{saved['custom']['image_id']}", headers=h)
    assert img.status_code == 200 and img.content[:2] == b"\xff\xd8"
    again = client.post(f"/api/designs/{d['id']}/edit", headers=h, json={"edits": edits[:1]}).json()   # undo one change, save
    assert again["custom"]["edits"] == edits[:1]
    assert client.get(f"/api/designs/{d['id']}/images/{saved['custom']['image_id']}", headers=h).status_code == 404  # old file gone
    cleared = client.delete(f"/api/designs/{d['id']}/edit", headers=h).json()
    assert "custom" not in cleared


def test_edit_validation_and_privacy(client, room_image_bytes):
    h, d = _design(client, room_image_bytes)
    url = f"/api/designs/{d['id']}/edit/preview"
    assert client.post(url, headers=h, json={"edits": [{"region": "walls", "colour": "blue"}]}).status_code == 422
    assert client.post(url, headers=h, json={"edits": [{"region": "spaceship-1", "colour": "#112233"}]}).status_code == 422
    assert client.post(f"/api/designs/{d['id']}/edit", headers=h, json={"edits": []}).status_code == 422
    other = register(client, email())
    assert client.get(f"/api/designs/{d['id']}/regions", headers=other).status_code == 404
    assert client.post(url, headers=other, json={"edits": []}).status_code == 404


def test_later_edit_of_same_region_wins():
    img = Image.new("RGB", (60, 40), (240, 240, 240))
    index = np.zeros((40, 60), dtype=np.uint8)
    index[:20] = 1
    regions = [{"index": 1, "key": "walls", "kind": "surface"}]
    out = room_editor.apply_edits(img, index, regions, [{"region": "walls", "colour": "#FF0000"},
                                                        {"region": "walls", "colour": "#0000FF"}])
    r, g, b = Image.open(io.BytesIO(out)).convert("RGB").getpixel((30, 5))
    assert b > r                                                         # blue, not red
