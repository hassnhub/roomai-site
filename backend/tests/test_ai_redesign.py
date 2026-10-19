"""AI redesign flow with a cloud provider: instant preview first, generative redesign on request.
The paid API is replaced by a stand-in so no network call or cost is involved."""
import io

import pytest
from PIL import Image

from app.config import get_settings
from app.services import visualiser
from conftest import register
from test_api import PREFS, create_design, email


def _fake_ai_image(image, prompt, negative):
    _fake_ai_image.calls.append(prompt)
    buf = io.BytesIO()
    Image.new("RGB", image.size, (200, 180, 150)).save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture
def cloud(monkeypatch):
    _fake_ai_image.calls = []
    monkeypatch.setattr(get_settings(), "visualiser", "stability")
    monkeypatch.setattr(visualiser, "_stability", _fake_ai_image)
    return _fake_ai_image


def _latest(design, index):
    return [r for r in design["renders"] if r["option_index"] == index][-1]


def test_create_gives_instant_previews_without_calling_paid_api(client, room_image_bytes, cloud):
    h = register(client, email())
    d = create_design(client, h, room_image_bytes).json()
    assert len(d["renders"]) == 3 and all(r["provider"] == "local" for r in d["renders"])
    assert cloud.calls == []


def test_generate_ai_redesign_replaces_only_that_option(client, room_image_bytes, cloud):
    h = register(client, email())
    d = create_design(client, h, room_image_bytes).json()
    r = client.post(f"/api/designs/{d['id']}/render", headers=h, json={"option_index": 1})
    assert r.status_code == 200
    d2 = r.json()
    assert len(d2["renders"]) == 3  # replaced, not accumulated
    assert _latest(d2, 1)["provider"] == "stability"
    assert _latest(d2, 0)["provider"] == "local" and _latest(d2, 2)["provider"] == "local"
    assert "all-new furniture" in cloud.calls[0]
    img = client.get(f"/api/designs/{d['id']}/images/{_latest(d2, 1)['image_id']}", headers=h)
    assert img.status_code == 200 and img.content[:2] == b"\xff\xd8"
    old = client.get(f"/api/designs/{d['id']}/images/{_latest(d, 1)['image_id']}", headers=h)
    assert old.status_code == 404  # previous image of that option was deleted


def test_changing_preferences_returns_to_previews(client, room_image_bytes, cloud):
    h = register(client, email())
    d = create_design(client, h, room_image_bytes).json()
    client.post(f"/api/designs/{d['id']}/render", headers=h, json={"option_index": 0})
    r = client.patch(f"/api/designs/{d['id']}", headers=h, json={"preferences": {**PREFS, "style": "industrial"}})
    assert all(x["provider"] == "local" for x in r.json()["renders"])
    assert len(cloud.calls) == 1


def test_provider_failure_falls_back_to_preview(client, room_image_bytes, monkeypatch):
    def broken(*_):
        raise visualiser.VisualisationError("Stability API 402: insufficient credits")

    monkeypatch.setattr(get_settings(), "visualiser", "stability")
    monkeypatch.setattr(visualiser, "_stability", broken)
    h = register(client, email())
    d = create_design(client, h, room_image_bytes).json()
    r = client.post(f"/api/designs/{d['id']}/render", headers=h, json={"option_index": 0})
    assert r.status_code == 200 and _latest(r.json(), 0)["provider"] == "local"


def test_render_rejects_unknown_option(client, room_image_bytes):
    h = register(client, email())
    d = create_design(client, h, room_image_bytes).json()
    assert client.post(f"/api/designs/{d['id']}/render", headers=h, json={"option_index": 2}).status_code == 200
    assert client.post(f"/api/designs/{d['id']}/render", headers=h, json={"option_index": 5}).status_code == 422


def test_prompt_asks_for_new_furniture_in_the_same_room():
    items = [{"colour_name": "Sage", "material": "velvet", "category": "sofa"},
             {"colour_name": "Oak", "material": "wood", "category": "coffee_table"}]
    prompt, negative = visualiser.build_prompt("living_room", "scandinavian", "neutral", items)
    assert "all-new furniture" in prompt and "sage velvet sofa" in prompt
    assert "Same walls, windows" in prompt and "warped walls" in negative
