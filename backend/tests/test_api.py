"""Integration tests for the REST API, including authentication and access-control checks."""
import json
import uuid

from conftest import register

PREFS = {"room_type": "auto", "style": "scandinavian", "scheme": "neutral",
         "length_m": 4.5, "width_m": 3.8, "budget": 2500, "keep_existing": False}


def email():
    return f"user_{uuid.uuid4().hex[:8]}@example.com"


def create_design(client, headers, image, prefs=PREFS):
    return client.post("/api/designs", headers=headers,
                       files={"image": ("room.jpg", image, "image/jpeg")},
                       data={"preferences": json.dumps(prefs)})


# --- Authentication -----------------------------------------------------------------------
def test_health_and_meta(client):
    assert client.get("/api/health").json()["database"] is True
    meta = client.get("/api/meta").json()
    assert len(meta["styles"]) == 9 and len(meta["schemes"]) == 7


def test_register_login_me(client):
    e = email()
    register(client, e)
    r = client.post("/api/auth/login", json={"email": e.upper(), "password": "Passw0rd!"})
    assert r.status_code == 200
    me = client.get("/api/auth/me", headers={"Authorization": f"Bearer {r.json()['token']}"})
    assert me.json()["email"] == e
    assert "password_hash" not in me.json()


def test_weak_password_and_missing_consent_rejected(client):
    r = client.post("/api/auth/register", json={"name": "A B", "email": email(), "password": "password", "consent": True})
    assert r.status_code == 422
    r = client.post("/api/auth/register", json={"name": "A B", "email": email(), "password": "Passw0rd!", "consent": False})
    assert r.status_code == 422


def test_duplicate_email_rejected(client):
    e = email()
    register(client, e)
    r = client.post("/api/auth/register", json={"name": "Dup", "email": e, "password": "Passw0rd!", "consent": True})
    assert r.status_code == 409


def test_passwords_are_hashed(client):
    from app.db import get_db
    e = email()
    register(client, e)
    stored = get_db().users.find_one({"email": e})["password_hash"]
    assert stored.startswith("$2") and "Passw0rd!" not in stored


def test_login_rate_limited_after_failures(client):
    e = email()
    register(client, e)
    codes = [client.post("/api/auth/login", json={"email": e, "password": "wrong"}).status_code for _ in range(6)]
    assert codes[:5] == [401] * 5 and codes[5] == 429


def test_protected_routes_require_token(client):
    assert client.get("/api/designs").status_code == 401
    assert client.get("/api/designs", headers={"Authorization": "Bearer not.a.jwt"}).status_code == 401


def test_security_headers_present(client):
    h = client.get("/api/health").headers
    assert h["x-content-type-options"] == "nosniff" and h["x-frame-options"] == "DENY"


# --- Catalogue ----------------------------------------------------------------------------
def test_catalogue_filters(client):
    r = client.get("/api/catalogue", params={"category": "sofa", "style": "industrial", "max_price": 600}).json()
    assert r["total"] >= 1
    assert all(p["category"] == "sofa" and p["style"] == "industrial" and p["price"] <= 600 for p in r["items"])


def test_catalogue_search_is_regex_safe(client):
    assert client.get("/api/catalogue", params={"q": ".*(["}).status_code == 200


# --- Design workflow ------------------------------------------------------------------------
def test_full_design_workflow(client, room_image_bytes):
    h = register(client, email())
    r = create_design(client, h, room_image_bytes)
    assert r.status_code == 201, r.text
    d = r.json()
    assert d["room_type"] == "living_room" and d["room_type_source"] in ("detected", "default")
    assert len(d["options"]) == 3 and all(o["total"] <= 2500 for o in d["options"])
    assert len(d["analysis"]["palette"]) == 5

    # Original image is retrievable (decrypted) only via the API.
    img = client.get(f"/api/designs/{d['id']}/images/{d['image_id']}", headers=h)
    assert img.status_code == 200 and img.headers["content-type"] == "image/jpeg"

    # Generate a visualisation (local provider in tests).
    r = client.post(f"/api/designs/{d['id']}/render", headers=h, json={"option_index": 1})
    render = r.json()["renders"][-1]
    assert render["provider"] == "local" and render["option_index"] == 1
    assert client.get(f"/api/designs/{d['id']}/images/{render['image_id']}", headers=h).status_code == 200

    # Changing preferences recomputes the recommendations.
    new_prefs = {**PREFS, "room_type": "bedroom", "style": "japandi", "budget": 1500}
    r = client.patch(f"/api/designs/{d['id']}", headers=h, json={"preferences": new_prefs, "title": "Guest room"})
    assert r.json()["room_type"] == "bedroom" and r.json()["title"] == "Guest room"
    assert any(i["category"] == "bed" for i in r.json()["options"][0]["items"])

    assert len(client.get("/api/designs", headers=h).json()) == 1
    assert client.delete(f"/api/designs/{d['id']}", headers=h).status_code == 204
    assert client.get(f"/api/designs/{d['id']}", headers=h).status_code == 404


def test_upload_validation_via_api(client):
    h = register(client, email())
    assert create_design(client, h, b"not an image").status_code == 422
    bad_prefs = {**PREFS, "budget": -5}
    assert client.post("/api/designs", headers=h, files={"image": ("r.jpg", b"x", "image/jpeg")},
                       data={"preferences": json.dumps(bad_prefs)}).status_code == 422


def test_users_cannot_access_each_others_designs(client, room_image_bytes):
    alice, bob = register(client, email()), register(client, email())
    d = create_design(client, alice, room_image_bytes).json()
    assert client.get(f"/api/designs/{d['id']}", headers=bob).status_code == 404
    assert client.get(f"/api/designs/{d['id']}/images/{d['image_id']}", headers=bob).status_code == 404
    assert client.delete(f"/api/designs/{d['id']}", headers=bob).status_code == 404
    assert client.get("/api/designs", headers=bob).json() == []


def test_images_encrypted_at_rest(client, room_image_bytes):
    from app.config import get_settings
    h = register(client, email())
    d = create_design(client, h, room_image_bytes).json()
    files = list(get_settings().storage_dir.rglob(f"{d['image_id']}.enc"))
    assert files and not files[0].read_bytes().startswith(b"\xff\xd8")  # not a readable JPEG


def test_gdpr_export_and_account_deletion(client, room_image_bytes):
    from bson import ObjectId

    from app.config import get_settings
    from app.db import get_db
    h = register(client, email())
    create_design(client, h, room_image_bytes)
    export = client.get("/api/account/export", headers=h).json()
    assert len(export["designs"]) == 1 and export["consent_at"]
    uid = export["user"]["id"]
    assert client.delete("/api/account", headers=h).status_code == 204
    assert not (get_settings().storage_dir / uid).exists()
    assert get_db().designs.count_documents({"user_id": ObjectId(uid)}) == 0
    assert client.get("/api/auth/me", headers=h).status_code == 401


# --- Redesign features: auto-render, segmentation, swap, live preview, profile ------------------
def test_create_auto_renders_all_options_and_segments_room(client, room_image_bytes):
    h = register(client, email())
    d = create_design(client, h, room_image_bytes).json()
    assert {r["option_index"] for r in d["renders"]} == {0, 1, 2}
    assert all(r["provider"] == "local" for r in d["renders"])
    labels = {e["label"] for e in d["analysis"]["elements"]}
    if d["analysis"]["segmenter_available"]:
        assert {"Walls", "Floor"} <= labels
        assert d["analysis"]["layout"]["windows"] >= 1
        ov = client.get(f"/api/designs/{d['id']}/overlay", headers=h)
        assert ov.status_code == 200 and ov.headers["content-type"] == "image/png"
    assert all("name" in det for det in d["analysis"]["detections"])


def test_swap_item_updates_totals(client, room_image_bytes):
    h = register(client, email())
    d = create_design(client, h, room_image_bytes).json()
    sofa = next(i for i in d["options"][0]["items"] if i["category"] == "sofa")
    alts = client.get(f"/api/designs/{d['id']}/options/0/alternatives", params={"category": "sofa"}, headers=h).json()
    assert alts["current"]["sku"] == sofa["sku"] and alts["alternatives"]
    new = alts["alternatives"][0]
    assert all(a["category"] == "sofa" and a["sku"] != sofa["sku"] for a in alts["alternatives"])
    r = client.post(f"/api/designs/{d['id']}/options/0/swap", headers=h, json={"category": "sofa", "sku": new["sku"]})
    assert r.status_code == 200
    opt = r.json()["options"][0]
    assert next(i for i in opt["items"] if i["category"] == "sofa")["sku"] == new["sku"]
    assert opt["total"] == round(d["options"][0]["total"] - sofa["line_total"] + new["line_total"], 2)
    # A product from another category cannot be swapped in.
    bad = client.post(f"/api/designs/{d['id']}/options/0/swap", headers=h, json={"category": "sofa", "sku": "RA-SCA-10-B"})
    assert bad.status_code == 422


def test_live_preview_does_not_save(client, room_image_bytes):
    h = register(client, email())
    d = create_design(client, h, room_image_bytes).json()
    prefs = {**PREFS, "style": "industrial", "budget": 6000}
    r = client.post(f"/api/designs/{d['id']}/preview", headers=h, json={"preferences": prefs}).json()
    assert len(r["options"]) == 3 and r["options"][0]["total"] <= 6000
    assert client.get(f"/api/designs/{d['id']}", headers=h).json()["preferences"]["style"] == "scandinavian"


def test_profile_update_and_password_change(client):
    e = email()
    h = register(client, e)
    assert client.patch("/api/account", headers=h, json={"name": "New Name"}).json()["name"] == "New Name"
    bad = client.post("/api/account/password", headers=h, json={"current_password": "nope", "new_password": "Better123"})
    assert bad.status_code == 400
    ok = client.post("/api/account/password", headers=h, json={"current_password": "Passw0rd!", "new_password": "Better123"})
    assert ok.status_code == 204
    assert client.post("/api/auth/login", json={"email": e, "password": "Better123"}).status_code == 200


def test_google_login_not_configured(client):
    r = client.post("/api/auth/google", json={"credential": "x" * 40})
    assert r.status_code == 501
    assert client.get("/api/meta").json()["google_client_id"] is None
