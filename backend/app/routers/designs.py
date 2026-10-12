import io
import json
from datetime import datetime, timezone

import numpy as np
from bson import ObjectId
from bson.errors import InvalidId
from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Response, UploadFile, status
from PIL import Image
from pydantic import ValidationError
from starlette.concurrency import run_in_threadpool

from .. import storage
from ..db import get_db
from ..domain import ROOM_TYPES
from ..schemas import DesignUpdateIn, PreferencesIn, PreviewIn, RenderIn, SwapIn
from ..security import current_user
from ..services import visualiser
from ..services.detection import map_to_categories
from ..services.imaging import ImageValidationError, load_upload, to_jpeg_bytes
from ..services.pipeline import analyse_image, get_recommender, prefs_for, recommend_for
from ..services.segmentation import overlay

router = APIRouter(prefix="/api/designs", tags=["designs"])


def serialize(d: dict) -> dict:
    out = {k: v for k, v in d.items() if k not in ("_id", "user_id")}
    out["id"] = str(d["_id"])
    for key in ("created_at", "updated_at"):
        if key in out:
            out[key] = out[key].isoformat()
    out["renders"] = [{**r, "created_at": r["created_at"].isoformat()} for r in d.get("renders", [])]
    return out


def _get_owned(design_id: str, user: dict) -> dict:
    try:
        oid = ObjectId(design_id)
    except InvalidId:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Design not found")
    # Ownership is part of the query, so users can never read another user's design.
    design = get_db().designs.find_one({"_id": oid, "user_id": user["_id"]})
    if not design:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Design not found")
    return design


# --- Images & segmentation helpers -----------------------------------------------------------
def _save_label_map(uid: str, label_map: np.ndarray | None) -> str | None:
    if label_map is None:
        return None
    buf = io.BytesIO()
    Image.fromarray(label_map, "L").save(buf, format="PNG", optimize=True)
    return storage.save_image(uid, buf.getvalue(), "seg")


def _load_inputs(design: dict, uid: str) -> tuple[Image.Image, np.ndarray | None]:
    raw = storage.load_image(uid, design["image_id"])
    if raw is None:
        raise HTTPException(status.HTTP_410_GONE, "Original image is no longer available")
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    label_map = None
    if design.get("seg_id"):
        seg = storage.load_image(uid, design["seg_id"])
        if seg is not None:
            label_map = np.asarray(Image.open(io.BytesIO(seg)))
    return img, label_map


def _render_option(design: dict, uid: str, index: int, img=None, label_map=None) -> dict:
    if img is None:
        img, label_map = _load_inputs(design, uid)
    option = design["options"][index]
    jpeg, provider, prompt = visualiser.render(
        img, design["room_type"], design["preferences"]["style"], design["preferences"]["scheme"], option, label_map,
        design.get("detected_items"))
    return {
        "image_id": storage.save_image(uid, jpeg, "render"),
        "option_index": index, "option_title": option["title"],
        "provider": provider, "prompt": prompt, "created_at": datetime.now(timezone.utc),
    }


def _replace_renders(design: dict, uid: str, indices: list[int]) -> list[dict]:
    """Re-render the given options, deleting their previous renders."""
    keep = []
    for r in design.get("renders", []):
        if r["option_index"] in indices:
            storage.delete_image(uid, r["image_id"])
        else:
            keep.append(r)
    img, label_map = _load_inputs(design, uid)
    return keep + [_render_option(design, uid, i, img, label_map) for i in indices]


# --- Endpoints ---------------------------------------------------------------------------------
@router.post("", status_code=201)
async def create_design(image: UploadFile = File(...), preferences: str = Form(...),
                        user: dict = Depends(current_user)):
    try:
        prefs = PreferencesIn.model_validate(json.loads(preferences))
    except (json.JSONDecodeError, ValidationError) as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, f"Invalid preferences: {exc}")
    data = await image.read()
    try:
        img = load_upload(data)
    except ImageValidationError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(exc))
    # Vision models are CPU-bound; run them off the event loop.
    return await run_in_threadpool(_create, img, prefs, user)


def _create(img: Image.Image, prefs: PreferencesIn, user: dict) -> dict:
    img, label_map, analysis = analyse_image(img)
    uid = str(user["_id"])
    image_id = storage.save_image(uid, to_jpeg_bytes(img), "orig")
    p = prefs.model_dump()
    room_type, source, options, rec_ms = recommend_for(p, analysis)
    analysis["timings"]["recommendation_ms"] = rec_ms
    now = datetime.now(timezone.utc)
    count = get_db().designs.count_documents({"user_id": user["_id"]})
    design = {
        "user_id": user["_id"],
        "title": f"{ROOM_TYPES[room_type]} #{count + 1}",
        "created_at": now, "updated_at": now,
        "preferences": p,
        "room_type": room_type, "room_type_source": source,
        "image_id": image_id,
        "seg_id": _save_label_map(uid, label_map),
        "analysis": analysis,
        "detected_items": map_to_categories(analysis["detections"], room_type),
        "options": options,
        "selected_option": 0,
        "renders": [],
        "favourite": False,
    }
    # The offline renderer takes about a second, so all three options are visualised straight away.
    # Paid cloud providers are only called when the user asks.
    if visualiser.active_provider() == "local":
        design["renders"] = [_render_option(design, uid, i, img, label_map) for i in range(len(options))]
    design["_id"] = get_db().designs.insert_one(design).inserted_id
    return serialize(design)


@router.get("")
def list_designs(user: dict = Depends(current_user)):
    cursor = get_db().designs.find(
        {"user_id": user["_id"]},
        {"title": 1, "created_at": 1, "updated_at": 1, "room_type": 1, "preferences": 1, "image_id": 1,
         "favourite": 1, "selected_option": 1, "options.total": 1, "options.title": 1, "renders": 1},
    ).sort("created_at", -1)
    return [serialize(d) for d in cursor]


@router.get("/{design_id}")
def get_design(design_id: str, user: dict = Depends(current_user)):
    return serialize(_get_owned(design_id, user))


@router.patch("/{design_id}")
def update_design(design_id: str, body: DesignUpdateIn, user: dict = Depends(current_user)):
    design = _get_owned(design_id, user)
    changes: dict = {"updated_at": datetime.now(timezone.utc)}
    if body.title is not None:
        changes["title"] = body.title.strip()
    if body.favourite is not None:
        changes["favourite"] = body.favourite
    if body.selected_option is not None:
        changes["selected_option"] = body.selected_option
    if body.preferences is not None:
        p = body.preferences.model_dump()
        room_type, source, options, _ = recommend_for(p, design["analysis"])
        changes.update({
            "preferences": p, "room_type": room_type, "room_type_source": source, "options": options,
            "detected_items": map_to_categories(design["analysis"]["detections"], room_type),
        })
        if visualiser.active_provider() == "local":
            changes["renders"] = _replace_renders({**design, **changes}, str(user["_id"]), list(range(len(options))))
    get_db().designs.update_one({"_id": design["_id"]}, {"$set": changes})
    return serialize({**design, **changes})


@router.post("/{design_id}/preview")
def preview_design(design_id: str, body: PreviewIn, user: dict = Depends(current_user)):
    """Live cost estimate for the Design Explorer - computes options without saving anything."""
    design = _get_owned(design_id, user)
    room_type, _, options, ms = recommend_for(body.preferences.model_dump(), design["analysis"])
    return {
        "room_type": room_type, "ms": ms,
        "options": [{"title": o["title"], "total": o["total"], "items": len(o["items"]), "match_score": o["match_score"],
                     "within_budget": o["within_budget"], "colours": [i["colour_hex"] for i in o["items"][:6]]}
                    for o in options],
    }


@router.post("/{design_id}/render")
def render_design(design_id: str, body: RenderIn, user: dict = Depends(current_user)):
    design = _get_owned(design_id, user)
    render = _render_option(design, str(user["_id"]), body.option_index)
    get_db().designs.update_one({"_id": design["_id"]},
                                {"$push": {"renders": render}, "$set": {"updated_at": render["created_at"]}})
    design.setdefault("renders", []).append(render)
    return serialize(design)


@router.get("/{design_id}/options/{index}/alternatives")
def alternatives(design_id: str, index: int, category: str = Query(..., max_length=40),
                 user: dict = Depends(current_user)):
    design = _get_owned(design_id, user)
    if not 0 <= index < len(design["options"]):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Option not found")
    current = next((i for i in design["options"][index]["items"] if i["category"] == category), None)
    if current is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Item not found in this option")
    prefs = prefs_for(design["preferences"], design["room_type"])
    alts = get_recommender().alternatives(prefs, category, current["qty"], current["essential"],
                                          design.get("detected_items"), design["analysis"].get("palette"))
    return {"current": current, "alternatives": [a for a in alts if a["sku"] != current["sku"]]}


@router.post("/{design_id}/options/{index}/swap")
def swap_item(design_id: str, index: int, body: SwapIn, user: dict = Depends(current_user)):
    design = _get_owned(design_id, user)
    if not 0 <= index < len(design["options"]):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Option not found")
    option = design["options"][index]
    pos = next((n for n, i in enumerate(option["items"]) if i["category"] == body.category), None)
    if pos is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Item not found in this option")
    prefs = prefs_for(design["preferences"], design["room_type"])
    current = option["items"][pos]
    alts = get_recommender().alternatives(prefs, body.category, current["qty"], current["essential"],
                                          design.get("detected_items"), design["analysis"].get("palette"), limit=50)
    replacement = next((a for a in alts if a["sku"] == body.sku), None)
    if replacement is None:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, "That product is not a suitable alternative")
    option["items"][pos] = replacement
    get_recommender().summarize(option, prefs)
    changes = {"options": design["options"], "updated_at": datetime.now(timezone.utc)}
    if visualiser.active_provider() == "local":
        changes["renders"] = _replace_renders(design, str(user["_id"]), [index])
    get_db().designs.update_one({"_id": design["_id"]}, {"$set": changes})
    return serialize({**design, **changes})


@router.get("/{design_id}/images/{image_id}")
def get_image(design_id: str, image_id: str, user: dict = Depends(current_user)):
    design = _get_owned(design_id, user)
    allowed = {design["image_id"], *(r["image_id"] for r in design.get("renders", []))}
    if image_id not in allowed:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Image not found")
    data = storage.load_image(str(user["_id"]), image_id)
    if data is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Image not found")
    return Response(data, media_type="image/jpeg", headers={"Cache-Control": "private, max-age=3600"})


@router.get("/{design_id}/overlay")
def get_overlay(design_id: str, user: dict = Depends(current_user)):
    """Colour-coded segmentation overlay (walls, floor, windows, furniture) as a transparent PNG."""
    design = _get_owned(design_id, user)
    seg = storage.load_image(str(user["_id"]), design["seg_id"]) if design.get("seg_id") else None
    if seg is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No segmentation for this design")
    buf = io.BytesIO()
    overlay(np.asarray(Image.open(io.BytesIO(seg)))).save(buf, format="PNG", optimize=True)
    return Response(buf.getvalue(), media_type="image/png", headers={"Cache-Control": "private, max-age=3600"})


@router.delete("/{design_id}", status_code=204)
def delete_design(design_id: str, user: dict = Depends(current_user)):
    design = _get_owned(design_id, user)
    uid = str(user["_id"])
    for file_id in [design["image_id"], design.get("seg_id"), *(r["image_id"] for r in design.get("renders", []))]:
        if file_id:
            storage.delete_image(uid, file_id)
    get_db().designs.delete_one({"_id": design["_id"]})
    return Response(status_code=204)
