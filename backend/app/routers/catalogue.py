import re

from fastapi import APIRouter, Query

from ..config import get_settings
from ..db import get_db
from ..services.visualiser import active_provider
from ..domain import CATEGORIES, COLOUR_SCHEMES, ROOM_TYPES, STYLES

router = APIRouter(prefix="/api", tags=["catalogue"])


@router.get("/meta")
def meta():
    """Options for the design form (public)."""
    return {
        "styles": [{"key": k, **v} for k, v in STYLES.items()],
        "schemes": [{"key": k, **v} for k, v in COLOUR_SCHEMES.items()],
        "room_types": [{"key": k, "label": v} for k, v in ROOM_TYPES.items()],
        "categories": [{"key": k, "label": v} for k, v in CATEGORIES.items()],
        "google_client_id": get_settings().google_client_id or None,
        "visualiser": active_provider(),
    }


@router.get("/catalogue")
def catalogue(
    category: str | None = None,
    style: str | None = None,
    tier: str | None = Query(default=None, pattern="^(budget|premium)$"),
    max_price: float | None = Query(default=None, ge=0),
    q: str | None = Query(default=None, max_length=60),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=24, ge=1, le=100),
):
    filt: dict = {}
    if category in CATEGORIES:
        filt["category"] = category
    if style in STYLES:
        filt["style"] = style
    if tier:
        filt["tier"] = tier
    if max_price is not None:
        filt["price"] = {"$lte": max_price}
    if q:
        filt["name"] = {"$regex": re.escape(q), "$options": "i"}  # escaped: no regex injection
    col = get_db().products
    total = col.count_documents(filt)
    items = list(col.find(filt, {"_id": 0}).sort([("category", 1), ("price", 1)]).skip((page - 1) * page_size).limit(page_size))
    return {"total": total, "page": page, "page_size": page_size, "items": items}
