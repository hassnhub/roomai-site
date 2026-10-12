"""Re-analyse and re-render designs created before room segmentation was added.

Adds walls/floor/window elements, layout measurements and the segmentation mask, refreshes the
recommendations and replaces the visualisations with the current renderer. Original photos are kept.

Usage:  .venv\\Scripts\\python.exe scripts\\reprocess_designs.py [--all]
"""
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image  # noqa: E402

from app import storage  # noqa: E402
from app.db import get_db  # noqa: E402
from app.domain import DISPLAY_NAMES  # noqa: E402
from app.routers.designs import _render_option, _save_label_map  # noqa: E402
from app.services import visualiser  # noqa: E402
from app.services.detection import map_to_categories  # noqa: E402
from app.services.pipeline import recommend_for  # noqa: E402
from app.services.segmentation import describe, segment  # noqa: E402


def main(reprocess_all: bool) -> None:
    db = get_db()
    query = {} if reprocess_all else {"seg_id": {"$exists": False}}
    for design in db.designs.find(query):
        uid = str(design["user_id"])
        raw = storage.load_image(uid, design["image_id"])
        if raw is None:
            print(f"skip {design['_id']}: original photo missing")
            continue
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        label_map = segment(img)
        analysis = design["analysis"]
        analysis["detections"] = [{**d, "name": DISPLAY_NAMES.get(d["label"], d["label"].title())} for d in analysis["detections"]]
        room = describe(img, label_map) if label_map is not None else {"elements": [], "layout": {}}
        analysis.update(room, segmenter_available=label_map is not None)
        room_type, source, options, _ = recommend_for(design["preferences"], analysis)
        if design.get("seg_id"):
            storage.delete_image(uid, design["seg_id"])
        updated = {
            **design, "analysis": analysis, "options": options, "room_type": room_type, "room_type_source": source,
            "detected_items": map_to_categories(analysis["detections"], room_type),
            "seg_id": _save_label_map(uid, label_map),
        }
        for r in design.get("renders", []):
            storage.delete_image(uid, r["image_id"])
        updated["renders"] = [_render_option(updated, uid, i, img, label_map) for i in range(len(options))] \
            if visualiser.active_provider() == "local" else []
        db.designs.replace_one({"_id": design["_id"]}, updated)
        print(f"reprocessed {design['_id']} ({design['title']})")


if __name__ == "__main__":
    main("--all" in sys.argv)
