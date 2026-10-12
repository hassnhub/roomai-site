"""Entry point for cPanel "Setup Python App" (Phusion Passenger, WSGI).

FastAPI is an ASGI app, so it is wrapped with a2wsgi. Passenger does not send ASGI lifespan
events, so the start-up work (database indexes, catalogue seeding) runs here instead.

cPanel settings:  Application startup file = passenger_wsgi.py   Entry point = application
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from a2wsgi import ASGIMiddleware  # noqa: E402

from app.db import ensure_indexes, get_db  # noqa: E402
from app.main import app, seed_catalogue_if_empty  # noqa: E402

ensure_indexes(get_db())
seed_catalogue_if_empty()


def _warm_up_models():
    # Loading YOLO + SegFormer takes ~30-40 s; doing it in the background at start-up stops the
    # first visitor's request from hitting the web server's timeout.
    from app.services.detection import _load_model
    from app.services.segmentation import _load

    _load_model()
    _load()


if os.environ.get("PRELOAD_MODELS", "1") == "1":
    import threading

    threading.Thread(target=_warm_up_models, daemon=True).start()

application = ASGIMiddleware(app)
