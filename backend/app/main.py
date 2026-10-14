"""AI Room Designer API - AI-powered personalised interior design and room transformation.

In development the React app runs on Vite (port 5173) and proxies /api here. In production a
single process serves both: the API under /api and the built React app for every other path,
optionally under a sub-path (BASE_PATH) such as /~stu191266/roomai.
"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from .catalogue_data import generate_products
from .config import get_settings
from .db import ensure_indexes, get_db
from .routers import account, auth, catalogue, designs
from .services import visualiser
from .services.detection import detector_status
from .services.segmentation import segmenter_status

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("roomai")
settings = get_settings()

if settings.is_production and (problems := settings.production_problems()):
    raise RuntimeError("Unsafe production configuration:\n  - " + "\n  - ".join(problems))


def seed_catalogue_if_empty() -> int:
    db = get_db()
    if db.products.estimated_document_count() == 0:
        db.products.insert_many(generate_products())
    return db.products.estimated_document_count()


@asynccontextmanager
async def lifespan(_: FastAPI):
    ensure_indexes(get_db())
    log.info("Catalogue ready with %d products (%s mode)", seed_catalogue_if_empty(), settings.environment)
    yield


api = FastAPI(
    title="AI Room Designer API", version="1.0.0",
    # Sub-applications don't run lifespan handlers, so it lives on whichever app is served.
    lifespan=None if settings.base_path else lifespan,
    docs_url=None if settings.is_production else "/api/docs",
    redoc_url=None, openapi_url=None if settings.is_production else "/api/openapi.json",
)

if not settings.is_production:
    api.add_middleware(
        CORSMiddleware,
        allow_origins=[o.strip() for o in settings.cors_origins.split(",") if o.strip()],
        allow_methods=["GET", "POST", "PATCH", "DELETE"],
        allow_headers=["Authorization", "Content-Type"],
    )


@api.middleware("http")
async def security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
    if settings.is_production:
        response.headers["Strict-Transport-Security"] = "max-age=31536000"
    return response


for router in (auth.router, catalogue.router, designs.router, account.router):
    api.include_router(router)


@api.get("/api/health")
def health():
    db_ok = True
    try:
        get_db().command("ping")
    except Exception:
        db_ok = False
    return {"status": "ok" if db_ok else "degraded", "database": db_ok, "environment": settings.environment,
            "detector": detector_status(), "segmenter": segmenter_status(), "visualiser": visualiser.active_provider()}


# --- Serve the built React app (single-origin deployment) ------------------------------------
_dist = settings.frontend_dir.resolve()
if (_dist / "index.html").exists():
    @api.get("/{path:path}", include_in_schema=False)
    def spa(path: str):
        if path.startswith("api/"):
            raise HTTPException(404, "Not found")
        file = (_dist / path).resolve()
        if path and file.is_file() and file.is_relative_to(_dist):
            # Hashed build assets never change; index.html must always be revalidated.
            cache = "public, max-age=31536000, immutable" if path.startswith("assets/") else "no-cache"
            return FileResponse(file, headers={"Cache-Control": cache})
        return FileResponse(_dist / "index.html", headers={"Cache-Control": "no-cache"})

    log.info("Serving frontend from %s", _dist)


# --- Optional sub-path hosting -------------------------------------------------------------------
if settings.base_path:
    app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
    app.mount(settings.base_path, api)
else:
    app = api
