"""Application configuration loaded from environment variables / .env file."""
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=BASE_DIR / ".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "RoomAI"
    # "development" or "production". Production refuses to start with default secrets.
    environment: str = "development"
    # URL prefix when hosted under a sub-path, e.g. "/~stu191266/roomai" (no trailing slash).
    base_path: str = ""
    # Built frontend (npm run build) to serve from the same server. Ignored if the folder is missing.
    frontend_dir: Path = BASE_DIR.parent / "frontend" / "dist"
    # Where images live: "disk" (storage_dir) or "gridfs" (inside MongoDB, for hosts without persistent disk).
    storage_backend: str = "disk"

    # Database: "mongo" (MongoDB / Atlas) or "sqlite", the built-in document store for hosts whose
    # firewall blocks outgoing connections to MongoDB. Both serve the same collections and queries.
    db_backend: str = "mongo"
    sqlite_path: Path = BASE_DIR / "data" / "roomai.sqlite3"
    mongo_uri: str = "mongodb://localhost:27017"
    mongo_db: str = "roomai"

    # Security
    jwt_secret: str = "change-me-in-production-please-use-a-long-random-string"
    jwt_algorithm: str = "HS256"
    jwt_expiry_minutes: int = 60 * 8
    # Fernet key used to encrypt uploaded room photos at rest. Generated on first run if blank.
    storage_key: str = ""

    # Upload limits
    max_upload_mb: int = 10
    max_image_side: int = 1280

    # Storage
    storage_dir: Path = BASE_DIR / "storage"

    # Object detection
    yolo_model: str = "yolov8n.pt"
    detection_confidence: float = 0.35

    # Visualisation provider: "auto" | "stability" | "replicate" | "local"
    visualiser: str = "auto"
    stability_api_key: str = ""
    replicate_api_token: str = ""
    replicate_model: str = "adirik/interior-design"

    # Google Sign-In (OAuth client ID from Google Cloud Console). Leave blank to hide the option.
    google_client_id: str = ""

    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"


    @property
    def is_production(self) -> bool:
        return self.environment.lower() == "production"

    def production_problems(self) -> list[str]:
        """Configuration that is unsafe to run publicly."""
        problems = []
        if len(self.jwt_secret) < 32 or self.jwt_secret.startswith("change-me"):
            problems.append("JWT_SECRET must be set to a random value of at least 32 characters")
        if not self.storage_key.strip():
            problems.append("STORAGE_KEY must be set (a Fernet key) so photos stay decryptable across deployments")
        elif not _valid_fernet_key(self.storage_key.strip()):
            problems.append("STORAGE_KEY is not a valid Fernet key (44 characters ending in '='); was it copied completely?")
        if self.storage_backend not in ("disk", "gridfs"):
            problems.append("STORAGE_BACKEND must be 'disk' or 'gridfs'")
        if self.db_backend not in ("mongo", "sqlite"):
            problems.append("DB_BACKEND must be 'mongo' or 'sqlite'")
        if self.db_backend == "sqlite" and self.storage_backend == "gridfs":
            problems.append("STORAGE_BACKEND=gridfs needs MongoDB; use 'disk' with DB_BACKEND=sqlite")
        if self.base_path and (not self.base_path.startswith("/") or self.base_path.endswith("/")):
            problems.append("BASE_PATH must start with '/' and have no trailing slash, e.g. /roomai")
        return problems


def _valid_fernet_key(key: str) -> bool:
    from cryptography.fernet import Fernet

    try:
        Fernet(key.encode())
        return True
    except ValueError:
        return False


@lru_cache
def get_settings() -> Settings:
    return Settings()
