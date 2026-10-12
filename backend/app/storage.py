"""Encrypted-at-rest image storage.

Room photos show the inside of people's homes, so every file is encrypted with Fernet
(AES-128-CBC + HMAC-SHA256) before it is stored, and is only decrypted when the owning
user requests it through an authenticated endpoint.

Two backends (STORAGE_BACKEND):
  * disk   - encrypted files under STORAGE_DIR/<user_id>/ (default)
  * gridfs - encrypted blobs inside MongoDB, for hosts without persistent disk
"""
import shutil
import uuid
from functools import lru_cache
from pathlib import Path

from cryptography.fernet import Fernet

from .config import BASE_DIR, get_settings


@lru_cache
def _fernet() -> Fernet:
    s = get_settings()
    key = s.storage_key.strip()
    if not key:
        key_file = BASE_DIR / ".storage_key"
        if key_file.exists():
            key = key_file.read_text().strip()
        else:
            key = Fernet.generate_key().decode()
            key_file.write_text(key)
    return Fernet(key.encode())


def _valid(file_id: str | None) -> bool:
    return bool(file_id) and file_id.replace("_", "").isalnum()  # blocks path traversal


def _gridfs():
    import gridfs

    from .db import get_db

    return gridfs.GridFS(get_db(), collection="images")


def _user_dir(user_id: str) -> Path:
    d = get_settings().storage_dir / user_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def _use_gridfs() -> bool:
    return get_settings().storage_backend == "gridfs"


def save_image(user_id: str, data: bytes, prefix: str = "img") -> str:
    """Encrypt and store image bytes; returns an opaque file id."""
    file_id = f"{prefix}_{uuid.uuid4().hex}"
    blob = _fernet().encrypt(data)
    if _use_gridfs():
        _gridfs().put(blob, filename=f"{user_id}/{file_id}", metadata={"user_id": user_id})
    else:
        (_user_dir(user_id) / f"{file_id}.enc").write_bytes(blob)
    return file_id


def load_image(user_id: str, file_id: str) -> bytes | None:
    if not _valid(file_id):
        return None
    if _use_gridfs():
        f = _gridfs().find_one({"filename": f"{user_id}/{file_id}"})
        return _fernet().decrypt(f.read()) if f else None
    path = _user_dir(user_id) / f"{file_id}.enc"
    return _fernet().decrypt(path.read_bytes()) if path.exists() else None


def delete_image(user_id: str, file_id: str) -> None:
    if not _valid(file_id):
        return
    if _use_gridfs():
        fs = _gridfs()
        for f in fs.find({"filename": f"{user_id}/{file_id}"}):
            fs.delete(f._id)
    else:
        (_user_dir(user_id) / f"{file_id}.enc").unlink(missing_ok=True)


def delete_user_files(user_id: str) -> None:
    if _use_gridfs():
        fs = _gridfs()
        for f in fs.find({"metadata.user_id": user_id}):
            fs.delete(f._id)
    else:
        shutil.rmtree(get_settings().storage_dir / user_id, ignore_errors=True)
