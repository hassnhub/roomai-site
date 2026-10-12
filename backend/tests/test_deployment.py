"""Deployment features: GridFS storage, production safety checks, single-origin SPA serving."""
import pytest

from app import storage
from app.config import Settings, get_settings


@pytest.mark.parametrize("backend", ["disk", "gridfs"])
def test_storage_backends_round_trip_and_delete(client, monkeypatch, backend):
    monkeypatch.setattr(get_settings(), "storage_backend", backend)
    fid = storage.save_image("u_test", b"\xff\xd8 fake jpeg", "orig")
    assert storage.load_image("u_test", fid) == b"\xff\xd8 fake jpeg"
    storage.delete_image("u_test", fid)
    assert storage.load_image("u_test", fid) is None
    fid2 = storage.save_image("u_test", b"another", "render")
    storage.delete_user_files("u_test")
    assert storage.load_image("u_test", fid2) is None


def test_gridfs_blobs_are_encrypted(client, monkeypatch):
    from app.db import get_db
    monkeypatch.setattr(get_settings(), "storage_backend", "gridfs")
    storage.save_image("u_enc", b"PLAINTEXT-ROOM-PHOTO", "orig")
    stored = get_db()["images.chunks"].find_one()["data"]
    assert b"PLAINTEXT-ROOM-PHOTO" not in stored
    storage.delete_user_files("u_enc")


def test_storage_rejects_path_traversal_ids():
    assert storage.load_image("u", "../../.env") is None


def test_production_safety_checks():
    weak = Settings(environment="production", jwt_secret="change-me", storage_key="")
    assert len(weak.production_problems()) == 2
    good = Settings(environment="production", jwt_secret="x" * 48,
                    storage_key="ZmFrZS10ZXN0LWtleS1mb3ItdW5pdC10ZXN0cy0xMjM=", base_path="/roomai")
    assert good.production_problems() == []
    assert Settings(base_path="roomai/").production_problems()
