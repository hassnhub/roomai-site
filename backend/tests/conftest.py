import os
import tempfile
from pathlib import Path

# Configure an isolated environment BEFORE the app is imported.
_tmp = tempfile.mkdtemp(prefix="roomai_test_")
os.environ["MONGO_DB"] = "roomai_test"
os.environ["STORAGE_DIR"] = _tmp
os.environ["VISUALISER"] = "local"
os.environ["STORAGE_KEY"] = "ZmFrZS10ZXN0LWtleS1mb3ItdW5pdC10ZXN0cy0xMjM="

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.catalogue_data import generate_products  # noqa: E402
from app.db import get_client  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def products():
    return generate_products()


@pytest.fixture(scope="session")
def client():
    get_client().drop_database("roomai_test")
    from app.main import app

    with TestClient(app) as c:
        yield c
    get_client().drop_database("roomai_test")


@pytest.fixture(scope="session")
def room_image_bytes():
    return (FIXTURES / "living_room.jpg").read_bytes()


def register(client, email, password="Passw0rd!", name="Test User"):
    r = client.post("/api/auth/register", json={"name": name, "email": email, "password": password, "consent": True})
    assert r.status_code == 201, r.text
    return {"Authorization": f"Bearer {r.json()['token']}"}
