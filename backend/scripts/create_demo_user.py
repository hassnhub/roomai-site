"""Create (or reset) a demo account for demonstrations and user testing.

Usage:  .venv\\Scripts\\python.exe scripts\\create_demo_user.py
"""
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.db import ensure_indexes, get_db  # noqa: E402
from app.security import hash_password  # noqa: E402

DEMO_NAME = "Demo User"
DEMO_EMAIL = "demo@example.com"
DEMO_PASSWORD = "RoomAIdemo2026"


def main():
    db = get_db()
    ensure_indexes(db)
    now = datetime.now(timezone.utc)
    db.users.update_one(
        {"email": DEMO_EMAIL},
        {"$set": {"name": DEMO_NAME, "password_hash": hash_password(DEMO_PASSWORD)},
         "$setOnInsert": {"email": DEMO_EMAIL, "created_at": now, "consent_at": now}},
        upsert=True,
    )
    print(f"Demo account ready: {DEMO_EMAIL}")


if __name__ == "__main__":
    main()
