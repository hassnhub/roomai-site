"""MongoDB connection helpers."""
from functools import lru_cache

from pymongo import ASCENDING, DESCENDING, MongoClient
from pymongo.database import Database

from .config import get_settings


@lru_cache
def get_client() -> MongoClient:
    return MongoClient(get_settings().mongo_uri, serverSelectionTimeoutMS=5000, tz_aware=True)


def get_db() -> Database:
    return get_client()[get_settings().mongo_db]


def ensure_indexes(db: Database) -> None:
    db.users.create_index([("email", ASCENDING)], unique=True)
    db.designs.create_index([("user_id", ASCENDING), ("created_at", DESCENDING)])
    db.products.create_index([("category", ASCENDING), ("style", ASCENDING)])
    db.products.create_index([("sku", ASCENDING)], unique=True)
