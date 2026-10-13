"""Built-in SQLite document store: behaves like the subset of MongoDB the application uses."""
from datetime import datetime, timedelta, timezone

import pytest
from bson import ObjectId
from pymongo.errors import DuplicateKeyError

from app.docstore import DocStore


@pytest.fixture
def db(tmp_path):
    return DocStore(tmp_path / "test.sqlite3")


def test_insert_and_find_round_trip_types(db):
    when = datetime(2026, 10, 13, 9, 30, 15, 123000, tzinfo=timezone.utc)
    uid = ObjectId()
    doc = {"user_id": uid, "created_at": when, "blob": b"\x00\xff", "price": 12.5, "count": 3,
           "nested": {"list": [1, "two", {"three": 3}]}, "flag": True, "missing": None}
    new_id = db.items.insert_one(doc).inserted_id
    assert isinstance(new_id, ObjectId) and doc["_id"] == new_id  # like PyMongo, the id is set on the input
    got = db.items.find_one({"_id": new_id})
    assert got == doc
    assert got["created_at"].tzinfo is not None


def test_filters_operators_and_arrays(db):
    db.p.insert_many([
        {"name": "Oak Chair", "price": 80, "tags": ["wood", "seat"], "style": "modern"},
        {"name": "Velvet Sofa", "price": 650, "tags": ["seat"], "style": "classic"},
        {"name": "Glass Table", "price": 220, "style": "modern"},
    ])
    names = lambda f: sorted(d["name"] for d in db.p.find(f))  # noqa: E731
    assert names({"price": {"$lte": 220}}) == ["Glass Table", "Oak Chair"]
    assert names({"price": {"$gt": 80, "$lt": 650}}) == ["Glass Table"]
    assert names({"style": "modern", "price": {"$gte": 100}}) == ["Glass Table"]
    assert names({"name": {"$regex": "SOFA", "$options": "i"}}) == ["Velvet Sofa"]
    assert names({"tags": "seat"}) == ["Oak Chair", "Velvet Sofa"]  # array contains
    assert names({"tags": {"$exists": False}}) == ["Glass Table"]
    assert names({"style": {"$in": ["classic"]}}) == ["Velvet Sofa"]
    assert names({"style": {"$ne": "modern"}}) == ["Velvet Sofa"]
    assert names({"$or": [{"price": 80}, {"price": 650}]}) == ["Oak Chair", "Velvet Sofa"]
    assert db.p.count_documents({"style": "modern"}) == 2
    assert db.p.estimated_document_count() == 3


def test_sort_skip_limit_and_projection(db):
    db.p.insert_many([{"category": c, "price": p, "sku": f"S{i}"} for i, (c, p) in
                      enumerate([("sofa", 300), ("bed", 500), ("sofa", 100), ("bed", 200)])])
    rows = list(db.p.find({}, {"_id": 0}).sort([("category", 1), ("price", 1)]).skip(1).limit(2))
    assert rows == [{"category": "bed", "price": 500, "sku": "S1"}, {"category": "sofa", "price": 100, "sku": "S2"}]
    newest = [d["price"] for d in db.p.find().sort("price", -1)]
    assert newest == [500, 300, 200, 100]


def test_inclusion_projection_into_arrays(db):
    db.d.insert_one({"title": "Lounge", "secret": 1,
                     "options": [{"title": "A", "total": 10, "items": [1]}, {"title": "B", "total": 20, "items": [2]}]})
    got = db.d.find_one({}, {"title": 1, "options.total": 1, "options.title": 1})
    assert set(got) == {"_id", "title", "options"}
    assert got["options"] == [{"title": "A", "total": 10}, {"title": "B", "total": 20}]


def test_update_operators_and_upsert(db):
    _id = db.d.insert_one({"name": "x", "renders": []}).inserted_id
    r = db.d.update_one({"_id": _id}, {"$push": {"renders": {"n": 1}}, "$set": {"meta.updated": True}, "$inc": {"views": 2}})
    assert (r.matched_count, r.modified_count) == (1, 1)
    assert db.d.find_one({"_id": _id}) == {"_id": _id, "name": "x", "renders": [{"n": 1}], "meta": {"updated": True}, "views": 2}
    assert db.d.update_one({"name": "nobody"}, {"$set": {"a": 1}}).matched_count == 0
    up = db.d.update_one({"email": "a@b.c"}, {"$set": {"name": "A"}, "$setOnInsert": {"created": 1}}, upsert=True)
    assert up.upserted_id is not None
    assert db.d.find_one({"email": "a@b.c"})["created"] == 1
    db.d.update_one({"email": "a@b.c"}, {"$set": {"name": "B"}, "$setOnInsert": {"created": 2}}, upsert=True)
    assert db.d.find_one({"email": "a@b.c"}) | {"_id": None} == {"_id": None, "email": "a@b.c", "name": "B", "created": 1}
    db.d.update_one({"_id": _id}, {"$unset": {"meta": ""}})
    assert "meta" not in db.d.find_one({"_id": _id})


def test_replace_and_delete(db):
    ids = db.d.insert_many([{"u": 1, "n": i} for i in range(3)] + [{"u": 2, "n": 9}]).inserted_ids
    db.d.replace_one({"_id": ids[0]}, {"u": 1, "n": 100})
    assert db.d.find_one({"_id": ids[0]}) == {"_id": ids[0], "u": 1, "n": 100}
    assert db.d.delete_one({"_id": ids[1]}).deleted_count == 1
    assert db.d.delete_many({"u": 1}).deleted_count == 2
    assert [d["n"] for d in db.d.find()] == [9]


def test_unique_index_rejects_duplicates_on_insert_and_update(db):
    db.users.create_index([("email", 1)], unique=True)
    db.users.insert_one({"email": "a@x.com"})
    b = db.users.insert_one({"email": "b@x.com"}).inserted_id
    with pytest.raises(DuplicateKeyError):
        db.users.insert_one({"email": "a@x.com"})
    with pytest.raises(DuplicateKeyError):
        db.users.update_one({"_id": b}, {"$set": {"email": "a@x.com"}})
    assert db.users.find_one({"_id": b})["email"] == "b@x.com"  # failed write rolled back
    db.users.delete_one({"email": "a@x.com"})
    db.users.insert_one({"email": "a@x.com"})  # key is free again after delete
    assert db.users.count_documents({}) == 2


def test_unique_index_created_after_data_and_persisted(tmp_path):
    path = tmp_path / "p.sqlite3"
    first = DocStore(path)
    first.p.insert_many([{"sku": "A"}, {"sku": "B"}])
    first.p.create_index([("sku", 1)], unique=True)
    second = DocStore(path)  # e.g. another Passenger worker process
    with pytest.raises(DuplicateKeyError):
        second.p.insert_one({"sku": "A"})
    assert second.p.count_documents({}) == 2


def test_datetime_ordering_and_ping(db):
    now = datetime.now(timezone.utc)
    for i in range(3):
        db.d.insert_one({"user_id": "u", "created_at": now + timedelta(minutes=i), "i": i})
    assert [d["i"] for d in db.d.find({"user_id": "u"}).sort("created_at", -1)] == [2, 1, 0]
    assert db.command("ping") == {"ok": 1.0}


def test_bulk_catalogue_seed_is_fast(db):
    import time

    from app.catalogue_data import generate_products

    db.products.create_index([("sku", 1)], unique=True)
    t = time.perf_counter()
    db.products.insert_many(generate_products())
    assert time.perf_counter() - t < 5
    assert db.products.estimated_document_count() == len(generate_products())
