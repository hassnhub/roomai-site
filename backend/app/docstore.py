"""Built-in document store on SQLite, for hosts that block outgoing connections to MongoDB.

It implements the part of the PyMongo API this application uses (find / find_one with filters,
projections, sort, skip and limit; insert, update with $set, $setOnInsert, $unset, $inc and $push,
replace, delete, count and unique indexes), so the routers run unchanged on either backend.
Each collection is a table of (id, document) rows; documents are stored as MongoDB Extended JSON,
which keeps ObjectId, timezone-aware datetime and binary values intact. Filtering happens in Python,
which is fast enough for this application's data volumes (hundreds of products, per-user designs).
"""
import copy
import re
import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from bson import ObjectId, json_util
from bson.json_util import JSONMode, JSONOptions
from pymongo.errors import DuplicateKeyError

_JSON = JSONOptions(json_mode=JSONMode.CANONICAL, tz_aware=True, tzinfo=timezone.utc)
_NAME = re.compile(r"^[A-Za-z0-9_.]+$")
_MISSING = object()


@dataclass
class InsertOneResult:
    inserted_id: object


@dataclass
class InsertManyResult:
    inserted_ids: list = field(default_factory=list)


@dataclass
class UpdateResult:
    matched_count: int
    modified_count: int
    upserted_id: object = None


@dataclass
class DeleteResult:
    deleted_count: int


# --- Query matching ------------------------------------------------------------------------------
def _values(doc, path: str) -> list:
    """All values at a dotted path, descending into arrays the way MongoDB does."""
    current = [doc]
    for part in path.split("."):
        found = []
        for value in current:
            if isinstance(value, dict):
                if part in value:
                    found.append(value[part])
            elif isinstance(value, list):
                if part.isdigit() and int(part) < len(value):
                    found.append(value[int(part)])
                else:
                    found.extend(item[part] for item in value if isinstance(item, dict) and part in item)
        current = found
    return current


def _candidates(values: list) -> list:
    """A field holding an array matches a condition on any of its elements."""
    out = []
    for value in values:
        out.append(value)
        if isinstance(value, list):
            out.extend(value)
    return out


def _equals(values: list, target) -> bool:
    if target is None and not values:
        return True
    return any(v == target for v in _candidates(values))


def _compare(values: list, target, test) -> bool:
    for v in _candidates(values):
        try:
            if v is not None and not isinstance(v, bool) and test(v, target):
                return True
        except TypeError:
            continue
    return False


def _regex(spec: dict, pattern) -> re.Pattern:
    if isinstance(pattern, re.Pattern):
        return pattern
    flags = 0
    for letter in spec.get("$options", ""):
        flags |= {"i": re.IGNORECASE, "m": re.MULTILINE, "s": re.DOTALL, "x": re.VERBOSE}.get(letter, 0)
    return re.compile(pattern, flags)


def _condition(values: list, cond) -> bool:
    if not (isinstance(cond, dict) and cond and all(k.startswith("$") for k in cond)):
        if isinstance(cond, re.Pattern):
            return any(isinstance(v, str) and cond.search(v) for v in _candidates(values))
        return _equals(values, cond)
    for op, arg in cond.items():
        if op == "$options":
            continue
        if op == "$eq":
            ok = _equals(values, arg)
        elif op == "$ne":
            ok = not _equals(values, arg)
        elif op == "$gt":
            ok = _compare(values, arg, lambda a, b: a > b)
        elif op == "$gte":
            ok = _compare(values, arg, lambda a, b: a >= b)
        elif op == "$lt":
            ok = _compare(values, arg, lambda a, b: a < b)
        elif op == "$lte":
            ok = _compare(values, arg, lambda a, b: a <= b)
        elif op == "$in":
            ok = any(_equals(values, a) for a in arg)
        elif op == "$nin":
            ok = not any(_equals(values, a) for a in arg)
        elif op == "$exists":
            ok = bool(values) == bool(arg)
        elif op == "$regex":
            rx = _regex(cond, arg)
            ok = any(isinstance(v, str) and rx.search(v) for v in _candidates(values))
        else:
            raise NotImplementedError(f"Query operator {op} is not supported by the built-in store")
        if not ok:
            return False
    return True


def matches(doc: dict, filt: dict | None) -> bool:
    for key, cond in (filt or {}).items():
        if key == "$and":
            ok = all(matches(doc, f) for f in cond)
        elif key == "$or":
            ok = any(matches(doc, f) for f in cond)
        elif key == "$nor":
            ok = not any(matches(doc, f) for f in cond)
        elif key.startswith("$"):
            raise NotImplementedError(f"Query operator {key} is not supported by the built-in store")
        else:
            ok = _condition(_values(doc, key), cond)
        if not ok:
            return False
    return True


# --- Projection, sorting and updates --------------------------------------------------------------
def _include(src: dict, dst: dict, parts: list[str]) -> None:
    key = parts[0]
    if key not in src:
        return
    value = src[key]
    if len(parts) == 1:
        dst[key] = value
    elif isinstance(value, dict):
        _include(value, dst.setdefault(key, {}), parts[1:])
    elif isinstance(value, list):
        elements = [v for v in value if isinstance(v, dict)]
        target = dst.setdefault(key, [{} for _ in elements])
        for s, d in zip(elements, target):
            _include(s, d, parts[1:])


def _exclude(doc, parts: list[str]) -> None:
    if isinstance(doc, list):
        for item in doc:
            _exclude(item, parts)
    elif isinstance(doc, dict) and parts[0] in doc:
        if len(parts) == 1:
            del doc[parts[0]]
        else:
            _exclude(doc[parts[0]], parts[1:])


def project(doc: dict, projection: dict | None) -> dict:
    if not projection:
        return doc
    fields = {k: v for k, v in projection.items() if k != "_id"}
    keep_id = bool(projection.get("_id", 1))
    if any(fields.values()):
        out = {}
        for path in fields:
            _include(doc, out, path.split("."))
        if keep_id and "_id" in doc:
            out["_id"] = doc["_id"]
        return out
    for path in fields:
        _exclude(doc, path.split("."))
    if not keep_id:
        doc.pop("_id", None)
    return doc


def _sort_key(value):
    """MongoDB's cross-type order: null < numbers < strings < objects < arrays < ObjectId < bool < dates."""
    if value is None:
        return (0, 0)
    if isinstance(value, bool):
        return (6, value)
    if isinstance(value, (int, float)):
        return (1, value)
    if isinstance(value, str):
        return (2, value)
    if isinstance(value, dict):
        return (3, str(value))
    if isinstance(value, list):
        return (4, str(value))
    if isinstance(value, ObjectId):
        return (5, str(value))
    if isinstance(value, datetime):
        return (7, value)
    return (8, str(value))


def _parent(doc: dict, path: str, create: bool):
    parts = path.split(".")
    node = doc
    for part in parts[:-1]:
        if isinstance(node, list) and part.isdigit():
            node = node[int(part)]
        elif isinstance(node, dict):
            if part not in node:
                if not create:
                    return None, parts[-1]
                node[part] = {}
            node = node[part]
        else:
            return None, parts[-1]
    return node, parts[-1]


def _get(doc: dict, path: str):
    node, last = _parent(doc, path, create=False)
    if isinstance(node, dict):
        return node.get(last, _MISSING)
    if isinstance(node, list) and last.isdigit() and int(last) < len(node):
        return node[int(last)]
    return _MISSING


def _set(doc: dict, path: str, value) -> None:
    node, last = _parent(doc, path, create=True)
    if isinstance(node, list) and last.isdigit():
        node[int(last)] = value
    elif isinstance(node, dict):
        node[last] = value
    else:
        raise ValueError(f"Cannot set field {path!r}")


def apply_update(doc: dict, update: dict, inserting: bool = False) -> None:
    if not update or not all(k.startswith("$") for k in update):
        raise ValueError("update requires operators such as $set; use replace_one to replace a document")
    for op, fields in update.items():
        for path, value in fields.items():
            if path == "_id" and op != "$setOnInsert":
                raise ValueError("The _id field cannot be changed")
            if op == "$set" or (op == "$setOnInsert" and inserting):
                _set(doc, path, copy.deepcopy(value))
            elif op == "$setOnInsert":
                continue
            elif op == "$unset":
                node, last = _parent(doc, path, create=False)
                if isinstance(node, dict):
                    node.pop(last, None)
            elif op == "$inc":
                current = _get(doc, path)
                _set(doc, path, (0 if current is _MISSING else current) + value)
            elif op == "$push":
                current = _get(doc, path)
                if current is _MISSING:
                    current = []
                    _set(doc, path, current)
                if not isinstance(current, list):
                    raise ValueError(f"Cannot $push to non-array field {path!r}")
                items = value["$each"] if isinstance(value, dict) and "$each" in value else [value]
                current.extend(copy.deepcopy(items))
            else:
                raise NotImplementedError(f"Update operator {op} is not supported by the built-in store")


def _seed_from_filter(filt: dict) -> dict:
    """Equality conditions of an upsert's filter become fields of the new document."""
    doc: dict = {}
    for key, cond in (filt or {}).items():
        if key.startswith("$") or (isinstance(cond, dict) and any(k.startswith("$") for k in cond)):
            continue
        _set(doc, key, copy.deepcopy(cond))
    return doc


# --- Storage -----------------------------------------------------------------------------------
def _row_id(value) -> str:
    if isinstance(value, ObjectId):
        return f"oid:{value}"
    return "json:" + json_util.dumps(value, json_options=_JSON)


def _dump(doc: dict) -> str:
    return json_util.dumps(doc, json_options=_JSON)


def _load(text: str) -> dict:
    return json_util.loads(text, json_options=_JSON)


class Cursor:
    def __init__(self, collection: "Collection", filt: dict | None, projection: dict | None):
        self._collection, self._filter, self._projection = collection, filt or {}, projection
        self._sort: list[tuple[str, int]] = []
        self._skip = 0
        self._limit = 0

    def sort(self, key, direction: int = 1) -> "Cursor":
        self._sort = [(key, direction)] if isinstance(key, str) else list(key)
        return self

    def skip(self, count: int) -> "Cursor":
        self._skip = count
        return self

    def limit(self, count: int) -> "Cursor":
        self._limit = count
        return self

    def __iter__(self):
        docs = self._collection._matching(self._filter)
        for path, direction in reversed(self._sort):
            docs.sort(key=lambda d, p=path: _sort_key((_values(d, p) or [None])[0]), reverse=direction < 0)
        docs = docs[self._skip:]
        if self._limit:
            docs = docs[:self._limit]
        return iter([project(d, self._projection) for d in docs])


class Collection:
    def __init__(self, store: "DocStore", name: str):
        if not _NAME.match(name):
            raise ValueError(f"Invalid collection name {name!r}")
        self._store, self.name = store, name
        store._ensure_table(name)

    # Reads ------------------------------------------------------------------------------------
    def _rows(self, filt: dict) -> list[dict]:
        conn = self._store._conn()
        target = filt.get("_id") if filt else None
        if target is not None and not isinstance(target, dict):
            rows = conn.execute(f'SELECT doc FROM "{self.name}" WHERE id = ?', (_row_id(target),)).fetchall()
        else:
            rows = conn.execute(f'SELECT doc FROM "{self.name}" ORDER BY rowid').fetchall()
        return [_load(r[0]) for r in rows]

    def _matching(self, filt: dict) -> list[dict]:
        return [d for d in self._rows(filt) if matches(d, filt)]

    def find(self, filter: dict | None = None, projection: dict | None = None) -> Cursor:
        return Cursor(self, filter, projection)

    def find_one(self, filter: dict | None = None, projection: dict | None = None):
        return next(iter(self.find(filter, projection).limit(1)), None)

    def count_documents(self, filter: dict) -> int:
        return len(self._matching(filter))

    def estimated_document_count(self) -> int:
        return self._store._conn().execute(f'SELECT COUNT(*) FROM "{self.name}"').fetchone()[0]

    # Indexes ----------------------------------------------------------------------------------
    def create_index(self, keys, unique: bool = False, **_) -> str:
        spec = [(keys, 1)] if isinstance(keys, str) else list(keys)
        name = "_".join(f"{k}_{d}" for k, d in spec)
        with self._store._write() as conn:
            exists = conn.execute("SELECT 1 FROM __indexes__ WHERE coll = ? AND name = ?", (self.name, name)).fetchone()
            if exists:
                return name
            conn.execute("INSERT INTO __indexes__ (coll, name, fields, is_unique) VALUES (?, ?, ?, ?)",
                         (self.name, name, ",".join(k for k, _ in spec), int(unique)))
            if unique:  # index documents that already exist
                for row in conn.execute(f'SELECT doc FROM "{self.name}"').fetchall():
                    self._index(conn, _load(row[0]))
        return name

    def _unique_indexes(self, conn) -> list[tuple[str, list[str]]]:
        rows = conn.execute("SELECT name, fields FROM __indexes__ WHERE coll = ? AND is_unique = 1", (self.name,))
        return [(name, fields.split(",")) for name, fields in rows]

    def _index(self, conn, doc: dict) -> None:
        """Record the document's unique-index keys; the table's primary key rejects duplicates."""
        row = _row_id(doc["_id"])
        for name, fields in self._unique_indexes(conn):
            key = json_util.dumps([(_values(doc, f) or [None])[0] for f in fields], json_options=_JSON)
            try:
                conn.execute("INSERT INTO __unique__ (coll, idx, key, row) VALUES (?, ?, ?, ?)", (self.name, name, key, row))
            except sqlite3.IntegrityError:
                raise DuplicateKeyError(f"E11000 duplicate key error collection: {self.name} index: {name}")

    def _unindex(self, conn, doc: dict) -> None:
        conn.execute("DELETE FROM __unique__ WHERE coll = ? AND row = ?", (self.name, _row_id(doc["_id"])))

    # Writes -----------------------------------------------------------------------------------
    def _insert(self, conn, doc: dict) -> object:
        doc.setdefault("_id", ObjectId())
        try:
            conn.execute(f'INSERT INTO "{self.name}" (id, doc) VALUES (?, ?)', (_row_id(doc["_id"]), _dump(doc)))
        except sqlite3.IntegrityError:
            raise DuplicateKeyError(f"E11000 duplicate key error collection: {self.name} index: _id_")
        self._index(conn, doc)
        return doc["_id"]

    def insert_one(self, document: dict) -> InsertOneResult:
        with self._store._write() as conn:
            return InsertOneResult(self._insert(conn, document))

    def insert_many(self, documents) -> InsertManyResult:
        with self._store._write() as conn:
            return InsertManyResult([self._insert(conn, d) for d in documents])

    def _save(self, conn, doc: dict) -> None:
        self._unindex(conn, doc)
        self._index(conn, doc)
        conn.execute(f'UPDATE "{self.name}" SET doc = ? WHERE id = ?', (_dump(doc), _row_id(doc["_id"])))

    def _update(self, filt: dict, update: dict, upsert: bool, many: bool) -> UpdateResult:
        with self._store._write() as conn:
            docs = self._matching(filt)
            if not many:
                docs = docs[:1]
            modified = 0
            for doc in docs:
                before = _dump(doc)
                apply_update(doc, update)
                if _dump(doc) != before:
                    self._save(conn, doc)
                    modified += 1
            if docs or not upsert:
                return UpdateResult(len(docs), modified)
            doc = _seed_from_filter(filt)
            apply_update(doc, update, inserting=True)
            return UpdateResult(0, 0, self._insert(conn, doc))

    def update_one(self, filter: dict, update: dict, upsert: bool = False) -> UpdateResult:
        return self._update(filter, update, upsert, many=False)

    def update_many(self, filter: dict, update: dict, upsert: bool = False) -> UpdateResult:
        return self._update(filter, update, upsert, many=True)

    def replace_one(self, filter: dict, replacement: dict, upsert: bool = False) -> UpdateResult:
        if any(k.startswith("$") for k in replacement):
            raise ValueError("replacement must not contain update operators")
        with self._store._write() as conn:
            found = self._matching(filter)[:1]
            if not found:
                if not upsert:
                    return UpdateResult(0, 0)
                doc = {**_seed_from_filter(filter), **copy.deepcopy(replacement)}
                return UpdateResult(0, 0, self._insert(conn, doc))
            doc = {**copy.deepcopy(replacement), "_id": found[0]["_id"]}
            self._save(conn, doc)
            return UpdateResult(1, int(_dump(doc) != _dump(found[0])))

    def _delete(self, filt: dict, many: bool) -> DeleteResult:
        with self._store._write() as conn:
            docs = self._matching(filt)
            if not many:
                docs = docs[:1]
            for doc in docs:
                conn.execute(f'DELETE FROM "{self.name}" WHERE id = ?', (_row_id(doc["_id"]),))
                self._unindex(conn, doc)
            return DeleteResult(len(docs))

    def delete_one(self, filter: dict) -> DeleteResult:
        return self._delete(filter, many=False)

    def delete_many(self, filter: dict) -> DeleteResult:
        return self._delete(filter, many=True)

    def drop(self) -> None:
        with self._store._write() as conn:
            conn.execute(f'DELETE FROM "{self.name}"')
            conn.execute("DELETE FROM __indexes__ WHERE coll = ?", (self.name,))
            conn.execute("DELETE FROM __unique__ WHERE coll = ?", (self.name,))


class DocStore:
    """A database: a single SQLite file holding one table per collection."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._tables: set[str] = set()
        with self._write() as conn:
            conn.execute("CREATE TABLE IF NOT EXISTS __indexes__ "
                         "(coll TEXT, name TEXT, fields TEXT, is_unique INTEGER, PRIMARY KEY (coll, name))")
            conn.execute("CREATE TABLE IF NOT EXISTS __unique__ "
                         "(coll TEXT, idx TEXT, key TEXT, row TEXT, PRIMARY KEY (coll, idx, key))")
            conn.execute("CREATE INDEX IF NOT EXISTS __unique_row__ ON __unique__ (coll, row)")

    def _conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            # Autocommit mode; writes take an explicit lock so read-check-write steps are atomic
            # across threads and Passenger worker processes.
            conn = sqlite3.connect(self.path, timeout=30, isolation_level=None)
            conn.execute("PRAGMA busy_timeout = 30000")
            self._local.conn = conn
        return conn

    @contextmanager
    def _write(self):
        conn = self._conn()
        conn.execute("BEGIN IMMEDIATE")
        try:
            yield conn
        except BaseException:
            conn.execute("ROLLBACK")
            raise
        conn.execute("COMMIT")

    def _ensure_table(self, name: str) -> None:
        if name not in self._tables:
            with self._write() as conn:
                conn.execute(f'CREATE TABLE IF NOT EXISTS "{name}" (id TEXT PRIMARY KEY, doc TEXT NOT NULL)')
            self._tables.add(name)

    def __getitem__(self, name: str) -> Collection:
        return Collection(self, name)

    def __getattr__(self, name: str) -> Collection:
        if name.startswith("_"):
            raise AttributeError(name)
        return Collection(self, name)

    def command(self, name: str) -> dict:
        if name != "ping":
            raise NotImplementedError(f"Command {name} is not supported by the built-in store")
        self._conn().execute("SELECT 1")
        return {"ok": 1.0}

    def list_collection_names(self) -> list[str]:
        rows = self._conn().execute("SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE '\\_\\_%' ESCAPE '\\'")
        return [r[0] for r in rows]

    def drop_all(self) -> None:
        for name in self.list_collection_names():
            self[name].drop()
