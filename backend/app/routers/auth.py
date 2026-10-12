import time
from collections import defaultdict, deque
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pymongo.errors import DuplicateKeyError

from ..db import get_db
from ..config import get_settings
from ..schemas import GoogleIn, LoginIn, RegisterIn, TokenOut
from ..security import create_token, current_user, hash_password, verify_password

router = APIRouter(prefix="/api/auth", tags=["auth"])

# Brute-force protection: max failed logins per (email, IP) within a sliding window.
MAX_FAILURES, WINDOW_SECONDS = 5, 15 * 60
_failures: dict[str, deque] = defaultdict(deque)


def _limited(key: str) -> bool:
    q = _failures[key]
    while q and q[0] < time.time() - WINDOW_SECONDS:
        q.popleft()
    return len(q) >= MAX_FAILURES


def public_user(u: dict) -> dict:
    return {"id": str(u["_id"]), "name": u["name"], "email": u["email"], "created_at": u["created_at"].isoformat(),
            "has_password": bool(u.get("password_hash")), "google_linked": bool(u.get("google_sub"))}


@router.post("/register", response_model=TokenOut, status_code=201)
def register(body: RegisterIn):
    now = datetime.now(timezone.utc)
    user = {
        "name": body.name, "email": body.email.lower(), "password_hash": hash_password(body.password),
        "created_at": now, "consent_at": now,
    }
    try:
        user["_id"] = get_db().users.insert_one(user).inserted_id
    except DuplicateKeyError:
        raise HTTPException(status.HTTP_409_CONFLICT, "An account with this email already exists")
    return {"token": create_token(str(user["_id"])), "user": public_user(user)}


@router.post("/login", response_model=TokenOut)
def login(body: LoginIn, request: Request):
    key = f"{body.email.lower()}|{request.client.host if request.client else '-'}"
    if _limited(key):
        raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS, "Too many failed attempts. Try again in 15 minutes.")
    user = get_db().users.find_one({"email": body.email.lower()})
    if not user or not user.get("password_hash") or not verify_password(body.password, user["password_hash"]):
        _failures[key].append(time.time())
        # Same message for unknown email and wrong password (prevents account enumeration).
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Incorrect email or password")
    _failures.pop(key, None)
    return {"token": create_token(str(user["_id"])), "user": public_user(user)}


@router.get("/me")
def me(user: dict = Depends(current_user)):
    return public_user(user)


@router.post("/google", response_model=TokenOut)
def google_login(body: GoogleIn):
    """Sign in with a Google ID token (from Google Identity Services); creates the account on first use."""
    client_id = get_settings().google_client_id
    if not client_id:
        raise HTTPException(status.HTTP_501_NOT_IMPLEMENTED, "Google sign-in is not configured on this server")
    try:
        from google.auth.transport import requests as google_requests
        from google.oauth2 import id_token

        info = id_token.verify_oauth2_token(body.credential, google_requests.Request(), client_id)
    except ValueError:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Google sign-in could not be verified")
    if not info.get("email_verified"):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Your Google email address is not verified")
    db = get_db()
    email = info["email"].lower()
    user = db.users.find_one({"email": email})
    now = datetime.now(timezone.utc)
    if user is None:
        user = {"name": (info.get("name") or email.split("@")[0])[:60], "email": email, "google_sub": info["sub"],
                "created_at": now, "consent_at": now}
        user["_id"] = db.users.insert_one(user).inserted_id
    elif not user.get("google_sub"):
        db.users.update_one({"_id": user["_id"]}, {"$set": {"google_sub": info["sub"]}})
        user["google_sub"] = info["sub"]
    return {"token": create_token(str(user["_id"])), "user": public_user(user)}
