"""GDPR rights: data export (right of access) and account erasure (right to be forgotten)."""
from fastapi import APIRouter, Depends, HTTPException, Response, status

from .. import storage
from ..db import get_db
from ..schemas import PasswordIn, ProfileIn
from ..security import current_user, hash_password, verify_password
from .auth import public_user
from .designs import serialize

router = APIRouter(prefix="/api/account", tags=["account"])


@router.get("/export")
def export_data(user: dict = Depends(current_user)):
    designs = [serialize(d) for d in get_db().designs.find({"user_id": user["_id"]})]
    return {"user": public_user(user), "consent_at": user["consent_at"].isoformat() if user.get("consent_at") else None, "designs": designs}


@router.patch("")
def update_profile(body: ProfileIn, user: dict = Depends(current_user)):
    get_db().users.update_one({"_id": user["_id"]}, {"$set": {"name": body.name}})
    return public_user({**user, "name": body.name})


@router.post("/password", status_code=204)
def change_password(body: PasswordIn, user: dict = Depends(current_user)):
    if not user.get("password_hash"):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "This account signs in with Google and has no password")
    if not verify_password(body.current_password, user["password_hash"]):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Current password is incorrect")
    get_db().users.update_one({"_id": user["_id"]}, {"$set": {"password_hash": hash_password(body.new_password)}})
    return Response(status_code=204)


@router.delete("", status_code=204)
def delete_account(user: dict = Depends(current_user)):
    db = get_db()
    db.designs.delete_many({"user_id": user["_id"]})
    db.users.delete_one({"_id": user["_id"]})
    storage.delete_user_files(str(user["_id"]))
    return Response(status_code=204)
