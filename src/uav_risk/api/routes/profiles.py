from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Body, HTTPException, status
from pydantic import ValidationError

from uav_risk.api.dependencies import get_profile_storage_root
from uav_risk.api.storage import LocalProfileStorage, ProfileStorageError
from uav_risk.core.contracts import DroneProfileRaw

router = APIRouter(prefix="/users/{user_id}/profiles", tags=["profiles"])



def _storage() -> LocalProfileStorage:
    return LocalProfileStorage(get_profile_storage_root())


def _validate_profile_payload(profile: Any) -> Any:
    try:
        return DroneProfileRaw.model_validate(profile)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("", status_code=status.HTTP_201_CREATED)
def create_profile(user_id: str, profile: Any = Body(...)) -> dict[str, object]:
    profile = _validate_profile_payload(profile)
    if user_id != profile.user_id:
        raise HTTPException(status_code=422, detail="Path user_id must match profile.user_id")
    try:
        stored = _storage().create_profile(user_id, profile)
    except ProfileStorageError as exc:
        status_code = status.HTTP_409_CONFLICT if "already exists" in str(exc) else 422
        raise HTTPException(status_code=status_code, detail=str(exc)) from exc
    return {"status": "created", "user_id": user_id, "profile_id": profile.profile_id, "profile": stored}


@router.get("")
def list_profiles(user_id: str) -> dict[str, object]:
    return {"user_id": user_id, "profiles": _storage().list_profiles(user_id)}


@router.get("/{profile_id}")
def get_profile(user_id: str, profile_id: str) -> dict[str, object]:
    profile = _storage().get_profile(user_id, profile_id)
    if profile is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Profile not found")
    return {"status": "found", "user_id": user_id, "profile_id": profile_id, "profile": profile}


@router.put("/{profile_id}")
def update_profile(user_id: str, profile_id: str, profile: Any = Body(...)) -> dict[str, object]:
    profile = _validate_profile_payload(profile)
    if user_id != profile.user_id or profile_id != profile.profile_id:
        raise HTTPException(status_code=422, detail="Path user_id/profile_id must match profile body")
    try:
        stored = _storage().update_profile(user_id, profile_id, profile)
    except ProfileStorageError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if stored is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Profile not found")
    return {"status": "updated", "user_id": user_id, "profile_id": profile_id, "profile": stored}


@router.delete("/{profile_id}")
def delete_profile(user_id: str, profile_id: str) -> dict[str, object]:
    deleted = _storage().delete_profile(user_id, profile_id)
    if not deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Profile not found")
    return {"status": "deleted", "user_id": user_id, "profile_id": profile_id}
