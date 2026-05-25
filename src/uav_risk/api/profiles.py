from typing import Dict, Any, Optional
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

from uav_risk.core.drone_profile_store import DroneProfileStore, DroneProfile

router = APIRouter()


class ProfileIn(BaseModel):
    profile_id: str
    profile_name: str
    uav_model_id: Optional[str] = None
    uav_model_spec: Optional[Dict[str, Any]] = None


class ProfileOut(BaseModel):
    profile_id: str
    profile_name: str
    uav_model_id: Optional[str] = None
    uav_model_spec: Optional[Dict[str, Any]] = None


@router.get("/profiles", response_model=Dict[str, ProfileOut])
def list_profiles():
    store = DroneProfileStore()
    return store.list_profiles()


@router.get("/profiles/{profile_id}", response_model=ProfileOut)
def get_profile(profile_id: str):
    store = DroneProfileStore()
    p = store.get_profile(profile_id)
    if not p:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Profile not found")
    return ProfileOut(
        profile_id=p.profile_id,
        profile_name=p.profile_name,
        uav_model_id=p.uav_model_id,
        uav_model_spec=p.uav_model_spec,
    )


@router.post("/profiles", status_code=status.HTTP_201_CREATED)
def upsert_profile(payload: ProfileIn):
    store = DroneProfileStore()
    profile = DroneProfile(profile_id=payload.profile_id, profile_name=payload.profile_name, uav_model_id=payload.uav_model_id, uav_model_spec=payload.uav_model_spec)
    store.upsert_profile(profile)
    return {"ok": True, "profile_id": payload.profile_id}


@router.delete("/profiles/{profile_id}")
def delete_profile(profile_id: str):
    store = DroneProfileStore()
    deleted = store.delete_profile(profile_id)
    if not deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Profile not found")
    return {"ok": True}
