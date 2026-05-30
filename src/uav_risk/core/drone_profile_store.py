from __future__ import annotations

# STAGE6_CLEANUP_REVIEW:
# Classification: LEGACY_PROFILE_STORE_REPLACED_BY_API_STORAGE
# Plan lineage: PLAN1_OR_PLAN2_RELIC
# Runtime status: Not part of current FastAPI profile persistence path.
# Legacy signal: Used by old stage2/pipeline.py profile/correction flow; current runtime uses api/storage.py.
# Replacement: src/uav_risk/api/storage.py LocalProfileStorage and profile routes.
# Action rule: Do not call from new code. Remove only after legacy pipeline.py is resolved.
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional


def _default_store_path() -> Path:
    return Path.home() / ".uav_risk" / "drone_profiles.json"


@dataclass
class DroneProfile:
    profile_id: str
    profile_name: str
    uav_model_id: Optional[str] = None
    uav_model_spec: Optional[Dict[str, Any]] = None


class DroneProfileStore:
    """Local per-user drone profile store.

    This is intentionally outside artifacts because it is user-owned data,
    not a packaged model asset.
    """

    def __init__(self, store_path: Optional[str] = None) -> None:
        self.store_path = Path(store_path) if store_path else _default_store_path()

    def _load(self) -> Dict[str, Any]:
        if not self.store_path.exists():
            return {}
        try:
            return json.loads(self.store_path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _save(self, data: Dict[str, Any]) -> None:
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        self.store_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    def list_profiles(self) -> Dict[str, Any]:
        return self._load()

    def get_profile(self, profile_id: str) -> Optional[DroneProfile]:
        data = self._load().get(profile_id)
        if not data:
            return None
        return DroneProfile(
            profile_id=profile_id,
            profile_name=data.get("profile_name", profile_id),
            uav_model_id=data.get("uav_model_id"),
            uav_model_spec=data.get("uav_model_spec"),
        )

    def upsert_profile(self, profile: DroneProfile) -> None:
        data = self._load()
        data[profile.profile_id] = asdict(profile)
        self._save(data)

    def delete_profile(self, profile_id: str) -> bool:
        data = self._load()
        if profile_id in data:
            del data[profile_id]
            self._save(data)
            return True
        return False
