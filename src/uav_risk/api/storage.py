from __future__ import annotations

import json
from pathlib import Path
from typing import Any

class ProfileStorageError(ValueError):
    pass


class LocalProfileStorage:
    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)

    def _validate_segment(self, value: str, label: str) -> None:
        if not value or "/" in value or "\\" in value or value in {".", ".."}:
            raise ProfileStorageError(f"Invalid {label}: {value!r}")

    def _profile_path(self, user_id: str, profile_id: str) -> Path:
        self._validate_segment(user_id, "user_id")
        self._validate_segment(profile_id, "profile_id")
        return self.root / user_id / f"{profile_id}.json"

    def _write_profile_json(self, path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        tmp_path.replace(path)

    def create_profile(self, user_id: str, profile: Any) -> dict[str, Any]:
        if user_id != profile.user_id:
            raise ProfileStorageError("Path user_id must match profile.user_id")
        path = self._profile_path(user_id, profile.profile_id)
        if path.exists():
            raise ProfileStorageError("Profile already exists")
        payload = profile.model_dump()
        self._write_profile_json(path, payload)
        return payload

    def list_profiles(self, user_id: str) -> list[dict[str, Any]]:
        self._validate_segment(user_id, "user_id")
        user_dir = self.root / user_id
        if not user_dir.exists():
            return []
        profiles = []
        for path in sorted(user_dir.glob("*.json")):
            profiles.append(json.loads(path.read_text(encoding="utf-8")))
        return profiles

    def get_profile(self, user_id: str, profile_id: str) -> dict[str, Any] | None:
        path = self._profile_path(user_id, profile_id)
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def update_profile(self, user_id: str, profile_id: str, profile: Any) -> dict[str, Any] | None:
        if user_id != profile.user_id or profile_id != profile.profile_id:
            raise ProfileStorageError("Path user_id/profile_id must match profile body")
        path = self._profile_path(user_id, profile_id)
        if not path.exists():
            return None
        payload = profile.model_dump()
        self._write_profile_json(path, payload)
        return payload

    def delete_profile(self, user_id: str, profile_id: str) -> bool:
        path = self._profile_path(user_id, profile_id)
        if not path.exists():
            return False
        path.unlink()
        return True
