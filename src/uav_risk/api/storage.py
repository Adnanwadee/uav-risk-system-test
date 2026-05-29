from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from uav_risk.api.schemas import AssessmentListItem, AssessmentRecord


class ProfileStorageError(ValueError):
    pass


class AssessmentStorageError(ValueError):
    pass


class AssessmentNotFoundError(AssessmentStorageError):
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


class LocalAssessmentStorage:
    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)

    def _validate_segment(self, value: str, label: str) -> None:
        if not value or "/" in value or "\\" in value or value in {".", ".."}:
            raise AssessmentStorageError(f"Invalid {label}: {value!r}")

    def _assessment_path(self, user_id: str, assessment_id: str) -> Path:
        self._validate_segment(user_id, "user_id")
        self._validate_segment(assessment_id, "assessment_id")
        return self.root / user_id / f"{assessment_id}.json"

    def _write_json(self, path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        tmp_path.replace(path)

    def save_assessment(self, record: AssessmentRecord) -> AssessmentRecord:
        self._validate_segment(record.user_id, "user_id")
        self._validate_segment(record.assessment_id, "assessment_id")
        self._validate_segment(record.profile_id, "profile_id")
        path = self._assessment_path(record.user_id, record.assessment_id)
        payload = record.model_dump()
        self._write_json(path, payload)
        return AssessmentRecord.model_validate(payload)

    def get_assessment(self, user_id: str, assessment_id: str) -> AssessmentRecord:
        path = self._assessment_path(user_id, assessment_id)
        if not path.exists():
            raise AssessmentNotFoundError("Assessment not found")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            return AssessmentRecord.model_validate(payload)
        except AssessmentStorageError:
            raise
        except Exception as exc:
            raise AssessmentStorageError("Failed to read assessment record") from exc

    def list_assessments(self, user_id: str, profile_id: str | None = None) -> list[AssessmentListItem]:
        self._validate_segment(user_id, "user_id")
        if profile_id is not None:
            self._validate_segment(profile_id, "profile_id")

        user_dir = self.root / user_id
        if not user_dir.exists():
            return []

        records: list[AssessmentRecord] = []
        for path in user_dir.glob("*.json"):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                record = AssessmentRecord.model_validate(payload)
            except Exception as exc:
                raise AssessmentStorageError(f"Failed to parse assessment record: {path.name}") from exc
            if profile_id is not None and record.profile_id != profile_id:
                continue
            records.append(record)

        records.sort(key=lambda item: item.created_at, reverse=True)

        items: list[AssessmentListItem] = []
        for record in records:
            summary = None
            if record.final_decision and record.decision_score is not None:
                summary = f"{record.final_decision} ({record.decision_score:.3f})"
            elif record.final_decision:
                summary = record.final_decision
            items.append(
                AssessmentListItem(
                    assessment_id=record.assessment_id,
                    user_id=record.user_id,
                    profile_id=record.profile_id,
                    created_at=record.created_at,
                    status=record.status,
                    final_decision=record.final_decision,
                    decision_score=record.decision_score,
                    confidence_level=record.confidence_level,
                    summary=summary,
                )
            )
        return items
