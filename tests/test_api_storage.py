from __future__ import annotations

import pytest

from uav_risk.api.schemas import AssessmentRecord
from uav_risk.api.storage import AssessmentNotFoundError, AssessmentStorageError, LocalAssessmentStorage


def _record(
    user_id: str = "user_1",
    profile_id: str = "profile_1",
    assessment_id: str = "a1",
    created_at: str = "2026-05-29T00:00:00Z",
) -> AssessmentRecord:
    return AssessmentRecord(
        assessment_id=assessment_id,
        user_id=user_id,
        profile_id=profile_id,
        created_at=created_at,
        status="completed",
        final_decision="caution",
        decision_score=0.42,
        confidence_level="medium",
        stage1={"ml": {"predicted_class": "Medium Risk"}},
        stage2={"decision": {"final_decision": "caution"}},
        report={"generated": True},
        system_work_trace={"entries": [], "summary": "ok", "public_safe": True},
        diagnostics={"llm_mode": "fallback"},
        warnings=[],
        errors=[],
    )


def test_local_assessment_storage_save_get_and_list(tmp_path):
    storage = LocalAssessmentStorage(tmp_path / "assessments")

    first = storage.save_assessment(_record(assessment_id="a1", created_at="2026-05-29T00:00:00Z"))
    second = storage.save_assessment(_record(assessment_id="a2", created_at="2026-05-29T01:00:00Z"))

    got = storage.get_assessment("user_1", first.assessment_id)
    assert got.assessment_id == "a1"

    items = storage.list_assessments("user_1")
    assert [item.assessment_id for item in items] == [second.assessment_id, first.assessment_id]


def test_local_assessment_storage_filters_by_profile_id(tmp_path):
    storage = LocalAssessmentStorage(tmp_path / "assessments")

    storage.save_assessment(_record(profile_id="profile_1", assessment_id="a1"))
    storage.save_assessment(_record(profile_id="profile_2", assessment_id="a2"))

    items = storage.list_assessments("user_1", profile_id="profile_2")
    assert len(items) == 1
    assert items[0].profile_id == "profile_2"


def test_local_assessment_storage_get_missing_raises_not_found(tmp_path):
    storage = LocalAssessmentStorage(tmp_path / "assessments")
    with pytest.raises(AssessmentNotFoundError):
        storage.get_assessment("user_1", "missing")


@pytest.mark.parametrize(
    "user_id,assessment_id,profile_id",
    [
        ("../bad", "a1", "profile_1"),
        ("user_1", "../bad", "profile_1"),
        ("user_1", "a1", "../bad"),
        ("", "a1", "profile_1"),
    ],
)
def test_local_assessment_storage_rejects_invalid_path_segments(tmp_path, user_id, assessment_id, profile_id):
    storage = LocalAssessmentStorage(tmp_path / "assessments")
    with pytest.raises(AssessmentStorageError):
        storage.save_assessment(_record(user_id=user_id, assessment_id=assessment_id, profile_id=profile_id))


def test_local_assessment_storage_list_rejects_invalid_user_or_profile(tmp_path):
    storage = LocalAssessmentStorage(tmp_path / "assessments")
    with pytest.raises(AssessmentStorageError):
        storage.list_assessments("../bad")
    with pytest.raises(AssessmentStorageError):
        storage.list_assessments("user_1", profile_id="../bad")
