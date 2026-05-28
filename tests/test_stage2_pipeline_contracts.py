from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from uav_risk.stage2.contracts import (
    AgentRecommendation,
    AgentResult,
    MLAssessmentSnapshot,
    PublicReasoningTrace,
    SHAPFeatureAttribution,
    Stage2AssessmentInput,
    Stage2AssessmentResult,
    Stage2Error,
    Stage2Status,
)


def _agent_result() -> AgentResult:
    return AgentResult(
        status=Stage2Status.COMPLETED,
        recommendation=AgentRecommendation.GO,
        confidence=0.7,
        findings=[
            {
                "finding_id": "f1",
                "finding_type": "tool_check",
                "severity": "info",
                "summary": "ok",
                "requires_evidence": False,
            }
        ],
        action_items=[],
        reasoning_trace=PublicReasoningTrace(),
        evidence_bundles=[],
        errors=[],
    )


def test_ml_snapshot_rejects_empty_predicted_class() -> None:
    with pytest.raises(ValidationError):
        MLAssessmentSnapshot(predicted_class=" ", probabilities={"x": 0.1})


def test_ml_snapshot_rejects_probability_below_zero() -> None:
    with pytest.raises(ValidationError):
        MLAssessmentSnapshot(predicted_class="High", probabilities={"x": -0.1})


def test_ml_snapshot_rejects_probability_above_one() -> None:
    with pytest.raises(ValidationError):
        MLAssessmentSnapshot(predicted_class="High", probabilities={"x": 1.1})


def test_ml_snapshot_rejects_negative_raw_feature_count() -> None:
    with pytest.raises(ValidationError):
        MLAssessmentSnapshot(predicted_class="High", probabilities={"x": 0.1}, raw_feature_count=-1)


def test_shap_feature_rejects_empty_feature() -> None:
    with pytest.raises(ValidationError):
        SHAPFeatureAttribution(feature=" ", value=1, importance=0.2)


def test_stage2_assessment_input_rejects_empty_user_id() -> None:
    with pytest.raises(ValidationError):
        Stage2AssessmentInput(
            user_id=" ",
            profile_id="p1",
            scenario_summary={},
            ml=MLAssessmentSnapshot(predicted_class="High", probabilities={"High": 1.0}),
        )


def test_stage2_assessment_input_rejects_empty_profile_id() -> None:
    with pytest.raises(ValidationError):
        Stage2AssessmentInput(
            user_id="u1",
            profile_id=" ",
            scenario_summary={},
            ml=MLAssessmentSnapshot(predicted_class="High", probabilities={"High": 1.0}),
        )


def test_stage2_assessment_result_completed_requires_agent_result() -> None:
    with pytest.raises(ValidationError):
        Stage2AssessmentResult(
            status=Stage2Status.COMPLETED,
            evidence_bundles=[],
            agent_result=None,
            errors=[],
        )


def test_stage2_assessment_result_failed_requires_errors() -> None:
    with pytest.raises(ValidationError):
        Stage2AssessmentResult(
            status=Stage2Status.FAILED,
            evidence_bundles=[],
            agent_result=None,
            errors=[],
        )


def test_stage2_assessment_result_json_serializable() -> None:
    result = Stage2AssessmentResult(
        status=Stage2Status.COMPLETED,
        assessment_id="a1",
        evidence_bundles=[],
        agent_result=_agent_result(),
        errors=[],
        metadata={},
    )
    encoded = json.dumps(result.model_dump())
    decoded = json.loads(encoded)
    assert decoded["status"] == "completed"
