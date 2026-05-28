from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.encoders import jsonable_encoder

from uav_risk.api.dependencies import get_profile_storage_root, get_stage1_bundle
from uav_risk.api.schemas import AssessmentRequest
from uav_risk.api.storage import LocalProfileStorage
from uav_risk.core.contracts import AssessmentCoreInput, DroneProfileRaw
from uav_risk.core.data_validator import run_structural_hard_veto
from uav_risk.ml.inference import run_stage1_inference
from uav_risk.ml.loader import ModelLoadError, assemble_raw_feature_vector
from uav_risk.ml.schemas import Stage1Bundle
from uav_risk.api.schemas import Stage2AssessmentRequest
from uav_risk.stage2.pipeline_v2 import Stage2PipelineV2
from uav_risk.stage2.contracts import (
    Stage2AssessmentInput,
    MLAssessmentSnapshot,
    EvidenceSupportStatus,
)
from uav_risk.stage2.rag.quality import build_runtime_rag_adapter_if_available
from uav_risk.stage2.agent.operational_agent import OperationalAgentV2
from uav_risk.stage2.llm.orchestrator import LLMOrchestrator
from uav_risk.stage2.reporting import build_operational_report, render_markdown_report

router = APIRouter(prefix="/users/{user_id}/profiles/{profile_id}/assessments", tags=["assessments"])


def _issue_payload(validation_result) -> list[dict[str, object]]:
    return [
        {
            "code": issue.code,
            "field": issue.field,
            "message": issue.message,
            "details": issue.details,
        }
        for issue in validation_result.issues
    ]


def _blocked(user_id: str, profile_id: str, issues: list[dict[str, object]]) -> dict[str, object]:
    return {"status": "blocked", "user_id": user_id, "profile_id": profile_id, "issues": issues, "ml": None}


def _blocked_stage2(user_id: str, profile_id: str, issues: list[dict[str, object]]) -> dict[str, object]:
    return {
        "status": "blocked",
        "user_id": user_id,
        "profile_id": profile_id,
        "issues": issues,
        "ml": None,
        "decision": {"final_decision": "no_go"},
    }


@router.post("")
def create_assessment(
    user_id: str,
    profile_id: str,
    request_body: AssessmentRequest,
    bundle: Stage1Bundle = Depends(get_stage1_bundle),
) -> dict[str, object]:
    stored_profile = LocalProfileStorage(get_profile_storage_root()).get_profile(user_id, profile_id)
    if stored_profile is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Profile not found")

    profile = DroneProfileRaw.model_validate(stored_profile)
    assessment = AssessmentCoreInput(
        user_id=user_id,
        profile_id=profile_id,
        drone_profile=profile,
        scenario=request_body.scenario,
        secondary_overrides=request_body.secondary_overrides,
    )
    veto = run_structural_hard_veto(assessment)
    if not veto.passed:
        return _blocked(user_id, profile_id, _issue_payload(veto))

    override_values = request_body.secondary_overrides.values
    try:
        raw_vector, metadata = assemble_raw_feature_vector(
            profile,
            request_body.scenario,
            overrides=override_values,
            bundle=bundle,
        )
    except ModelLoadError as exc:
        return _blocked(
            user_id,
            profile_id,
            [{"code": "RAW_ASSEMBLY_FAILED", "field": None, "message": str(exc), "details": {}}],
        )

    try:
        ml_result = run_stage1_inference(bundle, raw_vector, compute_shap=True)
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="ML inference failed") from exc

    raw_feature_map = metadata["raw_feature_map"]
    applied_overrides = {key: raw_feature_map[key] for key in override_values if key in raw_feature_map}
    response = {
        "status": "completed",
        "user_id": user_id,
        "profile_id": profile_id,
        "ml": {
            "predicted_class": ml_result.risk_class.value,
            "probabilities": ml_result.probabilities,
        },
        "shap": {"top_features": [feature.to_dict() for feature in ml_result.top_features]},
        "raw_feature_count": metadata["raw_feature_vector_length"],
        "processed_feature_count": metadata["processed_feature_vector_length"],
        "warnings": [],
        "operator_notes": request_body.operator_notes,
        "raw_feature_diagnostics": {
            "applied_secondary_overrides": applied_overrides,
        },
    }
    return jsonable_encoder(response)


@router.post("/stage2")
async def create_assessment_stage2(
    user_id: str,
    profile_id: str,
    request_body: Stage2AssessmentRequest,
    bundle: Stage1Bundle = Depends(get_stage1_bundle),
) -> dict[str, object]:
    stored_profile = LocalProfileStorage(get_profile_storage_root()).get_profile(user_id, profile_id)
    if stored_profile is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Profile not found")

    profile = DroneProfileRaw.model_validate(stored_profile)
    assessment = AssessmentCoreInput(
        user_id=user_id,
        profile_id=profile_id,
        drone_profile=profile,
        scenario=request_body.scenario,
        secondary_overrides=request_body.secondary_overrides,
    )
    veto = run_structural_hard_veto(assessment)
    if not veto.passed:
        return _blocked_stage2(user_id, profile_id, _issue_payload(veto))

    override_values = request_body.secondary_overrides.values
    try:
        raw_vector, metadata = assemble_raw_feature_vector(
            profile,
            request_body.scenario,
            overrides=override_values,
            bundle=bundle,
        )
    except ModelLoadError as exc:
        return _blocked_stage2(
            user_id,
            profile_id,
            [{"code": "RAW_ASSEMBLY_FAILED", "field": None, "message": str(exc), "details": {}}],
        )

    try:
        ml_result = run_stage1_inference(bundle, raw_vector, compute_shap=True)
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="ML inference failed") from exc

    # Build ML snapshot for Stage2
    shap_top = [
        {
            "feature": getattr(item, "feature_name", getattr(item, "feature", "")),
            "value": getattr(item, "feature_value", getattr(item, "value", None)),
            "importance": getattr(item, "shap_value", getattr(item, "importance", 0.0)),
            "direction": getattr(item, "direction", None),
            "metadata": {},
        }
        for item in ml_result.top_features
    ]

    ml_snapshot = MLAssessmentSnapshot(
        predicted_class=ml_result.risk_class.value,
        probabilities=dict(ml_result.probabilities),
        shap_top_features=shap_top,
        raw_feature_count=metadata.get("raw_feature_vector_length"),
        processed_feature_count=metadata.get("processed_feature_vector_length"),
        metadata={"model_version": ml_result.model_version},
    )

    stage2_input = Stage2AssessmentInput(
        assessment_id=None,
        user_id=user_id,
        profile_id=profile_id,
        scenario_summary=dict(request_body.scenario.model_dump()),
        ml=ml_snapshot,
        evidence_bundles=[],
        operator_notes=request_body.operator_notes,
        metadata={
            "raw_feature_count": metadata.get("raw_feature_vector_length"),
            "processed_feature_count": metadata.get("processed_feature_vector_length"),
            "model_version": ml_result.model_version,
        },
    )

    # Build runtime RAG adapter if available (may return None)
    rag_adapter = build_runtime_rag_adapter_if_available()

    # Build operational agent and LLM orchestrator (no external provider -> deterministic fallback)
    operational_agent = OperationalAgentV2(rag_adapter=rag_adapter)
    llm_orchestrator = LLMOrchestrator()

    pipeline = Stage2PipelineV2(
        rag_adapter=rag_adapter,
        operational_agent=operational_agent,
        llm_orchestrator=llm_orchestrator,
        max_evidence_queries=3,
    )

    stage2_result = await pipeline.run(stage2_input)

    report = build_operational_report(stage2_input, stage2_result)
    markdown = render_markdown_report(report)

    # Build response shape
    response: dict[str, object] = {
        "status": stage2_result.status.value,
        "user_id": user_id,
        "profile_id": profile_id,
        "assessment_id": stage2_result.assessment_id,
        "warnings": [],
        "errors": [err.model_dump() for err in stage2_result.errors],
        "stage1": {
            "ml": {
                "predicted_class": ml_result.risk_class.value,
                "probabilities": dict(ml_result.probabilities),
                "raw_feature_count": metadata.get("raw_feature_vector_length"),
                "processed_feature_count": metadata.get("processed_feature_vector_length"),
            },
            "shap": {"top_features": [f for f in shap_top]},
        },
        "stage2": {
            "rag": {
                "retrieval_usable": bool(rag_adapter),
                "rag_quality_is_proven": bool(stage2_result.metadata.get("rag_quality_is_proven") if isinstance(stage2_result.metadata, dict) else False),
                "evidence_bundle_count": len(stage2_result.evidence_bundles),
                "insufficient_evidence_count": sum(1 for b in stage2_result.evidence_bundles if b.support_status == EvidenceSupportStatus.INSUFFICIENT_EVIDENCE),
                "evidence_bundle_details": [
                    {"bundle_id": b.bundle_id, "query": b.query, "support_status": b.support_status.value} for b in stage2_result.evidence_bundles
                ],
            },
            "agent": {
                "recommendation": stage2_result.agent_result.recommendation.value if stage2_result.agent_result else None,
                "findings": [f.model_dump() for f in (stage2_result.agent_result.findings if stage2_result.agent_result else [])],
                "action_items": [a.model_dump() for a in (stage2_result.agent_result.action_items if stage2_result.agent_result else [])],
                "limitations": list(stage2_result.agent_result.reasoning_trace.limitations) if stage2_result.agent_result else [],
            },
            "decision": stage2_result.decision.model_dump() if stage2_result.decision else None,
            "llm_synthesis": stage2_result.llm_synthesis.model_dump() if stage2_result.llm_synthesis else {"status": "fallback"},
            "report": {"structured": report.model_dump(), "markdown": markdown},
        },
        "diagnostics": {
            "path_resolution_status": stage2_result.metadata.get("provenance_status") if isinstance(stage2_result.metadata, dict) else None,
            "index_provenance_status": stage2_result.metadata.get("provenance_status") if isinstance(stage2_result.metadata, dict) else None,
            "retrieval_usable": bool(rag_adapter),
            "rag_quality_is_proven": bool(stage2_result.metadata.get("rag_quality_is_proven") if isinstance(stage2_result.metadata, dict) else False),
            "scenario_evidence_complete": True if stage2_result.evidence_bundles else False,
            "llm_mode": "fallback",
            "external_llm_provider_used": False,
        },
    }

    return jsonable_encoder(response)
