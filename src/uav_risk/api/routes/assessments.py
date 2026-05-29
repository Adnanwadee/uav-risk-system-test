from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.encoders import jsonable_encoder

from uav_risk.api.dependencies import (
    get_assessment_storage_root,
    get_profile_storage_root,
    get_stage1_bundle,
)
from uav_risk.api.schemas import (
    AssessmentListItem,
    AssessmentRecord,
    AssessmentRequest,
    Stage2AISection,
    Stage2AgentSection,
    Stage2AssessmentRequest,
    Stage2AssessmentResponse,
    Stage2DecisionSection,
    Stage2DiagnosticsSection,
    Stage2LLMSection,
    Stage2PolicySection,
    Stage2RAGSection,
    Stage2ReportSection,
    Stage1AssessmentSection,
    Stage1MLSection,
    Stage1SHAPSection,
)
from uav_risk.api.storage import (
    AssessmentNotFoundError,
    AssessmentStorageError,
    LocalAssessmentStorage,
    LocalProfileStorage,
)
from uav_risk.core.contracts import AssessmentCoreInput, DroneProfileRaw
from uav_risk.core.data_validator import run_structural_hard_veto
from uav_risk.ml.inference import run_stage1_inference
from uav_risk.ml.loader import ModelLoadError, assemble_raw_feature_vector
from uav_risk.ml.schemas import Stage1Bundle
from uav_risk.stage2.agent.operational_agent import OperationalAgentV2
from uav_risk.stage2.contracts import (
    EvidenceSupportStatus,
    MLAssessmentSnapshot,
    Stage2AssessmentInput,
    Stage2ProfileContext,
    SystemWorkTrace,
    SystemWorkTraceEntry,
)
from uav_risk.stage2.llm.orchestrator import build_llm_orchestrator_from_env
from uav_risk.stage2.pipeline_v2 import Stage2PipelineV2
from uav_risk.stage2.rag.quality import build_runtime_rag_adapter_if_available
from uav_risk.stage2.reporting import build_operational_report, render_markdown_report
from uav_risk.utils.json_sanitize import (
    sanitize_system_work_trace_public,
    sanitize_tool_trace_public,
    sanitize_working_memory_public,
)

router = APIRouter(prefix="/users/{user_id}/profiles/{profile_id}/assessments", tags=["assessments"])
history_router = APIRouter(prefix="/users/{user_id}/assessments", tags=["assessments"])

FORBIDDEN_KEYS = {
    "reasoning_steps",
    "chain_of_thought",
    "reasoning_chain",
    "thoughts",
    "thought",
    "raw_prompt",
    "prompt",
    "raw_completion",
    "completion",
    "raw_llm_response",
    "tool_history",
    "internal_memory",
    "scratchpad",
    "hidden",
    "api_key",
    "secret",
    "token",
    "authorization",
    "internal_reasoning",
    "private_reasoning",
}


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


def _blocked_stage2(user_id: str, profile_id: str, issues: list[dict[str, object]]) -> Stage2AssessmentResponse:
    aid = str(uuid4())
    created_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    blocked_trace = SystemWorkTrace(
        entries=[],
        summary="Assessment blocked before Stage2 execution.",
        public_safe=True,
    ).model_dump()
    return Stage2AssessmentResponse(
        status="blocked",
        user_id=user_id,
        profile_id=profile_id,
        assessment_id=aid,
        created_at=created_at,
        persisted=False,
        persistence_status="not_persisted_blocked",
        system_work_trace=blocked_trace,
        warnings=[],
        errors=[{"code": "HARD_VETO", "message": "Core structural hard-veto blocked assessment.", "details": {"issues": issues}}],
        stage1=Stage1AssessmentSection(core={"blocked": True, "issues": issues}, ml=None, shap=None),
        stage2=Stage2AISection(
            policy=Stage2PolicySection(),
            rag=Stage2RAGSection(
                retrieval_usable=False,
                rag_quality_is_proven=False,
                evidence_bundle_count=0,
                insufficient_evidence_count=0,
                scenario_evidence_complete=None,
                scenario_evidence_status="blocked",
                evidence_bundle_details=[],
                citations=[],
                limitations=["Assessment blocked before Stage2 execution due to structural hard-veto."],
            ),
            agent=Stage2AgentSection(system_work_trace=blocked_trace),
            decision=Stage2DecisionSection(
                final_decision="no_go",
                decision_score=1.0,
                confidence_level="high",
                stage_weights={},
                stage_contributions=[],
                decision_reasons=["Core hard-veto blocked mission input."],
                blocking_reasons=["core_hard_veto"],
                required_actions=["Resolve hard-veto issues before retrying Stage2 assessment."],
                limitations=[],
            ),
            llm_synthesis=Stage2LLMSection(status="disabled", external_provider_used=False),
            report=Stage2ReportSection(markdown=None, sections=[], generated=False),
        ),
        diagnostics=Stage2DiagnosticsSection(
            persistence_status="not_persisted_blocked",
            retrieval_usable=False,
            rag_quality_is_proven=False,
            scenario_evidence_complete=None,
            llm_mode="disabled",
            external_llm_provider_used=False,
            warnings=[],
        ),
    )


def _build_stage2_profile_context(profile: DroneProfileRaw) -> Stage2ProfileContext:
    sensor_summary: list[str] = []
    sensor_map = {
        "gnss_available": getattr(profile, "uav_sensors_gnss", None),
        "camera_available": (
            bool(getattr(profile, "uav_sensors_camera_rgb", 0.0) or 0.0)
            or bool(getattr(profile, "uav_sensors_camera_thermal", 0.0) or 0.0)
        ),
        "lidar_available": getattr(profile, "uav_sensors_lidar", None),
        "radar_available": getattr(profile, "uav_sensors_radar", None),
        "parachute_available": getattr(profile, "uav_parachute_available", None),
        "detect_and_avoid_available": getattr(profile, "mission_detect_and_avoid_active", None),
    }
    for key, value in sensor_map.items():
        if isinstance(value, bool) and value:
            sensor_summary.append(key.replace("_available", ""))
        elif isinstance(value, (int, float)) and float(value) > 0.0:
            sensor_summary.append(key.replace("_available", ""))

    return Stage2ProfileContext(
        profile_id=getattr(profile, "profile_id", None),
        profile_name=getattr(profile, "profile_name", None),
        uav_mass_kg=getattr(profile, "uav_mass_kg", None),
        max_payload_kg=getattr(profile, "max_payload_kg", None),
        max_takeoff_mass_kg=getattr(profile, "max_takeoff_mass_kg", None),
        max_speed_mps=getattr(profile, "uav_max_speed_mps", None),
        max_flight_time_min=getattr(profile, "uav_max_flight_time_min", None),
        reserve_fraction=getattr(profile, "uav_reserve_fraction", None),
        hover_ceiling_m=getattr(profile, "uav_hover_ceiling_m", None),
        max_altitude_m=getattr(profile, "airspace_altitude_asl_max_m", None),
        swarm_capable=getattr(profile, "swarm_capable", None),
        max_swarm_size=getattr(profile, "max_swarm_size", None),
        runway_required=None,
        gnss_available=bool(getattr(profile, "uav_sensors_gnss", 0.0) or 0.0),
        camera_available=bool(getattr(profile, "uav_sensors_camera_rgb", 0.0) or 0.0) or bool(getattr(profile, "uav_sensors_camera_thermal", 0.0) or 0.0),
        lidar_available=bool(getattr(profile, "uav_sensors_lidar", 0.0) or 0.0),
        radar_available=bool(getattr(profile, "uav_sensors_radar", 0.0) or 0.0),
        parachute_available=bool(getattr(profile, "uav_parachute_available", 0.0) or 0.0),
        detect_and_avoid_available=bool(getattr(profile, "mission_detect_and_avoid_active", 0.0) or 0.0),
        sensor_summary=sorted(set(sensor_summary)),
        metadata={"runway_capable": bool(getattr(profile, "runway_capable", False))},
    )


def _clean_forbidden_keys(value):
    if isinstance(value, dict):
        out = {}
        for k, v in value.items():
            if str(k).lower() in FORBIDDEN_KEYS:
                continue
            out[k] = _clean_forbidden_keys(v)
        return out
    if isinstance(value, list):
        return [_clean_forbidden_keys(v) for v in value]
    return value



def _build_system_work_trace(
    agent_result,
    *,
    stage2_result=None,
) -> dict[str, object]:
    if agent_result is None and stage2_result is None:
        return SystemWorkTrace(
            entries=[],
            summary="No Stage2 execution trace was available.",
            public_safe=True,
        ).model_dump()

    entries: list[SystemWorkTraceEntry] = []
    if agent_result is not None:
        for idx, tool_call in enumerate(agent_result.tool_trace, start=1):
            metadata = tool_call.metadata if isinstance(tool_call.metadata, dict) else {}
            duration_value = metadata.get("duration_ms")
            duration_ms = int(duration_value) if isinstance(duration_value, (int, float)) and duration_value >= 0 else None
            warning_lines = [] if tool_call.status.lower() == "ok" else [f"tool_status={tool_call.status}"]

            entries.append(
                SystemWorkTraceEntry(
                    step_id=f"agent_step_{idx}",
                    stage="operational_agent_v2",
                    tool_name=tool_call.tool_name.value,
                    status=tool_call.status,
                    input_summary=tool_call.input_summary,
                    output_summary=tool_call.output_summary,
                    evidence_ids=list(tool_call.related_evidence_ids),
                    warnings=warning_lines,
                    started_at=metadata.get("started_at") if isinstance(metadata.get("started_at"), str) else None,
                    completed_at=metadata.get("completed_at") if isinstance(metadata.get("completed_at"), str) else None,
                    duration_ms=duration_ms,
                    public_safe=True,
                )
            )


    stage2_meta = {}
    if stage2_result is not None and isinstance(getattr(stage2_result, "metadata", None), dict):
        stage2_meta = dict(stage2_result.metadata)

    if stage2_meta:
        planned_queries = str(stage2_meta.get("planned_scenario_queries") or "")
        selected_queries = str(stage2_meta.get("selected_scenario_queries") or "")
        skipped_queries = str(stage2_meta.get("skipped_scenario_queries") or "")
        coverage_status = str(stage2_meta.get("corpus_coverage_status") or "unknown")
        reranker_reason = str(stage2_meta.get("reranker_reason") or "unknown")

        entries.append(
            SystemWorkTraceEntry(
                step_id=f"stage2_step_{len(entries) + 1}",
                stage="stage2_pipeline_v2",
                tool_name="scenario_rag_query_planner",
                status="ok",
                input_summary=f"planned_query_count={stage2_meta.get('planned_scenario_query_count')}",
                output_summary=f"selected_queries={selected_queries or 'none'} skipped_queries={skipped_queries or 'none'}",
                evidence_ids=[],
                warnings=[],
                public_safe=True,
            )
        )

        entries.append(
            SystemWorkTraceEntry(
                step_id=f"stage2_step_{len(entries) + 1}",
                stage="stage2_rag_adapter",
                tool_name="corpus_coverage_check",
                status="ok" if coverage_status in {"complete", "unknown"} else "partial",
                input_summary=f"expected_sources={stage2_meta.get('expected_source_count')}",
                output_summary=f"coverage_status={coverage_status} indexed_sources={stage2_meta.get('indexed_source_count')}",
                evidence_ids=[],
                warnings=[f"missing_sources={stage2_meta.get('missing_sources')}"] if stage2_meta.get('missing_sources') else [],
                public_safe=True,
            )
        )

        entries.append(
            SystemWorkTraceEntry(
                step_id=f"stage2_step_{len(entries) + 1}",
                stage="stage2_rag_adapter",
                tool_name="reranker_status",
                status="ok" if bool(stage2_meta.get("reranker_used")) else "partial",
                input_summary=f"configured={stage2_meta.get('reranker_configured')} available={stage2_meta.get('reranker_available')}",
                output_summary=f"used={stage2_meta.get('reranker_used')} reason={reranker_reason}",
                evidence_ids=[],
                warnings=[],
                public_safe=True,
            )
        )

    if stage2_result is not None and getattr(stage2_result, "decision", None) is not None:
        decision = stage2_result.decision
        entries.append(
            SystemWorkTraceEntry(
                step_id=f"stage2_step_{len(entries) + 1}",
                stage="decision_engine",
                tool_name="weighted_decision_engine",
                status="ok",
                input_summary=f"stage_contributions={len(decision.stage_contributions)}",
                output_summary=f"final_decision={decision.final_decision.value} decision_score={decision.decision_score}",
                evidence_ids=[ref.claim_id for ref in decision.evidence_refs[:8]],
                warnings=[],
                public_safe=True,
            )
        )

    if stage2_result is not None and getattr(stage2_result, "llm_synthesis", None) is not None:
        synthesis = stage2_result.llm_synthesis
        entries.append(
            SystemWorkTraceEntry(
                step_id=f"stage2_step_{len(entries) + 1}",
                stage="llm_orchestrator",
                tool_name="llm_synthesis",
                status="ok" if synthesis.status.value in {"generated", "fallback"} else "partial",
                input_summary=f"provider={synthesis.provider or 'fallback'} model={synthesis.model_name or 'not_configured'}",
                output_summary=f"synthesis_status={synthesis.status.value} warnings={len(synthesis.consistency_warnings)}",
                evidence_ids=list(synthesis.evidence_reference_ids[:8]),
                warnings=[w.warning_type for w in synthesis.consistency_warnings[:8]],
                public_safe=True,
            )
        )

    return SystemWorkTrace(
        entries=entries,
        summary=f"{len(entries)} summarized tool steps are available for public transparency.",
        public_safe=True,
    ).model_dump()


def _build_stage2_citations(evidence_bundles) -> list[dict[str, object]]:
    items: list[dict[str, object]] = []
    for bundle in evidence_bundles:
        for rank, citation in enumerate(bundle.citations, start=1):
            meta = citation.metadata if isinstance(citation.metadata, dict) else {}
            items.append(
                {
                    "citation_id": citation.citation_id,
                    "bundle_id": bundle.bundle_id,
                    "source_filename": meta.get("source_filename"),
                    "source_title": citation.source_title,
                    "page_start": meta.get("page_start", citation.page),
                    "page_end": meta.get("page_end"),
                    "section_title": meta.get("section_title", citation.section),
                    "quote": citation.quote,
                    "retrieval_score": citation.retrieval_score,
                    "confidence_label": meta.get("confidence_label"),
                    "rank": meta.get("rank", rank),
                    "retrieval_origin": meta.get("retrieval_origin") or (bundle.metadata.get("retrieval_origin") if isinstance(bundle.metadata, dict) else None),
                    "support_status": meta.get("support_status") or (bundle.metadata.get("evidence_status") if isinstance(bundle.metadata, dict) else None),
                    "synthetic": bool(meta.get("synthetic") if "synthetic" in meta else (bundle.metadata.get("synthetic") if isinstance(bundle.metadata, dict) else False)),
                }
            )
    return items


def _top_probability(probs: dict[str, float]) -> float | None:
    if not probs:
        return None
    return max(probs.values())


def _assessment_storage() -> LocalAssessmentStorage:
    return LocalAssessmentStorage(get_assessment_storage_root())


def _normalize_messages(value: list[object]) -> list[str]:
    messages: list[str] = []
    for item in value:
        if isinstance(item, dict):
            code = str(item.get("code", "")).strip()
            message = str(item.get("message", "")).strip()
            if code and message:
                messages.append(f"{code}: {message}")
            elif message:
                messages.append(message)
            elif code:
                messages.append(code)
            else:
                messages.append(str(item))
        else:
            messages.append(str(item))
    return messages


def _is_invalid_storage_segment(exc: AssessmentStorageError) -> bool:
    return str(exc).startswith("Invalid ")


def _build_assessment_record_from_response(
    *,
    user_id: str,
    profile_id: str,
    response_payload: dict[str, object],
    created_at: str,
) -> AssessmentRecord:
    stage2_payload = response_payload.get("stage2") if isinstance(response_payload.get("stage2"), dict) else {}
    stage2_agent = stage2_payload.get("agent") if isinstance(stage2_payload, dict) and isinstance(stage2_payload.get("agent"), dict) else {}
    stage2_decision = stage2_payload.get("decision") if isinstance(stage2_payload, dict) and isinstance(stage2_payload.get("decision"), dict) else {}

    return AssessmentRecord(
        assessment_id=str(response_payload.get("assessment_id", "")),
        user_id=user_id,
        profile_id=profile_id,
        created_at=created_at,
        status=str(response_payload.get("status", "")),
        final_decision=stage2_decision.get("final_decision"),
        decision_score=stage2_decision.get("decision_score"),
        confidence_level=stage2_decision.get("confidence_level"),
        stage1=_clean_forbidden_keys(response_payload.get("stage1") if isinstance(response_payload.get("stage1"), dict) else {}),
        stage2=_clean_forbidden_keys(stage2_payload if isinstance(stage2_payload, dict) else {}),
        report=_clean_forbidden_keys(stage2_payload.get("report") if isinstance(stage2_payload, dict) else None),
        system_work_trace=_clean_forbidden_keys(stage2_agent.get("system_work_trace") if isinstance(stage2_agent, dict) else None),
        diagnostics=_clean_forbidden_keys(response_payload.get("diagnostics") if isinstance(response_payload.get("diagnostics"), dict) else {}),
        warnings=_normalize_messages(response_payload.get("warnings") if isinstance(response_payload.get("warnings"), list) else []),
        errors=_normalize_messages(response_payload.get("errors") if isinstance(response_payload.get("errors"), list) else []),
    )


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


@router.post("/stage2", response_model=Stage2AssessmentResponse)
async def create_assessment_stage2(
    user_id: str,
    profile_id: str,
    request_body: Stage2AssessmentRequest,
    bundle: Stage1Bundle = Depends(get_stage1_bundle),
) -> Stage2AssessmentResponse:
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

    aid = str(uuid4())
    ml_snapshot = MLAssessmentSnapshot(
        predicted_class=ml_result.risk_class.value,
        probabilities=dict(ml_result.probabilities),
        shap_top_features=shap_top,
        raw_feature_count=metadata.get("raw_feature_vector_length"),
        processed_feature_count=metadata.get("processed_feature_vector_length"),
        metadata={"model_version": ml_result.model_version},
    )

    stage2_input = Stage2AssessmentInput(
        assessment_id=aid,
        user_id=user_id,
        profile_id=profile_id,
        scenario_summary=dict(request_body.scenario.model_dump()),
        profile_context=_build_stage2_profile_context(profile),
        ml=ml_snapshot,
        evidence_bundles=[],
        operator_notes=request_body.operator_notes,
        metadata={
            "raw_feature_count": metadata.get("raw_feature_vector_length"),
            "processed_feature_count": metadata.get("processed_feature_vector_length"),
            "model_version": ml_result.model_version,
        },
    )

    rag_adapter = build_runtime_rag_adapter_if_available()
    operational_agent = OperationalAgentV2(rag_adapter=rag_adapter)
    llm_orchestrator = build_llm_orchestrator_from_env()

    pipeline = Stage2PipelineV2(
        rag_adapter=rag_adapter,
        operational_agent=operational_agent,
        llm_orchestrator=llm_orchestrator,
        max_evidence_queries=3,
    )

    stage2_result = await pipeline.run(stage2_input)
    if not stage2_result.assessment_id:
        stage2_result = stage2_result.model_copy(update={"assessment_id": aid})

    report = build_operational_report(stage2_input, stage2_result)
    markdown = render_markdown_report(report)

    stage1_ml = Stage1MLSection(
        predicted_class=ml_result.risk_class.value,
        probabilities=dict(ml_result.probabilities),
        top_probability=_top_probability(dict(ml_result.probabilities)),
        raw_feature_count=metadata.get("raw_feature_vector_length"),
        processed_feature_count=metadata.get("processed_feature_vector_length"),
    )
    stage1_shap = Stage1SHAPSection(
        top_features=[f for f in shap_top],
        topic_count=len({str(item.get("feature", "")).split("_")[0] for item in shap_top if item.get("feature")}),
        notes=[],
    )

    evidence_bundles = list(stage2_result.evidence_bundles)
    insufficient_count = sum(1 for b in evidence_bundles if b.support_status == EvidenceSupportStatus.INSUFFICIENT_EVIDENCE)
    rag_limitations: list[str] = []
    for bundle in evidence_bundles:
        if bundle.support_status == EvidenceSupportStatus.INSUFFICIENT_EVIDENCE and (bundle.no_evidence_reason or "").strip():
            rag_limitations.append(f"{bundle.query}: {bundle.no_evidence_reason}")

    agent_result = stage2_result.agent_result
    wm = agent_result.working_memory if agent_result else None
    raw_tool_trace = [t.model_dump() for t in (agent_result.tool_trace if agent_result else [])]
    safe_tool_trace = sanitize_tool_trace_public(raw_tool_trace)
    safe_system_work_trace = sanitize_system_work_trace_public(
        _build_system_work_trace(agent_result, stage2_result=stage2_result)
    )

    working_memory_summary_payload = {
        "coverage_summary": dict(wm.coverage_summary),
        "reasoning_summary": wm.reasoning_summary,
        "limitations": list(wm.limitations),
        "selected_rag_queries": list(wm.selected_rag_queries),
        "skipped_rag_queries": list(wm.skipped_rag_queries),
    } if wm else {}
    safe_working_memory_summary = sanitize_working_memory_public(working_memory_summary_payload)
    safe_top_input_signals = sanitize_working_memory_public(
        {"top_input_signals": [item.model_dump() for item in (wm.input_signals[:12] if wm else [])]}
    ).get("top_input_signals", [])
    safe_top_feature_assessments = sanitize_working_memory_public(
        {"top_feature_assessments": [item.model_dump() for item in (wm.feature_assessments[:12] if wm else [])]}
    ).get("top_feature_assessments", [])

    stage2_agent = Stage2AgentSection(
        recommendation=agent_result.recommendation.value if agent_result else None,
        findings=_clean_forbidden_keys([f.model_dump() for f in (agent_result.findings if agent_result else [])]),
        action_items=_clean_forbidden_keys([a.model_dump() for a in (agent_result.action_items if agent_result else [])]),
        limitations=list(agent_result.reasoning_trace.limitations) if agent_result else [],
        tool_trace=safe_tool_trace,
        system_work_trace=safe_system_work_trace,
        working_memory_summary=safe_working_memory_summary,
        top_input_signals=safe_top_input_signals,
        top_feature_assessments=safe_top_feature_assessments,
        selected_rag_queries=list(safe_working_memory_summary.get("selected_rag_queries", [])),
        skipped_rag_queries=list(safe_working_memory_summary.get("skipped_rag_queries", [])),
    )

    decision = stage2_result.decision
    stage2_decision = Stage2DecisionSection(
        final_decision=decision.final_decision.value if decision else "caution",
        decision_score=float(decision.decision_score) if decision else 0.0,
        confidence_level=decision.confidence_level.value if decision else "low",
        stage_weights=dict(decision.stage_weights) if decision else {},
        stage_contributions=_clean_forbidden_keys([c.model_dump() for c in (decision.stage_contributions if decision else [])]),
        decision_reasons=list(decision.decision_reasons) if decision else [],
        blocking_reasons=list(decision.blocking_reasons) if decision else [],
        required_actions=list(decision.required_actions) if decision else [],
        limitations=list(decision.limitations) if decision else [],
    )

    llm = stage2_result.llm_synthesis
    llm_provider = llm.provider if llm else None
    llm_model = llm.model_name if llm else None
    external_used = bool(llm and llm.status.value == "generated" and llm.provider == "groq")
    stage2_llm = Stage2LLMSection(
        status=llm.status.value if llm else "disabled",
        provider=llm_provider,
        model_name=llm_model,
        external_provider_used=external_used,
        executive_summary=llm.executive_summary if llm else "",
        operational_interpretation=llm.operational_interpretation if llm else "",
        decision_explanation=llm.decision_explanation if llm else "",
        key_risk_drivers=list(llm.key_risk_drivers) if llm else [],
        mitigation_narrative=llm.mitigation_narrative if llm else "",
        consistency_warnings=_clean_forbidden_keys([w.model_dump() for w in (llm.consistency_warnings if llm else [])]),
    )

    policy_meta = dict(decision.metadata) if decision and isinstance(decision.metadata, dict) else {}
    policy_section = Stage2PolicySection(
        policy_name=policy_meta.get("policy_name"),
        policy_version=policy_meta.get("policy_version"),
        weights=dict(decision.stage_weights) if decision else {},
        go_threshold=policy_meta.get("go_threshold"),
        no_go_threshold=policy_meta.get("no_go_threshold"),
        weight_rationales=policy_meta.get("weight_rationales") or {},
    )

    rag_quality = bool(stage2_result.metadata.get("rag_quality_is_proven") if isinstance(stage2_result.metadata, dict) else False)
    scenario_evidence_complete = stage2_result.metadata.get("scenario_evidence_complete") if isinstance(stage2_result.metadata, dict) else None
    scenario_evidence_status = stage2_result.metadata.get("scenario_evidence_status") if isinstance(stage2_result.metadata, dict) else None
    corpus_coverage_status = stage2_result.metadata.get("corpus_coverage_status") if isinstance(stage2_result.metadata, dict) else None
    expected_source_count = stage2_result.metadata.get("expected_source_count") if isinstance(stage2_result.metadata, dict) else None
    indexed_source_count = stage2_result.metadata.get("indexed_source_count") if isinstance(stage2_result.metadata, dict) else None
    missing_sources_raw = stage2_result.metadata.get("missing_sources") if isinstance(stage2_result.metadata, dict) else None
    source_ids_raw = stage2_result.metadata.get("source_ids") if isinstance(stage2_result.metadata, dict) else None
    source_titles_raw = stage2_result.metadata.get("source_titles") if isinstance(stage2_result.metadata, dict) else None
    missing_sources = [item.strip() for item in str(missing_sources_raw or "").split(",") if item.strip()]
    source_ids = [item.strip() for item in str(source_ids_raw or "").split(",") if item.strip()]
    source_titles = [item.strip() for item in str(source_titles_raw or "").split(",") if item.strip()]
    reranker_configured = stage2_result.metadata.get("reranker_configured") if isinstance(stage2_result.metadata, dict) else None
    reranker_available = stage2_result.metadata.get("reranker_available") if isinstance(stage2_result.metadata, dict) else None
    reranker_used = stage2_result.metadata.get("reranker_used") if isinstance(stage2_result.metadata, dict) else None
    reranker_reason = stage2_result.metadata.get("reranker_reason") if isinstance(stage2_result.metadata, dict) else None

    created_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    response = Stage2AssessmentResponse(
        status=stage2_result.status.value,
        user_id=user_id,
        profile_id=profile_id,
        assessment_id=stage2_result.assessment_id or aid,
        created_at=created_at,
        persisted=True,
        persistence_status="saved",
        system_work_trace=safe_system_work_trace,
        warnings=[],
        errors=_clean_forbidden_keys([err.model_dump() for err in stage2_result.errors]),
        stage1=Stage1AssessmentSection(core={"structural_hard_veto_passed": True}, ml=stage1_ml, shap=stage1_shap),
        stage2=Stage2AISection(
            profile_context=stage2_input.profile_context.model_dump() if stage2_input.profile_context else None,
            policy=policy_section,
            rag=Stage2RAGSection(
                retrieval_usable=bool(rag_adapter),
                rag_quality_is_proven=rag_quality,
                evidence_bundle_count=len(evidence_bundles),
                insufficient_evidence_count=insufficient_count,
                scenario_evidence_complete=scenario_evidence_complete,
                scenario_evidence_status=scenario_evidence_status,
                corpus_coverage_status=str(corpus_coverage_status) if corpus_coverage_status is not None else None,
                expected_source_count=int(expected_source_count) if isinstance(expected_source_count, (int, float)) else None,
                indexed_source_count=int(indexed_source_count) if isinstance(indexed_source_count, (int, float)) else None,
                source_ids=source_ids,
                source_titles=source_titles,
                missing_sources=missing_sources,
                reranker_configured=bool(reranker_configured) if isinstance(reranker_configured, bool) else None,
                reranker_available=bool(reranker_available) if isinstance(reranker_available, bool) else None,
                reranker_used=bool(reranker_used) if isinstance(reranker_used, bool) else None,
                reranker_reason=str(reranker_reason) if isinstance(reranker_reason, str) else None,
                evidence_bundle_details=_clean_forbidden_keys([
                    {
                        "bundle_id": b.bundle_id,
                        "query": b.query,
                        "support_status": b.support_status.value,
                        "confidence": b.confidence,
                        "no_evidence_reason": b.no_evidence_reason,
                    }
                    for b in evidence_bundles
                ]),
                citations=_clean_forbidden_keys(_build_stage2_citations(evidence_bundles)),
                limitations=rag_limitations,
            ),
            agent=stage2_agent,
            decision=stage2_decision,
            llm_synthesis=stage2_llm,
            report=Stage2ReportSection(
                markdown=markdown,
                sections=_clean_forbidden_keys([s.model_dump() for s in report.sections]),
                generated=True,
            ),
        ),
        diagnostics=Stage2DiagnosticsSection(
            path_resolution_status=stage2_result.metadata.get("provenance_status") if isinstance(stage2_result.metadata, dict) else None,
            index_provenance_status=stage2_result.metadata.get("provenance_status") if isinstance(stage2_result.metadata, dict) else None,
            persistence_status="saved",
            retrieval_usable=bool(rag_adapter),
            rag_quality_is_proven=rag_quality,
            scenario_evidence_complete=scenario_evidence_complete,
            corpus_coverage_status=str(corpus_coverage_status) if corpus_coverage_status is not None else None,
            expected_source_count=int(expected_source_count) if isinstance(expected_source_count, (int, float)) else None,
            indexed_source_count=int(indexed_source_count) if isinstance(indexed_source_count, (int, float)) else None,
            source_ids=source_ids,
            source_titles=source_titles,
            reranker_configured=bool(reranker_configured) if isinstance(reranker_configured, bool) else None,
            reranker_available=bool(reranker_available) if isinstance(reranker_available, bool) else None,
            reranker_used=bool(reranker_used) if isinstance(reranker_used, bool) else None,
            reranker_reason=str(reranker_reason) if isinstance(reranker_reason, str) else None,
            llm_mode=stage2_llm.status,
            external_llm_provider_used=stage2_llm.external_provider_used,
            llm_provider=stage2_llm.provider,
            llm_model_name=stage2_llm.model_name,
            faiss_secret_configured=None,
            warnings=_clean_forbidden_keys([w.model_dump() for w in (llm.consistency_warnings if llm else [])]),
        ),
    )

    response_payload = jsonable_encoder(response)
    record = _build_assessment_record_from_response(
        user_id=user_id,
        profile_id=profile_id,
        response_payload=response_payload,
        created_at=created_at,
    )
    try:
        _assessment_storage().save_assessment(record)
    except AssessmentStorageError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Assessment persistence failed: {exc}",
        ) from exc

    return response_payload

@history_router.get("", response_model=list[AssessmentListItem])
def list_assessments_for_user(
    user_id: str,
    profile_id: str | None = Query(default=None),
) -> list[AssessmentListItem]:
    try:
        return _assessment_storage().list_assessments(user_id, profile_id=profile_id)
    except AssessmentStorageError as exc:
        if _is_invalid_storage_segment(exc):
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Assessment listing failed") from exc


@history_router.get("/{assessment_id}", response_model=AssessmentRecord)
def get_assessment_for_user(user_id: str, assessment_id: str) -> AssessmentRecord:
    try:
        return _assessment_storage().get_assessment(user_id, assessment_id)
    except AssessmentNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Assessment not found") from exc
    except AssessmentStorageError as exc:
        if _is_invalid_storage_segment(exc):
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Assessment retrieval failed") from exc
