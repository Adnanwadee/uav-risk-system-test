from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from typing import Any

from uav_risk.stage2.agent.operational_agent import OperationalAgentV2
from uav_risk.stage2.contracts import MLAssessmentSnapshot, SHAPFeatureAttribution, Stage2AssessmentInput
from uav_risk.stage2.llm.orchestrator import LLMOrchestrator, get_cached_llm_orchestrator_from_env
from uav_risk.stage2.pipeline_v2 import Stage2PipelineV2
from uav_risk.stage2.rag.quality import get_cached_runtime_rag_adapter
from uav_risk.stage2.rag.runtime_diagnostics import inspect_rag_index_provenance, run_rag_runtime_diagnostic
from uav_risk.stage2.reporting import build_operational_report


_AGENT_REQUESTED_COUNT_PATTERN = re.compile(r"agent_requested_query_count\s*=\s*(\d+)", re.IGNORECASE)


def _bundle_detail(bundle) -> dict[str, Any]:
    limitations = []
    for claim in getattr(bundle, "claims", []):
        for lim in getattr(claim, "limitations", []) or []:
            if isinstance(lim, str) and lim.strip():
                limitations.append(lim.strip())

    best_source = None
    best_score = None
    for citation in getattr(bundle, "citations", []):
        source = citation.metadata.get("source_filename") if isinstance(citation.metadata, dict) else None
        score = citation.metadata.get("final_score") if isinstance(citation.metadata, dict) else None
        try:
            score_f = float(score) if score is not None else None
        except Exception:
            score_f = None
        if best_score is None and score_f is not None:
            best_score = score_f
            best_source = source or citation.source_title
        elif score_f is not None and best_score is not None and score_f > best_score:
            best_score = score_f
            best_source = source or citation.source_title

    meta = getattr(bundle, "metadata", {}) if isinstance(getattr(bundle, "metadata", {}), dict) else {}
    reason = getattr(bundle, "no_evidence_reason", None)
    expected = None
    if getattr(bundle, "support_status", None) and bundle.support_status.value == "insufficient_evidence":
        expected = bool(
            reason
            and any(
                token in reason.lower()
                for token in ("outside", "scope", "no sufficient evidence", "failed retrieval safety")
            )
        )

    return {
        "query": bundle.query,
        "support_status": bundle.support_status.value,
        "no_evidence_reason": reason,
        "limitations": limitations,
        "source_intent": meta.get("intent_name"),
        "retrieval_origin": meta.get("retrieval_origin"),
        "synthetic": bool(meta.get("synthetic")),
        "domain_match": meta.get("domain_match"),
        "candidate_count": meta.get("candidate_count"),
        "top_candidate_source": best_source,
        "top_final_score": best_score,
        "insufficiency_expected": expected,
    }


def _llm_synthesis_summary(synthesis) -> dict[str, Any]:
    if synthesis is None:
        return {
            "llm_synthesis_status": None,
            "llm_provider": None,
            "llm_model_name": None,
            "external_llm_provider_used": False,
            "llm_executive_summary": None,
            "llm_operational_interpretation": None,
            "llm_decision_explanation": None,
            "llm_key_risk_drivers": [],
            "llm_mitigation_narrative": None,
            "llm_consistency_warnings": [],
        }

    return {
        "llm_synthesis_status": synthesis.status.value,
        "llm_provider": synthesis.provider,
        "llm_model_name": synthesis.model_name,
        "external_llm_provider_used": synthesis.status.value == "generated",
        "llm_executive_summary": synthesis.executive_summary,
        "llm_operational_interpretation": synthesis.operational_interpretation,
        "llm_decision_explanation": synthesis.decision_explanation,
        "llm_key_risk_drivers": list(synthesis.key_risk_drivers),
        "llm_mitigation_narrative": synthesis.mitigation_narrative,
        "llm_consistency_warnings": [
            {
                "warning_type": warning.warning_type,
                "message": warning.message,
                "related_ids": list(warning.related_ids),
                "metadata": dict(warning.metadata),
            }
            for warning in synthesis.consistency_warnings
        ],
    }


def _decision_summary(decision) -> dict[str, Any]:
    if decision is None:
        return {
            "final_decision": None,
            "decision_score": None,
            "decision_confidence_level": None,
            "decision_reasons": [],
            "blocking_reasons": [],
            "required_actions": [],
            "decision_limitations": [],
            "stage_weights": {},
            "stage_contributions": [],
        }

    return {
        "final_decision": decision.final_decision.value,
        "decision_score": decision.decision_score,
        "decision_confidence_level": decision.confidence_level.value,
        "decision_reasons": list(decision.decision_reasons),
        "blocking_reasons": list(decision.blocking_reasons),
        "required_actions": list(decision.required_actions),
        "decision_limitations": list(decision.limitations),
        "stage_weights": dict(decision.stage_weights),
        "stage_contributions": [
            {
                "stage": contribution.stage.value,
                "weight": contribution.weight,
                "contribution": contribution.contribution,
                "signal": contribution.signal,
                "summary": contribution.summary,
            }
            for contribution in decision.stage_contributions
        ],
    }


def _build_input() -> Stage2AssessmentInput:
    return Stage2AssessmentInput(
        assessment_id="stage2_smoke",
        user_id="demo_user",
        profile_id="demo_profile",
        scenario_summary={
            "environment_weather_wind_mps": 7.5,
            "airspace_altitude_agl_max_m": 100.0,
            "comms_uplink_ok": True,
        },
        ml=MLAssessmentSnapshot(
            predicted_class="Medium Risk",
            probabilities={"High Risk": 0.2, "Medium Risk": 0.6, "Low Risk": 0.2},
            shap_top_features=[
                SHAPFeatureAttribution(feature="environment_weather_wind_mps", value=7.5, importance=0.21),
            ],
            raw_feature_count=197,
            processed_feature_count=198,
        ),
        operator_notes="Smoke run",
    )


def _safe_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except Exception:
        return None


def _parse_missing_sources_count(metadata: dict[str, Any]) -> int | None:
    raw = metadata.get("missing_sources")
    if not isinstance(raw, str):
        return None
    return len([item.strip() for item in raw.split(",") if item.strip()])


def _extract_agent_requested_query_count(agent_result) -> int | None:
    if agent_result is None:
        return None
    for tool_call in getattr(agent_result, "tool_trace", []) or []:
        if getattr(tool_call, "tool_name", None) and tool_call.tool_name.value == "rag_retrieval":
            match = _AGENT_REQUESTED_COUNT_PATTERN.search(str(getattr(tool_call, "input_summary", "") or ""))
            if match:
                return int(match.group(1))
    return None


def _system_trace_summary_from_agent(agent_result) -> tuple[int, list[str]]:
    if agent_result is None:
        return 0, []
    tool_trace = list(getattr(agent_result, "tool_trace", []) or [])
    stages = sorted({str(getattr(item.tool_name, "value", "")) for item in tool_trace if getattr(item, "tool_name", None)})
    return len(tool_trace), stages


async def _run(use_real_rag: bool, use_llm_fallback: bool = False, use_env_llm: bool = False) -> dict[str, Any]:
    provenance = inspect_rag_index_provenance()

    rag_adapter = None
    if use_real_rag:
        rag_adapter = get_cached_runtime_rag_adapter()

    operational_agent = OperationalAgentV2(rag_adapter=rag_adapter) if use_real_rag else None
    if use_llm_fallback and use_env_llm:
        raise ValueError("--use-llm-fallback and --use-env-llm are mutually exclusive")

    if use_llm_fallback:
        llm_orchestrator = LLMOrchestrator()
    elif use_env_llm:
        llm_orchestrator = get_cached_llm_orchestrator_from_env()
    else:
        llm_orchestrator = None

    stage2_input = _build_input()
    pipeline = Stage2PipelineV2(
        rag_adapter=rag_adapter,
        operational_agent=operational_agent,
        llm_orchestrator=llm_orchestrator,
    )

    result = await pipeline.run(stage2_input)

    insufficient_count = sum(1 for b in result.evidence_bundles if b.support_status.value == "insufficient_evidence")
    scenario_evidence_complete = insufficient_count == 0
    scenario_evidence_status = "complete" if scenario_evidence_complete else "incomplete"

    rag_quality_is_proven = False
    retrieval_usable = any(b.support_status.value == "supported" for b in result.evidence_bundles)
    if use_real_rag:
        try:
            diag = await run_rag_runtime_diagnostic(run_quality=True)
            retrieval_usable = bool(diag.metadata.get("retrieval_usable", retrieval_usable))
            rag_quality_is_proven = bool(diag.metadata.get("rag_quality_is_proven", diag.metadata.get("quality_is_proven", False)))
        except Exception:
            rag_quality_is_proven = False

    bundle_details = [_bundle_detail(b) for b in result.evidence_bundles]
    decision_summary = _decision_summary(result.decision)
    llm_summary = _llm_synthesis_summary(result.llm_synthesis)

    metadata = dict(result.metadata or {}) if isinstance(result.metadata, dict) else {}
    retrieval_origins = sorted(
        {
            str((b.metadata or {}).get("retrieval_origin")).strip()
            for b in result.evidence_bundles
            if isinstance(b.metadata, dict) and str((b.metadata or {}).get("retrieval_origin", "")).strip()
        }
    )
    synthetic_bundle_count = sum(
        1
        for b in result.evidence_bundles
        if isinstance(b.metadata, dict) and bool((b.metadata or {}).get("synthetic"))
    )

    agent_requested_query_count = _extract_agent_requested_query_count(result.agent_result)
    if agent_requested_query_count is None:
        agent_requested_query_count = sum(
            1
            for b in result.evidence_bundles
            if isinstance(b.metadata, dict) and str((b.metadata or {}).get("retrieval_origin")) == "agent_requested"
        )

    system_work_trace_entry_count, system_work_trace_stages = _system_trace_summary_from_agent(result.agent_result)

    report_section_names: list[str] = []
    try:
        report = build_operational_report(stage2_input, result)
        report_section_names = [section.title for section in report.sections]
    except Exception:
        report_section_names = []

    return {
        "status": result.status.value,
        "assessment_id": result.assessment_id,
        "persisted": None,
        "persistence_status": None,
        "agent_recommendation": result.agent_result.recommendation.value if result.agent_result else None,
        "evidence_bundle_count": len(result.evidence_bundles),
        "insufficient_evidence_count": insufficient_count,
        "index_provenance_status": provenance.provenance_status,
        "resolved_dense_index_path": provenance.index_path,
        "resolved_sparse_index_path": provenance.sparse_index_path,
        "path_resolution_status": provenance.path_resolution_status,
        "retrieval_usable": retrieval_usable,
        "rag_quality_is_proven": rag_quality_is_proven,
        "quality_is_proven": rag_quality_is_proven,
        "scenario_evidence_complete": scenario_evidence_complete,
        "scenario_evidence_status": scenario_evidence_status,
        "corpus_coverage_status": metadata.get("corpus_coverage_status"),
        "expected_source_count": _safe_int(metadata.get("expected_source_count")),
        "indexed_source_count": _safe_int(metadata.get("indexed_source_count")),
        "missing_sources_count": _parse_missing_sources_count(metadata),
        "reranker_configured": metadata.get("reranker_configured"),
        "reranker_available": metadata.get("reranker_available"),
        "reranker_used": metadata.get("reranker_used"),
        "reranker_reason": metadata.get("reranker_reason"),
        "retrieval_origins": retrieval_origins,
        "synthetic_bundle_count": synthetic_bundle_count,
        "agent_requested_query_count": agent_requested_query_count,
        "system_work_trace_entry_count": system_work_trace_entry_count,
        "system_work_trace_stages": system_work_trace_stages,
        "report_section_names": report_section_names,
        "evidence_bundle_details": bundle_details,
        "errors": [e.model_dump() for e in result.errors],
    } | decision_summary | llm_summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Stage2 PipelineV2 smoke check.")
    parser.add_argument(
        "--use-real-rag",
        action="store_true",
        help="Use runtime RAG adapter if available.",
    )
    parser.add_argument(
        "--use-llm-fallback",
        action="store_true",
        help="Attach deterministic LLM fallback synthesis without a real provider.",
    )
    parser.add_argument(
        "--use-env-llm",
        action="store_true",
        help="Use environment-configured LLM orchestrator/provider if enabled; safely falls back otherwise.",
    )
    args = parser.parse_args()

    try:
        payload = asyncio.run(_run(args.use_real_rag, args.use_llm_fallback, args.use_env_llm))
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0
    except Exception:
        print(json.dumps({"status": "failed", "error": "stage2_pipeline_v2_smoke_failed"}, indent=2))
        return 1


if __name__ == "__main__":
    sys.exit(main())
