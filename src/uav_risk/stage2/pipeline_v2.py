from __future__ import annotations

from typing import Any

from uav_risk.stage2.agent.facade import AgentResultFacade
from uav_risk.stage2.agent.query_planner import build_agent_rag_query_plan
from uav_risk.stage2.decision_engine import evaluate_stage2_decision
from uav_risk.stage2.llm.orchestrator import LLMOrchestrator
from uav_risk.stage2.contracts import (
    AgentInput,
    AgentRAGQueryPlan,
    DecisionPolicyConfig,
    EvidenceBundle,
    EvidenceSupportStatus,
    Stage2AssessmentInput,
    Stage2AssessmentResult,
    Stage2Error,
    Stage2Status,
)


def build_stage2_evidence_query_plan(
    stage2_input: Stage2AssessmentInput, max_queries: int
) -> list[AgentRAGQueryPlan]:
    if max_queries < 0:
        raise ValueError("max_queries must be >= 0")
    if max_queries == 0:
        return []

    agent_input = AgentInput(
        assessment_id=stage2_input.assessment_id,
        scenario_summary=dict(stage2_input.scenario_summary),
        profile_context=stage2_input.profile_context,
        ml_prediction=stage2_input.ml.predicted_class,
        ml_probabilities=dict(stage2_input.ml.probabilities),
        shap_top_features=[
            {
                "feature": item.feature,
                "value": item.value,
                "importance": item.importance,
                "direction": item.direction,
                **({"metadata": item.metadata} if item.metadata else {}),
            }
            for item in stage2_input.ml.shap_top_features
        ],
        evidence_bundles=list(stage2_input.evidence_bundles),
        operator_notes=stage2_input.operator_notes,
        metadata=dict(stage2_input.metadata),
    )
    return build_agent_rag_query_plan(agent_input, max_queries=max_queries)


def build_stage2_evidence_queries(
    stage2_input: Stage2AssessmentInput, max_queries: int
) -> list[str]:
    return [item.query_text for item in build_stage2_evidence_query_plan(stage2_input, max_queries)]


def build_agent_input(
    stage2_input: Stage2AssessmentInput, evidence_bundles: list[EvidenceBundle]
) -> AgentInput:
    shap_top_features = [
        {
            "feature": item.feature,
            "value": item.value,
            "importance": item.importance,
            "direction": item.direction,
            **({"metadata": item.metadata} if item.metadata else {}),
        }
        for item in stage2_input.ml.shap_top_features
    ]
    return AgentInput(
        assessment_id=stage2_input.assessment_id,
        scenario_summary=dict(stage2_input.scenario_summary),
        profile_context=stage2_input.profile_context,
        ml_prediction=stage2_input.ml.predicted_class,
        ml_probabilities=dict(stage2_input.ml.probabilities),
        shap_top_features=shap_top_features,
        evidence_bundles=list(evidence_bundles),
        operator_notes=stage2_input.operator_notes,
        metadata=dict(stage2_input.metadata),
    )


def has_insufficient_evidence(evidence_bundles: list[EvidenceBundle]) -> bool:
    return any(
        bundle.support_status == EvidenceSupportStatus.INSUFFICIENT_EVIDENCE
        for bundle in evidence_bundles
    )


def _enrich_bundle_with_query_plan(bundle: EvidenceBundle, plan: AgentRAGQueryPlan) -> EvidenceBundle:
    merged = dict(bundle.metadata)
    merged.update(
        {
            "query_id": plan.query_id,
            "query_purpose": plan.query_purpose,
            "source_intent": plan.source_intent.value,
            "derived_from": plan.derived_from.value,
            "related_feature_names": ",".join(plan.related_feature_names),
            "query_priority": plan.priority,
            "retrieval_origin": str(merged.get("retrieval_origin") or "scenario_driven"),
        }
    )
    return bundle.model_copy(update={"metadata": merged})




def _bundle_metadata(bundle: EvidenceBundle) -> dict[str, Any]:
    return bundle.metadata if isinstance(bundle.metadata, dict) else {}


def _scenario_evidence_status(evidence_bundles: list[EvidenceBundle]) -> tuple[str, bool | None]:
    scenario_bundles = [
        b for b in evidence_bundles
        if str(_bundle_metadata(b).get("retrieval_origin", "scenario_driven")) == "scenario_driven"
    ]
    if not scenario_bundles:
        return "unavailable", None

    statuses = {b.support_status for b in scenario_bundles}
    if statuses == {EvidenceSupportStatus.SUPPORTED}:
        return "grounded", True
    if statuses.issubset({EvidenceSupportStatus.SUPPORTED, EvidenceSupportStatus.PARTIALLY_SUPPORTED}):
        return "partial", False
    if statuses == {EvidenceSupportStatus.INSUFFICIENT_EVIDENCE}:
        synthetic_only = any(bool(_bundle_metadata(b).get("synthetic")) for b in scenario_bundles)
        return ("synthetic_only" if synthetic_only else "insufficient"), False
    if EvidenceSupportStatus.INSUFFICIENT_EVIDENCE in statuses:
        return "partial", False
    return "unknown", None




async def _retrieve_evidence_with_origin(
    rag_adapter: Any,
    *,
    query_text: str,
    scenario_context: dict[str, Any],
    retrieval_origin: str,
) -> EvidenceBundle:
    try:
        return await rag_adapter.retrieve_evidence(
            query_text,
            scenario_context=scenario_context,
            retrieval_origin=retrieval_origin,
        )
    except TypeError:
        return await rag_adapter.retrieve_evidence(
            query_text,
            scenario_context=scenario_context,
        )


def _collect_rag_metadata(evidence_bundles: list[EvidenceBundle], *, query_plan: list[AgentRAGQueryPlan]) -> dict[str, Any]:
    scenario_status, scenario_complete = _scenario_evidence_status(evidence_bundles)
    first_meta = next((m for m in (_bundle_metadata(b) for b in evidence_bundles) if m), {})

    reranker_configured = first_meta.get("reranker_configured")
    reranker_available = first_meta.get("reranker_available")
    reranker_used = first_meta.get("reranker_used")
    reranker_reason = first_meta.get("reranker_reason")

    coverage_status = first_meta.get("corpus_coverage_status")
    expected_source_count = first_meta.get("expected_source_count")
    indexed_source_count = first_meta.get("indexed_source_count")
    missing_sources = first_meta.get("missing_sources")

    rag_queries = [item.query_text for item in query_plan]
    return {
        "scenario_evidence_status": scenario_status,
        "scenario_evidence_complete": scenario_complete,
        "planned_scenario_query_count": len(query_plan),
        "planned_scenario_queries": " | ".join(rag_queries[:8]),
        "selected_scenario_queries": " | ".join(rag_queries[:8]),
        "skipped_scenario_queries": "",
        "reranker_configured": bool(reranker_configured) if isinstance(reranker_configured, bool) else None,
        "reranker_available": bool(reranker_available) if isinstance(reranker_available, bool) else None,
        "reranker_used": bool(reranker_used) if isinstance(reranker_used, bool) else None,
        "reranker_reason": str(reranker_reason) if isinstance(reranker_reason, str) else None,
        "corpus_coverage_status": str(coverage_status) if isinstance(coverage_status, str) else None,
        "expected_source_count": int(expected_source_count) if isinstance(expected_source_count, (int, float)) else None,
        "indexed_source_count": int(indexed_source_count) if isinstance(indexed_source_count, (int, float)) else None,
        "missing_sources": str(missing_sources) if isinstance(missing_sources, str) else None,
    }


class Stage2PipelineV2:
    def __init__(
        self,
        *,
        rag_adapter: Any | None = None,
        agent_facade: AgentResultFacade | None = None,
        operational_agent: Any | None = None,
        llm_orchestrator: LLMOrchestrator | None = None,
        decision_policy: DecisionPolicyConfig | None = None,
        max_evidence_queries: int = 3,
    ) -> None:
        if max_evidence_queries < 0:
            raise ValueError("max_evidence_queries must be >= 0")
        self._rag_adapter = rag_adapter
        self._agent_facade = agent_facade or AgentResultFacade(agent=None)
        self._operational_agent = operational_agent
        self._llm_orchestrator = llm_orchestrator
        self._decision_policy = decision_policy
        self._max_evidence_queries = max_evidence_queries

    async def run(self, stage2_input: Stage2AssessmentInput) -> Stage2AssessmentResult:
        try:
            evidence_bundles = list(stage2_input.evidence_bundles)

            if self._rag_adapter is not None:
                query_plan = build_stage2_evidence_query_plan(
                    stage2_input, self._max_evidence_queries
                )
                for item in query_plan:
                    bundle = await _retrieve_evidence_with_origin(
                        self._rag_adapter,
                        query_text=item.query_text,
                        scenario_context=stage2_input.scenario_summary,
                        retrieval_origin="scenario_driven",
                    )
                    evidence_bundles.append(_enrich_bundle_with_query_plan(bundle, item))
            elif not evidence_bundles:
                from uav_risk.stage2.contracts import make_insufficient_evidence_bundle

                evidence_bundles.append(
                    make_insufficient_evidence_bundle(
                        query="stage2_evidence",
                        reason="RAG adapter is not configured.",
                    ).model_copy(update={"metadata": {"retrieval_origin": "fallback", "evidence_status": "unavailable"}})
                )

            agent_input = build_agent_input(stage2_input, evidence_bundles)
            if self._operational_agent is not None and callable(getattr(self._operational_agent, "run", None)):
                agent_result = self._operational_agent.run(agent_input)
                if hasattr(agent_result, "__await__"):
                    agent_result = await agent_result
            else:
                agent_result = await self._agent_facade.run(agent_input)

            if agent_result.status == Stage2Status.COMPLETED and not has_insufficient_evidence(
                evidence_bundles
            ):
                status = Stage2Status.COMPLETED
            elif agent_result.status in {Stage2Status.DEGRADED, Stage2Status.SKIPPED} or has_insufficient_evidence(
                evidence_bundles
            ):
                status = Stage2Status.DEGRADED
            else:
                status = Stage2Status.COMPLETED

            rag_metadata = _collect_rag_metadata(evidence_bundles, query_plan=query_plan if self._rag_adapter is not None else [])
            merged_metadata = dict(stage2_input.metadata)
            merged_metadata.update({k: v for k, v in rag_metadata.items() if v is not None})

            provisional_result = Stage2AssessmentResult(
                status=status,
                assessment_id=stage2_input.assessment_id,
                evidence_bundles=evidence_bundles,
                agent_result=agent_result,
                errors=list(agent_result.errors),
                metadata=merged_metadata,
            )
            decision = evaluate_stage2_decision(stage2_input, provisional_result, policy=self._decision_policy)
            result_with_decision = provisional_result.model_copy(update={"decision": decision})

            llm_synthesis = None
            if self._llm_orchestrator is not None:
                llm_synthesis = await self._llm_orchestrator.synthesize(
                    stage2_input,
                    result_with_decision,
                )

            if llm_synthesis is not None:
                return result_with_decision.model_copy(update={"llm_synthesis": llm_synthesis})
            return result_with_decision
        except Exception:
            return Stage2AssessmentResult(
                status=Stage2Status.FAILED,
                assessment_id=stage2_input.assessment_id,
                evidence_bundles=list(stage2_input.evidence_bundles),
                agent_result=None,
                errors=[
                    Stage2Error(
                        code="stage2_pipeline_failed",
                        message="Stage2 pipeline execution failed.",
                        details={},
                    )
                ],
                metadata=dict(stage2_input.metadata),
            )
