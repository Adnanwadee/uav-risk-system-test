from __future__ import annotations

from typing import Any

from uav_risk.stage2.agent.facade import AgentResultFacade
from uav_risk.stage2.agent.query_planner import build_agent_rag_query_plan
from uav_risk.stage2.decision_engine import evaluate_stage2_decision
from uav_risk.stage2.llm.orchestrator import LLMOrchestrator
from uav_risk.stage2.contracts import (
    AgentInput,
    AgentRAGQueryPlan,
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
        }
    )
    return bundle.model_copy(update={"metadata": merged})


class Stage2PipelineV2:
    def __init__(
        self,
        *,
        rag_adapter: Any | None = None,
        agent_facade: AgentResultFacade | None = None,
        operational_agent: Any | None = None,
        llm_orchestrator: LLMOrchestrator | None = None,
        max_evidence_queries: int = 3,
    ) -> None:
        if max_evidence_queries < 0:
            raise ValueError("max_evidence_queries must be >= 0")
        self._rag_adapter = rag_adapter
        self._agent_facade = agent_facade or AgentResultFacade(agent=None)
        self._operational_agent = operational_agent
        self._llm_orchestrator = llm_orchestrator
        self._max_evidence_queries = max_evidence_queries

    async def run(self, stage2_input: Stage2AssessmentInput) -> Stage2AssessmentResult:
        try:
            evidence_bundles = list(stage2_input.evidence_bundles)

            if self._rag_adapter is not None:
                query_plan = build_stage2_evidence_query_plan(
                    stage2_input, self._max_evidence_queries
                )
                for item in query_plan:
                    bundle = await self._rag_adapter.retrieve_evidence(
                        item.query_text,
                        scenario_context=stage2_input.scenario_summary,
                    )
                    evidence_bundles.append(_enrich_bundle_with_query_plan(bundle, item))
            elif not evidence_bundles:
                from uav_risk.stage2.contracts import make_insufficient_evidence_bundle

                evidence_bundles.append(
                    make_insufficient_evidence_bundle(
                        query="stage2_evidence",
                        reason="RAG adapter is not configured.",
                    )
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

            provisional_result = Stage2AssessmentResult(
                status=status,
                assessment_id=stage2_input.assessment_id,
                evidence_bundles=evidence_bundles,
                agent_result=agent_result,
                errors=list(agent_result.errors),
                metadata=dict(stage2_input.metadata),
            )
            decision = evaluate_stage2_decision(stage2_input, provisional_result)
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
