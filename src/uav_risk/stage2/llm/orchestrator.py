from __future__ import annotations

import inspect
import json
from dataclasses import dataclass
from typing import Any, Mapping, Protocol

from uav_risk.stage2.contracts import (
    AgentActionItem,
    AgentFinding,
    AgentResult,
    DecisionEngineResult,
    EvidenceBundle,
    LLMAgentSynthesis,
    LLMSynthesisStatus,
    LLMSynthesisWarning,
    Stage2AssessmentInput,
    Stage2AssessmentResult,
)

FORBIDDEN_OUTPUT_FIELDS = {
    "chain_of_thought",
    "reasoning_chain",
    "thought",
    "scratchpad",
    "internal_reasoning",
    "private_reasoning",
}


class LLMProviderProtocol(Protocol):
    async def generate_json(self, prompt: str, schema_name: str) -> Mapping[str, Any]:
        ...


@dataclass(frozen=True)
class LLMOrchestratorConfig:
    enabled: bool = True
    use_fallback_without_provider: bool = True
    model_name: str | None = None
    provider_name: str | None = None
    max_quote_preview_chars: int = 280


def _clean_text(value: Any, fallback: str) -> str:
    text = str(value or "").strip()
    return text or fallback


def _warning(warning_type: str, message: str, related_ids: list[str] | None = None) -> LLMSynthesisWarning:
    return LLMSynthesisWarning(
        warning_type=warning_type,
        message=message,
        related_ids=related_ids or [],
        metadata={},
    )


def _iter_citation_ids(evidence_bundles: list[EvidenceBundle]) -> set[str]:
    ids: set[str] = set()
    for bundle in evidence_bundles:
        ids.add(bundle.bundle_id)
        for citation in bundle.citations:
            ids.add(citation.citation_id)
        for claim in bundle.claims:
            ids.add(claim.claim_id)
            for citation in claim.citations:
                ids.add(citation.citation_id)
    return ids


def _finding_ids(agent_result: AgentResult | None) -> set[str]:
    return {finding.finding_id for finding in (agent_result.findings if agent_result else [])}


def _action_ids(agent_result: AgentResult | None) -> set[str]:
    return {item.action_id for item in (agent_result.action_items if agent_result else [])}


def _allowed_reference_ids(stage2_result: Stage2AssessmentResult) -> set[str]:
    ids = _iter_citation_ids(stage2_result.evidence_bundles)
    ids.update(_finding_ids(stage2_result.agent_result))
    ids.update(_action_ids(stage2_result.agent_result))
    if stage2_result.decision:
        for ref in stage2_result.decision.evidence_refs:
            ids.add(ref.claim_id)
            ids.update(ref.citation_ids)
    return ids


def _contains_forbidden_key(value: Any) -> bool:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if str(key).lower() in FORBIDDEN_OUTPUT_FIELDS:
                return True
            if _contains_forbidden_key(child):
                return True
    elif isinstance(value, list):
        return any(_contains_forbidden_key(item) for item in value)
    return False


def _safe_quote_preview(text: str, limit: int) -> str:
    compact = " ".join(str(text).split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3].rstrip() + "..."


def build_llm_synthesis_context(
    stage2_input: Stage2AssessmentInput,
    stage2_result: Stage2AssessmentResult,
    *,
    max_quote_preview_chars: int = 280,
) -> dict[str, Any]:
    decision = stage2_result.decision
    agent = stage2_result.agent_result

    evidence = []
    for bundle in stage2_result.evidence_bundles:
        citations = []
        for citation in bundle.citations[:5]:
            citations.append(
                {
                    "citation_id": citation.citation_id,
                    "source_id": citation.source_id,
                    "source_title": citation.source_title,
                    "page": citation.page,
                    "section": citation.section,
                    "quote_preview": _safe_quote_preview(citation.quote, max_quote_preview_chars),
                }
            )
        evidence.append(
            {
                "bundle_id": bundle.bundle_id,
                "query": bundle.query,
                "support_status": bundle.support_status.value,
                "confidence": bundle.confidence,
                "no_evidence_reason": bundle.no_evidence_reason,
                "citation_ids": [item["citation_id"] for item in citations],
                "citations": citations,
            }
        )

    findings = []
    action_items = []
    if agent:
        findings = [
            {
                "finding_id": finding.finding_id,
                "finding_type": finding.finding_type.value,
                "severity": finding.severity.value,
                "summary": finding.summary,
                "evidence_refs": [ref.model_dump() for ref in finding.evidence_references],
            }
            for finding in agent.findings
        ]
        action_items = [
            {
                "action_id": item.action_id,
                "priority": item.priority.value,
                "summary": item.summary,
                "evidence_refs": [ref.model_dump() for ref in item.evidence_references],
            }
            for item in agent.action_items
        ]

    return {
        "assessment_id": stage2_result.assessment_id or stage2_input.assessment_id,
        "operator_notes_untrusted_summary": _safe_quote_preview(stage2_input.operator_notes or "", 220),
        "decision": None
        if decision is None
        else {
            "final_decision": decision.final_decision.value,
            "decision_score": decision.decision_score,
            "confidence_level": decision.confidence_level.value,
            "decision_reasons": list(decision.decision_reasons),
            "blocking_reasons": list(decision.blocking_reasons),
            "required_actions": list(decision.required_actions),
            "limitations": list(decision.limitations),
            "stage_contributions": [
                {
                    "stage": item.stage.value,
                    "weight": item.weight,
                    "contribution": item.contribution,
                    "signal": item.signal,
                    "summary": item.summary,
                }
                for item in decision.stage_contributions
            ],
        },
        "ml": {
            "predicted_class": stage2_input.ml.predicted_class,
            "probabilities": dict(stage2_input.ml.probabilities),
        },
        "shap_top_features": [
            {
                "feature": item.feature,
                "importance": item.importance,
                "direction": item.direction,
                "value": item.value,
            }
            for item in stage2_input.ml.shap_top_features[:8]
        ],
        "evidence": evidence,
        "agent": None
        if agent is None
        else {
            "recommendation": agent.recommendation.value,
            "confidence": agent.confidence,
            "findings": findings,
            "action_items": action_items,
            "limitations": list(agent.reasoning_trace.limitations),
        },
        "allowed_reference_ids": sorted(_allowed_reference_ids(stage2_result)),
        "safety_rules": [
            "Use only provided structured context.",
            "Do not invent facts or citations.",
            "Do not change final decision or evidence support status.",
            "Do not include private reasoning or hidden analysis.",
        ],
    }


def _build_prompt(context: Mapping[str, Any]) -> str:
    return (
        "You are a constrained UAV operational report synthesis assistant. "
        "Use only the JSON context. Do not invent facts, citations, evidence, or support statuses. "
        "Do not change the final decision. Do not include chain-of-thought. "
        "Return valid JSON matching LLMAgentSynthesis.\n\n"
        + json.dumps(context, indent=2, sort_keys=True, ensure_ascii=False)
    )


def _fallback_synthesis(
    stage2_input: Stage2AssessmentInput,
    stage2_result: Stage2AssessmentResult,
    *,
    status: LLMSynthesisStatus,
    warnings: list[LLMSynthesisWarning] | None = None,
    config: LLMOrchestratorConfig | None = None,
) -> LLMAgentSynthesis:
    decision = stage2_result.decision
    agent = stage2_result.agent_result
    final_decision = decision.final_decision.value if decision else "unavailable"
    score = decision.decision_score if decision else None
    confidence = decision.confidence_level.value if decision else "unknown"

    decision_reasons = list(decision.decision_reasons[:4]) if decision else ["Decision Engine result was not available."]
    findings = list(agent.findings[:4]) if agent else []
    actions = list(agent.action_items[:5]) if agent else []

    risk_drivers = [finding.summary for finding in findings]
    if decision:
        risk_drivers.extend(
            f"{item.stage.value}: {item.signal}"
            for item in decision.stage_contributions
            if item.contribution > 0.0
        )
    risk_drivers = risk_drivers[:8] or ["No major operational risk drivers were available for synthesis."]

    mitigation = " ".join(item.summary for item in actions).strip()
    if not mitigation and decision:
        mitigation = " ".join(decision.required_actions).strip()
    if not mitigation:
        mitigation = "No specific mitigation actions were provided by the deterministic pipeline."

    synth_warnings = list(warnings or [])
    if status == LLMSynthesisStatus.DISABLED:
        synth_warnings.append(_warning("llm_disabled", "LLM synthesis is disabled; deterministic fallback text was produced."))
    elif status == LLMSynthesisStatus.FALLBACK:
        synth_warnings.append(_warning("llm_fallback", "LLM provider was unavailable or unsafe; deterministic fallback text was produced."))

    if agent and decision and agent.recommendation.value != final_decision:
        synth_warnings.append(
            _warning(
                "agent_decision_difference",
                "Agent provisional recommendation differs from the Decision Engine final decision.",
            )
        )
    if any(bundle.support_status.value == "insufficient_evidence" for bundle in stage2_result.evidence_bundles):
        synth_warnings.append(_warning("insufficient_evidence_present", "At least one evidence topic was insufficiently supported."))
    if not any(bundle.citations for bundle in stage2_result.evidence_bundles):
        synth_warnings.append(_warning("no_rag_citations", "No RAG citations were available for synthesis."))

    evidence_ids = sorted(_iter_citation_ids(stage2_result.evidence_bundles))
    finding_ids = sorted(_finding_ids(agent))
    action_ids = sorted(_action_ids(agent))

    provider = config.provider_name if config else None
    model_name = config.model_name if config else None
    return LLMAgentSynthesis(
        status=status,
        executive_summary=f"Final operational decision is {final_decision}. Confidence is {confidence}"
        + (f" with decision score {score:.3f}." if isinstance(score, float) else "."),
        operational_interpretation=" ".join(finding.summary for finding in findings)
        or "The deterministic pipeline did not provide detailed agent findings.",
        decision_explanation=" ".join(decision_reasons),
        key_risk_drivers=risk_drivers,
        mitigation_narrative=mitigation,
        consistency_warnings=synth_warnings,
        evidence_reference_ids=evidence_ids,
        finding_ids=finding_ids,
        action_item_ids=action_ids,
        limitation_ids=[warning.warning_type for warning in synth_warnings],
        model_name=model_name,
        provider=provider,
        metadata={"source": "deterministic_fallback", "llm_called": False},
    )


def validate_llm_synthesis_output(
    raw_output: Mapping[str, Any],
    *,
    stage2_result: Stage2AssessmentResult,
    provider_name: str | None = None,
    model_name: str | None = None,
) -> LLMAgentSynthesis:
    if _contains_forbidden_key(raw_output):
        raise ValueError("LLM output contains forbidden private reasoning field")

    decision = stage2_result.decision
    proposed_decision = raw_output.get("final_decision")
    if proposed_decision is not None and decision is not None and proposed_decision != decision.final_decision.value:
        raise ValueError("LLM output attempted to change final decision")

    payload = dict(raw_output)
    payload.pop("final_decision", None)
    payload["status"] = LLMSynthesisStatus.GENERATED.value
    payload.setdefault("model_name", model_name)
    payload.setdefault("provider", provider_name)
    payload.setdefault("metadata", {})
    payload["metadata"] = {**dict(payload.get("metadata") or {}), "llm_called": True}

    synthesis = LLMAgentSynthesis.model_validate(payload)
    allowed = _allowed_reference_ids(stage2_result)
    referenced = set(synthesis.evidence_reference_ids) | set(synthesis.finding_ids) | set(synthesis.action_item_ids)
    unknown = sorted(item for item in referenced if item not in allowed)
    if unknown:
        raise ValueError(f"LLM output referenced unknown IDs: {', '.join(unknown)}")
    return synthesis


class LLMOrchestrator:
    def __init__(self, provider: LLMProviderProtocol | None = None, config: LLMOrchestratorConfig | None = None) -> None:
        self.provider = provider
        self.config = config or LLMOrchestratorConfig()

    async def synthesize(
        self,
        stage2_input: Stage2AssessmentInput,
        stage2_result: Stage2AssessmentResult,
    ) -> LLMAgentSynthesis:
        if not self.config.enabled:
            return _fallback_synthesis(
                stage2_input,
                stage2_result,
                status=LLMSynthesisStatus.DISABLED,
                config=self.config,
            )

        if self.provider is None:
            fallback_status = LLMSynthesisStatus.FALLBACK if self.config.use_fallback_without_provider else LLMSynthesisStatus.DISABLED
            return _fallback_synthesis(
                stage2_input,
                stage2_result,
                status=fallback_status,
                config=self.config,
            )

        context = build_llm_synthesis_context(
            stage2_input,
            stage2_result,
            max_quote_preview_chars=self.config.max_quote_preview_chars,
        )
        prompt = _build_prompt(context)
        try:
            raw = self.provider.generate_json(prompt, "LLMAgentSynthesis")
            if inspect.isawaitable(raw):
                raw = await raw
            return validate_llm_synthesis_output(
                raw,
                stage2_result=stage2_result,
                provider_name=self.config.provider_name,
                model_name=self.config.model_name,
            )
        except Exception as exc:
            return _fallback_synthesis(
                stage2_input,
                stage2_result,
                status=LLMSynthesisStatus.FALLBACK,
                warnings=[_warning("llm_provider_invalid", "LLM provider output was unavailable or failed validation.")],
                config=self.config,
            )


async def synthesize_stage2_result(
    stage2_input: Stage2AssessmentInput,
    stage2_result: Stage2AssessmentResult,
    *,
    orchestrator: LLMOrchestrator | None = None,
) -> LLMAgentSynthesis:
    active = orchestrator or LLMOrchestrator()
    return await active.synthesize(stage2_input, stage2_result)
