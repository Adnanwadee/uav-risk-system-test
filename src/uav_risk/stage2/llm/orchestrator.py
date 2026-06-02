from __future__ import annotations

import inspect
import json
import os
import re
from dataclasses import dataclass
from threading import Lock
from typing import Any, Mapping, Protocol

from uav_risk.core.env import load_project_env

load_project_env()

from uav_risk.stage2.contracts import (
    AgentActionItem,
    AgentFinding,
    AgentResult,
    LLMRuntimeConfig,
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


def _env_true(name: str) -> bool:
    return str(os.getenv(name, "")).strip().lower() in {"1", "true", "yes", "on"}


def load_llm_runtime_config_from_env() -> LLMRuntimeConfig:
    enabled = _env_true("LLM_ENABLED")
    provider = str(os.getenv("LLM_PROVIDER", "fallback") or "fallback").strip().lower()
    model_name = str(os.getenv("GROQ_MODEL", "") or "").strip() or None

    api_key_present = bool(str(os.getenv("GROQ_API_KEY", "") or "").strip())
    externally_allowed = enabled and provider == "groq" and api_key_present

    return LLMRuntimeConfig(
        enabled=enabled,
        provider=provider if externally_allowed else "fallback",
        model_name=model_name,
        temperature=float(os.getenv("LLM_TEMPERATURE", "0.1") or 0.1),
        max_tokens=int(os.getenv("LLM_MAX_TOKENS", "1200") or 1200),
        timeout_seconds=float(os.getenv("LLM_TIMEOUT_SECONDS", "20.0") or 20.0),
        use_fallback_on_error=_env_true("LLM_USE_FALLBACK_ON_ERROR") if os.getenv("LLM_USE_FALLBACK_ON_ERROR") is not None else True,
        allow_external_provider=externally_allowed,
    )


@dataclass(frozen=True)
class LLMOrchestratorConfig:
    enabled: bool = True
    use_fallback_without_provider: bool = True
    model_name: str | None = None
    provider_name: str | None = None
    max_quote_preview_chars: int = 280


def _build_llm_orchestrator_from_env_uncached(runtime: LLMRuntimeConfig | None = None) -> "LLMOrchestrator":
    runtime = runtime or load_llm_runtime_config_from_env()
    config = LLMOrchestratorConfig(
        enabled=runtime.enabled,
        use_fallback_without_provider=runtime.use_fallback_on_error,
        model_name=runtime.model_name,
        provider_name=runtime.provider,
    )

    if not runtime.allow_external_provider or runtime.provider != "groq":
        return LLMOrchestrator(provider=None, config=config)

    try:
        from uav_risk.stage2.llm.groq_client import GroqLLMProvider

        provider = GroqLLMProvider(
            api_key=os.getenv("GROQ_API_KEY"),
            model_name=runtime.model_name,
            temperature=runtime.temperature,
            max_tokens=runtime.max_tokens,
            timeout_seconds=runtime.timeout_seconds,
        )
        return LLMOrchestrator(provider=provider, config=config)
    except Exception:
        return LLMOrchestrator(provider=None, config=config)


_LLM_ORCHESTRATOR_CACHE_LOCK = Lock()
_LLM_ORCHESTRATOR_CACHE: "LLMOrchestrator" | None = None
_LLM_ORCHESTRATOR_CACHE_KEY: tuple[Any, ...] | None = None


def _llm_orchestrator_cache_key(runtime: LLMRuntimeConfig) -> tuple[Any, ...]:
    return (
        runtime.enabled,
        runtime.provider,
        runtime.model_name,
        runtime.temperature,
        runtime.max_tokens,
        runtime.timeout_seconds,
        runtime.use_fallback_on_error,
        runtime.allow_external_provider,
    )


def build_llm_orchestrator_from_env() -> "LLMOrchestrator":
    global _LLM_ORCHESTRATOR_CACHE
    global _LLM_ORCHESTRATOR_CACHE_KEY

    runtime = load_llm_runtime_config_from_env()
    cache_key = _llm_orchestrator_cache_key(runtime)

    if _LLM_ORCHESTRATOR_CACHE is not None and _LLM_ORCHESTRATOR_CACHE_KEY == cache_key:
        return _LLM_ORCHESTRATOR_CACHE

    with _LLM_ORCHESTRATOR_CACHE_LOCK:
        if _LLM_ORCHESTRATOR_CACHE is None or _LLM_ORCHESTRATOR_CACHE_KEY != cache_key:
            _LLM_ORCHESTRATOR_CACHE = _build_llm_orchestrator_from_env_uncached(runtime=runtime)
            _LLM_ORCHESTRATOR_CACHE_KEY = cache_key

    return _LLM_ORCHESTRATOR_CACHE


def get_cached_llm_orchestrator_from_env() -> "LLMOrchestrator":
    return build_llm_orchestrator_from_env()


def clear_llm_orchestrator_cache_for_tests() -> None:
    global _LLM_ORCHESTRATOR_CACHE
    global _LLM_ORCHESTRATOR_CACHE_KEY

    with _LLM_ORCHESTRATOR_CACHE_LOCK:
        _LLM_ORCHESTRATOR_CACHE = None
        _LLM_ORCHESTRATOR_CACHE_KEY = None


def _clean_text(value: Any, fallback: str) -> str:
    text = str(value or "").strip()
    return text or fallback


def _warning(
    warning_type: str,
    message: str,
    related_ids: list[str] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> LLMSynthesisWarning:
    return LLMSynthesisWarning(
        warning_type=warning_type,
        message=message,
        related_ids=related_ids or [],
        metadata=dict(metadata or {}),
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
            "tool_trace": [
                {
                    "tool_name": item.tool_name.value,
                    "purpose": item.purpose,
                    "status": item.status,
                    "input_summary": item.input_summary,
                    "output_summary": item.output_summary,
                    "related_query_ids": list(item.related_query_ids),
                    "related_evidence_ids": list(item.related_evidence_ids),
                    "related_finding_ids": list(item.related_finding_ids),
                }
                for item in agent.tool_trace
            ],
            "working_memory": None
            if agent.working_memory is None
            else {
                "coverage_summary": dict(agent.working_memory.coverage_summary),
                "reasoning_summary": agent.working_memory.reasoning_summary,
                "selected_rag_queries": list(agent.working_memory.selected_rag_queries),
                "limitations": list(agent.working_memory.limitations),
                "top_input_signals": [
                    {
                        "signal_id": item.signal_id,
                        "source": item.source.value,
                        "name": item.name,
                        "topic": item.topic,
                        "priority": item.priority,
                        "risk_relevance": item.risk_relevance.value,
                        "needs_rag_evidence": item.needs_rag_evidence,
                    }
                    for item in agent.working_memory.input_signals[:10]
                ],
                "top_feature_assessments": [
                    {
                        "assessment_id": item.assessment_id,
                        "feature_name": item.feature_name,
                        "topic": item.topic,
                        "priority": item.priority,
                        "evidence_status": item.evidence_status,
                        "conclusion": item.conclusion,
                    }
                    for item in agent.working_memory.feature_assessments[:10]
                ],
            },
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
    allowed_fields = [
        "status",
        "executive_summary",
        "operational_interpretation",
        "decision_explanation",
        "key_risk_drivers",
        "mitigation_narrative",
        "consistency_warnings",
        "evidence_reference_ids",
        "finding_ids",
        "action_item_ids",
        "limitation_ids",
        "model_name",
        "provider",
        "metadata",
    ]
    return (
        "You are a constrained UAV operational safety report synthesis assistant and aviation operations analyst. "
        "Use only the JSON context produced by the deterministic UAV risk pipeline. "
        "Do not invent facts, citations, evidence, support statuses, scores, probabilities, counts, regulations, or legal conclusions. "
        "Do not change the final decision, confidence level, evidence support status, or DecisionEngine scoring. "
        "Treat RAG citations as the only evidence authority; your role is post-decision operational narrative synthesis. "
        "Write as a structured operational report narrative: executive summary, operational interpretation, decision explanation, "
        "key risk drivers, and mitigation narrative. "
        "Give practical UAV pre-flight recommendations only when grounded in provided required_actions, agent action_items, "
        "agent findings, RAG evidence, scenario/profile context, or deterministic decision reasons. "
        "When evidence is incomplete, state limitations clearly instead of filling gaps. "
        "Do not include chain-of-thought, hidden reasoning, private analysis, or unsupported assumptions. "
        "Return exactly one JSON object and no surrounding text. "
        "Allowed top-level fields only: " + ", ".join(allowed_fields) + ". "
        "Use only reference IDs provided in allowed_reference_ids; if unsure, return empty lists for reference IDs. "
        "Do not invent numeric values; use exact provided values only or omit numbers. "
        "Do not set provider, model_name, or runtime metadata; backend will attach runtime ownership fields.\n\n"
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



_DECISION_SCORE_PATTERN = re.compile(r"decision\s*score[^0-9]*([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)
_CONFIDENCE_PATTERN = re.compile(r"confidence(?:\s*level)?[^a-z]*(low|medium|high)", re.IGNORECASE)
_DECISION_WORD_PATTERN = re.compile(r"final\s+decision\s+(?:is|=)?\s*(go|caution|no[_\s-]?go)", re.IGNORECASE)


def _validate_generated_narrative_consistency(synthesis: LLMAgentSynthesis, decision: DecisionEngineResult | None) -> None:
    if decision is None:
        return

    combined = " ".join(
        [
            synthesis.executive_summary,
            synthesis.operational_interpretation,
            synthesis.decision_explanation,
            synthesis.mitigation_narrative,
        ]
    )

    score_match = _DECISION_SCORE_PATTERN.search(combined)
    if score_match is not None:
        try:
            mentioned = float(score_match.group(1))
        except Exception:
            mentioned = None
        if mentioned is not None and abs(mentioned - float(decision.decision_score)) > 0.02:
            raise ValueError("LLM narrative decision score inconsistent with backend decision score")

    confidence_match = _CONFIDENCE_PATTERN.search(combined)
    if confidence_match is not None:
        mentioned_conf = confidence_match.group(1).lower().strip()
        expected_conf = decision.confidence_level.value.lower().strip()
        if mentioned_conf != expected_conf:
            raise ValueError("LLM narrative confidence level inconsistent with backend confidence level")

    decision_word_match = _DECISION_WORD_PATTERN.search(combined)
    if decision_word_match is not None:
        mentioned_decision = decision_word_match.group(1).lower().replace("_", "-").replace(" ", "-").strip()
        expected_decision = decision.final_decision.value.lower().replace("_", "-").strip()
        if mentioned_decision != expected_decision:
            raise ValueError("LLM narrative final decision inconsistent with backend final decision")


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
    # Backend-owned runtime fields: always overwrite any provider/model proposed by LLM.
    payload["model_name"] = model_name
    payload["provider"] = provider_name
    payload.setdefault("metadata", {})
    metadata = dict(payload.get("metadata") or {})
    for blocked_key in (
        "final_decision",
        "decision_score",
        "confidence_level",
        "ml_probabilities",
        "evidence_count",
        "citation_count",
    ):
        metadata.pop(blocked_key, None)
    metadata["llm_called"] = True
    payload["metadata"] = metadata

    synthesis = LLMAgentSynthesis.model_validate(payload)
    _validate_generated_narrative_consistency(synthesis, decision)
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
            reason_code = str(getattr(exc, "reason_code", "provider_invalid") or "provider_invalid")
            safe_message = str(getattr(exc, "safe_message", "") or "").strip()
            raw_message = safe_message or str(exc or "provider error")
            lowered = raw_message.lower()

            reason_to_short = {
                "sdk_missing": "provider sdk missing",
                "client_init_error": "provider client init error",
                "network_call_error": "provider network call error",
                "empty_response": "provider empty response",
                "invalid_json": "invalid json",
                "response_parse_error": "provider response parse error",
                "unsupported_response_format": "unsupported response format",
                "model_error": "provider model error",
                "auth_error": "provider auth error",
                "rate_limit": "provider rate limited",
                "timeout": "provider timeout",
                "schema_unavailable": "provider schema unavailable",
                "unknown_api_error": "provider api error",
                "provider_invalid": "provider response invalid",
            }
            redacted_message = reason_to_short.get(reason_code, "provider response invalid")
            if "forbidden" in lowered:
                redacted_message = "forbidden field detected"
            elif "unknown id" in lowered or "referenced unknown" in lowered:
                redacted_message = "unknown reference ids"
            elif "json" in lowered and reason_code == "provider_invalid":
                redacted_message = "invalid json"

            invalid_reference_count = 0
            if "unknown IDs:" in raw_message:
                suffix = raw_message.split("unknown IDs:", 1)[-1].strip()
                invalid_reference_count = len([item for item in suffix.split(",") if item.strip()])

            warning_meta = {
                "provider_error_type": reason_code,
                "provider_error_message_short": redacted_message,
                "validation_error_type": type(exc).__name__,
                "invalid_reference_count": invalid_reference_count,
                "forbidden_field_detected": "forbidden" in lowered,
            }
            return _fallback_synthesis(
                stage2_input,
                stage2_result,
                status=LLMSynthesisStatus.FALLBACK,
                warnings=[
                    _warning(
                        "llm_provider_invalid",
                        "LLM provider output was unavailable or failed validation.",
                        metadata=warning_meta,
                    )
                ],
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
