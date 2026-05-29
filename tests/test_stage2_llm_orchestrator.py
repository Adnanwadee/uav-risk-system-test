from __future__ import annotations

import json

import pytest

from uav_risk.stage2.contracts import (
    AgentActionItem,
    AgentFeatureAssessment,
    AgentInputSignal,
    AgentRiskRelevance,
    AgentSignalSource,
    AgentWorkingMemory,
    AgentEvidenceReference,
    AgentFinding,
    AgentFindingSeverity,
    AgentFindingType,
    AgentRecommendation,
    AgentResult,
    DecisionConfidenceLevel,
    DecisionEngineResult,
    EvidenceBundle,
    EvidenceCitation,
    EvidenceOrigin,
    EvidenceSourceType,
    EvidenceSupportStatus,
    FinalDecision,
    LLMSynthesisStatus,
    LLMRuntimeConfig,
    MLAssessmentSnapshot,
    PublicReasoningTrace,
    Stage2AssessmentInput,
    Stage2AssessmentResult,
    Stage2Status,
)
from uav_risk.stage2.llm.groq_client import GroqLLMProvider, _extract_json_text, GroqProviderError
from uav_risk.stage2.llm.orchestrator import (
    FORBIDDEN_OUTPUT_FIELDS,
    LLMOrchestrator,
    LLMOrchestratorConfig,
    build_llm_orchestrator_from_env,
    build_llm_synthesis_context,
    load_llm_runtime_config_from_env,
)


def _input() -> Stage2AssessmentInput:
    return Stage2AssessmentInput(
        assessment_id="a1",
        user_id="u1",
        profile_id="p1",
        scenario_summary={"environment_weather_wind_mps": 7.0},
        ml=MLAssessmentSnapshot(
            predicted_class="Medium Risk",
            probabilities={"Medium Risk": 0.6, "High Risk": 0.2, "Low Risk": 0.2},
            shap_top_features=[],
        ),
        operator_notes="Operator reports gusty winds near the site.",
    )


def _result() -> Stage2AssessmentResult:
    citation = EvidenceCitation(
        citation_id="cit1",
        source_id="src1",
        source_title="AC_107-2A",
        source_type=EvidenceSourceType.ADVISORY_CIRCULAR,
        origin=EvidenceOrigin.LOCAL_DOCUMENT,
        page=4,
        quote="Remote pilots should evaluate weather before operation.",
    )
    bundle = EvidenceBundle(
        bundle_id="bundle1",
        query="AC 107-2A preflight weather assessment small UAS wind conditions",
        claims=[],
        citations=[citation],
        support_status=EvidenceSupportStatus.SUPPORTED,
        confidence=0.8,
    )
    ref = AgentEvidenceReference(claim_id="claim1", citation_ids=["cit1"], summary="weather citation")
    agent = AgentResult(
        status=Stage2Status.COMPLETED,
        recommendation=AgentRecommendation.CAUTION,
        confidence=0.7,
        findings=[
            AgentFinding(
                finding_id="finding1",
                finding_type=AgentFindingType.EVIDENCE_BACKED,
                severity=AgentFindingSeverity.MEDIUM,
                summary="Weather and wind conditions require preflight assessment.",
                evidence_references=[ref],
                requires_evidence=True,
            )
        ],
        action_items=[
            AgentActionItem(
                action_id="action1",
                summary="Review weather and define wind limits.",
                priority=AgentFindingSeverity.HIGH,
                evidence_references=[ref],
            )
        ],
        reasoning_trace=PublicReasoningTrace(limitations=[]),
        working_memory=AgentWorkingMemory(
            input_signals=[
                AgentInputSignal(
                    signal_id="sig1",
                    source=AgentSignalSource.SHAP,
                    name="environment_weather_wind_mps",
                    value_summary="importance=0.3",
                    topic="weather",
                    priority=0.8,
                    risk_relevance=AgentRiskRelevance.HIGH,
                    needs_rag_evidence=True,
                )
            ],
            feature_assessments=[
                AgentFeatureAssessment(
                    assessment_id="fa1",
                    signal_id="sig1",
                    feature_name="environment_weather_wind_mps",
                    source=AgentSignalSource.SHAP,
                    topic="weather",
                    priority=0.8,
                    risk_relevance=AgentRiskRelevance.HIGH,
                    raw_value_summary="importance=0.3",
                    rag_query="AC 107-2A preflight weather assessment small UAS wind conditions",
                    evidence_status="supported_concern",
                    evidence_bundle_ids=["bundle1"],
                    finding_ids=["finding1"],
                    action_item_ids=["action1"],
                    conclusion="supported concern",
                )
            ],
            selected_rag_queries=["AC 107-2A preflight weather assessment small UAS wind conditions"],
            skipped_rag_queries=[],
            reasoning_summary="test working memory",
            coverage_summary={"input_signal_count": 1, "feature_assessment_count": 1},
            limitations=[],
        ),
        evidence_bundles=[bundle],
        errors=[],
    )
    decision = DecisionEngineResult(
        final_decision=FinalDecision.CAUTION,
        decision_score=0.44,
        confidence_level=DecisionConfidenceLevel.MEDIUM,
        stage_weights={"ml": 0.22, "rag": 0.28, "agent": 0.25},
        stage_contributions=[],
        decision_reasons=["Weighted decision score requires caution."],
        blocking_reasons=[],
        required_actions=["Review weather and define wind limits."],
        limitations=[],
        evidence_refs=[ref],
    )
    return Stage2AssessmentResult(
        status=Stage2Status.COMPLETED,
        assessment_id="a1",
        evidence_bundles=[bundle],
        agent_result=agent,
        decision=decision,
        errors=[],
    )


class ValidProvider:
    async def generate_json(self, prompt: str, schema_name: str):
        assert schema_name == "LLMAgentSynthesis"
        assert "final decision" in prompt.lower()
        return {
            "executive_summary": "Mission should proceed only with caution.",
            "operational_interpretation": "Wind creates a preflight review concern.",
            "decision_explanation": "Decision Engine remains caution.",
            "key_risk_drivers": ["weather", "ml_signal"],
            "mitigation_narrative": "Review weather and define wind limits.",
            "consistency_warnings": [],
            "evidence_reference_ids": ["cit1"],
            "finding_ids": ["finding1"],
            "action_item_ids": ["action1"],
            "limitation_ids": [],
        }


class UnknownCitationProvider(ValidProvider):
    async def generate_json(self, prompt: str, schema_name: str):
        payload = await super().generate_json(prompt, schema_name)
        payload["evidence_reference_ids"] = ["made_up_citation"]
        return payload


class PrivateReasoningProvider(ValidProvider):
    async def generate_json(self, prompt: str, schema_name: str):
        payload = await super().generate_json(prompt, schema_name)
        payload["chain_of_thought"] = "hidden reasoning"
        return payload


class ChangedDecisionProvider(ValidProvider):
    async def generate_json(self, prompt: str, schema_name: str):
        payload = await super().generate_json(prompt, schema_name)
        payload["final_decision"] = "go"
        return payload


class ProviderSpoofingProvider(ValidProvider):
    async def generate_json(self, prompt: str, schema_name: str):
        payload = await super().generate_json(prompt, schema_name)
        payload["provider"] = "UAV Operational Report Synthesis Assistant"
        payload["model_name"] = "UAV Operational Report Synthesis Model"
        payload["metadata"] = {"decision_score": 0.99, "confidence_level": "high"}
        return payload


class InconsistentNumericNarrativeProvider(ValidProvider):
    async def generate_json(self, prompt: str, schema_name: str):
        payload = await super().generate_json(prompt, schema_name)
        payload["executive_summary"] = "Final decision is caution with decision score 0.99 and confidence high."
        return payload


class ExplodingProvider:
    async def generate_json(self, prompt: str, schema_name: str):
        raise RuntimeError("provider unavailable")


@pytest.mark.asyncio
async def test_missing_provider_returns_fallback_synthesis() -> None:
    synthesis = await LLMOrchestrator().synthesize(_input(), _result())
    assert synthesis.status == LLMSynthesisStatus.FALLBACK
    assert synthesis.metadata["llm_called"] is False
    assert synthesis.executive_summary


@pytest.mark.asyncio
async def test_disabled_orchestrator_returns_disabled_synthesis() -> None:
    synthesis = await LLMOrchestrator(config=LLMOrchestratorConfig(enabled=False)).synthesize(_input(), _result())
    assert synthesis.status == LLMSynthesisStatus.DISABLED


@pytest.mark.asyncio
async def test_fake_provider_valid_json_returns_generated_synthesis() -> None:
    synthesis = await LLMOrchestrator(
        provider=ValidProvider(),
        config=LLMOrchestratorConfig(provider_name="fake", model_name="fake-model"),
    ).synthesize(_input(), _result())
    assert synthesis.status == LLMSynthesisStatus.GENERATED
    assert synthesis.provider == "fake"
    assert synthesis.model_name == "fake-model"
    assert synthesis.metadata["llm_called"] is True




@pytest.mark.asyncio
async def test_generated_provider_and_model_are_backend_owned() -> None:
    synthesis = await LLMOrchestrator(
        provider=ProviderSpoofingProvider(),
        config=LLMOrchestratorConfig(provider_name="groq", model_name="llama-3.3-70b-versatile"),
    ).synthesize(_input(), _result())
    assert synthesis.status == LLMSynthesisStatus.GENERATED
    assert synthesis.provider == "groq"
    assert synthesis.model_name == "llama-3.3-70b-versatile"
    assert synthesis.metadata.get("decision_score") is None
    assert synthesis.metadata.get("confidence_level") is None


@pytest.mark.asyncio
async def test_inconsistent_numeric_narrative_falls_back() -> None:
    synthesis = await LLMOrchestrator(
        provider=InconsistentNumericNarrativeProvider(),
        config=LLMOrchestratorConfig(provider_name="groq", model_name="llama-3.3-70b-versatile"),
    ).synthesize(_input(), _result())
    assert synthesis.status == LLMSynthesisStatus.FALLBACK
    warning = next(w for w in synthesis.consistency_warnings if w.warning_type == "llm_provider_invalid")
    assert warning.metadata.get("provider_error_type") == "provider_invalid"

@pytest.mark.asyncio
async def test_unknown_citation_id_is_rejected_and_falls_back() -> None:
    synthesis = await LLMOrchestrator(provider=UnknownCitationProvider()).synthesize(_input(), _result())
    assert synthesis.status == LLMSynthesisStatus.FALLBACK
    assert any(w.warning_type == "llm_provider_invalid" for w in synthesis.consistency_warnings)


@pytest.mark.asyncio
async def test_private_reasoning_field_is_rejected_and_falls_back() -> None:
    synthesis = await LLMOrchestrator(provider=PrivateReasoningProvider()).synthesize(_input(), _result())
    assert synthesis.status == LLMSynthesisStatus.FALLBACK


@pytest.mark.asyncio
async def test_provider_cannot_change_final_decision() -> None:
    synthesis = await LLMOrchestrator(provider=ChangedDecisionProvider()).synthesize(_input(), _result())
    assert synthesis.status == LLMSynthesisStatus.FALLBACK


@pytest.mark.asyncio
async def test_provider_exception_falls_back() -> None:
    synthesis = await LLMOrchestrator(provider=ExplodingProvider()).synthesize(_input(), _result())
    assert synthesis.status == LLMSynthesisStatus.FALLBACK
    warning = next(w for w in synthesis.consistency_warnings if w.warning_type == "llm_provider_invalid")
    assert warning.metadata.get("validation_error_type")


def test_context_includes_decision_findings_evidence_actions_and_limitations() -> None:
    context = build_llm_synthesis_context(_input(), _result())
    assert context["decision"]["final_decision"] == "caution"
    assert context["evidence"][0]["citation_ids"] == ["cit1"]
    assert context["agent"]["findings"][0]["finding_id"] == "finding1"
    assert context["agent"]["action_items"][0]["action_id"] == "action1"
    assert "cit1" in context["allowed_reference_ids"]


def test_context_does_not_include_forbidden_private_reasoning_fields() -> None:
    encoded = json.dumps(build_llm_synthesis_context(_input(), _result())).lower()
    for token in FORBIDDEN_OUTPUT_FIELDS:
        assert token not in encoded


def test_load_llm_runtime_config_from_env_defaults_disabled(monkeypatch) -> None:
    monkeypatch.delenv("LLM_ENABLED", raising=False)
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    cfg = load_llm_runtime_config_from_env()
    assert isinstance(cfg, LLMRuntimeConfig)
    assert cfg.enabled is False
    assert cfg.provider == "fallback"
    assert cfg.allow_external_provider is False


def test_missing_groq_api_key_keeps_fallback(monkeypatch) -> None:
    monkeypatch.setenv("LLM_ENABLED", "true")
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    orch = build_llm_orchestrator_from_env()
    assert orch.provider is None
    assert orch.config.provider_name == "fallback"


def test_enabled_groq_env_builds_provider_without_exposing_key(monkeypatch) -> None:
    monkeypatch.setenv("LLM_ENABLED", "true")
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "test-secret-key")
    monkeypatch.setenv("GROQ_MODEL", "test-model")
    orch = build_llm_orchestrator_from_env()
    assert orch.provider is not None
    assert orch.config.provider_name == "groq"
    assert "secret" not in repr(orch.config).lower()


def test_context_includes_agent_tool_trace_summaries() -> None:
    result = _result()
    assert result.agent_result is not None
    result.agent_result.tool_trace = []
    from uav_risk.stage2.contracts import AgentToolCall, AgentToolName

    result.agent_result.tool_trace.append(
        AgentToolCall(
            tool_name=AgentToolName.RAG_RETRIEVAL,
            purpose="Retrieve evidence",
            input_summary="query_count=1",
            output_summary="bundle_count=1",
            status="ok",
            related_query_ids=["q1"],
            related_evidence_ids=["bundle1"],
            related_finding_ids=["finding1"],
            metadata={},
        )
    )
    context = build_llm_synthesis_context(_input(), result)
    assert context["agent"]["tool_trace"][0]["tool_name"] == "rag_retrieval"



def test_groq_json_extraction_accepts_raw_json_text() -> None:
    text = '{"a": 1}'
    assert _extract_json_text(text) == '{"a": 1}'


def test_groq_json_extraction_accepts_json_fenced_text() -> None:
    text = """```json
{"a": 1}
```"""
    assert _extract_json_text(text) == '{"a": 1}'


def test_groq_json_extraction_rejects_empty_content() -> None:
    with pytest.raises(GroqProviderError) as exc:
        _extract_json_text("   ")
    assert exc.value.reason_code == "empty_response"


class InvalidJSONProvider:
    async def generate_json(self, prompt: str, schema_name: str):
        del prompt, schema_name
        raise RuntimeError("invalid json response")


class SecretLeakyProvider:
    async def generate_json(self, prompt: str, schema_name: str):
        del prompt, schema_name
        raise RuntimeError("timeout while using key sk-test-secret")


@pytest.mark.asyncio
async def test_invalid_json_falls_back_with_safe_diagnostics() -> None:
    synthesis = await LLMOrchestrator(provider=InvalidJSONProvider()).synthesize(_input(), _result())
    assert synthesis.status == LLMSynthesisStatus.FALLBACK
    warning = next(w for w in synthesis.consistency_warnings if w.warning_type == "llm_provider_invalid")
    assert warning.metadata.get("provider_error_message_short") in {"invalid json", "provider response invalid"}
    assert "provider_error_type" in warning.metadata


@pytest.mark.asyncio
async def test_safe_diagnostics_do_not_include_raw_response_or_secrets() -> None:
    synthesis = await LLMOrchestrator(provider=SecretLeakyProvider()).synthesize(_input(), _result())
    assert synthesis.status == LLMSynthesisStatus.FALLBACK
    warning = next(w for w in synthesis.consistency_warnings if w.warning_type == "llm_provider_invalid")
    dumped = json.dumps(warning.model_dump()).lower()
    assert "sk-test-secret" not in dumped


class ReasonCodeProvider:
    async def generate_json(self, prompt: str, schema_name: str):
        del prompt, schema_name

        class E(Exception):
            reason_code = "auth_error"
            safe_message = "Provider authentication failed."

        raise E()


@pytest.mark.asyncio
async def test_reason_code_is_propagated_in_safe_metadata() -> None:
    synthesis = await LLMOrchestrator(provider=ReasonCodeProvider()).synthesize(_input(), _result())
    assert synthesis.status == LLMSynthesisStatus.FALLBACK
    warning = next(w for w in synthesis.consistency_warnings if w.warning_type == "llm_provider_invalid")
    assert warning.metadata.get("provider_error_type") == "auth_error"
    assert warning.metadata.get("provider_error_message_short") == "provider auth error"




class _FakeMsg:
    def __init__(self, content):
        self.content = content


class _FakeChoice:
    def __init__(self, content):
        self.message = _FakeMsg(content)


class _FakeResp:
    def __init__(self, content):
        self.choices = [_FakeChoice(content)]

@pytest.mark.asyncio
async def test_groq_provider_direct_style_success(monkeypatch) -> None:
    class _FakeClient:
        class _Chat:
            class _Completions:
                def create(self, **kwargs):
                    return _FakeResp('{"pong":"world"}')
            completions = _Completions()
        chat = _Chat()

    monkeypatch.setattr(GroqLLMProvider, "_build_client", lambda self: _FakeClient())
    provider = GroqLLMProvider(api_key="x", model_name="m", temperature=0, max_tokens=64)
    payload = await provider.generate_json('{"ping":"hello"}', "Probe")
    assert payload == {"pong": "world"}


@pytest.mark.asyncio
async def test_groq_provider_invalid_json_maps_to_invalid_json(monkeypatch) -> None:
    class _FakeClient:
        class _Chat:
            class _Completions:
                def create(self, **kwargs):
                    return _FakeResp('not-json')
            completions = _Completions()
        chat = _Chat()

    monkeypatch.setattr(GroqLLMProvider, "_build_client", lambda self: _FakeClient())
    provider = GroqLLMProvider(api_key="x", model_name="m")
    with pytest.raises(GroqProviderError) as exc:
        await provider.generate_json('{"ping":"hello"}', "Probe")
    assert exc.value.reason_code == "invalid_json"


@pytest.mark.asyncio
async def test_groq_provider_auth_exception_maps_to_auth_error(monkeypatch) -> None:
    class _FakeClient:
        class _Chat:
            class _Completions:
                def create(self, **kwargs):
                    from groq import AuthenticationError
                    import httpx
                    req = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
                    resp = httpx.Response(401, request=req)
                    raise AuthenticationError("auth failed", response=resp, body=None)
            completions = _Completions()
        chat = _Chat()

    monkeypatch.setattr(GroqLLMProvider, "_build_client", lambda self: _FakeClient())
    provider = GroqLLMProvider(api_key="x", model_name="m")
    with pytest.raises(GroqProviderError) as exc:
        await provider.generate_json('{"ping":"hello"}', "Probe")
    assert exc.value.reason_code == "auth_error"


@pytest.mark.asyncio
async def test_groq_provider_non_auth_exception_not_mapped_to_auth(monkeypatch) -> None:
    class _FakeClient:
        class _Chat:
            class _Completions:
                def create(self, **kwargs):
                    raise RuntimeError("boom")
            completions = _Completions()
        chat = _Chat()

    monkeypatch.setattr(GroqLLMProvider, "_build_client", lambda self: _FakeClient())
    provider = GroqLLMProvider(api_key="x", model_name="m")
    with pytest.raises(GroqProviderError) as exc:
        await provider.generate_json('{"ping":"hello"}', "Probe")
    assert exc.value.reason_code != "auth_error"


def test_context_includes_working_memory_summary() -> None:
    context = build_llm_synthesis_context(_input(), _result())
    assert context["agent"]["working_memory"]["coverage_summary"]["input_signal_count"] == 1
    assert context["agent"]["working_memory"]["top_input_signals"][0]["signal_id"] == "sig1"

