"""Constrained Stage2 LLM synthesis helpers.

This package contains provider-injected orchestration only. It does not call
external LLM APIs unless a caller supplies a provider explicitly.
"""

from uav_risk.stage2.llm.orchestrator import (
    LLMOrchestrator,
    LLMOrchestratorConfig,
    LLMProviderProtocol,
    build_llm_synthesis_context,
    synthesize_stage2_result,
    validate_llm_synthesis_output,
)

__all__ = [
    "LLMOrchestrator",
    "LLMOrchestratorConfig",
    "LLMProviderProtocol",
    "build_llm_synthesis_context",
    "synthesize_stage2_result",
    "validate_llm_synthesis_output",
]
