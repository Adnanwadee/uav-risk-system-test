"""Constrained Stage2 LLM synthesis helpers.

This package contains provider-injected orchestration only. It does not call
external LLM APIs unless explicitly enabled by runtime environment settings.
"""

from uav_risk.stage2.llm.orchestrator import (
    LLMOrchestrator,
    LLMOrchestratorConfig,
    LLMProviderProtocol,
    build_llm_orchestrator_from_env,
    get_cached_llm_orchestrator_from_env,
    clear_llm_orchestrator_cache_for_tests,
    build_llm_synthesis_context,
    load_llm_runtime_config_from_env,
    synthesize_stage2_result,
    validate_llm_synthesis_output,
)

__all__ = [
    "LLMOrchestrator",
    "LLMOrchestratorConfig",
    "LLMProviderProtocol",
    "build_llm_orchestrator_from_env",
    "get_cached_llm_orchestrator_from_env",
    "clear_llm_orchestrator_cache_for_tests",
    "build_llm_synthesis_context",
    "load_llm_runtime_config_from_env",
    "synthesize_stage2_result",
    "validate_llm_synthesis_output",
]
