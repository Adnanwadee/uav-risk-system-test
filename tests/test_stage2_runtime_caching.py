from __future__ import annotations

from uav_risk.stage2.llm.orchestrator import (
    LLMOrchestrator,
    build_llm_orchestrator_from_env,
    clear_llm_orchestrator_cache_for_tests,
)
from uav_risk.stage2.rag.quality import (
    build_runtime_rag_adapter_if_available,
    clear_runtime_rag_adapter_cache_for_tests,
)


def test_rag_runtime_adapter_builder_is_cached_per_process(monkeypatch):
    import uav_risk.stage2.rag.quality as quality

    calls = {"count": 0}

    def fake_uncached_builder():
        calls["count"] += 1
        return None

    clear_runtime_rag_adapter_cache_for_tests()
    monkeypatch.setattr(quality, "_build_runtime_rag_adapter_uncached", fake_uncached_builder)

    first = build_runtime_rag_adapter_if_available()
    second = build_runtime_rag_adapter_if_available()

    assert first is None
    assert second is None
    assert calls["count"] == 1


def test_llm_orchestrator_builder_cache_reuses_and_rebuilds_on_env_change(monkeypatch):
    import uav_risk.stage2.llm.orchestrator as orch

    calls = {"count": 0}

    def fake_uncached_builder(runtime=None):
        calls["count"] += 1
        return LLMOrchestrator()

    clear_llm_orchestrator_cache_for_tests()
    monkeypatch.setattr(orch, "_build_llm_orchestrator_from_env_uncached", fake_uncached_builder)

    monkeypatch.setenv("LLM_ENABLED", "false")
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    _ = build_llm_orchestrator_from_env()
    _ = build_llm_orchestrator_from_env()
    assert calls["count"] == 1

    monkeypatch.setenv("LLM_ENABLED", "true")
    _ = build_llm_orchestrator_from_env()
    assert calls["count"] == 2
