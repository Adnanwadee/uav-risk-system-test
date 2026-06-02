from __future__ import annotations

import inspect
import sys

from uav_risk.stage2 import pipeline_v2
from uav_risk.stage2.contracts import Stage2Status
from uav_risk.stage2.reporting import build_operational_report, render_markdown_report


def test_importing_pipeline_v2_does_not_import_legacy_pipeline_module() -> None:
    assert "uav_risk.stage2.pipeline" not in sys.modules


def test_importing_pipeline_v2_does_not_instantiate_rag_core() -> None:
    source = inspect.getsource(pipeline_v2)
    assert "AsyncRAGCore" not in source
    assert "rag_core_v3" not in source


def test_importing_pipeline_v2_does_not_instantiate_llm_or_groq() -> None:
    """Importing pipeline_v2 may expose an injected LLM boundary, but must not load provider clients."""

    before = set(sys.modules)
    import uav_risk.stage2.pipeline_v2 as module

    after = set(sys.modules)
    newly_loaded = after - before

    assert "uav_risk.stage2.llm.groq_client" not in newly_loaded
    assert "groq" not in newly_loaded

    source = inspect.getsource(module)
    forbidden_runtime_calls = (
        "build_llm_orchestrator_from_env(",
        "get_cached_llm_orchestrator_from_env(",
        "GroqLLMProvider(",
        "AsyncGroq(",
    )
    for token in forbidden_runtime_calls:
        assert token not in source

def test_importing_pipeline_v2_does_not_load_heavy_resources() -> None:
    """Importing pipeline_v2 must not initialize RAG/vector/model runtime resources."""

    before = set(sys.modules)
    import uav_risk.stage2.pipeline_v2 as module

    after = set(sys.modules)
    newly_loaded = after - before

    forbidden_modules = {
        "faiss",
        "sentence_transformers",
        "onnxruntime",
        "torch",
        "transformers",
        "groq",
    }
    assert forbidden_modules.isdisjoint(newly_loaded)

    source = inspect.getsource(module)
    forbidden_runtime_calls = (
        "build_runtime_rag_adapter_if_available(",
        "get_cached_runtime_rag_adapter(",
        "HybridRetriever(",
        "AsyncRAGCoreV3(",
        "FAISSIndexVerifier(",
        "load_stage1_bundle(",
    )
    for token in forbidden_runtime_calls:
        assert token not in source

def test_pipeline_v2_source_does_not_reference_masterflightpayload() -> None:
    source = inspect.getsource(pipeline_v2)
    assert "MasterFlightPayload" not in source


def test_pipeline_v2_source_does_not_reference_featurerouter() -> None:
    source = inspect.getsource(pipeline_v2)
    assert "FeatureRouter" not in source


def test_pipeline_v2_source_does_not_reference_generate_all_features_map() -> None:
    source = inspect.getsource(pipeline_v2)
    assert "generate_all_features_map" not in source


def test_pipeline_v2_source_does_not_reference_feature_defs_as_policy_authority() -> None:
    source = inspect.getsource(pipeline_v2)
    assert "feature_defs" not in source


def test_report_rendering_does_not_execute_llm_or_rag() -> None:
    import uav_risk.stage2.reporting as reporting

    source = inspect.getsource(reporting).lower()
    assert "groq" not in source
    assert "retrieve_evidence(" not in source


def test_safety_tests_are_behavior_revealing_not_shallow() -> None:
    assert Stage2Status.DEGRADED.value == "degraded"
    assert callable(build_operational_report)
    assert callable(render_markdown_report)
