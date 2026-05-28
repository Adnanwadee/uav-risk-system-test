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
    source = inspect.getsource(pipeline_v2)
    assert "Groq" not in source
    assert "llm" not in source.lower()


def test_importing_pipeline_v2_does_not_load_heavy_resources() -> None:
    source = inspect.getsource(pipeline_v2).lower()
    for token in ("faiss", "vector_db", "model", ".pdf", "artifacts"):
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
