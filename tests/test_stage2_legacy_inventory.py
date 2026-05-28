from __future__ import annotations

from uav_risk.stage2.legacy_inventory import (
    Stage2FileStatus,
    get_stage2_legacy_inventory,
)


def _by_path(path: str):
    for item in get_stage2_legacy_inventory():
        if item.path == path:
            return item
    raise AssertionError(f"missing inventory path: {path}")


def test_inventory_marks_pipeline_py_legacy_do_not_use() -> None:
    item = _by_path("src/uav_risk/stage2/pipeline.py")
    assert item.status == Stage2FileStatus.LEGACY_DO_NOT_USE


def test_inventory_marks_pipeline_v2_active() -> None:
    item = _by_path("src/uav_risk/stage2/pipeline_v2.py")
    assert item.status == Stage2FileStatus.ACTIVE_V2


def test_inventory_marks_rag_adapter_active() -> None:
    item = _by_path("src/uav_risk/stage2/rag/adapter.py")
    assert item.status == Stage2FileStatus.ACTIVE_V2


def test_inventory_marks_rag_schemas_compatibility_bridge() -> None:
    item = _by_path("src/uav_risk/stage2/rag/schemas.py")
    assert item.status == Stage2FileStatus.COMPATIBILITY_BRIDGE


def test_inventory_marks_agent_schemas_compatibility_bridge() -> None:
    item = _by_path("src/uav_risk/stage2/agent/agent_schemas.py")
    assert item.status == Stage2FileStatus.COMPATIBILITY_BRIDGE


def test_inventory_marks_docs_models_vector_db_heavy_resources() -> None:
    inv = get_stage2_legacy_inventory()
    paths = {item.path: item.status for item in inv}
    assert paths["src/uav_risk/stage2/docs/*"] == Stage2FileStatus.HEAVY_RUNTIME_RESOURCE
    assert paths["src/uav_risk/stage2/knowledge/models/*"] == Stage2FileStatus.HEAVY_RUNTIME_RESOURCE
    assert paths["src/uav_risk/stage2/knowledge/vectdb/*"] == Stage2FileStatus.HEAVY_RUNTIME_RESOURCE


def test_inventory_does_not_import_heavy_modules() -> None:
    import inspect
    import uav_risk.stage2.legacy_inventory as module

    source = inspect.getsource(module)
    assert "from uav_risk.stage2.rag.groq_llm" not in source
    assert "import groq" not in source.lower()


def test_inventory_recommends_replacements_for_legacy_pipeline_and_agent() -> None:
    pipeline_item = _by_path("src/uav_risk/stage2/pipeline.py")
    agent_item = _by_path("src/uav_risk/stage2/agent/ace_agent.py")
    assert pipeline_item.replacement == "src/uav_risk/stage2/pipeline_v2.py"
    assert agent_item.replacement == "src/uav_risk/stage2/agent/operational_agent.py"
