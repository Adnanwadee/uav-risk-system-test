from __future__ import annotations

from pathlib import Path

from uav_risk.stage2.rag.config_v3 import (
    RAGConfig,
    get_dense_index_path,
    get_docs_dir,
    get_index_dir,
    get_models_dir,
    get_sparse_index_path,
)
from uav_risk.stage2.rag.rag_core_v3 import AsyncRAGCoreV3
from uav_risk.stage2.rag.runtime_diagnostics import inspect_rag_index_provenance


def test_default_index_dir_is_project_local_vectdb(monkeypatch):
    monkeypatch.delenv("UAV_RAG_INDEX_DIR", raising=False)
    resolved = get_index_dir()
    assert str(resolved).endswith("src/uav_risk/stage2/knowledge/vectdb")


def test_default_docs_dir_is_project_local_docs(monkeypatch):
    monkeypatch.delenv("UAV_RAG_DOCS_DIR", raising=False)
    resolved = get_docs_dir()
    assert str(resolved).endswith("src/uav_risk/stage2/docs")


def test_default_models_dir_is_project_local_models(monkeypatch):
    monkeypatch.delenv("UAV_RAG_MODELS_DIR", raising=False)
    resolved = get_models_dir()
    assert str(resolved).endswith("src/uav_risk/stage2/knowledge/models")


def test_uav_rag_index_dir_override_is_respected(monkeypatch):
    monkeypatch.setenv("UAV_RAG_INDEX_DIR", "/tmp/uav_rag_custom_index")
    resolved = get_index_dir()
    assert str(resolved) == "/tmp/uav_rag_custom_index"


def test_no_default_fallback_to_home_indices(monkeypatch):
    monkeypatch.delenv("UAV_RAG_INDEX_DIR", raising=False)
    resolved = get_index_dir()
    assert "/home/vscode/.uav_rag/indices" not in str(resolved)


def test_no_default_fallback_to_bare_indices(monkeypatch):
    monkeypatch.delenv("UAV_RAG_INDEX_DIR", raising=False)
    resolved = get_index_dir()
    assert Path(resolved).name != "indices"


def test_ragconfig_exposes_index_dir_compatible_with_helper():
    assert Path(RAGConfig.INDEX_DIR) == Path(get_index_dir())


def test_rag_core_resolver_not_bare_indices_when_helper_exists(monkeypatch):
    monkeypatch.delenv("UAV_RAG_INDEX_DIR", raising=False)
    core = AsyncRAGCoreV3(config_module=RAGConfig)
    resolved = core._resolve_index_dir()
    assert resolved == get_index_dir()
    assert resolved.name == "vectdb"


def test_runtime_diag_and_core_match_dense_index_path():
    core = AsyncRAGCoreV3(config_module=RAGConfig)
    dense_from_core = core._resolve_index_dir() / "dense_index.faiss"
    provenance = inspect_rag_index_provenance()
    assert provenance.index_path is not None
    assert Path(provenance.index_path) == dense_from_core


def test_relative_index_dir_override_resolves_from_cwd(monkeypatch):
    monkeypatch.setenv("UAV_RAG_INDEX_DIR", "relative_idx")
    resolved = get_index_dir()
    assert resolved == (Path.cwd() / "relative_idx").resolve()


def test_dense_and_sparse_paths_under_index_dir(monkeypatch):
    monkeypatch.delenv("UAV_RAG_INDEX_DIR", raising=False)
    index_dir = get_index_dir()
    assert get_dense_index_path().parent == index_dir
    assert get_sparse_index_path().parent == index_dir


def test_no_faiss_binary_opened():
    dense = get_dense_index_path()
    assert dense.suffix == ".faiss"
