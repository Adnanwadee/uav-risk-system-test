from __future__ import annotations

from pathlib import Path


def _read(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def test_build_index_canonical_output_names_present() -> None:
    src = _read("src/uav_risk/stage2/rag/build_index.py")
    assert "get_dense_index_path" in src
    assert "get_dense_mapping_path" in src
    assert "get_sparse_index_path" in src
    assert "get_index_metadata_path" in src


def test_build_index_uses_config_helpers_for_output_directory() -> None:
    src = _read("src/uav_risk/stage2/rag/build_index.py")
    assert "get_index_dir" in src
    assert "get_dense_index_path" in src
    assert "get_sparse_index_path" in src


def test_build_index_does_not_default_to_langchain_compat_outputs() -> None:
    src = _read("src/uav_risk/stage2/rag/build_index.py")
    assert "save_local(" not in src


def test_dense_mapping_schema_contains_chunks_authoritative_format() -> None:
    src = _read("src/uav_risk/stage2/rag/build_index.py")
    assert '"chunks"' in src
    assert '"vector_id"' in src
    assert '"source_id"' in src
    assert '"text_sha256"' in src


def test_rebuild_script_defaults_to_canonical_build() -> None:
    src = _read("scripts/rebuild_stage2_rag_index.py")
    assert "build_rag_index(force=args.force)" in src
    assert "--repair-from-existing" in src


def test_rebuild_script_writes_to_project_local_vectdb_by_default() -> None:
    from uav_risk.stage2.rag.config_v3 import get_index_dir

    assert str(get_index_dir()).endswith("src/uav_risk/stage2/knowledge/vectdb")


def test_rebuild_script_not_defaulting_to_home_indices() -> None:
    from uav_risk.stage2.rag.config_v3 import get_index_dir

    assert "/home/vscode/.uav_rag/indices" not in str(get_index_dir())


def test_rebuild_script_no_groq_or_llm_imports() -> None:
    src = _read("scripts/rebuild_stage2_rag_index.py").lower()
    assert "groq" not in src
    assert "llm" not in src


def test_rebuild_script_does_not_touch_core_ml_api() -> None:
    src = _read("scripts/rebuild_stage2_rag_index.py")
    assert "uav_risk.core" not in src
    assert "uav_risk.ml" not in src
    assert "uav_risk.api" not in src
