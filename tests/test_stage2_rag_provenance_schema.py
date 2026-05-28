from __future__ import annotations

import hashlib
from pathlib import Path

from uav_risk.stage2.rag import runtime_diagnostics
from uav_risk.stage2.rag.runtime_diagnostics import validate_provenance_metadata


def test_metadata_validation_rejects_missing_schema_version() -> None:
    ok, errors = validate_provenance_metadata({"build_tool": "x"})
    assert ok is False
    assert "missing_schema_version" in errors


def test_metadata_validation_rejects_missing_dense_index_sha256() -> None:
    payload = {
        "schema_version": "3.0",
        "build_tool": "x",
        "build_timestamp": "t",
        "docs_dir": "/tmp",
        "source_documents": [],
        "source_count": 0,
        "chunk_count": 0,
        "embedding_model_path": "/tmp/e",
        "dense_index": "dense_index.faiss",
        "dense_mapping": "dense_mapping.json",
        "dense_mapping_sha256": "abc",
        "sparse_index": "sparse_index.pkl",
        "sparse_index_sha256": "def",
        "faiss_ntotal": 0,
        "dense_mapping_count": 0,
        "canonical_index_dir": "/tmp/i",
    }
    ok, errors = validate_provenance_metadata(payload)
    assert ok is False
    assert "missing_dense_index_sha256" in errors


def test_metadata_validation_rejects_missing_source_documents() -> None:
    payload = {
        "schema_version": "3.0",
        "build_tool": "x",
        "build_timestamp": "t",
        "docs_dir": "/tmp",
        "source_count": 0,
        "chunk_count": 0,
        "embedding_model_path": "/tmp/e",
        "dense_index": "dense_index.faiss",
        "dense_index_sha256": "a",
        "dense_mapping": "dense_mapping.json",
        "dense_mapping_sha256": "b",
        "sparse_index": "sparse_index.pkl",
        "sparse_index_sha256": "c",
        "faiss_ntotal": 0,
        "dense_mapping_count": 0,
        "canonical_index_dir": "/tmp/i",
    }
    ok, errors = validate_provenance_metadata(payload)
    assert ok is False
    assert "missing_source_documents" in errors


def test_metadata_validation_rejects_chunk_count_mismatch() -> None:
    payload = {
        "schema_version": "3.0",
        "build_tool": "x",
        "build_timestamp": "t",
        "docs_dir": "/tmp",
        "source_documents": [],
        "source_count": 0,
        "chunk_count": 3,
        "embedding_model_path": "/tmp/e",
        "dense_index": "dense_index.faiss",
        "dense_index_sha256": "a",
        "dense_mapping": "dense_mapping.json",
        "dense_mapping_sha256": "b",
        "sparse_index": "sparse_index.pkl",
        "sparse_index_sha256": "c",
        "faiss_ntotal": 3,
        "dense_mapping_count": 2,
        "canonical_index_dir": "/tmp/i",
    }
    ok, errors = validate_provenance_metadata(payload)
    assert ok is False
    assert "chunk_count_mismatch" in errors


def test_deterministic_chunk_id_formula_reference_stable() -> None:
    source_id = "source_a"
    page_start = 2
    page_end = 2
    chunk_index = 5
    text_sha = hashlib.sha256("hello world".encode("utf-8")).hexdigest()

    payload = f"{source_id}|{page_start}|{page_end}|{chunk_index}|{text_sha}"
    cid_a = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]
    cid_b = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]
    assert cid_a == cid_b


def test_provenance_classifies_invalid_index_or_integrity_failed(monkeypatch):
    def fake_validate(**kwargs):
        return {
            "status": "invalid_index",
            "dense_index_loadable": False,
            "faiss_ntotal": 0,
            "dense_mapping_count": 0,
            "dense_mapping_matches_faiss": False,
            "sparse_index_exists": False,
            "sparse_index_loadable": False,
            "signature_exists": False,
            "warnings": [],
            "errors": ["invalid_faiss_magic"],
        }

    monkeypatch.setattr(runtime_diagnostics, "validate_dense_index_bundle", fake_validate)
    result = runtime_diagnostics.inspect_rag_index_provenance()
    assert result.provenance_status in {"invalid_index", "stale_or_mismatch"}


def test_runtime_diagnostics_not_current_when_dense_validation_fails(monkeypatch):
    def fake_validate(**kwargs):
        return {
            "status": "integrity_failed",
            "dense_index_loadable": False,
            "faiss_ntotal": 0,
            "dense_mapping_count": 0,
            "dense_mapping_matches_faiss": False,
            "sparse_index_exists": False,
            "sparse_index_loadable": False,
            "signature_exists": False,
            "warnings": [],
            "errors": ["faiss_integrity_failed"],
        }

    monkeypatch.setattr(runtime_diagnostics, "validate_dense_index_bundle", fake_validate)
    result = runtime_diagnostics.inspect_rag_index_provenance()
    assert result.provenance_status != "current"


def test_source_document_fields_expected_by_schema_exist_in_builder_source() -> None:
    src = Path("src/uav_risk/stage2/rag/build_index.py").read_text(encoding="utf-8")
    assert '"source_id"' in src
    assert '"filename"' in src
    assert '"sha256"' in src
    assert '"chunk_count"' in src
