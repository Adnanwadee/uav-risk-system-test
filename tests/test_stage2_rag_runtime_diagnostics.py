from __future__ import annotations

import inspect

import pytest

from uav_risk.stage2.rag import runtime_diagnostics
from uav_risk.stage2.rag.config_v3 import get_dense_index_path
from uav_risk.stage2.rag.runtime_diagnostics import (
    inspect_rag_index_provenance,
    run_rag_runtime_diagnostic,
)


def _fake_bundle_status(status: str = "current"):
    return {
        "status": status,
        "dense_index_exists": status != "missing_index",
        "dense_index_loadable": status == "current",
        "faiss_ntotal": 10 if status == "current" else 0,
        "dense_mapping_exists": status == "current",
        "dense_mapping_count": 10 if status == "current" else 0,
        "dense_mapping_schema_version": "3.0" if status == "current" else None,
        "dense_mapping_matches_faiss": status == "current",
        "chunk_provenance_complete": status == "current",
        "sparse_index_exists": True,
        "sparse_index_loadable": True,
        "sparse_index_count_matches_dense": status == "current",
        "metadata_exists": True,
        "metadata_valid": status == "current",
        "all_source_documents_have_chunks": status == "current",
        "source_count": 9 if status == "current" else 0,
        "chunk_count": 10 if status == "current" else 0,
        "docs_pdf_count": 9,
        "signature_exists": False,
        "warnings": [],
        "errors": [],
    }


def test_inspect_resources_does_not_initialize_rag_core() -> None:
    source = inspect.getsource(runtime_diagnostics)
    assert "initialize(" not in source


def test_inspect_resources_does_not_load_faiss_vector_models() -> None:
    source = inspect.getsource(runtime_diagnostics)
    prefix = source.split("def run_rag_runtime_diagnostic", 1)[0].lower()
    assert "verify_and_safely_load_faiss" not in prefix


def test_inspect_index_provenance_uses_canonical_metadata_when_present(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runtime_diagnostics, "validate_dense_index_bundle", lambda **kwargs: _fake_bundle_status("current"))
    result = inspect_rag_index_provenance()
    assert result.path_resolution_status == "canonical"
    assert result.metadata_path is not None
    assert result.metadata_path.endswith("metadata.json")


@pytest.mark.asyncio
async def test_run_diagnostic_without_quality_does_not_build_runtime_adapter(monkeypatch: pytest.MonkeyPatch) -> None:
    called = {"v": False}

    def fake_builder():
        called["v"] = True
        return None

    monkeypatch.setattr(runtime_diagnostics, "build_runtime_rag_adapter_if_available", fake_builder)
    _ = await run_rag_runtime_diagnostic(run_quality=False)
    assert called["v"] is False


@pytest.mark.asyncio
async def test_run_diagnostic_without_quality_returns_result() -> None:
    result = await run_rag_runtime_diagnostic(run_quality=False)
    assert result.status.value in {"completed", "degraded"}


@pytest.mark.asyncio
async def test_run_diagnostic_with_quality_handles_builder_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runtime_diagnostics, "build_runtime_rag_adapter_if_available", lambda: None)
    result = await run_rag_runtime_diagnostic(run_quality=True)
    assert result.errors


@pytest.mark.asyncio
async def test_run_diagnostic_with_quality_all_insufficient_sets_quality_not_proven(monkeypatch: pytest.MonkeyPatch) -> None:
    from uav_risk.stage2.rag.quality import RAGQualityReport
    from uav_risk.stage2.contracts import Stage2Status

    class DummyAdapter:
        pass

    async def fake_quality(*args, **kwargs):
        return RAGQualityReport(
            status=Stage2Status.DEGRADED,
            metadata={
                "quality_is_proven": False,
                "retrieval_usable": False,
                "supported_cases": 0,
            },
        )

    monkeypatch.setattr(runtime_diagnostics, "build_runtime_rag_adapter_if_available", lambda: DummyAdapter())
    monkeypatch.setattr(runtime_diagnostics, "evaluate_rag_adapter_quality", fake_quality)
    monkeypatch.setattr(runtime_diagnostics, "inspect_rag_index_provenance", lambda: inspect_rag_index_provenance())
    result = await run_rag_runtime_diagnostic(run_quality=True)
    assert result.metadata["quality_is_proven"] is False


@pytest.mark.asyncio
async def test_run_diagnostic_with_quality_catches_quality_failure_without_stacktrace(monkeypatch: pytest.MonkeyPatch) -> None:
    from uav_risk.stage2.rag.runtime_diagnostics import RAGIndexProvenanceStatus

    class DummyAdapter:
        pass

    async def fake_quality(*args, **kwargs):
        raise RuntimeError("traceback: internal details")

    monkeypatch.setattr(runtime_diagnostics, "build_runtime_rag_adapter_if_available", lambda: DummyAdapter())
    monkeypatch.setattr(runtime_diagnostics, "evaluate_rag_adapter_quality", fake_quality)
    monkeypatch.setattr(
        runtime_diagnostics,
        "inspect_rag_index_provenance",
        lambda: RAGIndexProvenanceStatus(
            index_path="/tmp/dense_index.faiss",
            sparse_index_path="/tmp/sparse_index.pkl",
            dense_mapping_path="/tmp/dense_mapping.json",
            metadata_path="/tmp/metadata.json",
            metadata_exists=True,
            index_exists=True,
            provenance_status="current",
            path_resolution_status="canonical",
            validation={"dense_index_loadable": True},
        ),
    )
    result = await run_rag_runtime_diagnostic(run_quality=True)
    assert any(e.code == "rag_quality_failed" for e in result.errors)
    assert all("traceback" not in e.message.lower() for e in result.errors)


@pytest.mark.asyncio
async def test_run_quality_true_with_invalid_index_skips_retrieval(monkeypatch: pytest.MonkeyPatch) -> None:
    from uav_risk.stage2.rag.runtime_diagnostics import RAGIndexProvenanceStatus

    monkeypatch.setattr(
        runtime_diagnostics,
        "inspect_rag_index_provenance",
        lambda: RAGIndexProvenanceStatus(
            index_path="/tmp/dense_index.faiss",
            sparse_index_path="/tmp/sparse_index.pkl",
            dense_mapping_path="/tmp/dense_mapping.json",
            metadata_path="/tmp/metadata.json",
            metadata_exists=False,
            index_exists=False,
            provenance_status="invalid_index",
            path_resolution_status="canonical",
            validation={"dense_index_loadable": False},
        ),
    )
    result = await run_rag_runtime_diagnostic(run_quality=True)
    assert any(e.code == "rag_quality_skipped_index_invalid" for e in result.errors)


def test_runtime_diagnostics_surfaces_path_resolution_status(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runtime_diagnostics, "validate_dense_index_bundle", lambda **kwargs: _fake_bundle_status("current"))
    result = inspect_rag_index_provenance()
    assert result.path_resolution_status == "canonical"


def test_runtime_diagnostics_dense_path_matches_helper(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runtime_diagnostics, "validate_dense_index_bundle", lambda **kwargs: _fake_bundle_status("current"))
    result = inspect_rag_index_provenance()
    assert result.index_path is not None
    assert result.index_path.endswith(get_dense_index_path().name)


def test_runtime_diagnostics_reports_new_integrity_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runtime_diagnostics, "validate_dense_index_bundle", lambda **kwargs: _fake_bundle_status("current"))
    result = inspect_rag_index_provenance()
    assert "chunk_provenance_complete" in result.validation
    assert "source_count" in result.validation
    assert "docs_pdf_count" in result.validation


def test_runtime_diagnostics_no_groq_or_llm_imports() -> None:
    source = inspect.getsource(runtime_diagnostics).lower()
    assert "groq" not in source
    assert "report_writer" not in source


def test_runtime_diagnostics_does_not_touch_artifacts_paths() -> None:
    source = inspect.getsource(runtime_diagnostics).lower()
    assert "artifacts" not in source


@pytest.mark.asyncio
async def test_runtime_diagnostic_exposes_rag_quality_alias(monkeypatch: pytest.MonkeyPatch) -> None:
    from uav_risk.stage2.rag.quality import RAGQualityReport
    from uav_risk.stage2.contracts import Stage2Status

    class DummyAdapter:
        pass

    async def fake_quality(*args, **kwargs):
        return RAGQualityReport(
            status=Stage2Status.COMPLETED,
            metadata={
                "quality_is_proven": True,
                "retrieval_usable": True,
            },
        )

    monkeypatch.setattr(runtime_diagnostics, "build_runtime_rag_adapter_if_available", lambda: DummyAdapter())
    monkeypatch.setattr(runtime_diagnostics, "evaluate_rag_adapter_quality", fake_quality)
    monkeypatch.setattr(runtime_diagnostics, "inspect_rag_index_provenance", lambda: inspect_rag_index_provenance())
    result = await run_rag_runtime_diagnostic(run_quality=True)
    assert result.metadata["rag_quality_is_proven"] is True
    assert result.metadata["quality_is_proven"] is True
