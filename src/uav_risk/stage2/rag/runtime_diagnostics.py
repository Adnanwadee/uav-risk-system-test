from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from uav_risk.stage2.contracts import Stage2Error, Stage2Status
from uav_risk.stage2.rag.faiss_security import validate_dense_index_bundle
from uav_risk.stage2.rag.quality import (
    RAGQualityReport,
    build_runtime_rag_adapter_if_available,
    evaluate_rag_adapter_quality,
)

JsonScalar = str | int | float | bool | None


class RAGRuntimeResourceStatus(BaseModel):
    name: str
    path: str
    exists: bool
    is_dir: bool
    file_count: int | None
    warning: str | None = None


class RAGIndexProvenanceStatus(BaseModel):
    index_path: str | None
    sparse_index_path: str | None = None
    dense_mapping_path: str | None = None
    metadata_path: str | None
    metadata_exists: bool
    index_exists: bool
    provenance_status: str
    warning: str | None = None
    metadata_summary: dict[str, JsonScalar] = Field(default_factory=dict)
    path_resolution_status: str = "unknown"
    validation: dict[str, JsonScalar] = Field(default_factory=dict)


class RAGRuntimeDiagnosticResult(BaseModel):
    status: Stage2Status
    resources: list[RAGRuntimeResourceStatus] = Field(default_factory=list)
    index_provenance: RAGIndexProvenanceStatus
    quality_report: RAGQualityReport | None = None
    errors: list[Stage2Error] = Field(default_factory=list)
    metadata: dict[str, JsonScalar] = Field(default_factory=dict)


def _resource_status(name: str, path: Path) -> RAGRuntimeResourceStatus:
    exists = path.exists()
    is_dir = path.is_dir() if exists else False
    file_count: int | None = None
    warning: str | None = None
    if exists and is_dir:
        try:
            file_count = sum(1 for _ in path.iterdir())
        except Exception:
            file_count = None
            warning = "Could not count files in directory."
    elif exists and not is_dir:
        warning = "Resource path exists but is not a directory."
    else:
        warning = "Resource path does not exist."

    return RAGRuntimeResourceStatus(
        name=name,
        path=str(path),
        exists=exists,
        is_dir=is_dir,
        file_count=file_count,
        warning=warning,
    )


def inspect_rag_runtime_resources() -> list[RAGRuntimeResourceStatus]:
    """Inspect resource paths only; no heavy runtime loading."""
    from uav_risk.stage2.rag.config_v3 import (
        DOCS_PATH,
        EMBEDDING_PATH,
        RERANKER_PATH,
        get_index_dir,
    )

    legacy_vector_db = Path("src/uav_risk/stage2/knowledge/vector_db")

    return [
        _resource_status("docs", Path(DOCS_PATH).expanduser().resolve()),
        _resource_status("embedding_model", Path(EMBEDDING_PATH).expanduser().resolve()),
        _resource_status("reranker_model", Path(RERANKER_PATH).expanduser().resolve()),
        _resource_status("configured_index_dir", get_index_dir()),
        _resource_status("legacy_vector_db", legacy_vector_db),
    ]


def _safe_read_json(path: Path) -> dict[str, JsonScalar] | None:
    try:
        if not path.exists() or not path.is_file():
            return None
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            out: dict[str, JsonScalar] = {}
            for key, value in data.items():
                if isinstance(value, (str, int, float, bool)) or value is None:
                    out[str(key)] = value
                elif isinstance(value, list):
                    out[str(key)] = len(value)
                elif isinstance(value, dict):
                    out[str(key)] = str(type(value).__name__)
                else:
                    out[str(key)] = str(type(value).__name__)
            return out
        return None
    except Exception:
        return None


def validate_provenance_metadata(metadata: dict[str, Any]) -> tuple[bool, list[str]]:
    required_keys = [
        "schema_version",
        "build_tool",
        "build_timestamp",
        "docs_dir",
        "source_documents",
        "source_count",
        "chunk_count",
        "embedding_model_path",
        "dense_index",
        "dense_index_sha256",
        "dense_mapping",
        "dense_mapping_sha256",
        "sparse_index",
        "sparse_index_sha256",
        "faiss_ntotal",
        "dense_mapping_count",
        "canonical_index_dir",
    ]
    errors: list[str] = []
    for key in required_keys:
        if key not in metadata:
            errors.append(f"missing_{key}")

    src_docs = metadata.get("source_documents")
    if "source_documents" in metadata and not isinstance(src_docs, list):
        errors.append("invalid_source_documents")

    if "source_count" in metadata and isinstance(src_docs, list):
        if int(metadata.get("source_count", -1)) != len(src_docs):
            errors.append("source_count_mismatch")

    if "faiss_ntotal" in metadata and "dense_mapping_count" in metadata:
        if int(metadata.get("faiss_ntotal", -1)) != int(metadata.get("dense_mapping_count", -2)):
            errors.append("chunk_count_mismatch")

    return (len(errors) == 0), errors


def inspect_rag_index_provenance() -> RAGIndexProvenanceStatus:
    from uav_risk.stage2.rag.config_v3 import (
        DOCS_PATH,
        EMBEDDING_PATH,
        RERANKER_PATH,
        get_dense_index_path,
        get_dense_mapping_path,
        get_index_dir,
        get_index_metadata_path,
        get_sparse_index_path,
    )

    dense_path = get_dense_index_path()
    sparse_path = get_sparse_index_path()
    mapping_path = get_dense_mapping_path()
    metadata_path = get_index_metadata_path()

    path_resolution_status = "canonical"
    warning: str | None = None

    bundle = validate_dense_index_bundle(
        dense_index_path=dense_path,
        dense_mapping_path=mapping_path,
        sparse_index_path=sparse_path,
        metadata_path=metadata_path,
        allow_unsigned=True,
    )

    metadata_exists = metadata_path.exists()
    metadata_summary = _safe_read_json(metadata_path) or {}

    if bundle["status"] in {"missing_index", "invalid_index"}:
        provenance_status = bundle["status"]
    elif bundle["status"] == "integrity_failed":
        provenance_status = "stale_or_mismatch"
    elif not metadata_exists:
        provenance_status = "missing_index"
    elif bundle["status"] == "current":
        provenance_status = "current"
    else:
        provenance_status = "stale_or_mismatch"

    if bundle.get("warnings"):
        msg = ", ".join(str(x) for x in bundle["warnings"])
        warning = (warning + " " if warning else "") + msg

    docs_dir = Path(DOCS_PATH).expanduser().resolve()
    docs_pdf_count = len(list(docs_dir.glob("*.pdf"))) if docs_dir.exists() else 0

    return RAGIndexProvenanceStatus(
        index_path=str(dense_path),
        sparse_index_path=str(sparse_path),
        dense_mapping_path=str(mapping_path),
        metadata_path=str(metadata_path),
        metadata_exists=metadata_exists,
        index_exists=dense_path.exists(),
        provenance_status=provenance_status,
        warning=warning,
        metadata_summary=metadata_summary,
        path_resolution_status=path_resolution_status,
        validation={
            "docs_dir_exists": docs_dir.exists(),
            "docs_pdf_count": docs_pdf_count,
            "models_dir_exists": Path(EMBEDDING_PATH).expanduser().resolve().parent.exists()
            and Path(RERANKER_PATH).expanduser().resolve().parent.exists(),
            "canonical_index_dir": str(get_index_dir()),
            "dense_index_exists": bool(bundle.get("dense_index_exists")),
            "dense_mapping_exists": bool(bundle.get("dense_mapping_exists")),
            "sparse_index_exists": bool(bundle.get("sparse_index_exists")),
            "metadata_exists": bool(bundle.get("metadata_exists")),
            "dense_mapping_schema_version": bundle.get("dense_mapping_schema_version"),
            "chunk_provenance_complete": bool(bundle.get("chunk_provenance_complete")),
            "source_count": int(bundle.get("source_count") or 0),
            "chunk_count": int(bundle.get("chunk_count") or 0),
            "faiss_ntotal": int(bundle.get("faiss_ntotal") or 0),
            "dense_mapping_count": int(bundle.get("dense_mapping_count") or 0),
            "all_source_documents_have_chunks": bool(bundle.get("all_source_documents_have_chunks")),
            "dense_index_loadable": bool(bundle.get("dense_index_loadable")),
            "dense_mapping_matches_faiss": bool(bundle.get("dense_mapping_matches_faiss")),
            "sparse_index_loadable": bool(bundle.get("sparse_index_loadable")),
            "sparse_index_count_matches_dense": bool(bundle.get("sparse_index_count_matches_dense")),
            "metadata_valid": bool(bundle.get("metadata_valid")),
            "signature_exists": bool(bundle.get("signature_exists")),
        },
    )


async def run_rag_runtime_diagnostic(*, run_quality: bool = False) -> RAGRuntimeDiagnosticResult:
    resources = inspect_rag_runtime_resources()
    provenance = inspect_rag_index_provenance()
    errors: list[Stage2Error] = []
    quality_report: RAGQualityReport | None = None

    missing_critical = any(
        (item.name in {"docs", "embedding_model", "reranker_model"} and not item.exists)
        for item in resources
    )

    status = Stage2Status.COMPLETED
    if missing_critical or provenance.provenance_status in {
        "missing_index",
        "invalid_index",
        "stale_or_mismatch",
        "unknown",
    }:
        status = Stage2Status.DEGRADED

    if run_quality:
        if provenance.provenance_status in {"missing_index", "invalid_index", "stale_or_mismatch"}:
            errors.append(
                Stage2Error(
                    code="rag_quality_skipped_index_invalid",
                    message="Quality retrieval skipped because canonical index validation failed; rebuild required.",
                    details={
                        "provenance_status": provenance.provenance_status,
                        "resolved_dense_index_path": provenance.index_path,
                        "path_resolution_status": provenance.path_resolution_status,
                    },
                )
            )
        else:
            try:
                adapter = build_runtime_rag_adapter_if_available()
                if adapter is None:
                    errors.append(
                        Stage2Error(
                            code="rag_runtime_unavailable",
                            message="Runtime RAG adapter is not available from local resources.",
                            details={
                                "resolved_dense_index_path": provenance.index_path,
                                "path_resolution_status": provenance.path_resolution_status,
                            },
                        )
                    )
                else:
                    quality_report = await evaluate_rag_adapter_quality(
                        adapter,
                        provenance_status=provenance.provenance_status,
                        extra_metadata={
                            "resolved_dense_index_path": provenance.index_path,
                            "resolved_sparse_index_path": provenance.sparse_index_path,
                            "path_resolution_status": provenance.path_resolution_status,
                        },
                    )
                    if quality_report.status != Stage2Status.COMPLETED:
                        status = Stage2Status.DEGRADED
            except Exception:
                errors.append(
                    Stage2Error(
                        code="rag_quality_failed",
                        message="Quality retrieval run failed.",
                        details={
                            "resolved_dense_index_path": provenance.index_path,
                            "path_resolution_status": provenance.path_resolution_status,
                        },
                    )
                )

    infrastructure_valid = bool(
        provenance.validation.get("docs_dir_exists")
        and provenance.validation.get("models_dir_exists")
    )
    index_valid = bool(
        provenance.validation.get("dense_index_loadable")
        and provenance.validation.get("dense_mapping_matches_faiss")
        and provenance.validation.get("sparse_index_loadable")
    )
    provenance_valid = bool(
        provenance.validation.get("metadata_valid")
        and provenance.validation.get("chunk_provenance_complete")
        and provenance.validation.get("all_source_documents_have_chunks")
    )

    metadata: dict[str, JsonScalar] = {
        "run_quality": run_quality,
        "resource_count": len(resources),
        "provenance_status": provenance.provenance_status,
        "resolved_dense_index_path": provenance.index_path,
        "resolved_sparse_index_path": provenance.sparse_index_path,
        "path_resolution_status": provenance.path_resolution_status,
        "infrastructure_valid": infrastructure_valid,
        "index_valid": index_valid,
        "provenance_valid": provenance_valid,
        "retrieval_usable": False,
        "rag_quality_is_proven": False,
        "quality_is_proven": False,
        "docs_pdf_count": provenance.validation.get("docs_pdf_count"),
        "source_count": provenance.validation.get("source_count"),
        "chunk_count": provenance.validation.get("chunk_count"),
        "faiss_ntotal": provenance.validation.get("faiss_ntotal"),
        "dense_mapping_count": provenance.validation.get("dense_mapping_count"),
        "chunk_provenance_complete": provenance.validation.get("chunk_provenance_complete"),
    }

    if quality_report is not None:
        metadata["retrieval_usable"] = bool(quality_report.metadata.get("retrieval_usable", False))
        metadata["rag_quality_is_proven"] = bool(quality_report.metadata.get("rag_quality_is_proven", quality_report.metadata.get("quality_is_proven", False)))
        metadata["quality_is_proven"] = bool(metadata["rag_quality_is_proven"])

    if errors:
        status = Stage2Status.DEGRADED

    return RAGRuntimeDiagnosticResult(
        status=status,
        resources=resources,
        index_provenance=provenance,
        quality_report=quality_report,
        errors=errors,
        metadata=metadata,
    )
