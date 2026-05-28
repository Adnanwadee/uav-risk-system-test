from __future__ import annotations

import json
import sys

from uav_risk.stage2.rag.runtime_diagnostics import inspect_rag_index_provenance


def main() -> int:
    try:
        provenance = inspect_rag_index_provenance()
        payload = {
            "provenance_status": provenance.provenance_status,
            "dense_index_loadable": provenance.validation.get("dense_index_loadable", False),
            "faiss_ntotal": provenance.validation.get("faiss_ntotal", 0),
            "dense_mapping_count": provenance.validation.get("dense_mapping_count", 0),
            "chunk_count": provenance.validation.get("chunk_count", 0),
            "source_count": provenance.validation.get("source_count", 0),
            "docs_pdf_count": provenance.validation.get("docs_pdf_count", 0),
            "chunk_provenance_complete": provenance.validation.get("chunk_provenance_complete", False),
            "all_source_documents_have_chunks": provenance.validation.get("all_source_documents_have_chunks", False),
            "dense_mapping_schema_version": provenance.validation.get("dense_mapping_schema_version"),
            "sparse_index_exists": provenance.validation.get("sparse_index_exists", False),
            "sparse_index_loadable": provenance.validation.get("sparse_index_loadable", False),
            "metadata_valid": provenance.validation.get("metadata_valid", False),
            "resolved_dense_index_path": provenance.index_path,
            "resolved_sparse_index_path": provenance.sparse_index_path,
            "path_resolution_status": provenance.path_resolution_status,
            "warnings": [provenance.warning] if provenance.warning else [],
            "errors": [],
        }
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0
    except Exception:
        print(json.dumps({"status": "failed", "error": "stage2_rag_validate_script_failed"}, indent=2))
        return 1


if __name__ == "__main__":
    sys.exit(main())
