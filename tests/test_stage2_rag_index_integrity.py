from __future__ import annotations

import hashlib
import json
from pathlib import Path

import faiss
import numpy as np

from uav_risk.stage2.rag.faiss_security import sha256_file, validate_dense_index_bundle


def _make_chunk(i: int, *, source_id: str = "src_a") -> dict:
    txt = f"chunk text {i}"
    return {
        "vector_id": i,
        "chunk_id": f"chunk_{i}",
        "source_id": source_id,
        "source_filename": "a.pdf",
        "source_title": "a",
        "page_start": 1,
        "page_end": 1,
        "section_title": None,
        "text": txt,
        "text_sha256": hashlib.sha256(txt.encode("utf-8")).hexdigest(),
        "char_count": len(txt),
        "token_count": 3,
    }


def _write_sparse(path: Path, n: int) -> None:
    import pickle

    payload = {
        "doc_ids": [f"chunk_{i}" for i in range(n)],
        "chunk_ids": [f"chunk_{i}" for i in range(n)],
        "vector_ids": list(range(n)),
        "doc_texts": [f"t{i}" for i in range(n)],
        "doc_sources": ["a.pdf" for _ in range(n)],
        "term_doc_freq": {"a": {0: 1}},
        "idf": {"a": 1.0},
        "doc_lengths": [1 for _ in range(n)],
        "avg_doc_length": 1.0,
        "k1": 1.5,
        "b": 0.75,
        "N": n,
    }
    with path.open("wb") as f:
        pickle.dump(payload, f)


def _write_mapping(path: Path, chunks: list[dict], dim: int = 3) -> None:
    mapping = {
        "schema_version": "3.0",
        "embedding_dimension": dim,
        "count": len(chunks),
        "chunks": chunks,
        "dimension": dim,
        "doc_ids": [c["chunk_id"] for c in chunks],
        "texts": [c["text"] for c in chunks],
        "sources": [c["source_filename"] for c in chunks],
        "pages": [c["page_start"] or 0 for c in chunks],
    }
    path.write_text(json.dumps(mapping), encoding="utf-8")


def _write_metadata(path: Path, *, dense: Path, mapping: Path, sparse: Path, docs_dir: Path, source_docs: list[dict], n: int) -> None:
    metadata = {
        "schema_version": "3.0",
        "build_tool": "test",
        "build_timestamp": "2026-01-01T00:00:00Z",
        "docs_dir": str(docs_dir),
        "canonical_index_dir": str(path.parent),
        "source_documents": source_docs,
        "source_count": len(source_docs),
        "chunk_count": n,
        "embedding_model_path": "/tmp/embed",
        "reranker_model_path": "/tmp/rerank",
        "dense_index": dense.name,
        "dense_index_sha256": sha256_file(dense),
        "dense_mapping": mapping.name,
        "dense_mapping_sha256": sha256_file(mapping),
        "sparse_index": sparse.name,
        "sparse_index_sha256": sha256_file(sparse),
        "faiss_ntotal": n,
        "dense_mapping_count": n,
        "chunking_config": {"chunk_size": 800},
    }
    path.write_text(json.dumps(metadata), encoding="utf-8")


def _make_docs_dir(tmp_path: Path) -> Path:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.pdf").write_bytes(b"%PDF-test")
    return docs


def test_tiny_temp_faiss_written_by_faiss_write_index_passes_validation(tmp_path: Path) -> None:
    dense = tmp_path / "dense_index.faiss"
    mapping = tmp_path / "dense_mapping.json"
    sparse = tmp_path / "sparse_index.pkl"
    metadata = tmp_path / "metadata.json"

    idx = faiss.IndexFlatIP(3)
    xb = np.array([[0.1, 0.2, 0.3], [0.2, 0.1, 0.4]], dtype="float32")
    faiss.normalize_L2(xb)
    idx.add(xb)
    faiss.write_index(idx, str(dense))

    chunks = [_make_chunk(0), _make_chunk(1)]
    _write_mapping(mapping, chunks)
    _write_sparse(sparse, 2)
    docs = _make_docs_dir(tmp_path)
    _write_metadata(
        metadata,
        dense=dense,
        mapping=mapping,
        sparse=sparse,
        docs_dir=docs,
        source_docs=[{"source_id": "src_a", "filename": "a.pdf", "sha256": "x", "chunk_count": 2}],
        n=2,
    )

    result = validate_dense_index_bundle(
        dense_index_path=dense,
        dense_mapping_path=mapping,
        sparse_index_path=sparse,
        metadata_path=metadata,
        allow_unsigned=True,
    )

    assert result["dense_index_loadable"] is True
    assert result["faiss_ntotal"] == 2
    assert result["chunk_provenance_complete"] is True


def test_random_bytes_named_dense_index_fails_invalid_index(tmp_path: Path) -> None:
    dense = tmp_path / "dense_index.faiss"
    mapping = tmp_path / "dense_mapping.json"
    sparse = tmp_path / "sparse_index.pkl"
    metadata = tmp_path / "metadata.json"

    dense.write_bytes(b"not-a-faiss-index")
    _write_mapping(mapping, [])
    _write_sparse(sparse, 0)
    docs = _make_docs_dir(tmp_path)
    _write_metadata(metadata, dense=dense, mapping=mapping, sparse=sparse, docs_dir=docs, source_docs=[], n=0)

    result = validate_dense_index_bundle(
        dense_index_path=dense,
        dense_mapping_path=mapping,
        sparse_index_path=sparse,
        metadata_path=metadata,
        allow_unsigned=True,
    )
    assert result["status"] == "invalid_index"


def test_validation_detects_missing_dense_mapping(tmp_path: Path) -> None:
    dense = tmp_path / "dense_index.faiss"
    sparse = tmp_path / "sparse_index.pkl"
    metadata = tmp_path / "metadata.json"

    idx = faiss.IndexFlatIP(2)
    x = np.array([[0.1, 0.2]], dtype="float32")
    faiss.normalize_L2(x)
    idx.add(x)
    faiss.write_index(idx, str(dense))
    _write_sparse(sparse, 1)
    docs = _make_docs_dir(tmp_path)
    missing_mapping = tmp_path / "missing.json"
    missing_mapping.write_text("{}", encoding="utf-8")
    _write_metadata(metadata, dense=dense, mapping=missing_mapping, sparse=sparse, docs_dir=docs, source_docs=[{"source_id": "src_a", "filename": "a.pdf", "sha256": "x", "chunk_count": 1}], n=1)

    result = validate_dense_index_bundle(
        dense_index_path=dense,
        dense_mapping_path=tmp_path / "really_missing.json",
        sparse_index_path=sparse,
        metadata_path=metadata,
        allow_unsigned=True,
    )
    assert "missing_dense_mapping" in result["errors"]


def test_validation_detects_mapping_count_mismatch(tmp_path: Path) -> None:
    dense = tmp_path / "dense_index.faiss"
    mapping = tmp_path / "dense_mapping.json"
    sparse = tmp_path / "sparse_index.pkl"
    metadata = tmp_path / "metadata.json"

    idx = faiss.IndexFlatIP(2)
    x = np.array([[0.1, 0.2], [0.2, 0.1]], dtype="float32")
    faiss.normalize_L2(x)
    idx.add(x)
    faiss.write_index(idx, str(dense))

    _write_mapping(mapping, [_make_chunk(0)])
    _write_sparse(sparse, 1)
    docs = _make_docs_dir(tmp_path)
    _write_metadata(metadata, dense=dense, mapping=mapping, sparse=sparse, docs_dir=docs, source_docs=[{"source_id": "src_a", "filename": "a.pdf", "sha256": "x", "chunk_count": 1}], n=2)

    result = validate_dense_index_bundle(
        dense_index_path=dense,
        dense_mapping_path=mapping,
        sparse_index_path=sparse,
        metadata_path=metadata,
        allow_unsigned=True,
    )
    assert "dense_mapping_count_mismatch" in result["errors"]


def test_validation_detects_missing_source_id(tmp_path: Path) -> None:
    dense = tmp_path / "dense_index.faiss"
    mapping = tmp_path / "dense_mapping.json"
    sparse = tmp_path / "sparse_index.pkl"
    metadata = tmp_path / "metadata.json"

    idx = faiss.IndexFlatIP(2)
    x = np.array([[0.1, 0.2]], dtype="float32")
    faiss.normalize_L2(x)
    idx.add(x)
    faiss.write_index(idx, str(dense))

    bad = _make_chunk(0)
    bad.pop("source_id")
    _write_mapping(mapping, [bad])
    _write_sparse(sparse, 1)
    docs = _make_docs_dir(tmp_path)
    _write_metadata(metadata, dense=dense, mapping=mapping, sparse=sparse, docs_dir=docs, source_docs=[{"source_id": "src_a", "filename": "a.pdf", "sha256": "x", "chunk_count": 1}], n=1)

    result = validate_dense_index_bundle(
        dense_index_path=dense,
        dense_mapping_path=mapping,
        sparse_index_path=sparse,
        metadata_path=metadata,
        allow_unsigned=True,
    )
    assert "dense_mapping_chunk_missing_source_id" in result["errors"]


def test_validation_detects_source_document_zero_chunks(tmp_path: Path) -> None:
    dense = tmp_path / "dense_index.faiss"
    mapping = tmp_path / "dense_mapping.json"
    sparse = tmp_path / "sparse_index.pkl"
    metadata = tmp_path / "metadata.json"

    idx = faiss.IndexFlatIP(2)
    x = np.array([[0.1, 0.2]], dtype="float32")
    faiss.normalize_L2(x)
    idx.add(x)
    faiss.write_index(idx, str(dense))

    _write_mapping(mapping, [_make_chunk(0)])
    _write_sparse(sparse, 1)
    docs = _make_docs_dir(tmp_path)
    _write_metadata(metadata, dense=dense, mapping=mapping, sparse=sparse, docs_dir=docs, source_docs=[{"source_id": "src_a", "filename": "a.pdf", "sha256": "x", "chunk_count": 0}], n=1)

    result = validate_dense_index_bundle(
        dense_index_path=dense,
        dense_mapping_path=mapping,
        sparse_index_path=sparse,
        metadata_path=metadata,
        allow_unsigned=True,
    )
    assert "metadata_source_zero_chunks" in result["errors"]


def test_sparse_count_aligns_with_dense_mapping_when_possible(tmp_path: Path) -> None:
    dense = tmp_path / "dense_index.faiss"
    mapping = tmp_path / "dense_mapping.json"
    sparse = tmp_path / "sparse_index.pkl"
    metadata = tmp_path / "metadata.json"

    idx = faiss.IndexFlatIP(2)
    x = np.array([[0.1, 0.2], [0.2, 0.1]], dtype="float32")
    faiss.normalize_L2(x)
    idx.add(x)
    faiss.write_index(idx, str(dense))

    _write_mapping(mapping, [_make_chunk(0), _make_chunk(1)])
    _write_sparse(sparse, 2)
    docs = _make_docs_dir(tmp_path)
    _write_metadata(metadata, dense=dense, mapping=mapping, sparse=sparse, docs_dir=docs, source_docs=[{"source_id": "src_a", "filename": "a.pdf", "sha256": "x", "chunk_count": 2}], n=2)

    result = validate_dense_index_bundle(
        dense_index_path=dense,
        dense_mapping_path=mapping,
        sparse_index_path=sparse,
        metadata_path=metadata,
        allow_unsigned=True,
    )
    assert result["sparse_index_count_matches_dense"] is True


def test_insecure_default_signature_warns_not_silent(tmp_path: Path) -> None:
    dense = tmp_path / "dense_index.faiss"
    mapping = tmp_path / "dense_mapping.json"
    sparse = tmp_path / "sparse_index.pkl"
    metadata = tmp_path / "metadata.json"

    idx = faiss.IndexFlatIP(2)
    x = np.array([[0.1, 0.2]], dtype="float32")
    faiss.normalize_L2(x)
    idx.add(x)
    faiss.write_index(idx, str(dense))

    _write_mapping(mapping, [_make_chunk(0)])
    _write_sparse(sparse, 1)
    docs = _make_docs_dir(tmp_path)
    _write_metadata(metadata, dense=dense, mapping=mapping, sparse=sparse, docs_dir=docs, source_docs=[{"source_id": "src_a", "filename": "a.pdf", "sha256": "x", "chunk_count": 1}], n=1)

    result = validate_dense_index_bundle(
        dense_index_path=dense,
        dense_mapping_path=mapping,
        sparse_index_path=sparse,
        metadata_path=metadata,
        allow_unsigned=True,
    )
    assert any("default_faiss_secret_in_use" == w for w in result["warnings"])


def test_validate_script_exists_and_contains_expected_fields() -> None:
    src = Path("scripts/validate_stage2_rag_index.py").read_text(encoding="utf-8")
    assert "provenance_status" in src
    assert "dense_index_loadable" in src
    assert "faiss_ntotal" in src
    assert "chunk_provenance_complete" in src
