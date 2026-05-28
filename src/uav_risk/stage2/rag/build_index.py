"""
Module: src/uav_risk/stage2/rag/build_index.py
Description: Build canonical dense+sparse RAG indices and provenance metadata.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import pickle
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config_v3 import (
    DOCS_PATH,
    EMBEDDING_PATH,
    RERANKER_PATH,
    RAGConfig,
    get_dense_index_path,
    get_dense_mapping_path,
    get_index_dir,
    get_index_metadata_path,
    get_sparse_index_path,
)
from .faiss_security import FAISSIndexVerifier, sha256_file

logger = logging.getLogger(__name__)


@dataclass
class BuildOutputs:
    dense_index_path: Path
    dense_mapping_path: Path
    sparse_index_path: Path
    metadata_path: Path
    index_dir: Path


def _canonical_outputs() -> BuildOutputs:
    return BuildOutputs(
        dense_index_path=get_dense_index_path(),
        dense_mapping_path=get_dense_mapping_path(),
        sparse_index_path=get_sparse_index_path(),
        metadata_path=get_index_metadata_path(),
        index_dir=get_index_dir(),
    )


def _normalize_source_id(filename: str) -> str:
    stem = Path(filename).stem.lower()
    norm = re.sub(r"[^a-z0-9]+", "_", stem).strip("_") or "source"
    suffix = hashlib.sha256(filename.encode("utf-8")).hexdigest()[:8]
    return f"{norm}_{suffix}"


def _normalize_text(text: str) -> str:
    return "\n".join(line.rstrip() for line in text.strip().splitlines()).strip()


def _chunk_id(source_id: str, page_start: int | None, page_end: int | None, chunk_index: int, text_sha: str) -> str:
    payload = f"{source_id}|{page_start}|{page_end}|{chunk_index}|{text_sha}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def _collect_source_documents(docs_dir: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for pdf in sorted(docs_dir.glob("*.pdf")):
        stat = pdf.stat()
        out.append(
            {
                "source_id": _normalize_source_id(pdf.name),
                "filename": pdf.name,
                "path": str(pdf.resolve()),
                "sha256": sha256_file(pdf),
                "size_bytes": int(stat.st_size),
                "modified_time": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
                "chunk_count": 0,
            }
        )
    return out


def _clear_canonical_outputs(outputs: BuildOutputs) -> None:
    for path in [
        outputs.dense_index_path,
        outputs.dense_index_path.with_suffix(".faiss.sig"),
        outputs.dense_mapping_path,
        outputs.sparse_index_path,
        outputs.metadata_path,
    ]:
        if path.exists():
            path.unlink()


def _ensure_overwrite_policy(outputs: BuildOutputs, force: bool) -> None:
    existing = [
        outputs.dense_index_path,
        outputs.dense_mapping_path,
        outputs.sparse_index_path,
        outputs.metadata_path,
    ]
    if force:
        _clear_canonical_outputs(outputs)
        return
    if any(p.exists() for p in existing):
        raise FileExistsError(
            "Canonical index outputs already exist. Re-run with force=True to overwrite canonical outputs."
        )


def _build_sparse_index(chunk_rows: list[dict[str, Any]], sparse_path: Path) -> dict[str, Any]:
    doc_ids: list[str] = []
    doc_texts: list[str] = []
    doc_sources: list[str] = []
    doc_tokens_list: list[list[str]] = []
    doc_lengths: list[int] = []
    vector_ids: list[int] = []

    for row in chunk_rows:
        doc_ids.append(row["chunk_id"])
        doc_texts.append(row["text"])
        doc_sources.append(row["source_filename"])
        vector_ids.append(int(row["vector_id"]))
        text_clean = re.sub(r"[^a-z0-9\s]", " ", row["text"].lower())
        tokens = [t for t in text_clean.split() if len(t) > 2]
        doc_tokens_list.append(tokens)
        doc_lengths.append(len(tokens))

    N = len(doc_ids)
    avg_doc_length = sum(doc_lengths) / N if N else 0.0

    from collections import defaultdict

    term_doc_freq = defaultdict(lambda: defaultdict(int))
    for doc_idx, tokens in enumerate(doc_tokens_list):
        for token in tokens:
            term_doc_freq[token][doc_idx] += 1

    idf = {}
    for term, freqs in term_doc_freq.items():
        df = len(freqs)
        idf[term] = float(np.log((N - df + 0.5) / (df + 0.5) + 1.0))

    sparse_index = {
        "schema_version": "3.0",
        "doc_ids": doc_ids,
        "chunk_ids": doc_ids,
        "vector_ids": vector_ids,
        "doc_texts": doc_texts,
        "doc_sources": doc_sources,
        "term_doc_freq": dict(term_doc_freq),
        "idf": idf,
        "doc_lengths": doc_lengths,
        "avg_doc_length": avg_doc_length,
        "k1": 1.5,
        "b": 0.75,
        "N": N,
    }

    with sparse_path.open("wb") as f:
        pickle.dump(sparse_index, f)

    return sparse_index


def _write_native_faiss(chunk_rows: list[dict[str, Any]], dense_index_path: Path) -> dict[str, Any]:
    import faiss

    embeddings_np = np.array([row["embedding"] for row in chunk_rows], dtype="float32")
    if embeddings_np.ndim != 2 or embeddings_np.shape[0] == 0:
        raise ValueError("No embeddings generated for dense FAISS index.")

    dimension = int(embeddings_np.shape[1])
    native_index = faiss.IndexFlatIP(dimension)
    faiss.normalize_L2(embeddings_np)
    native_index.add(embeddings_np)
    faiss.write_index(native_index, str(dense_index_path))

    return {
        "count": len(chunk_rows),
        "dimension": dimension,
        "faiss_ntotal": int(native_index.ntotal),
    }


def _write_dense_mapping(chunk_rows: list[dict[str, Any]], dense_stats: dict[str, Any], dense_mapping_path: Path) -> None:
    chunks: list[dict[str, Any]] = []
    doc_ids: list[str] = []
    texts: list[str] = []
    sources: list[str] = []
    pages: list[int] = []

    for row in chunk_rows:
        chunks.append(
            {
                "vector_id": int(row["vector_id"]),
                "chunk_id": row["chunk_id"],
                "source_id": row["source_id"],
                "source_filename": row["source_filename"],
                "source_title": row["source_title"],
                "page_start": row["page_start"],
                "page_end": row["page_end"],
                "section_title": row["section_title"],
                "text": row["text"],
                "text_sha256": row["text_sha256"],
                "char_count": int(row["char_count"]),
                "token_count": int(row["token_count"]),
            }
        )
        doc_ids.append(row["chunk_id"])
        texts.append(row["text"])
        sources.append(row["source_filename"])
        pages.append(int(row["page_start"] or 0))

    mapping = {
        "schema_version": "3.0",
        "embedding_dimension": int(dense_stats["dimension"]),
        "count": int(dense_stats["count"]),
        "chunks": chunks,
        # Backward-compatible fields:
        "dimension": int(dense_stats["dimension"]),
        "doc_ids": doc_ids,
        "texts": texts,
        "sources": sources,
        "pages": pages,
    }
    dense_mapping_path.write_text(json.dumps(mapping, indent=2, ensure_ascii=False), encoding="utf-8")


def _build_metadata(
    *,
    docs_dir: Path,
    source_documents: list[dict[str, Any]],
    chunk_count: int,
    outputs: BuildOutputs,
    dense_stats: dict[str, Any],
    chunk_size: int,
    chunk_overlap: int,
) -> dict[str, Any]:
    dense_sig = outputs.dense_index_path.with_suffix(".faiss.sig")

    metadata: dict[str, Any] = {
        "schema_version": "3.0",
        "build_tool": "uav_risk.stage2.rag.build_index",
        "build_timestamp": datetime.now(timezone.utc).isoformat(),
        "docs_dir": str(docs_dir.resolve()),
        "canonical_index_dir": str(outputs.index_dir.resolve()),
        "source_documents": source_documents,
        "source_count": len(source_documents),
        "chunk_count": int(chunk_count),
        "embedding_model_path": str(Path(EMBEDDING_PATH).resolve()),
        "reranker_model_path": str(Path(RERANKER_PATH).resolve()),
        "dense_index": outputs.dense_index_path.name,
        "dense_index_sha256": sha256_file(outputs.dense_index_path),
        "dense_mapping": outputs.dense_mapping_path.name,
        "dense_mapping_sha256": sha256_file(outputs.dense_mapping_path),
        "sparse_index": outputs.sparse_index_path.name,
        "sparse_index_sha256": sha256_file(outputs.sparse_index_path),
        "faiss_ntotal": int(dense_stats["faiss_ntotal"]),
        "dense_mapping_count": int(dense_stats["count"]),
        "chunking_config": {
            "chunk_size": int(chunk_size),
            "chunk_overlap": int(chunk_overlap),
            "splitter": "RecursiveCharacterTextSplitter",
        },
        "signature": {
            "dense_index_sig": dense_sig.name,
            "dense_index_sig_exists": dense_sig.exists(),
        },
        "compatibility_outputs": {
            "langchain_index_faiss": "index.faiss",
            "langchain_index_pkl": "index.pkl",
            "canonical": False,
            "generated_by_default": False,
        },
    }

    return metadata


def build_rag_index(
    config: RAGConfig | None = None,
    chunk_size: int = 800,
    chunk_overlap: int = 150,
    *,
    force: bool = False,
) -> dict[str, Any]:
    """Build canonical dense/sparse index artifacts and provenance metadata."""
    start = time.perf_counter()
    _ = config or RAGConfig()

    outputs = _canonical_outputs()
    docs_dir = Path(DOCS_PATH).expanduser().resolve()
    outputs.index_dir.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {
        "status": "failed",
        "canonical_index_dir": str(outputs.index_dir),
        "dense_index": str(outputs.dense_index_path),
        "dense_mapping": str(outputs.dense_mapping_path),
        "sparse_index": str(outputs.sparse_index_path),
        "metadata": str(outputs.metadata_path),
    }

    try:
        _ensure_overwrite_policy(outputs, force=force)

        if not docs_dir.exists():
            raise FileNotFoundError(f"Documents directory not found: {docs_dir}")

        source_documents = _collect_source_documents(docs_dir)
        if not source_documents:
            raise FileNotFoundError(f"No PDF files found in: {docs_dir}")

        embedding_model_dir = Path(EMBEDDING_PATH).expanduser().resolve()
        if not embedding_model_dir.exists():
            raise FileNotFoundError(f"Embedding model not found: {embedding_model_dir}")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\nArticle ", "\nSection ", "\n§ ", "\n\n", "\n", ". ", " "],
            keep_separator=True,
        )

        embeddings = HuggingFaceEmbeddings(
            model_name=str(embedding_model_dir),
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

        source_by_filename = {entry["filename"]: entry for entry in source_documents}
        chunk_rows: list[dict[str, Any]] = []

        for src in source_documents:
            pdf_path = docs_dir / src["filename"]
            loader = PyPDFLoader(str(pdf_path))
            pages = loader.load()
            if not pages:
                raise ValueError(f"PDF parsed with zero pages: {src['filename']}")

            for page_idx, page in enumerate(pages):
                page.metadata = {
                    "source_file": src["filename"],
                    "page_number": page_idx + 1,
                    "total_pages": len(pages),
                }

            chunks = splitter.split_documents(pages)
            if not chunks:
                raise ValueError(f"PDF produced zero chunks: {src['filename']}")

            chunk_texts: list[str] = []
            for chunk in chunks:
                text = _normalize_text(chunk.page_content)
                if not text:
                    continue
                chunk_texts.append(text)

            if not chunk_texts:
                raise ValueError(f"PDF produced only empty chunks: {src['filename']}")

            chunk_embeddings = embeddings.embed_documents(chunk_texts)

            text_idx = 0
            for local_idx, chunk in enumerate(chunks):
                text = _normalize_text(chunk.page_content)
                if not text:
                    continue
                emb = chunk_embeddings[text_idx]
                text_idx += 1

                page_num_raw = chunk.metadata.get("page_number")
                page_num = int(page_num_raw) if isinstance(page_num_raw, int) else None
                text_sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
                source_id = src["source_id"]
                chunk_id = _chunk_id(source_id, page_num, page_num, local_idx, text_sha)
                token_count = len([t for t in re.split(r"\s+", text) if t])

                row = {
                    "vector_id": len(chunk_rows),
                    "chunk_id": chunk_id,
                    "source_id": source_id,
                    "source_filename": src["filename"],
                    "source_title": Path(src["filename"]).stem,
                    "page_start": page_num,
                    "page_end": page_num,
                    "section_title": None,
                    "text": text,
                    "text_sha256": text_sha,
                    "char_count": len(text),
                    "token_count": token_count,
                    "embedding": emb,
                }
                chunk_rows.append(row)

                source_entry = source_by_filename[src["filename"]]
                source_entry["chunk_count"] = int(source_entry.get("chunk_count", 0)) + 1

        if not chunk_rows:
            raise ValueError("No non-empty chunks were produced from the docs corpus.")

        for src in source_documents:
            if int(src.get("chunk_count", 0)) <= 0:
                raise ValueError(f"Source has zero chunks: {src['filename']}")

        dense_stats = _write_native_faiss(chunk_rows, outputs.dense_index_path)
        _write_dense_mapping(chunk_rows, dense_stats, outputs.dense_mapping_path)
        sparse_index = _build_sparse_index(chunk_rows, outputs.sparse_index_path)

        verifier = FAISSIndexVerifier()
        verifier.sign_index(
            outputs.dense_index_path,
            metadata={
                "sources": [x["filename"] for x in source_documents],
                "chunks": len(chunk_rows),
                "embedding_model": str(embedding_model_dir),
            },
        )

        metadata = _build_metadata(
            docs_dir=docs_dir,
            source_documents=source_documents,
            chunk_count=len(chunk_rows),
            outputs=outputs,
            dense_stats=dense_stats,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        outputs.metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

        elapsed_ms = (time.perf_counter() - start) * 1000
        report.update(
            {
                "status": "success",
                "elapsed_ms": round(elapsed_ms, 2),
                "source_count": len(source_documents),
                "chunk_count": len(chunk_rows),
                "faiss_ntotal": dense_stats["faiss_ntotal"],
                "dense_mapping_count": dense_stats["count"],
                "sparse_N": sparse_index.get("N", 0),
                "metadata_schema_version": metadata.get("schema_version"),
            }
        )
        return report

    except Exception as exc:
        report["error_summary"] = str(exc)
        return report



def repair_canonical_index_from_existing(*, force: bool = False) -> dict[str, Any]:
    """Repair canonical index artifacts from existing on-disk indices without PDF parsing."""
    import faiss

    outputs = _canonical_outputs()
    outputs.index_dir.mkdir(parents=True, exist_ok=True)

    if not force and outputs.metadata_path.exists():
        return {"status": "failed", "error_summary": "metadata_exists_use_force"}

    compat_dense = outputs.index_dir / "index.faiss"
    compat_mapping = outputs.index_dir / "dense_mapping.json"

    source_dense = outputs.dense_index_path if outputs.dense_index_path.exists() else compat_dense
    if not source_dense.exists():
        return {"status": "failed", "error_summary": "no_dense_index_found"}

    idx = faiss.read_index(str(source_dense))
    faiss.write_index(idx, str(outputs.dense_index_path))

    mapping_obj: dict[str, Any]
    if compat_mapping.exists():
        mapping_obj = json.loads(compat_mapping.read_text(encoding="utf-8"))
        chunks = mapping_obj.get("chunks")
        if isinstance(chunks, list):
            mapping_obj["count"] = len(chunks)
        else:
            doc_ids = mapping_obj.get("doc_ids", [])
            if not isinstance(doc_ids, list) or len(doc_ids) != int(idx.ntotal):
                doc_ids = [f"chunk_{i}" for i in range(int(idx.ntotal))]
                mapping_obj["doc_ids"] = doc_ids
                mapping_obj["count"] = len(doc_ids)
    else:
        mapping_obj = {
            "schema_version": "3.0",
            "count": int(idx.ntotal),
            "embedding_dimension": int(idx.d),
            "chunks": [],
            "dimension": int(idx.d),
            "doc_ids": [f"chunk_{i}" for i in range(int(idx.ntotal))],
            "texts": [],
            "sources": [],
            "pages": [],
        }

    outputs.dense_mapping_path.write_text(json.dumps(mapping_obj, indent=2, ensure_ascii=False), encoding="utf-8")

    if not outputs.sparse_index_path.exists():
        return {"status": "failed", "error_summary": "missing_sparse_index"}

    verifier = FAISSIndexVerifier()
    verifier.sign_index(outputs.dense_index_path, metadata={"repair": True, "ntotal": int(idx.ntotal)})

    docs_dir = Path(DOCS_PATH).expanduser().resolve()
    source_documents = _collect_source_documents(docs_dir) if docs_dir.exists() else []

    metadata = {
        "schema_version": "3.0",
        "build_tool": "uav_risk.stage2.rag.build_index.repair_canonical_index_from_existing",
        "build_timestamp": datetime.now(timezone.utc).isoformat(),
        "docs_dir": str(docs_dir),
        "source_documents": source_documents,
        "source_count": len(source_documents),
        "chunk_count": int(idx.ntotal),
        "embedding_model_path": str(Path(EMBEDDING_PATH).resolve()),
        "reranker_model_path": str(Path(RERANKER_PATH).resolve()),
        "canonical_index_dir": str(outputs.index_dir),
        "dense_index": outputs.dense_index_path.name,
        "dense_index_sha256": sha256_file(outputs.dense_index_path),
        "dense_mapping": outputs.dense_mapping_path.name,
        "dense_mapping_sha256": sha256_file(outputs.dense_mapping_path),
        "sparse_index": outputs.sparse_index_path.name,
        "sparse_index_sha256": sha256_file(outputs.sparse_index_path),
        "faiss_ntotal": int(idx.ntotal),
        "dense_mapping_count": int(mapping_obj.get("count", len(mapping_obj.get("doc_ids", [])))),
        "chunking_config": {"mode": "repaired_from_existing_artifacts"},
        "signature": {
            "dense_index_sig": outputs.dense_index_path.with_suffix(".faiss.sig").name,
            "dense_index_sig_exists": outputs.dense_index_path.with_suffix(".faiss.sig").exists(),
        },
        "compatibility_outputs": {
            "langchain_index_faiss": "index.faiss",
            "langchain_index_pkl": "index.pkl",
            "canonical": False,
            "generated_by_default": False,
        },
    }

    outputs.metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    return {
        "status": "success",
        "canonical_index_dir": str(outputs.index_dir),
        "dense_index": str(outputs.dense_index_path),
        "dense_mapping": str(outputs.dense_mapping_path),
        "sparse_index": str(outputs.sparse_index_path),
        "metadata": str(outputs.metadata_path),
        "faiss_ntotal": int(idx.ntotal),
        "dense_mapping_count": int(mapping_obj.get("count", len(mapping_obj.get("doc_ids", [])))),
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build canonical Stage2 RAG indices.")
    parser.add_argument("--force", action="store_true", help="Overwrite canonical outputs if they already exist.")
    args = parser.parse_args()

    result = build_rag_index(force=args.force)
    print(json.dumps(result, indent=2, ensure_ascii=False))
