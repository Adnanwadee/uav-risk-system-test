"""
FAISS Security Module - Safe Index Loading with Integrity Verification
V3.1 FIX: Replaced pickle with JSON for signature files (prevents RCE)
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import pickle
from pathlib import Path
from typing import Any, Optional, Tuple

logger = logging.getLogger(__name__)


class FAISSSecurityError(Exception):
    """Raised when FAISS index fails security checks"""


class FAISSIndexVerifier:
    """
    Verifies and safely loads FAISS indices with HMAC integrity checks.
    Prevents deserialization attacks and index tampering.

    V3.1: Uses JSON instead of pickle for signatures (anti-RCE)
    """

    def __init__(self, secret_key: Optional[str] = None):
        raw_secret = secret_key if secret_key is not None else os.getenv("UAV_FAISS_SECRET", "")
        self._using_default_secret = not bool(raw_secret)
        self.secret_key = raw_secret.encode() if raw_secret else b"uav_rag_default_secret_change_me"
        if self._using_default_secret:
            logger.warning("No FAISS secret key configured - using default (INSECURE)")

    def _compute_hmac(self, data: bytes) -> str:
        return hmac.new(self.secret_key, data, hashlib.sha256).hexdigest()[:32]

    def _verify_magic_bytes(self, data: bytes) -> bool:
        if len(data) < 4:
            return False
        faiss_magics = [b"h", b"Iw", b"IM", b"IH", b"IxF", b"IxM"]
        return any(data.startswith(magic) for magic in faiss_magics)

    def sign_index(self, index_path: Path, metadata: Optional[dict[str, Any]] = None) -> Path:
        index_path = Path(index_path)
        if not index_path.exists():
            raise FAISSSecurityError(f"Index file not found: {index_path}")

        with index_path.open("rb") as f:
            index_data = f.read()

        sig_payload = {
            "hmac": self._compute_hmac(index_data),
            "size": len(index_data),
            "metadata": metadata or {},
            "algorithm": "HMAC-SHA256",
            "version": "3.1",
        }

        sig_path = index_path.with_suffix(".faiss.sig")
        with sig_path.open("w", encoding="utf-8") as f:
            json.dump(sig_payload, f, indent=2)

        logger.info(f"Signed index: {index_path.name} -> {sig_path.name}")
        return sig_path

    def verify_and_load(self, index_path: Path, allow_unsigned: bool = False) -> Tuple[Optional[object], dict[str, Any]]:
        import faiss

        index_path = Path(index_path)
        if not index_path.exists():
            raise FAISSSecurityError(f"Index file not found: {index_path}")

        with index_path.open("rb") as f:
            index_data = f.read()

        if not self._verify_magic_bytes(index_data):
            raise FAISSSecurityError(
                f"Invalid FAISS magic bytes in {index_path}. "
                "Possible pickle injection attack or corrupted file."
            )

        sig_path = index_path.with_suffix(".faiss.sig")
        metadata: dict[str, Any] = {}

        if sig_path.exists():
            try:
                with sig_path.open("r", encoding="utf-8") as f:
                    sig_payload = json.load(f)
            except json.JSONDecodeError as exc:
                raise FAISSSecurityError(f"Invalid signature file format: {sig_path}") from exc

            computed_hmac = self._compute_hmac(index_data)
            if not hmac.compare_digest(computed_hmac, sig_payload["hmac"]):
                raise FAISSSecurityError(f"HMAC mismatch for {index_path}. Index may be tampered with.")

            if len(index_data) != sig_payload.get("size", len(index_data)):
                raise FAISSSecurityError(
                    f"Size mismatch for {index_path}. Expected {sig_payload['size']}, got {len(index_data)}"
                )

            metadata = sig_payload.get("metadata", {})
            logger.info(f"Index verified: {index_path.name}")
        else:
            if not allow_unsigned:
                raise FAISSSecurityError(
                    f"No signature found for {index_path}. "
                    f"Run sign_index() first or set allow_unsigned=True (DANGEROUS)"
                )
            logger.warning(f"Loading UNSIGNED index: {index_path.name}")

        tmp_path = None
        try:
            import tempfile

            with tempfile.NamedTemporaryFile(delete=False, suffix=".faiss") as tmp:
                tmp.write(index_data)
                tmp_path = tmp.name

            index = faiss.read_index(tmp_path)
            logger.info(f"Safely loaded index: {index_path.name} (ntotal={index.ntotal})")
            return index, metadata

        except Exception as e:
            raise FAISSSecurityError(f"Failed to load FAISS index: {e}") from e
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_sparse_index_file(path: Path) -> tuple[bool, str | None, dict[str, Any] | None]:
    p = Path(path)
    if not p.exists():
        return False, "sparse_index_missing", None
    try:
        with p.open("rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, dict):
            return False, "sparse_index_not_dict", None
        if "doc_ids" not in obj or "N" not in obj:
            return False, "sparse_index_missing_keys", None
        return True, None, obj
    except Exception:
        return False, "sparse_index_unloadable", None


def _validate_mapping_schema(mapping: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    required_top = ["schema_version", "count", "chunks"]
    for key in required_top:
        if key not in mapping:
            errors.append(f"dense_mapping_missing_{key}")

    chunks = mapping.get("chunks")
    if not isinstance(chunks, list):
        return errors + ["dense_mapping_chunks_not_list"]

    if int(mapping.get("count", -1)) != len(chunks):
        errors.append("dense_mapping_count_field_mismatch")

    required_chunk = [
        "vector_id",
        "chunk_id",
        "source_id",
        "source_filename",
        "source_title",
        "text",
        "text_sha256",
    ]
    expected_vector = 0
    seen_sources: set[str] = set()

    for idx, chunk in enumerate(chunks):
        if not isinstance(chunk, dict):
            errors.append("dense_mapping_chunk_not_dict")
            continue
        for key in required_chunk:
            if key not in chunk:
                errors.append(f"dense_mapping_chunk_missing_{key}")

        vid = chunk.get("vector_id")
        if not isinstance(vid, int):
            errors.append("dense_mapping_chunk_invalid_vector_id")
        elif vid != expected_vector:
            errors.append("dense_mapping_vector_id_sequence_invalid")
        expected_vector += 1

        sid = chunk.get("source_id")
        if not isinstance(sid, str) or not sid.strip():
            errors.append("dense_mapping_chunk_missing_source_id")
        else:
            seen_sources.add(sid.strip())

        sfile = chunk.get("source_filename")
        if not isinstance(sfile, str) or not sfile.strip():
            errors.append("dense_mapping_chunk_missing_source_filename")

        txt = chunk.get("text")
        if not isinstance(txt, str) or not txt.strip():
            errors.append("dense_mapping_chunk_empty_text")

        tsha = chunk.get("text_sha256")
        if not isinstance(tsha, str) or not tsha.strip():
            errors.append("dense_mapping_chunk_missing_text_sha256")

        pstart = chunk.get("page_start")
        pend = chunk.get("page_end")
        if pstart is not None:
            if not isinstance(pstart, int) or pstart < 1:
                errors.append("dense_mapping_chunk_invalid_page_start")
        if pend is not None:
            if not isinstance(pend, int) or pend < 1:
                errors.append("dense_mapping_chunk_invalid_page_end")
        if isinstance(pstart, int) and isinstance(pend, int) and pend < pstart:
            errors.append("dense_mapping_chunk_invalid_page_range")

    if not seen_sources:
        errors.append("dense_mapping_no_sources")

    return sorted(set(errors))


def _validate_metadata_against_artifacts(
    metadata_path: Path,
    dense_index_path: Path,
    dense_mapping_path: Path,
    sparse_index_path: Path,
    mapping: dict[str, Any] | None,
) -> tuple[list[str], dict[str, Any]]:
    errors: list[str] = []
    details: dict[str, Any] = {
        "chunk_provenance_complete": False,
        "all_source_documents_have_chunks": False,
        "source_count": 0,
        "chunk_count": 0,
        "docs_pdf_count": None,
        "dense_mapping_schema_version": None,
    }

    if not metadata_path.exists():
        errors.append("missing_metadata")
        return errors, details

    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception:
        return ["metadata_unreadable"], details

    if not isinstance(metadata, dict):
        return ["metadata_not_dict"], details

    details["source_count"] = int(metadata.get("source_count", 0))
    details["chunk_count"] = int(metadata.get("chunk_count", 0))

    required_meta = [
        "schema_version",
        "build_tool",
        "build_timestamp",
        "docs_dir",
        "canonical_index_dir",
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
    ]
    for key in required_meta:
        if key not in metadata:
            errors.append(f"metadata_missing_{key}")

    if dense_index_path.exists() and metadata.get("dense_index_sha256") != sha256_file(dense_index_path):
        errors.append("metadata_dense_index_sha256_mismatch")
    if dense_mapping_path.exists() and metadata.get("dense_mapping_sha256") != sha256_file(dense_mapping_path):
        errors.append("metadata_dense_mapping_sha256_mismatch")
    if sparse_index_path.exists() and metadata.get("sparse_index_sha256") != sha256_file(sparse_index_path):
        errors.append("metadata_sparse_index_sha256_mismatch")

    source_documents = metadata.get("source_documents")
    if not isinstance(source_documents, list):
        errors.append("metadata_invalid_source_documents")
        source_documents = []

    source_ids: set[str] = set()
    all_source_docs_have_chunks = True
    for source in source_documents:
        if not isinstance(source, dict):
            errors.append("metadata_source_document_not_dict")
            all_source_docs_have_chunks = False
            continue
        for key in ["source_id", "filename", "sha256", "chunk_count"]:
            if key not in source:
                errors.append(f"metadata_source_missing_{key}")
        sid = source.get("source_id")
        if isinstance(sid, str) and sid.strip():
            source_ids.add(sid.strip())
        if int(source.get("chunk_count", 0)) <= 0:
            errors.append("metadata_source_zero_chunks")
            all_source_docs_have_chunks = False

    details["all_source_documents_have_chunks"] = all_source_docs_have_chunks and len(source_documents) > 0

    docs_dir_raw = metadata.get("docs_dir")
    if isinstance(docs_dir_raw, str) and docs_dir_raw:
        docs_dir = Path(docs_dir_raw)
        if docs_dir.exists() and docs_dir.is_dir():
            docs_pdf_count = len(list(docs_dir.glob("*.pdf")))
            details["docs_pdf_count"] = docs_pdf_count
            if int(metadata.get("source_count", -1)) != docs_pdf_count:
                errors.append("metadata_source_count_docs_mismatch")

    if mapping and isinstance(mapping, dict):
        details["dense_mapping_schema_version"] = mapping.get("schema_version")
        chunks = mapping.get("chunks", [])
        missing_source_link = False
        for c in chunks if isinstance(chunks, list) else []:
            sid = c.get("source_id") if isinstance(c, dict) else None
            if not isinstance(sid, str) or sid.strip() not in source_ids:
                missing_source_link = True
                break
        if missing_source_link:
            errors.append("chunk_source_missing_in_metadata")

        details["chunk_provenance_complete"] = not any(
            x in errors
            for x in [
                "dense_mapping_chunk_missing_source_id",
                "dense_mapping_chunk_missing_source_filename",
                "dense_mapping_chunk_empty_text",
                "dense_mapping_chunk_missing_text_sha256",
                "chunk_source_missing_in_metadata",
            ]
        )

    return sorted(set(errors)), details


def validate_dense_index_bundle(
    *,
    dense_index_path: Path,
    dense_mapping_path: Path,
    sparse_index_path: Path,
    metadata_path: Path,
    allow_unsigned: bool = True,
) -> dict[str, Any]:
    """Validate dense+sparse+mapping+metadata bundle without assuming trust by existence only."""
    result: dict[str, Any] = {
        "dense_index_path": str(dense_index_path),
        "dense_index_exists": Path(dense_index_path).exists(),
        "dense_index_loadable": False,
        "faiss_ntotal": None,
        "dense_mapping_path": str(dense_mapping_path),
        "dense_mapping_exists": Path(dense_mapping_path).exists(),
        "dense_mapping_count": None,
        "dense_mapping_schema_version": None,
        "dense_mapping_matches_faiss": False,
        "chunk_provenance_complete": False,
        "sparse_index_path": str(sparse_index_path),
        "sparse_index_exists": Path(sparse_index_path).exists(),
        "sparse_index_loadable": False,
        "sparse_index_count_matches_dense": False,
        "metadata_path": str(metadata_path),
        "metadata_exists": Path(metadata_path).exists(),
        "metadata_valid": False,
        "all_source_documents_have_chunks": False,
        "source_count": 0,
        "chunk_count": 0,
        "docs_pdf_count": None,
        "signature_exists": Path(dense_index_path).with_suffix(".faiss.sig").exists(),
        "warnings": [],
        "errors": [],
        "status": "unknown",
    }

    verifier = FAISSIndexVerifier()

    if verifier._using_default_secret:
        result["warnings"].append("default_faiss_secret_in_use")

    if not result["dense_index_exists"]:
        result["errors"].append("missing_dense_index")
        result["status"] = "missing_index"
        return result

    try:
        index, _ = verifier.verify_and_load(Path(dense_index_path), allow_unsigned=allow_unsigned)
        result["dense_index_loadable"] = True
        result["faiss_ntotal"] = int(getattr(index, "ntotal", 0))
        if result["faiss_ntotal"] <= 0:
            result["errors"].append("faiss_ntotal_zero")
    except FAISSSecurityError as exc:
        msg = str(exc).lower()
        if "magic bytes" in msg:
            result["status"] = "invalid_index"
            result["errors"].append("invalid_faiss_magic")
        elif "hmac mismatch" in msg or "signature" in msg or "size mismatch" in msg:
            result["status"] = "integrity_failed"
            result["errors"].append("faiss_integrity_failed")
        else:
            result["status"] = "invalid_index"
            result["errors"].append("faiss_unloadable")
        return result

    mapping: dict[str, Any] | None = None
    if not result["dense_mapping_exists"]:
        result["errors"].append("missing_dense_mapping")
    else:
        try:
            mapping = json.loads(Path(dense_mapping_path).read_text(encoding="utf-8"))
            if not isinstance(mapping, dict):
                result["errors"].append("dense_mapping_not_dict")
                mapping = None
            else:
                result["dense_mapping_schema_version"] = mapping.get("schema_version")
                m_errors = _validate_mapping_schema(mapping)
                result["errors"].extend(m_errors)
                chunks = mapping.get("chunks", []) if isinstance(mapping.get("chunks"), list) else []
                result["dense_mapping_count"] = len(chunks)
                result["dense_mapping_matches_faiss"] = result["faiss_ntotal"] == result["dense_mapping_count"]
                if not result["dense_mapping_matches_faiss"]:
                    result["errors"].append("dense_mapping_count_mismatch")
        except Exception:
            result["errors"].append("dense_mapping_unreadable")

    sparse_ok, sparse_err, sparse_obj = validate_sparse_index_file(Path(sparse_index_path))
    result["sparse_index_loadable"] = sparse_ok
    if not sparse_ok and sparse_err:
        result["errors"].append(sparse_err)
    if sparse_ok and isinstance(sparse_obj, dict):
        sparse_count = int(sparse_obj.get("N", 0))
        result["sparse_index_count_matches_dense"] = result["dense_mapping_count"] == sparse_count
        if not result["sparse_index_count_matches_dense"]:
            result["errors"].append("sparse_dense_count_mismatch")

    meta_errors, meta_details = _validate_metadata_against_artifacts(
        metadata_path=Path(metadata_path),
        dense_index_path=Path(dense_index_path),
        dense_mapping_path=Path(dense_mapping_path),
        sparse_index_path=Path(sparse_index_path),
        mapping=mapping,
    )
    result["errors"].extend(meta_errors)
    result["metadata_valid"] = len(meta_errors) == 0
    result["chunk_provenance_complete"] = bool(meta_details.get("chunk_provenance_complete", False))
    result["all_source_documents_have_chunks"] = bool(meta_details.get("all_source_documents_have_chunks", False))
    result["source_count"] = int(meta_details.get("source_count", 0))
    result["chunk_count"] = int(meta_details.get("chunk_count", 0))
    result["docs_pdf_count"] = meta_details.get("docs_pdf_count")
    if result["dense_mapping_schema_version"] is None and meta_details.get("dense_mapping_schema_version") is not None:
        result["dense_mapping_schema_version"] = meta_details.get("dense_mapping_schema_version")

    if not result["signature_exists"]:
        result["warnings"].append("missing_faiss_signature")

    if result["errors"]:
        if "missing_dense_index" in result["errors"]:
            result["status"] = "missing_index"
        elif any(err in result["errors"] for err in ["invalid_faiss_magic", "faiss_unloadable"]):
            result["status"] = "invalid_index"
        else:
            result["status"] = "integrity_failed"
    else:
        result["status"] = "current"

    return result


def verify_index_integrity(index_path: Path) -> bool:
    try:
        FAISSIndexVerifier().verify_and_load(index_path, allow_unsigned=False)
        return True
    except FAISSSecurityError:
        return False


# Convenience function
def verify_and_safely_load_faiss(
    index_path: str,
    secret_key: Optional[str] = None,
    allow_unsigned: bool = False,
) -> Tuple[Optional[object], dict[str, Any]]:
    verifier = FAISSIndexVerifier(secret_key)
    return verifier.verify_and_load(Path(index_path), allow_unsigned)
