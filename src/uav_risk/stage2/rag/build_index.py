"""
Module: src/uav_risk/stage2/rag/build_index.py
Author: Elite Technical Partner
Description: Production-ready pipeline to build, partition, and verify the FAISS Vector Index locally.
"""

import os
import json
import time
from typing import Dict, Any
import structlog
import hmac
import hashlib
import base64
from datetime import datetime
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# استيراد كلاس الإعدادات الموحد لمنع تشتت المسارات
from uav_risk.stage2.rag.config import RAGConfig

# استدعاء الـ Logger المركزي للمنظومة الجوية
logger = structlog.get_logger()

def build_rag_index(
    config: RAGConfig = None,
    chunk_size: int = 800,
    chunk_overlap: int = 150
) -> Dict[str, Any]:
    """
    Loads aviation policy PDFs, splits them into semantic legislative fragments,
    builds a vector index via local embeddings, and performs an integrated safety validation.
    """
    start_time = time.perf_counter()
    
    # في حال لم يتم تمرير إعدادات، يتم توليد الكائن الافتراضي الذكي ديناميكياً
    if config is None:
        config = RAGConfig()

    # سحب المسارات الحية والمكتشفة بصرياً من كلاس الـ Config
    docs_dir = str(config.DOCS_PATH)
    index_dir = str(config.INDEX_PATH)
    embedding_model_dir = str(config.EMBEDDING_PATH)

    logger.info("rag_index_build_started", docs_dir=docs_dir, index_dir=index_dir)

    report: Dict[str, Any] = {
        "status": "failed",
        "num_documents": 0,
        "num_chunks": 0,
        "index_path": index_dir,
        "doc_details": []
    }

    try:
        # 1. التحقق من وجود المجلدات والملفات التنظيمية الحتمية
        if not os.path.exists(docs_dir):
            error_msg = f"Regulatory documents directory not found at: {docs_dir}"
            logger.error("rag_index_build_failed", reason=error_msg)
            return report

        if not os.path.exists(embedding_model_dir):
            error_msg = f"Offline Embedding Model weights folder not found at: {embedding_model_dir}"
            logger.error("rag_index_build_failed", reason=error_msg)
            return report

        pdf_files = [f for f in os.listdir(docs_dir) if f.lower().endswith('.pdf')]
        if not pdf_files:
            error_msg = f"No PDF files found inside the directory: {docs_dir}"
            logger.error("rag_index_build_failed", reason=error_msg)
            return report

        # 2. تحميل كتل المستندات وحقن الـ Metadata التشريعية الصارمة
        all_pages = []
        for pdf_file in pdf_files:
            pdf_path = os.path.join(docs_dir, pdf_file)
            logger.info("loading_aviation_pdf", file_name=pdf_file)
            
            try:
                loader = PyPDFLoader(pdf_path)
                pages = loader.load()
                
                # تعديل الـ Metadata لضمان وجود حقول تتبع جنائية موثقة
                for page_idx, page in enumerate(pages):
                    page.metadata = {
                        "source_file": pdf_file,
                        "page_number": page_idx + 1,
                        "total_pages": len(pages)
                    }
                
                all_pages.extend(pages)
                report["doc_details"].append({
                    "file": pdf_file,
                    "pages": len(pages)
                })
                logger.info("loaded_pdf_success", file_name=pdf_file, pages_count=len(pages))
            except Exception as exc:
                logger.error("pdf_loading_error", file_name=pdf_file, error=str(exc))
                raise exc

        report["num_documents"] = len(pdf_files)

        # 3. التقطيع الدلالي الذكي وفق القواطع التشريعية المنظمة [cite: 2]
        aviation_separators = ["\nArticle ", "\nSection ", "\n§ ", "\n\n", "\n", ". ", " "] 
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=aviation_separators,
            keep_separator=True
        )
        
        all_chunks = splitter.split_documents(all_pages)
        report["num_chunks"] = len(all_chunks)
        logger.info("text_semantic_splitting_complete", total_chunks=len(all_chunks))

        # تأكيد سلامة البصمة والمصدر لكل كسر قبل تشفير المصفوفة المتجهية
        for idx, chunk in enumerate(all_chunks):
            if "source_file" not in chunk.metadata:
                chunk.metadata["source_file"] = "unknown_regulatory_doc"
            if "page_number" not in chunk.metadata:
                chunk.metadata["page_number"] = 0

        # 4. التضمين المحلي المجهول (100% Offline) عبر مسار النموذج الموحد
        logger.info("loading_local_embedding_model", model_path=embedding_model_dir)
        embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model_dir,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )

        # 5. بناء جراف الفهرس وحفظه محلياً في المجلد الفعلي المستقر
        logger.info("building_faiss_graph")
        vector_db = FAISS.from_documents(all_chunks, embeddings)
        
        os.makedirs(index_dir, exist_ok=True)
        vector_db.save_local(index_dir)
        # بعد الحفظ، أنشئ توقيع HMAC للملفات الأساسية داخل الفهرس
        try:
            from uav_risk.stage2.rag.key_manager import get_signing_key
            signing_key = get_signing_key()
            if signing_key:
                sign_faiss_index(index_dir, signing_key)
                logger.info("index_signed", index_dir=index_dir)
            else:
                logger.warning("no_faiss_signing_key_present_index_unsigned", index_dir=index_dir)
        except Exception as sig_exc:
            logger.error("index_signing_failed", error=str(sig_exc))
        
        # كتابة ملف الميتا-داتا المرجعي لحفظ التاريخ والتدقيق
        metadata_payload = {
            "num_chunks": len(all_chunks),
            "sources": pdf_files,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        with open(os.path.join(index_dir, "metadata.json"), "w", encoding="utf-8") as meta_file:
            json.dump(metadata_payload, meta_file, indent=4, ensure_ascii=False)

        # 6. الفحص الجنائي التلقائي المدمج لضمان سلامة جودة الأبعاد واسترجاع البيانات [cite: 6]
        logger.info("running_integrated_sanity_verification")
        # تحقق من التوقيع قبل التحميل؛ دالة التحقق ترفع SecurityError عند عدم المطابقة
        from uav_risk.stage2.rag.build_index import verify_and_safely_load_faiss
        verified_db = verify_and_safely_load_faiss(index_dir, embeddings)
        test_query = "drone maximum altitude operation"
        test_results = verified_db.similarity_search(test_query, k=3)

        if len(test_results) == 0:
            raise RuntimeError("Sanity check failed: Vector DB returned empty result pool for test query.")
        
        for idx, res in enumerate(test_results):
            if "source_file" not in res.metadata or "page_number" not in res.metadata:
                raise KeyError(f"Sanity check failed: Chunk metadata missing tracking keys at index {idx}.")

        elapsed_time = (time.perf_counter() - start_time) * 1000
        logger.info("rag_index_build_complete_success", elapsed_ms=f"{elapsed_time:.2f}ms")
        
        report["status"] = "success"
        return report

    except Exception as general_exc:
        logger.critical("fatal_pipeline_failure_in_rag_indexing", error=str(general_exc))
        report["status"] = "failed"
        report["error_summary"] = str(general_exc)
        return report

# =====================================================================
# Stage 2 Architectural Dependency Comment Block:
# This file constructs the core Vector database artifact used offline.
# Dependencies: src/uav_risk/stage2/rag/config.py
# Dependent Files: Verified during server lifespan setup via tests/unit/test_rag_core.py
# =====================================================================


class SecurityError(Exception):
    pass


def _collect_index_file_paths(index_dir: str):
    """Return a sorted list of file paths that constitute the FAISS index on disk."""
    files = []
    for root, _, filenames in os.walk(index_dir):
        for fn in sorted(filenames):
            # ignore signature file itself when calculating digest
            if fn in ("index.signature",):
                continue
            files.append(os.path.join(root, fn))
    return files


def calculate_file_signature(file_path: str, key: bytes) -> str:
    """Calculate HMAC-SHA256 over a single file and return hex digest."""
    h = hmac.new(key, digestmod=hashlib.sha256)
    with open(file_path, "rb") as fh:
        while True:
            chunk = fh.read(8192)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def sign_faiss_index(index_dir: str, key: bytes) -> None:
    """Create a signature file for the FAISS index directory using HMAC-SHA256.

    The signature is computed deterministically over all files (sorted by path)
    present in the index directory, excluding the signature file itself.
    """
    files = _collect_index_file_paths(index_dir)
    if not files:
        raise RuntimeError("No files found to sign in index directory")

    # Compute HMAC over concatenated file contents in deterministic order
    h = hmac.new(key, digestmod=hashlib.sha256)
    for fp in files:
        with open(fp, "rb") as fh:
            while True:
                chunk = fh.read(8192)
                if not chunk:
                    break
                h.update(chunk)

    signature_hex = h.hexdigest()
    sig_payload = {
        "signature": signature_hex,
        "algo": "HMAC-SHA256",
        "created_at": datetime.utcnow().isoformat() + "Z",
        "files": [os.path.relpath(p, index_dir) for p in files]
    }

    sig_path = os.path.join(index_dir, "index.signature")
    with open(sig_path, "w", encoding="utf-8") as sf:
        json.dump(sig_payload, sf, indent=2)


def verify_and_safely_load_faiss(index_dir: str, embeddings) -> FAISS:
    """Verify signature of index directory and load FAISS only on success.

    Behavior:
      - If `FAISS_SIGNING_KEY` env exists: enforce verification and raise SecurityError on mismatch.
      - If key absent but `FAISS_ALLOW_UNSIGNED` == "1": allow load with warning.
      - Otherwise: raise SecurityError preventing load.
    """
    sig_path = os.path.join(index_dir, "index.signature")
    key_env = os.getenv("FAISS_SIGNING_KEY")
    allow_unsigned = os.getenv("FAISS_ALLOW_UNSIGNED", "0") == "1"

    if not key_env:
        if allow_unsigned:
            logger.warning("loading_unsigned_index_allowed_by_env", index_dir=index_dir)
            return FAISS.load_local(folder_path=index_dir, embeddings=embeddings, allow_dangerous_deserialization=False)
        else:
            raise SecurityError("FAISS signing key not present and unsigned indexes not allowed")

    key = key_env.encode("utf-8")

    # perform signature-only verification first
    verify_index_signature(index_dir, key)

    # load the FAISS index safely now that signature verified
    return FAISS.load_local(folder_path=index_dir, embeddings=embeddings, allow_dangerous_deserialization=False)


def verify_index_signature(index_dir: str, key: bytes) -> bool:
    """Verify only the signature of an index directory. Returns True or raises SecurityError."""
    sig_path = os.path.join(index_dir, "index.signature")
    if not os.path.exists(sig_path):
        raise SecurityError(f"Signature file missing for index at {index_dir}")

    with open(sig_path, "r", encoding="utf-8") as sf:
        payload = json.load(sf)
    expected_sig = payload.get("signature")
    if not expected_sig:
        raise SecurityError("Signature file malformed or missing 'signature' field")

    # compute digest over files
    files = _collect_index_file_paths(index_dir)
    h = hmac.new(key, digestmod=hashlib.sha256)
    for fp in files:
        with open(fp, "rb") as fh:
            while True:
                chunk = fh.read(8192)
                if not chunk:
                    break
                h.update(chunk)

    actual_sig = h.hexdigest()
    if not hmac.compare_digest(actual_sig, expected_sig):
        raise SecurityError("FAISS index signature mismatch - possible tampering detected")

    return True