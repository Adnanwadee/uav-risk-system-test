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
        verified_db = FAISS.load_local(index_dir, embeddings, allow_dangerous_deserialization=True)
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