"""
Module: src/uav_risk/stage2/rag/build_index.py
Author: Elite Technical Partner + V3.1 Production Fix
Description: Production-ready pipeline to build dense + sparse indices.
Supports: PDF loading, chunking, HMAC signing, sparse BM25 index.
Uses local offline models from knowledge/models/embedding.
"""

import os
import json
import time
import pickle
import re
import logging
from typing import Dict, Any, List, Tuple
from pathlib import Path
from collections import defaultdict
from datetime import datetime
import hmac
import hashlib
import numpy as np

# LangChain imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS as LangchainFAISS

from .config_v3 import RAGConfig, INDEX_DIR, BASE_DIR, DOCS_PATH, EMBEDDING_PATH
from .faiss_security import FAISSIndexVerifier

logger = logging.getLogger(__name__)


def build_rag_index(
    config: RAGConfig = None,
    chunk_size: int = 800,
    chunk_overlap: int = 150
) -> Dict[str, Any]:
    """
    Loads aviation policy PDFs, splits into chunks,
    builds BOTH dense (FAISS) + sparse (BM25) indices.
    Uses local offline embedding model from knowledge/models/embedding.
    """
    start_time = time.perf_counter()

    if config is None:
        config = RAGConfig()

    # Use paths from config_v3 (production paths)
    docs_dir = str(DOCS_PATH)
    index_dir = str(INDEX_DIR)
    embedding_model_dir = str(EMBEDDING_PATH)

    logger.info("rag_index_build_started", docs_dir=docs_dir, index_dir=index_dir, model_dir=embedding_model_dir)

    report: Dict[str, Any] = {
        "status": "failed",
        "num_documents": 0,
        "num_chunks": 0,
        "index_path": index_dir,
        "doc_details": []
    }

    try:
        # 1. Verify directories
        if not os.path.exists(docs_dir):
            error_msg = f"Documents directory not found: {docs_dir}"
            logger.error(error_msg)
            return report

        pdf_files = [f for f in os.listdir(docs_dir) if f.lower().endswith('.pdf')]
        if not pdf_files:
            error_msg = f"No PDF files found: {docs_dir}"
            logger.error(error_msg)
            return report

        # 2. Load PDFs
        all_pages = []
        for pdf_file in pdf_files:
            pdf_path = os.path.join(docs_dir, pdf_file)
            logger.info("loading_pdf", file=pdf_file)

            try:
                loader = PyPDFLoader(pdf_path)
                pages = loader.load()

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
                logger.info("loaded_pdf", file=pdf_file, pages=len(pages))
            except Exception as exc:
                logger.error("pdf_loading_error", file=pdf_file, error=str(exc))
                raise exc

        report["num_documents"] = len(pdf_files)

        # 3. Semantic splitting
        aviation_separators = ["\nArticle ", "\nSection ", "\n§ ", "\n\n", "\n", ". ", " "]
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=aviation_separators,
            keep_separator=True
        )

        all_chunks = splitter.split_documents(all_pages)
        report["num_chunks"] = len(all_chunks)
        logger.info("splitting_complete", chunks=len(all_chunks))

        # Ensure metadata
        for chunk in all_chunks:
            if "source_file" not in chunk.metadata:
                chunk.metadata["source_file"] = "unknown"
            if "page_number" not in chunk.metadata:
                chunk.metadata["page_number"] = 0

        # 4. Load embeddings from LOCAL offline model
        logger.info("loading_embeddings", model=embedding_model_dir)

        # Verify local model exists
        if not os.path.exists(embedding_model_dir):
            logger.error(f"Local embedding model not found at: {embedding_model_dir}")
            logger.error("Run force_download.py first to download models.")
            raise FileNotFoundError(f"Embedding model not found: {embedding_model_dir}")

        embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model_dir,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )

        # 5. Build DENSE index (LangChain FAISS)
        logger.info("building_dense_index")
        vector_db = LangchainFAISS.from_documents(all_chunks, embeddings)

        os.makedirs(index_dir, exist_ok=True)
        vector_db.save_local(index_dir)

        # 5b. Convert to native FAISS for V3.1 retriever
        logger.info("converting_to_native_faiss")
        _convert_to_native_faiss(vector_db, all_chunks, index_dir)

        # 5c. Sign the native FAISS index
        try:
            verifier = FAISSIndexVerifier()
            native_faiss_path = Path(index_dir) / "dense_index.faiss"
            if native_faiss_path.exists():
                verifier.sign_index(native_faiss_path, metadata={
                    "sources": pdf_files,
                    "chunks": len(all_chunks),
                    "model": embedding_model_dir
                })
                logger.info("faiss_index_signed")
        except Exception as sig_exc:
            logger.error("faiss_signing_error", error=str(sig_exc))

        # 6. Build SPARSE index (BM25)
        logger.info("building_sparse_index")
        _build_sparse_index(all_chunks, index_dir)

        # 7. Save metadata
        metadata_payload = {
            "num_chunks": len(all_chunks),
            "sources": pdf_files,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "dense_index": "dense_index.faiss",
            "sparse_index": "sparse_index.pkl",
            "embedding_model": embedding_model_dir
        }
        with open(os.path.join(index_dir, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(metadata_payload, f, indent=4, ensure_ascii=False)

        # 8. Sanity check
        logger.info("running_sanity_check")
        _sanity_check(index_dir, embeddings)

        elapsed = (time.perf_counter() - start_time) * 1000
        logger.info("build_complete", elapsed_ms=f"{elapsed:.2f}")

        report["status"] = "success"
        report["embedding_model"] = embedding_model_dir
        return report

    except Exception as e:
        logger.critical("build_failed", error=str(e))
        report["error_summary"] = str(e)
        return report


def _convert_to_native_faiss(vector_db, chunks, index_dir):
    """
    Convert LangChain FAISS to native faiss.IndexFlatIP for V3.1 retriever.
    Includes proper doc_id mapping via dense_mapping.json.
    """
    try:
        import faiss

        # Get document store
        docstore = vector_db.docstore
        index_to_docstore_id = vector_db.index_to_docstore_id

        # Extract texts and embeddings
        texts = []
        doc_ids = []

        for idx, doc_id in index_to_docstore_id.items():
            doc = docstore.search(doc_id)
            texts.append(doc.page_content)
            doc_ids.append(f"chunk_{idx}")  # Match sparse index naming

        # Generate embeddings
        embeddings_model = vector_db.embeddings
        embeddings_list = embeddings_model.embed_documents(texts)
        embeddings_np = np.array(embeddings_list).astype('float32')

        # Build native FAISS index
        dimension = embeddings_np.shape[1]
        native_index = faiss.IndexFlatIP(dimension)
        faiss.normalize_L2(embeddings_np)
        native_index.add(embeddings_np)

        # Save
        faiss_path = Path(index_dir) / "dense_index.faiss"
        faiss.write_index(native_index, str(faiss_path))

        # Save mapping with matching doc_ids
        mapping = {
            "doc_ids": doc_ids,
            "texts": texts,
            "dimension": dimension,
            "count": len(doc_ids)
        }
        with open(Path(index_dir) / "dense_mapping.json", "w", encoding="utf-8") as f:
            json.dump(mapping, f, indent=2)

        logger.info("native_faiss_saved", path=str(faiss_path), docs=len(doc_ids))

    except Exception as e:
        logger.error("native_faiss_conversion_failed", error=str(e))
        raise


def _build_sparse_index(chunks, index_dir):
    """
    Build BM25 sparse index from chunks.
    """
    try:
        # Prepare documents
        documents = []
        for i, chunk in enumerate(chunks):
            doc_id = f"chunk_{i}"  # Match dense index naming
            text = chunk.page_content
            source = chunk.metadata.get("source_file", "unknown")
            documents.append((doc_id, text, source))

        # Build BM25
        N = len(documents)
        doc_ids = []
        doc_texts = []
        doc_sources = []
        doc_tokens_list = []
        doc_lengths = []

        for doc_id, text, source in documents:
            doc_ids.append(doc_id)
            doc_texts.append(text)
            doc_sources.append(source)

            # Tokenize
            text_clean = text.lower()
            text_clean = re.sub(r"[^a-z0-9\s]", " ", text_clean)
            tokens = [t for t in text_clean.split() if len(t) > 2]

            doc_tokens_list.append(tokens)
            doc_lengths.append(len(tokens))

        avg_doc_length = sum(doc_lengths) / N if N > 0 else 0

        # Inverted index
        term_doc_freq = defaultdict(lambda: defaultdict(int))
        for doc_idx, tokens in enumerate(doc_tokens_list):
            for token in tokens:
                term_doc_freq[token][doc_idx] += 1

        # IDF
        idf = {}
        for term, doc_freqs in term_doc_freq.items():
            df = len(doc_freqs)
            idf[term] = np.log((N - df + 0.5) / (df + 0.5) + 1.0)

        sparse_index = {
            "doc_ids": doc_ids,
            "doc_texts": doc_texts,
            "doc_sources": doc_sources,
            "term_doc_freq": dict(term_doc_freq),
            "idf": idf,
            "doc_lengths": doc_lengths,
            "avg_doc_length": avg_doc_length,
            "k1": 1.5,
            "b": 0.75,
            "N": N
        }

        # Save
        sparse_path = Path(index_dir) / "sparse_index.pkl"
        with open(sparse_path, "wb") as f:
            pickle.dump(sparse_index, f)

        logger.info("sparse_index_saved", path=str(sparse_path), docs=N, terms=len(idf))

    except Exception as e:
        logger.error("sparse_index_build_failed", error=str(e))
        raise


def _sanity_check(index_dir, embeddings):
    """Verify indices work correctly"""
    try:
        from .faiss_security import verify_and_safely_load_faiss

        faiss_path = Path(index_dir) / "dense_index.faiss"

        if faiss_path.exists():
            index, _ = verify_and_safely_load_faiss(
                str(faiss_path),
                allow_unsigned=True  # During build, may not be signed yet
            )
            logger.info("dense_sanity_passed", ntotal=index.ntotal)

        # Test sparse
        sparse_path = Path(index_dir) / "sparse_index.pkl"
        if sparse_path.exists():
            with open(sparse_path, "rb") as f:
                sidx = pickle.load(f)
            logger.info("sparse_sanity_passed", docs=sidx.get("N", 0))

        # Test LangChain FAISS
        test_query = "drone maximum altitude operation"
        db = LangchainFAISS.load_local(index_dir, embeddings, allow_dangerous_deserialization=False)
        results = db.similarity_search(test_query, k=3)

        if len(results) == 0:
            raise RuntimeError("Sanity check: empty results")

        logger.info("sanity_check_passed", results=len(results))

    except Exception as e:
        logger.error("sanity_check_failed", error=str(e))
        raise

#python -m uav_risk.stage2.rag.build_index

