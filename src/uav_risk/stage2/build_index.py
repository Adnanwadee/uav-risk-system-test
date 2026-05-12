#!/usr/bin/env python3
"""
Index Builder for UAV RAG System - Enhanced
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


def build_index():
    BASE_DIR = Path(__file__).parent
    DOCS_DIR = BASE_DIR / "docs"
    INDEX_DIR = BASE_DIR / "knowledge" / "vector_db"
    MODELS_DIR = BASE_DIR / "knowledge" / "models" / "embedding"
    
    print("=" * 60)
    print("Building RAG Knowledge Base - Enhanced")
    print("=" * 60)
    
    if not DOCS_DIR.exists():
        print(f"docs folder not found: {DOCS_DIR}")
        return
    
    pdf_files = list(DOCS_DIR.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDF files found in {DOCS_DIR}")
        return
    
    print(f"\nFound {len(pdf_files)} PDF files")
    
    all_documents = []
    for pdf_file in pdf_files:
        print(f"   Processing: {pdf_file.name}")
        loader = PyPDFLoader(str(pdf_file))
        documents = loader.load()
        
        for doc in documents:
            if "FAA" in pdf_file.name or "AC_107" in pdf_file.name:
                doc.metadata["source"] = "FAA AC 107-2A"
                doc.metadata["regulation"] = "14 CFR Part 107"
            elif "EASA" in pdf_file.name or "EAR" in pdf_file.name:
                doc.metadata["source"] = "EASA Easy Access Rules"
                doc.metadata["regulation"] = "EU 2019/947 & EU 2019/945"
            else:
                doc.metadata["source"] = pdf_file.stem
        
        all_documents.extend(documents)
    
    print(f"\nTotal pages loaded: {len(all_documents)}")
    
    # chunking محسن
    text_splitter = RecursiveCharacterTextSplitter(
        # غيّر هذين السطرين
        chunk_size=1200,      # بدلاً من 800
        chunk_overlap=300,    # بدلاً من 150
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    
    chunks = text_splitter.split_documents(all_documents)
    print(f"Created {len(chunks)} chunks")
    
    print(f"\nLoading embedding model from: {MODELS_DIR}")
    embeddings = HuggingFaceEmbeddings(
        model_name=str(MODELS_DIR),
        model_kwargs={'device': 'cpu', 'local_files_only': True}
    )
    
    print("Building FAISS index...")
    vector_store = FAISS.from_documents(chunks, embeddings)
    
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    vector_store.save_local(str(INDEX_DIR))
    
    print("\n" + "=" * 60)
    print("INDEX BUILT SUCCESSFULLY")
    print("=" * 60)
    print(f"Path: {INDEX_DIR}")
    print(f"Files: {len(pdf_files)}")
    print(f"Pages: {len(all_documents)}")
    print(f"Chunks: {len(chunks)}")
    print("=" * 60)


if __name__ == "__main__":
    build_index()