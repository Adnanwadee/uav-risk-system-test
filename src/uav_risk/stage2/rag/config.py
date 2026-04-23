import os

class Config:
    # تحديد المسار ديناميكياً ليتجه مباشرة إلى src/uav_risk/stage2/knowledge/vector_db
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    INDEX_PATH = os.path.join(BASE_DIR, "knowledge", "vector_db")
    
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    INITIAL_K = 10
    FINAL_K = 3