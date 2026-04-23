from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from sentence_transformers import CrossEncoder
from .config import Config

class RAGCore:
    def __init__(self):
        print("⏳ search engine and rules are loading ⌛")
        # 1. تحميل موديل الـ Embedding
        self.embeddings = HuggingFaceEmbeddings(model_name=Config.EMBEDDING_MODEL)
        
        # 2. تحميل قاعدة FAISS من مجلد data
        self.vector_db = FAISS.load_local(
            Config.INDEX_PATH, 
            self.embeddings, 
            allow_dangerous_deserialization=True
        )
        
        # 3. تحميل الـ Re-ranker لزيادة الدقة
        self.reranker = CrossEncoder(Config.RERANKER_MODEL)
        print("Data base loaded successfully✅")

    def retrieve_optimized_context(self, query):
        """ تنفيذ البحث الأولي ثم إعادة الترتيب الذكي """
        # بحث أولي (Similarity Search)
        docs = self.vector_db.similarity_search(query, k=Config.INITIAL_K)
        
        # تحضير البيانات للـ Re-ranker (الترتيب الدقيق)
        pairs = [[query, doc.page_content] for doc in docs]
        scores = self.reranker.predict(pairs)
        
        # ربط السكور بكل قطعة وترتيبهم من الأعلى للأقل
        for i, doc in enumerate(docs):
            doc.metadata['rerank_score'] = scores[i]
            
        sorted_docs = sorted(docs, key=lambda x: x.metadata['rerank_score'], reverse=True)
        return sorted_docs[:Config.FINAL_K]