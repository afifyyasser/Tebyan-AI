import torch
import os
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
import numpy as np

class RAGService:
    def __init__(self):
        # تم ضبط الجهاز على CPU إجبارياً لتفادي أخطاء الـ Kernel مع كروت RTX 5060 حالياً
        # معالج Intel Ultra 7 سيعطي أداءً ممتازاً جداً في هذه المهمة
        self.device = "cpu"
        print(f"⚙️ RAG Service is firing up on: {self.device} (Stability Mode)")
        
        # 1. تحميل موديل الـ Embedding (BGE-M3) بدقة 1024
        # هذا الموديل هو الأفضل حالياً في فهم السياق الشرعي واللغة العربية
        print("📥 Loading BGE-M3 Semantic Model...")
        self.model = SentenceTransformer('BAAI/bge-m3', device=self.device)
        
        # 2. تحميل موديل الـ Re-ranker لفلترة النتائج المستخرجة
        print("📥 Loading Cross-Encoder Re-ranker...")
        self.ranker = CrossEncoder('BAAI/bge-reranker-v2-m3', device=self.device)

        self.bm25 = None
        self.documents = []

    def embed_text(self, text):
        """تحويل السؤال إلى فيكتور لفهمه دلالياً"""
        # الـ CPU سيعالج هذه العملية بدقة متناهية دون الحاجة لـ CUDA Kernels
        return self.model.encode([text], normalize_embeddings=True)[0].tolist()

    def setup_bm25(self, documents):
        """تجهيز محرك البحث التقليدي بالكلمات المفتاحية"""
        if not documents:
            print("⚠️ Warning: No documents found to index.")
            return
        self.documents = documents
        tokenized_corpus = [doc.split(" ") for doc in documents]
        self.bm25 = BM25Okapi(tokenized_corpus)
        print(f"✅ BM25 indexed {len(documents)} documents.")

    def hybrid_search_with_rrf(self, query, collection, top_k=50):
        """البحث الهجين: دمج البحث الدلالي مع البحث بالكلمات"""
        # أ. البحث الدلالي (عبر ChromaDB)
        query_vec = self.embed_text(query)
        results = collection.query(
            query_embeddings=[query_vec], 
            n_results=top_k
        )
        semantic_results = results['documents'][0] if results['documents'] else []
        
        # ب. البحث بالكلمات المفتاحية (BM25)
        keyword_results = self.bm25.get_top_n(query.split(" "), self.documents, n=top_k) if self.bm25 else []
        
        # ج. دمج النتائج بخوارزمية RRF لضمان ظهور النص الأدق أولاً
        scores = {}
        for rank, doc in enumerate(semantic_results):
            scores[doc] = scores.get(doc, 0) + 1 / (60 + rank)
        for rank, doc in enumerate(keyword_results):
            scores[doc] = scores.get(doc, 0) + 1 / (60 + rank)
        
        # ترتيب النتائج النهائية بناءً على الدمج
        fused_docs = [doc for doc, score in sorted(scores.items(), key=lambda x: x[1], reverse=True)]
        return fused_docs

    def rerank(self, query, docs):
        """إعادة الترتيب (Re-ranking) لضمان أن الإجابة هي الأدق شرعياً"""
        if not docs:
            return []
        
        # نأخذ أفضل 30 نتيجة ونعيد فحصهم بدقة بالميكروسكوب
        top_n = docs[:30]
        pairs = [[query, doc] for doc in top_n]
        
        # حساب سكور التشابه لكل نص مع السؤال
        scores = self.ranker.predict(pairs)
        
        scored_docs = sorted(zip(top_n, scores), key=lambda x: x[1], reverse=True)
        return [doc for doc, score in scored_docs]

# إنشاء نسخة الخدمة لاستدعائها في main.py
rag_handler = RAGService()