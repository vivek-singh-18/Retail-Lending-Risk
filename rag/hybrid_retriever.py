"""
Hybrid Retrieval Module using Reciprocal Rank Fusion (RRF).

Combines Dense Vector Search (ChromaDB) + Sparse Keyword Search (BM25)
to optimize recall and precision across policy terms and semantic contexts.
"""

from rag.vector_store import get_vector_store
from rag.bm25_retriever import get_bm25_retriever

class HybridRetriever:
    def __init__(self, vector_store=None, bm25_retriever=None):
        self.vector_store = vector_store if vector_store else get_vector_store()
        self.bm25_retriever = bm25_retriever if bm25_retriever else get_bm25_retriever()
        
    def retrieve_hybrid(self, query_text, top_k=10, k_rrf=60, metadata_filter=None):
        """
        Executes hybrid retrieval fusing Dense + BM25 results via RRF.
        """
        dense_results = self.vector_store.dense_search(query_text, top_k=top_k*2, metadata_filter=metadata_filter)
        bm25_results = self.bm25_retriever.bm25_search(query_text, top_k=top_k*2, metadata_filter=metadata_filter)
        
        rrf_scores = {}
        doc_map = {}
        
        # Rank Dense Results
        for rank, res in enumerate(dense_results):
            doc_id = res["text"]
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (k_rrf + (rank + 1)))
            doc_map[doc_id] = res
            
        # Rank BM25 Results
        for rank, res in enumerate(bm25_results):
            doc_id = res["text"]
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (k_rrf + (rank + 1)))
            doc_map[doc_id] = res
            
        sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        
        fused_results = []
        for doc_id, rrf_score in sorted_docs[:top_k]:
            item = doc_map[doc_id].copy()
            item["rrf_score"] = round(float(rrf_score), 5)
            fused_results.append(item)
            
        return fused_results

def get_hybrid_retriever():
    return HybridRetriever()
