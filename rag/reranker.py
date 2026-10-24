"""
Reranking Module.

Reranks candidate retrieved chunks using cosine similarity / cross-encoder scores
to refine relevance precision before presenting policy context to the decision engine & generator.
"""

import numpy as np
from rag.embeddings import get_policy_embedder

class PolicyReranker:
    def __init__(self):
        self.embedder = get_policy_embedder()
        
    def rerank(self, query_text, candidate_chunks, top_k=5):
        if not candidate_chunks:
            return []
            
        query_emb = np.array(self.embedder.embed_texts([query_text])[0])
        chunk_texts = [c["text"] for c in candidate_chunks]
        chunk_embs = np.array(self.embedder.embed_texts(chunk_texts))
        
        # Calculate Cosine Similarities
        norm_query = query_emb / (np.linalg.norm(query_emb) + 1e-9)
        norm_chunks = chunk_embs / (np.linalg.norm(chunk_embs, axis=1, keepdims=True) + 1e-9)
        
        similarities = np.dot(norm_chunks, norm_query)
        
        reranked = []
        for idx, sim in enumerate(similarities):
            item = candidate_chunks[idx].copy()
            item["rerank_score"] = round(float(sim), 4)
            reranked.append(item)
            
        reranked.sort(key=lambda x: x["rerank_score"], reverse=True)
        return reranked[:top_k]

def get_policy_reranker():
    return PolicyReranker()
