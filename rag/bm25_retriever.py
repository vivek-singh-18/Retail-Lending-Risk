"""
BM25 Keyword Retriever.

Provides RankBM25 sparse keyword retrieval for financial policy terms
(e.g., DTI, DPD, LTV, delinquency, underwriting, credit score).
"""

import re
from rank_bm25 import BM25Okapi

def tokenize_text(text):
    return re.findall(r"\w+", text.lower())

class BM25Retriever:
    def __init__(self, chunks=None):
        self.chunks = chunks if chunks else []
        self.corpus = [tokenize_text(c["text"]) for c in self.chunks]
        self.bm25 = BM25Okapi(self.corpus) if self.corpus else None
        
    def index_chunks(self, chunks):
        self.chunks = chunks
        self.corpus = [tokenize_text(c["text"]) for c in chunks]
        self.bm25 = BM25Okapi(self.corpus)
        
    def bm25_search(self, query_text, top_k=10, metadata_filter=None):
        if not self.bm25 or not self.chunks:
            return []
            
        tokenized_query = tokenize_text(query_text)
        scores = self.bm25.get_scores(tokenized_query)
        
        indexed_scores = list(enumerate(scores))
        indexed_scores.sort(key=lambda x: x[1], reverse=True)
        
        results = []
        for idx, score in indexed_scores:
            chunk = self.chunks[idx]
            
            # Apply metadata filtering if specified
            if metadata_filter:
                match = True
                for k, v in metadata_filter.items():
                    if chunk["metadata"].get(k) != v:
                        match = False
                        break
                if not match:
                    continue
                    
            results.append({
                "text": chunk["text"],
                "metadata": chunk["metadata"],
                "score": round(float(score), 4)
            })
            if len(results) >= top_k:
                break
                
        return results

_bm25_instance = None

def get_bm25_retriever(chunks=None):
    global _bm25_instance
    if chunks is None and (_bm25_instance is None or not _bm25_instance.chunks):
        from rag.chunking import load_all_policy_chunks
        chunks = load_all_policy_chunks()
        _bm25_instance = BM25Retriever(chunks)
    elif chunks is not None:
        _bm25_instance = BM25Retriever(chunks)
    return _bm25_instance
