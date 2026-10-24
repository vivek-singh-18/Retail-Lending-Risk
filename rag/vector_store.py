"""
ChromaDB Vector Store Manager.

Indexes policy document chunks and embeddings into ChromaDB storage.
Provides high-performance vector search with metadata filtering support.
"""

import os
import chromadb
from rag.embeddings import get_policy_embedder

class VectorStoreManager:
    def __init__(self, chroma_dir=None):
        if chroma_dir is None:
            chroma_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "chroma_db")
            
        self.chroma_dir = chroma_dir
        try:
            self.client = chromadb.PersistentClient(path=self.chroma_dir)
        except Exception:
            self.client = chromadb.Client()
            
        self.collection_name = "policy_documents"
        self.collection = self.client.get_or_create_collection(name=self.collection_name)
        self.embedder = get_policy_embedder()
        self.chunks_cache = []
        
    def add_chunks(self, chunks):
        if not chunks:
            return
            
        self.chunks_cache = chunks
        ids = [f"chunk_{i:04d}_{chunk['metadata']['document']}" for i, chunk in enumerate(chunks)]
        texts = [chunk["text"] for chunk in chunks]
        metadatas = [chunk["metadata"] for chunk in chunks]
        
        embeddings = self.embedder.embed_texts(texts)
        
        # Clear existing to avoid duplicate ID errors
        try:
            existing = self.collection.get()
            if existing and existing.get("ids"):
                self.collection.delete(ids=existing["ids"])
        except Exception:
            pass
            
        self.collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas
        )
        print(f"[VECTOR_STORE] Indexed {len(chunks)} document chunks into ChromaDB collection '{self.collection_name}'.")
        
    def dense_search(self, query_text, top_k=10, metadata_filter=None):
        query_emb = self.embedder.embed_texts([query_text])[0]
        
        kwargs = {
            "query_embeddings": [query_emb],
            "n_results": min(top_k, max(self.collection.count(), 1))
        }
        if metadata_filter:
            # Format where filter for ChromaDB
            formatted_filter = {}
            for k, v in metadata_filter.items():
                formatted_filter[k] = {"$eq": v}
            kwargs["where"] = formatted_filter
            
        try:
            results = self.collection.query(**kwargs)
            documents = results.get("documents", [[]])[0]
            metadatas = results.get("metadatas", [[]])[0]
            distances = results.get("distances", [[]])[0]
            
            search_results = []
            for text, meta, dist in zip(documents, metadatas, distances):
                search_results.append({
                    "text": text,
                    "metadata": meta,
                    "score": round(float(1.0 / (1.0 + dist)), 4)
                })
            return search_results
        except Exception as e:
            # Fallback if metadata filter yields no match or syntax error
            results = self.collection.query(query_embeddings=[query_emb], n_results=top_k)
            documents = results.get("documents", [[]])[0]
            metadatas = results.get("metadatas", [[]])[0]
            distances = results.get("distances", [[]])[0]
            
            search_results = []
            for text, meta, dist in zip(documents, metadatas, distances):
                search_results.append({
                    "text": text,
                    "metadata": meta,
                    "score": round(float(1.0 / (1.0 + dist)), 4)
                })
            return search_results

_vstore_instance = None

def get_vector_store():
    global _vstore_instance
    if _vstore_instance is None:
        _vstore_instance = VectorStoreManager()
    return _vstore_instance
