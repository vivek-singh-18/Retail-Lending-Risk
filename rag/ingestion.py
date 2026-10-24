"""
Policy Ingestion Orchestrator.

Reads PDF policy documents from `knowledge_base/`, parses semantic chunks,
indexes vectors in ChromaDB, and builds BM25 keyword index.
"""

import os
from rag.chunking import load_all_policy_chunks
from rag.vector_store import get_vector_store
from rag.bm25_retriever import get_bm25_retriever

def run_ingestion():
    kb_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "knowledge_base")
    print("=" * 60)
    print("STARTING RAG POLICY DOCUMENT INGESTION")
    print("=" * 60)
    
    chunks = load_all_policy_chunks(kb_dir)
    
    # 1. Index in ChromaDB
    vstore = get_vector_store()
    vstore.add_chunks(chunks)
    
    # 2. Index in BM25
    bm25 = get_bm25_retriever(chunks)
    
    print(f"[SUCCESS] Policy Ingestion Pipeline Completed! {len(chunks)} chunks indexed.")
    return len(chunks)

if __name__ == "__main__":
    run_ingestion()
