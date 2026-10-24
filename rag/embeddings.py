"""
Embeddings Layer with Graceful Fallback.

Provides high-speed vector embeddings using SentenceTransformers (all-MiniLM-L6-v2)
or lightweight TF-IDF dense projection fallback to ensure instant, zero-latency search.
"""

import os
import numpy as np

class PolicyEmbedder:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.st_model = None
        self.tfidf_vectorizer = None
        
        try:
            from sentence_transformers import SentenceTransformer
            print(f"Loading SentenceTransformer embedding model '{model_name}'...")
            self.st_model = SentenceTransformer(model_name)
            print("[SUCCESS] SentenceTransformer model ready!")
        except Exception as e:
            print(f"[WARN] SentenceTransformer initialization note: {e}. Utilizing lightweight TF-IDF dense projection.")
            from sklearn.feature_extraction.text import TfidfVectorizer
            self.tfidf_vectorizer = TfidfVectorizer(max_features=384)
            # Fit on standard policy terms
            standard_terms = [
                "borrower debt to income dti credit score delinquency loan amount income",
                "underwriting guidelines manual review approval decline tenure interest rate",
                "personal loan housing loan car loan property valuation legal document",
                "days past due dpd recent delinquency active loan credit utilization"
            ]
            self.tfidf_vectorizer.fit(standard_terms)
            
    def embed_texts(self, texts):
        if isinstance(texts, str):
            texts = [texts]
            
        if self.st_model is not None:
            embeddings = self.st_model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
            return embeddings.tolist()
        else:
            if not hasattr(self.tfidf_vectorizer, "vocabulary_"):
                from sklearn.feature_extraction.text import TfidfVectorizer
                self.tfidf_vectorizer = TfidfVectorizer(max_features=384)
                self.tfidf_vectorizer.fit(texts)
            else:
                # Update fit if new vocabulary seen
                try:
                    mat = self.tfidf_vectorizer.transform(texts).toarray()
                except Exception:
                    self.tfidf_vectorizer.fit(texts)
                    mat = self.tfidf_vectorizer.transform(texts).toarray()
            mat = self.tfidf_vectorizer.transform(texts).toarray()
            # Pad or truncate to 384 dimensions
            if mat.shape[1] < 384:
                pad = np.zeros((mat.shape[0], 384 - mat.shape[1]))
                mat = np.hstack([mat, pad])
            return mat.tolist()

_embedder_instance = None

def get_policy_embedder():
    global _embedder_instance
    if _embedder_instance is None:
        _embedder_instance = PolicyEmbedder()
    return _embedder_instance
