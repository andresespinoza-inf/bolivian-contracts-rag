"""
Bolivian Contracts RAG — TfidfVectorIndex: a REAL, classical stand-in for the production
semantic index (indexing/vector_index.py, which needs a downloaded
embedding model and can't run in this sandbox).

This is not a mock — it's actual term-frequency vector math (scikit-learn,
no pretrained weights, no network access needed) implementing the exact
same VectorIndex interface. It's weaker than a learned embedding model at
catching paraphrase/synonymy, but it's genuinely running real retrieval
math, not returning canned answers — which is why it's in indexing/
alongside the production implementation rather than in tests/stubs.py.

Swapping this for SentenceTransformerVectorIndex in HybridRetriever's
constructor is the only change needed to move to the production-grade
semantic branch — that's the point of the VectorIndex interface.
"""

from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from contracts_rag.core.models import Chunk, SubChunk
from contracts_rag.indexing.vector_index import VectorIndex


class TfidfVectorIndex(VectorIndex):
    def __init__(self) -> None:
        self._vectorizer = TfidfVectorizer()
        self._units: list[Chunk | SubChunk] = []
        self._matrix = None

    def index(self, leaf_units: list[Chunk | SubChunk]) -> None:
        self._units = list(leaf_units)
        texts = [unit.full_text if isinstance(unit, Chunk) else unit.text for unit in self._units]
        self._matrix = self._vectorizer.fit_transform(texts)

    def search(self, query: str, top_k: int) -> list[tuple[Chunk | SubChunk, float]]:
        if self._matrix is None:
            raise RuntimeError("TfidfVectorIndex.search() called before index()")

        query_vec = self._vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self._matrix)[0]
        ranked = sorted(zip(self._units, scores), key=lambda pair: pair[1], reverse=True)
        return [pair for pair in ranked if pair[1] > 0][:top_k]
