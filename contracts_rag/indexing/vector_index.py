"""
Bolivian Contracts RAG — Vector (semantic) index.

Real implementation, meant to run in an environment with unrestricted
internet access (downloads embedding weights from Hugging Face on first
use — same constraint as verifiers/nli_verifier.py). Not runnable inside
this sandbox; see tests/stubs.py for the test double used here instead.

Install:
    pip install sentence-transformers

Model choice per the ADR stack decision: voyage-law-2 or text-embedding-3-large
are commercial APIs; BAAI/bge-m3 is the named open-source alternative, used
as the default here so this runs with no API key required.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from contracts_rag.core.models import Chunk, SubChunk

DEFAULT_MODEL_NAME = "BAAI/bge-m3"


class VectorIndex(ABC):
    @abstractmethod
    def index(self, leaf_units: list[Chunk | SubChunk]) -> None:
        raise NotImplementedError

    @abstractmethod
    def search(self, query: str, top_k: int) -> list[tuple[Chunk | SubChunk, float]]:
        """Returns (unit, cosine_similarity) pairs, highest score first."""
        raise NotImplementedError


class SentenceTransformerVectorIndex(VectorIndex):
    def __init__(self, model_name: str = DEFAULT_MODEL_NAME) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "SentenceTransformerVectorIndex requires sentence-transformers. "
                "Install with: pip install sentence-transformers"
            ) from exc

        self._model = SentenceTransformer(model_name)
        self._units: list[Chunk | SubChunk] = []
        self._embeddings = None  # numpy array, shape (n_units, dim)

    def index(self, leaf_units: list[Chunk | SubChunk]) -> None:
        import numpy as np

        self._units = list(leaf_units)
        texts = [unit.full_text if isinstance(unit, Chunk) else unit.text for unit in self._units]
        self._embeddings = np.asarray(
            self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        )

    def search(self, query: str, top_k: int) -> list[tuple[Chunk | SubChunk, float]]:
        import numpy as np

        if self._embeddings is None:
            raise RuntimeError("SentenceTransformerVectorIndex.search() called before index()")

        query_vec = np.asarray(
            self._model.encode([query], normalize_embeddings=True, show_progress_bar=False)
        )[0]
        # Embeddings are normalized, so dot product == cosine similarity.
        scores = self._embeddings @ query_vec
        ranked_idx = np.argsort(-scores)[:top_k]
        return [(self._units[i], float(scores[i])) for i in ranked_idx]
