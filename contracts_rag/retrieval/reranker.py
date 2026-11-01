"""
Bolivian Contracts RAG — Cross-encoder reranker (ADR Section 5).

Real implementation, same network/model-download constraint as
nli_verifier.py and vector_index.py — not runnable in this sandbox.

Only ever called on the ~20 RRF-fused candidates, never the full corpus:
cross-encoders score a (query, document) pair jointly with full attention,
which is accurate but far more expensive per-pair than a bi-encoder
similarity lookup — the ADR's explicit reason for bounding this step.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from contracts_rag.core.models import Chunk, SubChunk

DEFAULT_MODEL_NAME = "BAAI/bge-reranker-base"


class Reranker(ABC):
    @abstractmethod
    def rerank(
        self, query: str, candidates: list[Chunk | SubChunk], top_k: int
    ) -> list[tuple[Chunk | SubChunk, float]]:
        """Returns (unit, rerank_score) pairs, highest score first."""
        raise NotImplementedError


class CrossEncoderReranker(Reranker):
    def __init__(self, model_name: str = DEFAULT_MODEL_NAME) -> None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise ImportError(
                "CrossEncoderReranker requires sentence-transformers. "
                "Install with: pip install sentence-transformers"
            ) from exc

        self._model = CrossEncoder(model_name)

    def rerank(
        self, query: str, candidates: list[Chunk | SubChunk], top_k: int
    ) -> list[tuple[Chunk | SubChunk, float]]:
        pairs = [
            (query, unit.full_text if isinstance(unit, Chunk) else unit.text) for unit in candidates
        ]
        scores = self._model.predict(pairs)
        ranked = sorted(zip(candidates, scores), key=lambda pair: pair[1], reverse=True)
        return [(unit, float(score)) for unit, score in ranked[:top_k]]
