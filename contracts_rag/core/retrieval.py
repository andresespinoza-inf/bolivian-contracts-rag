"""
Bolivian Contracts RAG — Indexer / HybridRetriever contracts (ADR Section 5 and Section 7).

Only leaf units (Chunk if atomic, SubChunk otherwise) are ever indexed or
retrieved — never a parent Chunk alongside its own SubChunks (ADR 2.4).
Callers should pass chunk.leaf_units() output, not raw Chunk lists.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from contracts_rag.core.models import Chunk, SubChunk

LeafUnit = "Chunk | SubChunk"


@dataclass
class RankedChunk:
    """
    A retrieval result. Carries the intermediate per-signal ranks/scores
    (not just the final one) deliberately — a RAG claiming "citation
    grounding" as a differentiator should be able to show, for any result,
    which retrieval signal actually surfaced it. Purely diagnostic; nothing
    downstream is required to read these fields.
    """

    unit: Chunk | SubChunk
    final_rank: int
    final_score: float
    bm25_rank: int | None = None
    vector_rank: int | None = None
    rrf_score: float | None = None
    rerank_score: float | None = None


class Indexer(ABC):
    @abstractmethod
    def index(self, leaf_units: list[Chunk | SubChunk]) -> None:
        """Populate whatever indices this implementation maintains (vector, lexical, or both)."""
        raise NotImplementedError


class HybridRetriever(ABC):
    @abstractmethod
    def retrieve(self, query: str, top_k_final: int = 8) -> list[RankedChunk]:
        """
        ADR Section 5 pipeline: top-50 vector + top-50 BM25 -> RRF fusion
        (k=60) -> top-20 -> cross-encoder re-rank -> top_k_final. The
        intermediate numbers are configuration on the concrete
        implementation, not part of this method's signature.
        """
        raise NotImplementedError
