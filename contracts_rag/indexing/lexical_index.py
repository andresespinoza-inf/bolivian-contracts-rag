"""
Bolivian Contracts RAG — Lexical (BM25) index.

This is one of the two branches HybridRetriever fuses (ADR Section 5),
split into its own small interface rather than buried inside a monolithic
retriever — same "one responsibility per class" principle as the rest of
the project.

Unlike the vector index and the cross-encoder reranker, this one needs no
pretrained model weights — rank_bm25 is pure Python/NumPy term statistics,
so it's the one piece of the retrieval layer that's fully real and testable
inside this sandbox.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod

from rank_bm25 import BM25Okapi

from contracts_rag.core.models import Chunk, SubChunk

_TOKEN_PATTERN = re.compile(r"\w+", re.UNICODE)


def _tokenize(text: str) -> list[str]:
    # Deliberately simple: lowercase + word-boundary split, no stemming.
    # Spanish legal text would benefit from stemming/lemmatization (e.g.
    # "cláusula"/"cláusulas" currently count as different terms) — flagged
    # here as a known, documented limitation rather than silently accepted,
    # consistent with the project's "never hide known gaps" rule.
    return _TOKEN_PATTERN.findall(text.lower())


class LexicalIndex(ABC):
    @abstractmethod
    def index(self, leaf_units: list[Chunk | SubChunk]) -> None:
        raise NotImplementedError

    @abstractmethod
    def search(self, query: str, top_k: int) -> list[tuple[Chunk | SubChunk, float]]:
        """Returns (unit, bm25_score) pairs, highest score first."""
        raise NotImplementedError


class BM25LexicalIndex(LexicalIndex):
    def __init__(self) -> None:
        self._units: list[Chunk | SubChunk] = []
        self._bm25: BM25Okapi | None = None

    def index(self, leaf_units: list[Chunk | SubChunk]) -> None:
        self._units = list(leaf_units)
        corpus_text = [
            unit.full_text if isinstance(unit, Chunk) else unit.text for unit in self._units
        ]
        tokenized_corpus = [_tokenize(text) for text in corpus_text]
        self._bm25 = BM25Okapi(tokenized_corpus)

    def search(self, query: str, top_k: int) -> list[tuple[Chunk | SubChunk, float]]:
        if self._bm25 is None:
            raise RuntimeError("BM25LexicalIndex.search() called before index()")

        scores = self._bm25.get_scores(_tokenize(query))
        ranked = sorted(zip(self._units, scores), key=lambda pair: pair[1], reverse=True)
        return [pair for pair in ranked if pair[1] > 0][:top_k]
