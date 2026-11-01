"""
Bolivian Contracts RAG — Reciprocal Rank Fusion (ADR Section 5).

Fuses ranked lists using only rank position, not raw scores — deliberately
NOT a weighted sum, because BM25 scores (unbounded TF-IDF) and cosine
similarity (bounded [-1, 1]) live on incomparable scales. RRF sidesteps
that entirely.

    RRF_score(d) = sum over lists containing d of 1 / (k + rank_i(d))

Pure function, no model dependency — this is the one piece of the hybrid
retrieval design that needed zero sandbox workarounds.
"""

from __future__ import annotations

from typing import Hashable, TypeVar

T = TypeVar("T", bound=Hashable)

DEFAULT_RRF_K = 60


def reciprocal_rank_fusion(
    ranked_lists: list[list[T]],
    k: int = DEFAULT_RRF_K,
) -> list[tuple[T, float]]:
    """
    Each input list is already sorted best-first (rank 0 = best). Items are
    compared by equality/hash, so callers must pass a stable identifier
    (e.g. chunk_id), not the object itself, unless the object's __eq__/__hash__
    are meaningful.

    Returns (item, rrf_score) pairs sorted by rrf_score descending.
    """
    scores: dict[T, float] = {}
    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank + 1)

    return sorted(scores.items(), key=lambda pair: pair[1], reverse=True)
