"""
Bolivian Contracts RAG — NaiveOverlapReranker: a REAL, heuristic stand-in for
CrossEncoderReranker (which needs a downloaded model and can't run here).

Scores each candidate by normalized token overlap with the query — real,
deterministic, generalizes to any query (unlike a hardcoded canned
answer), but has none of a cross-encoder's actual semantic joint
attention over the (query, document) pair. It will reliably separate
"shares several words with the query" from "shares none", which is
enough to validate that HybridRetriever's final re-ranking step narrows
candidates and populates RankedChunk correctly — not enough to claim
production-quality relevance ranking.
"""

from __future__ import annotations

import re

from contracts_rag.core.models import Chunk, SubChunk
from contracts_rag.retrieval.reranker import Reranker

_TOKEN_PATTERN = re.compile(r"\w+", re.UNICODE)

# Minimal Spanish stopword list — just enough that overlap scoring isn't
# dominated by "de", "la", "el", "que", etc. Not linguistically complete.
_STOPWORDS = {
    "de", "la", "el", "en", "y", "a", "que", "los", "las", "un", "una",
    "por", "con", "del", "se", "su", "al", "es", "lo", "como", "para",
}


def _content_tokens(text: str) -> set[str]:
    tokens = _TOKEN_PATTERN.findall(text.lower())
    return {t for t in tokens if t not in _STOPWORDS}


class NaiveOverlapReranker(Reranker):
    def rerank(
        self, query: str, candidates: list[Chunk | SubChunk], top_k: int
    ) -> list[tuple[Chunk | SubChunk, float]]:
        query_tokens = _content_tokens(query)

        scored = []
        for unit in candidates:
            text = unit.full_text if isinstance(unit, Chunk) else unit.text
            doc_tokens = _content_tokens(text)
            overlap = len(query_tokens & doc_tokens)
            score = overlap / max(len(query_tokens), 1)
            scored.append((unit, score))

        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:top_k]
