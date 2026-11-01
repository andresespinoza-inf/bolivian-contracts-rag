"""
Bolivian Contracts RAG — RRFHybridRetriever: the actual HybridRetriever from ADR Section 5.

Pipeline: top-50 vector + top-50 BM25 -> RRF fusion (k=60) -> top-20 ->
cross-encoder rerank -> top_k_final. Every intermediate number is a
constructor default, not hardcoded in the method body, so it can be
tuned against the demo datasets without touching this class's logic.
"""

from __future__ import annotations

from contracts_rag.core.models import Chunk, SubChunk
from contracts_rag.core.retrieval import HybridRetriever, RankedChunk
from contracts_rag.indexing.lexical_index import LexicalIndex
from contracts_rag.indexing.vector_index import VectorIndex
from contracts_rag.retrieval.reranker import Reranker
from contracts_rag.retrieval.rrf import DEFAULT_RRF_K, reciprocal_rank_fusion


def _unit_id(unit: Chunk | SubChunk) -> str:
    return unit.chunk_id if isinstance(unit, Chunk) else unit.subchunk_id


class RRFHybridRetriever(HybridRetriever):
    def __init__(
        self,
        vector_index: VectorIndex,
        lexical_index: LexicalIndex,
        reranker: Reranker,
        vector_top_k: int = 50,
        bm25_top_k: int = 50,
        rrf_k: int = DEFAULT_RRF_K,
        fusion_top_k: int = 20,
    ) -> None:
        self.vector_index = vector_index
        self.lexical_index = lexical_index
        self.reranker = reranker
        self.vector_top_k = vector_top_k
        self.bm25_top_k = bm25_top_k
        self.rrf_k = rrf_k
        self.fusion_top_k = fusion_top_k

    def retrieve(self, query: str, top_k_final: int = 8) -> list[RankedChunk]:
        vector_hits = self.vector_index.search(query, self.vector_top_k)
        bm25_hits = self.lexical_index.search(query, self.bm25_top_k)

        units_by_id = {_unit_id(u): u for u, _ in vector_hits}
        units_by_id.update({_unit_id(u): u for u, _ in bm25_hits})

        vector_ranked_ids = [_unit_id(u) for u, _ in vector_hits]
        bm25_ranked_ids = [_unit_id(u) for u, _ in bm25_hits]

        fused = reciprocal_rank_fusion([vector_ranked_ids, bm25_ranked_ids], k=self.rrf_k)
        fused_top = fused[: self.fusion_top_k]

        vector_rank_by_id = {uid: i + 1 for i, uid in enumerate(vector_ranked_ids)}
        bm25_rank_by_id = {uid: i + 1 for i, uid in enumerate(bm25_ranked_ids)}

        candidate_units = [units_by_id[uid] for uid, _ in fused_top]
        rrf_score_by_id = dict(fused_top)

        reranked = self.reranker.rerank(query, candidate_units, top_k_final)

        results = []
        for final_rank, (unit, rerank_score) in enumerate(reranked, start=1):
            uid = _unit_id(unit)
            results.append(
                RankedChunk(
                    unit=unit,
                    final_rank=final_rank,
                    final_score=rerank_score,
                    bm25_rank=bm25_rank_by_id.get(uid),
                    vector_rank=vector_rank_by_id.get(uid),
                    rrf_score=rrf_score_by_id.get(uid),
                    rerank_score=rerank_score,
                )
            )
        return results
