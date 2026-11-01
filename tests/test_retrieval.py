from contracts_rag.core.models import Chunk
from contracts_rag.indexing.combined_indexer import CombinedIndexer
from contracts_rag.indexing.lexical_index import BM25LexicalIndex
from contracts_rag.indexing.tfidf_vector_index import TfidfVectorIndex
from contracts_rag.retrieval.hybrid_retriever import RRFHybridRetriever
from contracts_rag.retrieval.overlap_reranker import NaiveOverlapReranker


def _label(unit):
    return unit.label if isinstance(unit, Chunk) else unit.text[:40]


def _build(units):
    indexer = CombinedIndexer(TfidfVectorIndex(), BM25LexicalIndex())
    indexer.index(units)
    return RRFHybridRetriever(indexer.vector_index, indexer.lexical_index, NaiveOverlapReranker())


def test_bm25_finds_the_registry_publicity_article_by_its_vocabulary(all_leaf_units):
    index = BM25LexicalIndex()
    index.index(all_leaf_units)
    hits = index.search("derechos reales sobre inmuebles contra terceros publicidad", top_k=5)
    assert any(_label(u).startswith("Artículo 1538") for u, _ in hits)


def test_hybrid_retriever_returns_ranked_results_with_signal_breakdown(all_leaf_units):
    results = _build(all_leaf_units).retrieve("nulidad del contrato por vicios del consentimiento", top_k_final=5)
    assert len(results) == 5
    assert [r.final_rank for r in results] == [1, 2, 3, 4, 5]
    assert all(r.rrf_score is not None for r in results)
    assert any(r.bm25_rank is not None or r.vector_rank is not None for r in results)
    assert any("Artículo 5" in _label(r.unit) or "Artículo 47" in _label(r.unit) for r in results)


def test_results_contain_no_duplicate_units(all_leaf_units):
    results = _build(all_leaf_units).retrieve("precio de la compraventa", top_k_final=8)
    ids = [getattr(r.unit, "chunk_id", None) or r.unit.subchunk_id for r in results]
    assert len(ids) == len(set(ids))


def test_parent_and_its_subchunks_are_never_both_indexed(all_leaf_units, es_contract):
    parents = {c.chunk_id for c in es_contract.all_chunks() if not c.is_atomic}
    indexed_chunk_ids = {u.chunk_id for u in all_leaf_units if isinstance(u, Chunk)}
    assert parents.isdisjoint(indexed_chunk_ids)
