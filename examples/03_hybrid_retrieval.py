"""
Hybrid retrieval (BM25 + vector → RRF → rerank) over the whole demo corpus.

    python examples/03_hybrid_retrieval.py                 # classical stack, no downloads
    python examples/03_hybrid_retrieval.py --real-models   # bge-m3 + bge-reranker (needs `pip install .[models]`)
"""
import sys
import time

from _common import label_of, leaf_units, load_corpus

from contracts_rag.indexing.combined_indexer import CombinedIndexer
from contracts_rag.indexing.lexical_index import BM25LexicalIndex
from contracts_rag.retrieval.hybrid_retriever import RRFHybridRetriever

QUERIES = [
    "¿Qué pasa si el contrato de compraventa no consta en documento público?",
    "¿Cuál es el precio de la compraventa y cómo se paga?",
    "nulidad del contrato por vicios del consentimiento",
]

units = leaf_units(load_corpus().values())
lexical = BM25LexicalIndex()
if "--real-models" in sys.argv:
    from contracts_rag.indexing.vector_index import SentenceTransformerVectorIndex
    from contracts_rag.retrieval.reranker import CrossEncoderReranker
    vector, reranker, mode = SentenceTransformerVectorIndex(), CrossEncoderReranker(), "bge-m3 + bge-reranker-base"
else:
    from contracts_rag.indexing.tfidf_vector_index import TfidfVectorIndex
    from contracts_rag.retrieval.overlap_reranker import NaiveOverlapReranker
    vector, reranker, mode = TfidfVectorIndex(), NaiveOverlapReranker(), "TF-IDF + token-overlap (classical stand-ins)"

t0 = time.perf_counter()
CombinedIndexer(vector, lexical).index(units)
retriever = RRFHybridRetriever(vector, lexical, reranker)
print(f"Indexed {len(units)} leaf units in {time.perf_counter() - t0:.2f}s — stack: {mode}")

for query in QUERIES:
    print(f"\nQ: {query}")
    for r in retriever.retrieve(query, top_k_final=4):
        print(f"  #{r.final_rank}  bm25={str(r.bm25_rank):>4} vec={str(r.vector_rank):>4}  {label_of(r.unit, 62)}")
