"""
Bolivian Contracts RAG — CombinedIndexer: the actual Indexer from ADR Section 7.

Composes a VectorIndex and a LexicalIndex behind the single Indexer
interface everything else depends on. Either branch can be a real
implementation or a test double without this class caring.
"""

from __future__ import annotations

from contracts_rag.core.models import Chunk, SubChunk
from contracts_rag.core.retrieval import Indexer
from contracts_rag.indexing.lexical_index import LexicalIndex
from contracts_rag.indexing.vector_index import VectorIndex


class CombinedIndexer(Indexer):
    def __init__(self, vector_index: VectorIndex, lexical_index: LexicalIndex) -> None:
        self.vector_index = vector_index
        self.lexical_index = lexical_index

    def index(self, leaf_units: list[Chunk | SubChunk]) -> None:
        self.vector_index.index(leaf_units)
        self.lexical_index.index(leaf_units)
