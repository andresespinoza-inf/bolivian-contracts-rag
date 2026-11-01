"""
Bolivian Contracts RAG — StructureExtractor contract (Section 3 of the ADR).

This is the single point of extension for "how do we get structure out of
raw text". RegexStructureExtractor (extractors/regex_extractor.py) is the
first and only implementation for v1. A future StructuredSourceAdapter
(for a pre-parsed JSON/XML legal source, if one ever becomes reliably
available across jurisdictions) would implement this same interface and
slot in without any change to the layers above it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from contracts_rag.core.models import Document, DocumentType


class StructureExtractor(ABC):
    @abstractmethod
    def extract(
        self,
        raw_text: str,
        jurisdiction: str,
        document_type: DocumentType,
        title: str,
        source: str,
        effective_date=None,
    ) -> Document:
        """
        Turn raw text into a fully populated Document -> Section? -> Chunk -> SubChunk?
        tree.

        Guarantees:
          - Every Chunk has chunk_type, label, order, and extraction_confidence set.
          - No embeddings are computed here (that's the Indexer's job).

        Does NOT guarantee perfect extraction. Callers must be prepared to see
        ChunkType.FRAGMENTO_SIN_CLASIFICAR chunks with ExtractionConfidence.BAJA
        when no recognizable marker is found — this is the deliberate fallback
        rule from the ADR, not a bug.
        """
        raise NotImplementedError
