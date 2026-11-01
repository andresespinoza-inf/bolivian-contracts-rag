"""
Bolivian Contracts RAG — Metadata schema (Section 2 of the Architecture Decision Record).

These classes are intentionally "dumb" data containers: they carry no
extraction logic, no embedding logic, and no retrieval logic. Each layer
downstream (StructureExtractor, Indexer, HybridRetriever, etc.) depends
on this shape but never on how an instance was produced.

Design rules encoded here (see docs/architecture-decisions.md for the why):
  - Section is optional: a Chunk may attach directly to a Document.
  - SubChunk exists only when a Chunk exceeds the split threshold.
  - Only leaf units (atomic Chunk, or its SubChunks) are ever indexed.
  - extraction_confidence and vigencia_no_verificada are first-class,
    not afterthoughts: the system must be able to say "I'm not sure"
    about its own structural extraction and about legal validity.
"""

from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from enum import Enum


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class DocumentType(str, Enum):
    CONTRATO = "contrato"
    LEY = "ley"
    SENTENCIA = "sentencia"
    NORMA_REGLAMENTARIA = "norma_reglamentaria"


class ChunkType(str, Enum):
    ARTICULO = "articulo"
    CLAUSULA = "clausula"
    ANTECEDENTE = "antecedente"          # a.k.a. "expositivo"
    CONSIDERANDO = "considerando"
    POR_TANTO = "por_tanto"
    HECHO_PROBADO = "hecho_probado"
    FRAGMENTO_SIN_CLASIFICAR = "fragmento_sin_clasificar"


class ExtractionConfidence(str, Enum):
    ALTA = "alta"
    BAJA = "baja"


class SplitReason(str, Enum):
    INCISO = "inciso"
    NUMERAL = "numeral"
    PARRAFO = "parrafo"
    VENTANA_FORZADA = "ventana_forzada"


# ---------------------------------------------------------------------------
# Document
# ---------------------------------------------------------------------------

@dataclass
class Document:
    jurisdiction: str  # e.g. "BO", "ES", "MX"
    document_type: DocumentType
    title: str
    source: str
    raw_text: str = field(repr=False)
    language: str = "es"
    effective_date: date | None = None
    upload_date: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    document_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    # Populated by StructureExtractor.extract(); not set at construction time.
    sections: list["Section"] = field(default_factory=list)
    # Chunks with no Section parent (the common case for simple contracts).
    chunks: list["Chunk"] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.raw_text_hash = hashlib.sha256(self.raw_text.encode("utf-8")).hexdigest()

        requires_effective_date = self.document_type in (
            DocumentType.LEY,
            DocumentType.NORMA_REGLAMENTARIA,
        )
        # Rule from the ADR: never assume a norm is in force by default.
        self.vigencia_no_verificada = requires_effective_date and self.effective_date is None

    def all_chunks(self) -> list["Chunk"]:
        """Every Chunk in the document, whether attached to a Section or not."""
        result = list(self.chunks)
        for section in self.sections:
            result.extend(section.chunks)
        return result


# ---------------------------------------------------------------------------
# Section (optional intermediate level — chapter/title)
# ---------------------------------------------------------------------------

@dataclass
class Section:
    document_id: str
    label: str
    order: int
    section_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    chunks: list["Chunk"] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Chunk (atomic unit of legal meaning — article / clause / recital / ...)
# ---------------------------------------------------------------------------

@dataclass
class Chunk:
    document_id: str
    chunk_type: ChunkType
    label: str
    order: int
    full_text: str
    extraction_confidence: ExtractionConfidence
    section_id: str | None = None
    chunk_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    references_to: list[str] = field(default_factory=list)
    embedding_id: str | None = None
    sub_chunks: list["SubChunk"] = field(default_factory=list)

    @property
    def token_count(self) -> int:
        # Placeholder heuristic until a real tokenizer is wired in (indexing layer).
        # Kept as a property (not a stored field) so it's always in sync with full_text.
        return len(self.full_text.split())

    @property
    def is_atomic(self) -> bool:
        return len(self.sub_chunks) == 0

    def leaf_units(self) -> list["Chunk | SubChunk"]:
        """
        The unit(s) that should actually be indexed for retrieval.
        Enforces the ADR rule: never index a parent Chunk alongside its
        own SubChunks — only leaves.
        """
        return [self] if self.is_atomic else list(self.sub_chunks)


# ---------------------------------------------------------------------------
# SubChunk (only when a Chunk exceeds the split threshold)
# ---------------------------------------------------------------------------

@dataclass
class SubChunk:
    parent_chunk_id: str
    position_in_parent: int
    text: str
    split_reason: SplitReason
    subchunk_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    embedding_id: str | None = None

    @property
    def token_count(self) -> int:
        return len(self.text.split())
