"""Shared helpers for the examples: load the bundled demo corpus."""
from datetime import date
from pathlib import Path

from contracts_rag.core.models import Chunk, DocumentType
from contracts_rag.extractors.regex_extractor import RegexStructureExtractor

DATA = Path(__file__).resolve().parent.parent / "data"


def load_corpus():
    ex = RegexStructureExtractor()
    read = lambda rel: (DATA / rel).read_text(encoding="utf-8")
    return {
        "bo_contract": ex.extract(read("synthetic/contrato_bo_sintetico.txt"), "BO", DocumentType.CONTRATO,
                                  "Bolivian purchase contract (synthetic)", "synthetic fixture"),
        "es_contract": ex.extract(read("synthetic/contrato_es_sintetico.txt"), "ES", DocumentType.CONTRATO,
                                  "Spanish purchase contract (synthetic)", "synthetic fixture"),
        "bo_civil_code": ex.extract(read("legal/codigo_civil_bo_extracto.txt"), "BO", DocumentType.LEY,
                                    "Bolivian Civil Code — extract", "Código Civil de Bolivia",
                                    effective_date=date(1976, 4, 2)),
    }


def leaf_units(documents):
    return [u for d in documents for c in d.all_chunks() for u in c.leaf_units()]


def label_of(unit, width=70):
    text = unit.label if isinstance(unit, Chunk) else "  ↳ sub-chunk: " + unit.text
    return " ".join(text.split())[:width]
