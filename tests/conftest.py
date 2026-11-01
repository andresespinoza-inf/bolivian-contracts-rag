from datetime import date
from pathlib import Path

import pytest

from contracts_rag.core.models import DocumentType
from contracts_rag.extractors.regex_extractor import RegexStructureExtractor

DATA = Path(__file__).resolve().parent.parent / "data"


def _extract(rel_path, jurisdiction, doc_type, **kwargs):
    text = (DATA / rel_path).read_text(encoding="utf-8")
    return RegexStructureExtractor().extract(
        raw_text=text, jurisdiction=jurisdiction, document_type=doc_type,
        title=rel_path, source="test fixture", **kwargs,
    )


@pytest.fixture(scope="session")
def bo_contract():
    return _extract("synthetic/contrato_bo_sintetico.txt", "BO", DocumentType.CONTRATO)


@pytest.fixture(scope="session")
def es_contract():
    return _extract("synthetic/contrato_es_sintetico.txt", "ES", DocumentType.CONTRATO)


@pytest.fixture(scope="session")
def bo_civil_code():
    return _extract(
        "legal/codigo_civil_bo_extracto.txt", "BO", DocumentType.LEY,
        effective_date=date(1976, 4, 2),
    )


@pytest.fixture(scope="session")
def all_leaf_units(bo_contract, es_contract, bo_civil_code):
    return [u for d in (bo_contract, es_contract, bo_civil_code)
            for c in d.all_chunks() for u in c.leaf_units()]
