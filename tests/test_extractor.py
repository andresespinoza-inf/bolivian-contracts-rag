from contracts_rag.core.models import ChunkType, DocumentType, ExtractionConfidence, SplitReason
from contracts_rag.extractors.regex_extractor import RegexStructureExtractor


def _by_type(doc, chunk_type):
    return [c for c in doc.all_chunks() if c.chunk_type == chunk_type]


# --- Bolivian contract -------------------------------------------------------

def test_bo_contract_has_ten_clauses_and_a_flagged_preamble(bo_contract):
    assert len(_by_type(bo_contract, ChunkType.CLAUSULA)) == 10
    pre = _by_type(bo_contract, ChunkType.FRAGMENTO_SIN_CLASIFICAR)
    assert len(pre) == 1
    assert pre[0].extraction_confidence == ExtractionConfidence.BAJA
    assert bo_contract.sections == []          # section level is optional


def test_preamble_is_kept_because_it_states_the_document_is_private(bo_contract):
    pre = _by_type(bo_contract, ChunkType.FRAGMENTO_SIN_CLASIFICAR)[0]
    assert "documento privado" in pre.full_text


def test_heading_wrapped_across_two_lines_is_still_detected(bo_contract):
    # In the PDF-style text, "SEXTA (OBLIGACIONES" and "DE LAS PARTES).-" are on different lines
    labels = [c.label for c in _by_type(bo_contract, ChunkType.CLAUSULA)]
    assert any(label.startswith("SEXTA (OBLIGACIONES DE LAS PARTES)") for label in labels)


def test_lowercase_cross_reference_does_not_start_a_new_chunk(bo_contract):
    # "...la forma acordada en la cláusula\nquinta del presente contrato" must stay inside SEXTA
    sexta = next(c for c in bo_contract.all_chunks() if c.label.startswith("SEXTA"))
    assert "cláusula quinta" in sexta.references_to
    assert len(_by_type(bo_contract, ChunkType.CLAUSULA)) == 10


# --- Spanish contract (different structure, same engine) ---------------------

def test_es_contract_has_recitals_and_compound_ordinals(es_contract):
    assert len(_by_type(es_contract, ChunkType.ANTECEDENTE)) == 3
    clauses = _by_type(es_contract, ChunkType.CLAUSULA)
    assert len(clauses) == 12
    labels = " | ".join(c.label for c in clauses)
    assert "DÉCIMO PRIMERA" in labels and "DECIMO SEGUNDA" in labels   # accent-inconsistent source


def test_long_clause_is_split_on_its_own_numbered_items(es_contract):
    segunda = next(c for c in es_contract.all_chunks() if c.label.startswith("SEGUNDA"))
    assert not segunda.is_atomic
    reasons = {s.split_reason for s in segunda.sub_chunks}
    assert SplitReason.NUMERAL in reasons
    assert SplitReason.VENTANA_FORZADA not in reasons   # natural markers preferred over windows


def test_subchunks_never_lose_text_regression(es_contract, bo_contract):
    """Regression: text before the first numbered item (the sentence that states the
    price) used to be dropped silently when a clause was split."""
    for doc in (es_contract, bo_contract):
        for chunk in doc.all_chunks():
            if not chunk.is_atomic:
                rebuilt = " ".join(s.text for s in chunk.sub_chunks).split()
                assert rebuilt == chunk.full_text.split(), chunk.label


def test_price_sentence_is_still_reachable_after_splitting(es_contract):
    leaf_text = " ".join(
        getattr(u, "text", None) or u.full_text
        for c in es_contract.all_chunks() for u in c.leaf_units()
    )
    assert "TRESCIENTOS MIL EUROS" in leaf_text


def test_only_leaf_units_are_indexable(es_contract):
    for chunk in es_contract.all_chunks():
        leaves = chunk.leaf_units()
        assert leaves == [chunk] if chunk.is_atomic else leaves == chunk.sub_chunks


# --- Civil code extract ------------------------------------------------------

def test_civil_code_articles_and_sections(bo_civil_code):
    arts = _by_type(bo_civil_code, ChunkType.ARTICULO)
    assert len(arts) == 122
    assert len(bo_civil_code.sections) == 23
    assert all(a.extraction_confidence == ExtractionConfidence.ALTA for a in arts)
    assert not bo_civil_code.vigencia_no_verificada


def test_article_476_is_present_regression(bo_civil_code):
    # It was once missing from the source extract by accident; keep it covered.
    assert any(a.label.startswith("Artículo 476") for a in bo_civil_code.all_chunks())


def test_norm_without_effective_date_is_flagged_not_assumed_in_force():
    doc = RegexStructureExtractor().extract(
        "Artículo 1°.- (Prueba) Texto.", "BO", DocumentType.LEY, "x", "x",
    )
    assert doc.vigencia_no_verificada is True


# --- Fallback rule -----------------------------------------------------------

def test_text_with_no_markers_becomes_one_low_confidence_fragment():
    doc = RegexStructureExtractor().extract(
        "Texto libre sin ninguna cláusula numerada.", "BO", DocumentType.CONTRATO, "x", "x",
    )
    chunks = doc.all_chunks()
    assert len(chunks) == 1
    assert chunks[0].chunk_type == ChunkType.FRAGMENTO_SIN_CLASIFICAR
    assert chunks[0].extraction_confidence == ExtractionConfidence.BAJA
