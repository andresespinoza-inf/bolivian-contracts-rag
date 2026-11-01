from contracts_rag.analysis.party_consistency import check_party_consistency, extract_person_names
from contracts_rag.core.models import DocumentType
from contracts_rag.extractors.regex_extractor import RegexStructureExtractor


def test_flagship_case_parties_clause_vs_signature_block_mismatch(bo_contract):
    report = check_party_consistency(bo_contract)
    assert report.status == "inconsistent"
    assert report.names_in_parties_clause == ["ROBERTO CALLE MAMANI", "SILVIA CALLE MAMANI"]
    assert set(report.names_in_closing) == {"ELIAS FLORES CONDORI", "TERESA ROJAS LIMACHI"}
    assert set(report.only_in_parties_clause) == set(report.names_in_parties_clause)
    assert len(report.evidence_chunk_ids) == 2           # both sources are traceable
    assert "flag, not a finding of invalidity" in report.explanation


def test_consistent_contract_is_reported_consistent():
    text = (
        "PRIMERA (PARTES).-\n1. JUAN PEREZ RIOS, mayor de edad; VENDEDOR.\n"
        "2. ANA SOTO VEGA, mayor de edad; COMPRADORA.\n"
        "DÉCIMA (CONFORMIDAD).- JUAN PEREZ RIOS y ANA SOTO VEGA manifiestan su conformidad.\n"
        "ANA SOTO VEGA\nCOMPRADORA\nJUAN PEREZ RIOS\nVENDEDOR\n"
    )
    doc = RegexStructureExtractor().extract(text, "BO", DocumentType.CONTRATO, "x", "x")
    assert check_party_consistency(doc).status == "consistent"


def test_accents_and_case_do_not_create_false_mismatches():
    assert extract_person_names("JOSÉ MUÑOZ ROJAS") == extract_person_names("JOSE MUÑOZ ROJAS")


def test_role_words_are_not_treated_as_names():
    assert extract_person_names("COMPRADORA\nELIAS FLORES CONDORI\nVENDEDOR") == ["ELIAS FLORES CONDORI"]


def test_other_jurisdictions_are_not_evaluated_not_guessed(es_contract):
    report = check_party_consistency(es_contract)
    assert report.status == "not_evaluable"
    assert "no calibration" in report.explanation
