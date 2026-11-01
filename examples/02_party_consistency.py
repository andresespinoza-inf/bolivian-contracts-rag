"""Deterministic consent-formation check: parties named at the start vs. at the signature."""
from _common import load_corpus

from contracts_rag.analysis.party_consistency import check_party_consistency

report = check_party_consistency(load_corpus()["bo_contract"])
print(f"status: {report.status.upper()}")
print(f"parties clause : {report.names_in_parties_clause}")
print(f"closing clause : {report.names_in_closing}")
print(f"only in parties clause: {report.only_in_parties_clause}")
print(f"only in closing clause: {report.only_in_closing}")
print(f"\n{report.explanation}")
print(f"evidence chunk ids (for highlighting): {report.evidence_chunk_ids}")
