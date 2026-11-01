"""
Tier-1 claim verification with a REAL NLI model (needs `pip install .[models]`; downloads ~0.7 GB).
No LLM is used: ambiguous cases are reported as 'not verified (tier2_unavailable)'.
"""
from _common import load_corpus

from contracts_rag.verifiers.nli_verifier import NLIEntailmentVerifier
from contracts_rag.verifiers.two_tier_verifier import TwoTierEntailmentVerifier

docs = load_corpus()
find = lambda doc, prefix: next(c for c in doc.all_chunks() if c.label.startswith(prefix))
cases = [
    ("Price stated in the contract",
     "El precio de la compraventa es de Bs 46.000.", find(docs["bo_contract"], "QUINTA")),
    ("Over-claim: contract is public and registered",
     "El contrato consta en documento público e inscrito en Derechos Reales.",
     find(docs["bo_contract"], "CONTRATO DE COMPRAVENTA")),
    ("Registry publicity rule (Art. 1538)",
     "Aunque el contrato no conste en documento público, es válido entre las partes, "
     "pero no será oponible a terceros hasta su inscripción en el registro.",
     find(docs["bo_civil_code"], "Artículo 1538")),
]

verifier = TwoTierEntailmentVerifier(tier1_verifier=NLIEntailmentVerifier())   # no Tier 2 → no LLM needed
for title, claim, chunk in cases:
    r = verifier.verify(claim, chunk)
    note = f"  [{r.inconclusive_reason}]" if r.inconclusive_reason else ""
    print(f"{title}\n   claim: {claim}\n   → {r.label.value.upper()} (confidence {r.confidence:.2f}){note}\n")
