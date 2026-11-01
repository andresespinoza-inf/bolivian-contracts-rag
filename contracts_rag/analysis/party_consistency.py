"""
Bolivian Contracts RAG — deterministic party-consistency check (no model, no LLM).

Question it answers: do the people named in the contract's opening "parties"
clause (PRIMERA) match the people named in its closing conformity/signature
clause? In Bolivian practice (and in the project owner's doctrinal framing of
*consentimiento*), the signatures must coincide with the parties identified at
the start; a mismatch is a consent-formation red flag a lawyer should review.

This is a transparent heuristic, not a legal conclusion:
  - It extracts runs of 2–4 UPPERCASE words (the convention used for personal
    names in the Bolivian contracts it was calibrated on), drops role words
    (VENDEDOR, COMPRADORA, ...), and compares the two name sets after
    normalising accents/case.
  - It reports `inconsistent` when names appear on only one side. It NEVER says
    a contract is void or invalid — that is a legal judgement for a lawyer
    (see Civil Code Arts. 473-482 and 549-559 for the legal framing).
  - Known limits: names with lowercase particles ("MARIA DE LA CRUZ") are split;
    contracts that write names in mixed case are not evaluable; it is only
    enabled for jurisdiction "BO" because no other format has been calibrated.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Literal

from contracts_rag.core.models import Chunk, ChunkType, Document

# Spaces/tabs only (not newlines): a name must not be glued to the next line's
# role label, e.g. "COMPRADORA\nELIAS FLORES" in a signature block.
_NAME_RUN = re.compile(r"[A-ZÁÉÍÓÚÑ]{3,}(?:[ \t]+[A-ZÁÉÍÓÚÑ]{2,}){1,3}")

_ROLE_WORDS = {
    "VENDEDOR", "VENDEDORA", "COMPRADOR", "COMPRADORA",
    "ARRENDADOR", "ARRENDADORA", "ARRENDATARIO", "ARRENDATARIA",
    "PRESTAMISTA", "PRESTATARIO", "PRESTATARIA", "DEUDOR", "DEUDORA",
    "ACREEDOR", "ACREEDORA", "PARTES", "CLAUSULA", "CLÁUSULA",
}

_SUPPORTED_JURISDICTIONS = {"BO"}

Status = Literal["consistent", "inconsistent", "not_evaluable"]


def _normalize(name: str) -> str:
    decomposed = unicodedata.normalize("NFD", name)
    stripped = "".join(c for c in decomposed if unicodedata.category(c) != "Mn")
    return re.sub(r"\s+", " ", stripped).strip().upper()


def extract_person_names(text: str) -> list[str]:
    names: list[str] = []
    for match in _NAME_RUN.finditer(text):
        tokens = [t for t in match.group(0).split() if _normalize(t) not in _ROLE_WORDS]
        if len(tokens) >= 2:
            normalized = _normalize(" ".join(tokens))
            if normalized not in names:
                names.append(normalized)
    return names


@dataclass
class PartyConsistencyReport:
    status: Status
    explanation: str
    parties_clause_label: str | None = None
    closing_clause_label: str | None = None
    names_in_parties_clause: list[str] = field(default_factory=list)
    names_in_closing: list[str] = field(default_factory=list)
    only_in_parties_clause: list[str] = field(default_factory=list)
    only_in_closing: list[str] = field(default_factory=list)
    # Chunk ids so a UI can highlight both sources (traceability).
    evidence_chunk_ids: list[str] = field(default_factory=list)


def _short_label(chunk: Chunk) -> str:
    return re.sub(r"\s+", " ", chunk.full_text[:40]).strip()


def check_party_consistency(document: Document) -> PartyConsistencyReport:
    if document.jurisdiction not in _SUPPORTED_JURISDICTIONS:
        return PartyConsistencyReport(
            status="not_evaluable",
            explanation=(
                f"Heuristic calibrated only for jurisdictions {sorted(_SUPPORTED_JURISDICTIONS)}; "
                f"no calibration exists for {document.jurisdiction!r}."
            ),
        )

    clauses = [c for c in document.all_chunks() if c.chunk_type == ChunkType.CLAUSULA]
    parties = next((c for c in clauses if re.match(r"\s*PRIMERA\b", c.full_text)), None)
    closing = clauses[-1] if clauses else None
    if parties is None or closing is None or parties is closing:
        return PartyConsistencyReport(
            status="not_evaluable",
            explanation="Could not locate both a PRIMERA (parties) clause and a distinct closing clause.",
        )

    in_parties = extract_person_names(parties.full_text)
    in_closing = extract_person_names(closing.full_text)
    base = dict(
        parties_clause_label=_short_label(parties),
        closing_clause_label=_short_label(closing),
        names_in_parties_clause=in_parties,
        names_in_closing=in_closing,
        evidence_chunk_ids=[parties.chunk_id, closing.chunk_id],
    )

    if not in_parties or not in_closing:
        return PartyConsistencyReport(
            status="not_evaluable",
            explanation="No uppercase personal names found on one side; format not recognised.",
            **base,
        )

    only_p = [n for n in in_parties if n not in in_closing]
    only_c = [n for n in in_closing if n not in in_parties]
    if only_p or only_c:
        return PartyConsistencyReport(
            status="inconsistent",
            explanation=(
                "Names identified in the parties clause do not match the names in the closing/"
                "signature clause. Requires lawyer review (possible consent-formation issue); "
                "this is a flag, not a finding of invalidity."
            ),
            only_in_parties_clause=only_p,
            only_in_closing=only_c,
            **base,
        )

    return PartyConsistencyReport(
        status="consistent",
        explanation="Every name in the parties clause also appears in the closing clause, and vice versa.",
        **base,
    )
