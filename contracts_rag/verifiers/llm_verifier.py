"""
Bolivian Contracts RAG — Tier 2 verifier: constrained LLM fallback (ADR Section 4).

Only invoked by TwoTierEntailmentVerifier when Tier 1 (NLI) returns
`neutral` or low confidence — this is deliberately the expensive,
rarely-used path.

The LLM client is injected as a plain callable (`LLMClient` protocol)
rather than hardcoding a specific SDK here. That keeps this module
testable without real API credentials (see tests/stubs.py) and makes
swapping providers (Claude, GPT-4, etc.) a one-line change at the call
site, not a rewrite of this class.
"""

from __future__ import annotations

import re
from typing import Protocol

from contracts_rag.core.models import Chunk, SubChunk
from contracts_rag.core.verification import (
    EntailmentVerifier,
    VerificationLabel,
    VerificationMethod,
    VerificationResult,
)

_PROMPT_TEMPLATE = """\
Fragmento fuente:
\"\"\"
{source_text}
\"\"\"

Afirmación a verificar:
\"\"\"
{claim}
\"\"\"

¿Cuál es la relación entre el fragmento fuente y la afirmación?
Responde en exactamente dos líneas, sin texto adicional:
RESPUESTA: RESPALDA, CONTRADICE o NO_APLICA
CITA: <el fragmento exacto del texto fuente que respalda o refuta la afirmación, \
o "ninguno" si RESPUESTA es NO_APLICA>
"""


class LLMClient(Protocol):
    """Anything that turns a prompt string into a completion string."""

    def __call__(self, prompt: str) -> str: ...


class LLMEntailmentVerifier(EntailmentVerifier):
    def __init__(self, llm_client: LLMClient) -> None:
        self._llm_client = llm_client

    def verify(self, claim: str, source_chunk: Chunk | SubChunk) -> VerificationResult:
        source_text = source_chunk.full_text if isinstance(source_chunk, Chunk) else source_chunk.text

        prompt = _PROMPT_TEMPLATE.format(source_text=source_text, claim=claim)
        raw_response = self._llm_client(prompt)

        return self._parse_response(raw_response)

    _LABEL_MAP = {
        "RESPALDA": VerificationLabel.ENTAILMENT,
        "CONTRADICE": VerificationLabel.CONTRADICTION,
        "NO_APLICA": VerificationLabel.NEUTRAL,
    }

    @classmethod
    def _parse_response(cls, raw_response: str) -> VerificationResult:
        answer_match = re.search(r"RESPUESTA:\s*(RESPALDA|CONTRADICE|NO_APLICA)", raw_response, re.IGNORECASE)
        quote_match = re.search(r"CITA:\s*(.+)", raw_response, re.IGNORECASE)

        if not answer_match:
            # Model didn't follow the constrained format — treat as neutral
            # rather than guessing. An unparseable verification response is
            # itself a signal the claim shouldn't be shown as verified.
            return VerificationResult(
                label=VerificationLabel.NEUTRAL,
                confidence=0.0,
                method=VerificationMethod.LLM_FALLBACK,
            )

        label = cls._LABEL_MAP[answer_match.group(1).strip().upper()]
        quote = quote_match.group(1).strip() if quote_match else None
        if quote and quote.lower() in ("ninguno", "ninguna", "none"):
            quote = None

        return VerificationResult(
            label=label,
            # This tier only runs on ambiguous cases, and the prompt forces
            # a single committed choice among three options — there's no
            # finer-grained confidence signal available than "the model
            # committed to this answer instead of leaving it ambiguous".
            confidence=0.75,
            method=VerificationMethod.LLM_FALLBACK,
            supporting_span=quote,
        )
