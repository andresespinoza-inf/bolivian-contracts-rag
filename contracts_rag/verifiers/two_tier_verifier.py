"""
Bolivian Contracts RAG — TwoTierEntailmentVerifier (ADR Section 4).

Every other layer should depend on this class, not on the individual tiers.
It composes two verifiers behind the same EntailmentVerifier interface, so
either one can be swapped (different NLI model, different LLM provider, or a
test fake) without touching this orchestration logic.

Tier 2 is OPTIONAL. Without it (no LLM available — the default for anyone who
just cloned the repo), cases that would have been escalated are returned as
NEUTRAL with `inconclusive_reason="tier2_unavailable"`. A low-confidence
"entailment" or "contradiction" from the NLI model is deliberately NOT
promoted to a verified result: the system's rule is that anything it cannot
back with enough confidence is shown as "not verified", never as verified.
"""

from __future__ import annotations

from contracts_rag.core.models import Chunk, SubChunk
from contracts_rag.core.verification import (
    EntailmentVerifier,
    VerificationLabel,
    VerificationResult,
)

DEFAULT_CONFIDENCE_THRESHOLD = 0.6


class TwoTierEntailmentVerifier(EntailmentVerifier):
    def __init__(
        self,
        tier1_verifier: EntailmentVerifier,
        tier2_verifier: EntailmentVerifier | None = None,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    ) -> None:
        self._tier1 = tier1_verifier
        self._tier2 = tier2_verifier
        self._confidence_threshold = confidence_threshold

    def verify(self, claim: str, source_chunk: Chunk | SubChunk) -> VerificationResult:
        tier1_result = self._tier1.verify(claim, source_chunk)

        escalate = (
            tier1_result.label == VerificationLabel.NEUTRAL
            or tier1_result.confidence < self._confidence_threshold
        )
        if not escalate:
            return tier1_result

        if self._tier2 is None:
            return VerificationResult(
                label=VerificationLabel.NEUTRAL,
                confidence=tier1_result.confidence,
                method=tier1_result.method,
                inconclusive_reason="tier2_unavailable",
            )

        return self._tier2.verify(claim, source_chunk)
