"""
Bolivian Contracts RAG — Anti-hallucination verification contract (ADR Section 4).

Two-tier design, reused later for contradiction detection (ADR Section 6):
  Tier 1 (NLI, cheap/deterministic) always runs first.
  Tier 2 (LLM) runs only when Tier 1 is inconclusive (label=neutral or
  confidence below threshold).

This module defines the shape everyone depends on. It contains no model
code — see verifiers/ for concrete implementations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

from contracts_rag.core.models import Chunk, SubChunk


class VerificationLabel(str, Enum):
    ENTAILMENT = "entailment"
    NEUTRAL = "neutral"
    CONTRADICTION = "contradiction"


class VerificationMethod(str, Enum):
    NLI = "nli"
    LLM_FALLBACK = "llm_fallback"


@dataclass
class VerificationResult:
    label: VerificationLabel
    confidence: float  # 0.0–1.0
    method: VerificationMethod
    # Populated only when method == LLM_FALLBACK and the model quoted a
    # specific span as support/refutation (ADR 4: "quote the exact
    # supporting or refuting span").
    supporting_span: str | None = None
    # Set when the result is deliberately downgraded to NEUTRAL because the
    # case needed escalation but no Tier 2 verifier is configured (e.g. no
    # LLM available). Lets callers/UI say *why* a claim is "not verified".
    inconclusive_reason: str | None = None


class EntailmentVerifier(ABC):
    @abstractmethod
    def verify(self, claim: str, source_chunk: Chunk | SubChunk) -> VerificationResult:
        """
        Does source_chunk's text support (entailment), contradict
        (contradiction), or neither confirm nor deny (neutral) `claim`?

        `claim` is deliberately just a string, not tied to AnswerGenerator
        output — this lets the exact same verifier be reused by
        ContradictionDetector, where `claim` is simply the text of another
        Chunk being checked against this one (ADR Section 6: "anti-
        hallucination checking and contradiction detection are,
        structurally, the same problem").
        """
        raise NotImplementedError
