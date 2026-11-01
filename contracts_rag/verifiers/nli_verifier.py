"""
Bolivian Contracts RAG — Tier 1 verifier: NLI cross-encoder (ADR Section 4).

Real implementation, meant to run in an environment with unrestricted
internet access (it downloads model weights from Hugging Face on first
use). This is NOT runnable inside the sandbox this project was designed
in — see tests/stubs.py for the test double used there to validate
orchestration logic instead.

Install:
    pip install sentence-transformers

The model classifies a (claim, chunk_text) pair into one of
{entailment, neutral, contradiction} with a confidence score — a
non-generative classifier, architecturally distinct from whatever LLM
is used for generation and for Tier 2. That distinction is the whole
point (ADR Section 4): the same kind of model that can hallucinate
should not be the one certifying its own claims.
"""

from __future__ import annotations

from contracts_rag.core.models import Chunk, SubChunk
from contracts_rag.core.verification import (
    EntailmentVerifier,
    VerificationLabel,
    VerificationMethod,
    VerificationResult,
)

DEFAULT_MODEL_NAME = "cross-encoder/nli-deberta-v3-base"

# Label order returned by this specific model's logits — cross-encoder/nli-*
# models from the sentence-transformers hub follow this convention
# (contradiction, entailment, neutral). Kept as an explicit constant,
# not a magic index, so a different model swap only requires updating
# this one place.
_LABEL_ORDER = [
    VerificationLabel.CONTRADICTION,
    VerificationLabel.ENTAILMENT,
    VerificationLabel.NEUTRAL,
]


class NLIEntailmentVerifier(EntailmentVerifier):
    def __init__(self, model_name: str = DEFAULT_MODEL_NAME) -> None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise ImportError(
                "NLIEntailmentVerifier requires sentence-transformers. "
                "Install with: pip install sentence-transformers"
            ) from exc

        self._model = CrossEncoder(model_name)

    def verify(self, claim: str, source_chunk: Chunk | SubChunk) -> VerificationResult:
        source_text = source_chunk.full_text if isinstance(source_chunk, Chunk) else source_chunk.text

        # NLI convention: (premise, hypothesis) — the source text is what we
        # know to be true; the claim is what we're checking against it.
        scores = self._model.predict([(source_text, claim)])[0]

        probs = _softmax(scores)
        best_idx = max(range(len(probs)), key=lambda i: probs[i])

        return VerificationResult(
            label=_LABEL_ORDER[best_idx],
            confidence=float(probs[best_idx]),
            method=VerificationMethod.NLI,
        )


def _softmax(scores) -> list[float]:
    import math

    max_score = max(scores)
    exps = [math.exp(s - max_score) for s in scores]
    total = sum(exps)
    return [e / total for e in exps]
