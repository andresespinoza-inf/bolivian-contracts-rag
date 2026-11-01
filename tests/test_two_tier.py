from contracts_rag.core.models import Chunk, ChunkType, ExtractionConfidence
from contracts_rag.core.verification import (
    EntailmentVerifier, VerificationLabel as L, VerificationMethod as M, VerificationResult,
)
from contracts_rag.verifiers.two_tier_verifier import TwoTierEntailmentVerifier


class Fixed(EntailmentVerifier):
    """Test fake: always returns the same result and counts calls."""

    def __init__(self, result):
        self.result, self.calls = result, 0

    def verify(self, claim, source_chunk):
        self.calls += 1
        return self.result


CHUNK = Chunk("d", ChunkType.CLAUSULA, "X", 1, "texto", ExtractionConfidence.ALTA)
T1 = lambda label, conf: Fixed(VerificationResult(label, conf, M.NLI))
T2 = lambda label: Fixed(VerificationResult(label, 0.75, M.LLM_FALLBACK, supporting_span="cita"))


def test_confident_tier1_result_is_returned_without_calling_tier2():
    t1, t2 = T1(L.ENTAILMENT, 0.94), T2(L.CONTRADICTION)
    result = TwoTierEntailmentVerifier(t1, t2).verify("c", CHUNK)
    assert result.label == L.ENTAILMENT and result.method == M.NLI and t2.calls == 0


def test_neutral_tier1_escalates_to_tier2():
    t2 = T2(L.CONTRADICTION)
    result = TwoTierEntailmentVerifier(T1(L.NEUTRAL, 0.95), t2).verify("c", CHUNK)
    assert result.label == L.CONTRADICTION and result.method == M.LLM_FALLBACK and t2.calls == 1


def test_low_confidence_escalates_even_if_label_is_not_neutral():
    t2 = T2(L.ENTAILMENT)
    result = TwoTierEntailmentVerifier(T1(L.CONTRADICTION, 0.40), t2).verify("c", CHUNK)
    assert result.method == M.LLM_FALLBACK and t2.calls == 1


def test_without_tier2_ambiguous_cases_are_never_promoted_to_verified():
    verifier = TwoTierEntailmentVerifier(T1(L.ENTAILMENT, 0.40))      # no LLM configured
    result = verifier.verify("c", CHUNK)
    assert result.label == L.NEUTRAL
    assert result.inconclusive_reason == "tier2_unavailable"


def test_without_tier2_confident_results_still_pass_through():
    result = TwoTierEntailmentVerifier(T1(L.CONTRADICTION, 0.92)).verify("c", CHUNK)
    assert result.label == L.CONTRADICTION and result.inconclusive_reason is None


def test_threshold_is_configurable():
    strict = TwoTierEntailmentVerifier(T1(L.ENTAILMENT, 0.8), T2(L.NEUTRAL), confidence_threshold=0.9)
    assert strict.verify("c", CHUNK).method == M.LLM_FALLBACK
