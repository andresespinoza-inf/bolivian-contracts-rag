# Bolivian Contracts RAG — Architecture Decision Record (ADR)

**Status:** Target architecture — partially implemented (see README for the current status table)
**Scope:** Retrieval and verification system for contracts (court rulings are out of scope for now, though the schema reserves their chunk types), designed for multi-jurisdiction use from the ground up.
**Non-goal:** This is not a client-specific or company-specific system. It is a portfolio-grade reference implementation demonstrating architectural judgment applied to a legal-domain RAG problem.

---

> **Status note.** This ADR describes the *target* architecture. Implemented and tested today:
> structure extraction (Sec. 2–3), hybrid retrieval pieces (Sec. 5), the two-tier verifier orchestration
> (Sec. 4) and a deterministic party-consistency check. **Not implemented yet:** contradiction detection
> (Sec. 6), answer generation, the segmented API / UI (Sec. 8), pgvector storage and the FastAPI/Vue stack.
> See the repository README for the honest, current status table.


## 1. Problem statement — why not a generic RAG

A tutorial-grade RAG (fixed-size chunking + default embeddings + one-line prompt) fails on legal documents for domain-specific reasons, not generic ones:

| Generic RAG assumption | Why it breaks in law |
|---|---|
| Text can be split by token windows | Legal documents have hierarchical structure (clauses, articles, sections) that carries legal meaning; splitting blindly can separate a rule from its exception within the same article |
| Semantic similarity is sufficient for retrieval | Exact term/number matches (e.g. "Artículo 15") matter as much as, or more than, semantic proximity |
| The LLM's citation can be trusted | A fabricated citation to a nonexistent law or ruling is not a generic error — it is a professional liability |
| One flat index is enough | Legal documents have parts with distinct legal function (facts vs. holding vs. reasoning in a ruling; recitals vs. operative clauses in a contract) that must be treated differently in generation and in risk analysis |

Every design decision below states explicitly what generic alternative it rejects and why.

---

## 2. Metadata schema

### 2.1 `Document`

| Field | Type | Purpose |
|---|---|---|
| `document_id` | UUID | Unique identifier |
| `jurisdiction` | str (ISO-like code, e.g. `BO`, `ES`, `MX`) | Key used to load jurisdiction-specific regex patterns and comparison rules |
| `document_type` | enum: `contrato`, `ley`, `sentencia`, `norma_reglamentaria` | Determines which `StructureExtractor` config and which section rules apply |
| `title` | str | Human-readable name |
| `source` | str | Origin (e.g. "Official Gazette", "User-uploaded contract") |
| `effective_date` | date \| null | Effective date. **Required** for `document_type in {ley, norma_reglamentaria}` — without it, temporal precedence in contradiction resolution cannot be evaluated. If absent, the document must be flagged `vigencia_no_verificada`; the system never assumes a norm is in force by default |
| `upload_date` | datetime | Audit trail |
| `raw_text_hash` | str | Duplicate-upload detection |
| `language` | str | Extensibility (e.g. Catalan, Basque, regional variants) |

### 2.2 `Section` (chapter/title — optional intermediate level)

| Field | Type | Purpose |
|---|---|---|
| `section_id` | UUID | — |
| `document_id` | FK | — |
| `label` | str (e.g. "Capítulo III") | — |
| `order` | int | Order of appearance |

**Rule:** `Section` is intentionally optional. A simple contract with no chapters attaches its `Chunk`s directly to the `Document`. Forcing a hierarchy the source document doesn't have is the same failure mode as generic fixed-size chunking.

### 2.3 `Chunk` (article / clause / recital — atomic unit of legal meaning)

| Field | Type | Purpose |
|---|---|---|
| `chunk_id` | UUID | — |
| `document_id` | FK | — |
| `section_id` | FK \| null | — |
| `chunk_type` | enum: `articulo`, `clausula`, `antecedente`/`expositivo`, `considerando`, `por_tanto`, `hecho_probado`, `fragmento_sin_clasificar` | Determines how the chunk is treated in generation and risk analysis (e.g. a narrated fact is not an applicable rule; a recital establishes *causa*, not an obligation) |
| `label` | str (e.g. "Artículo 15", "Cláusula Tercera") | — |
| `order` | int | Order within the document |
| `full_text` | str | Complete text of the chunk (the "parent") |
| `token_count` | int | Drives the sub-chunking decision |
| `is_atomic` | bool | `True` if never sub-divided |
| `references_to` | list[str] | Detected cross-references (e.g. "conforme al Artículo 10", or a conformity clause implicitly referencing the parties clause) |
| `extraction_confidence` | enum: `alta`, `baja` | Set by the `StructureExtractor`. Low confidence must be surfaced downstream (verification, UI) — the system never hides uncertainty about its own structural extraction |
| `embedding_id` | str \| null | Populated only if `is_atomic = True` |

### 2.4 `SubChunk` (only when `Chunk.token_count` exceeds the split threshold)

| Field | Type | Purpose |
|---|---|---|
| `subchunk_id` | UUID | — |
| `parent_chunk_id` | FK | — |
| `position_in_parent` | int | Reconstructs order; supports "show surrounding context" in the UI highlight |
| `text` | str | — |
| `split_reason` | enum: `inciso`, `numeral`, `parrafo`, `ventana_forzada` | Quality audit signal — a high rate of `ventana_forzada` for a jurisdiction indicates the regex config needs better sub-marker detection there |
| `embedding_id` | str | — |

**Indexing rule:** only leaf units are indexed for retrieval — a `Chunk` if atomic, or its `SubChunk`s if split. The parent `Chunk` is never indexed alongside its own sub-chunks (would duplicate signal and bias ranking). The parent is used exclusively to expand context at generation time.

**Split threshold:** 500 tokens as a starting default, configurable per `document_type` (e.g. a `considerando` in a ruling is naturally longer than a contract clause). To be tuned against the demo datasets.

**Split priority:** natural sub-markers within the unit (incisos `a)`, `b)`; numerals `1.`, `2.`; paragraph breaks) are tried first. A forced token-window split is the last resort, and is logged as `ventana_forzada` for later review — never applied silently as the default strategy.

---

## 3. Structure extraction contract

```
StructureExtractor.extract(raw_text: str, jurisdiction: str, document_type: DocumentType) -> Document
```

- **Input:** raw text + jurisdiction + document type.
- **Output:** a fully populated `Document → Section? → Chunk → SubChunk?` tree. No embeddings are computed here — embedding generation is a separate responsibility (indexing layer), never coupled to structure extraction.
- **Guarantees:** every `Chunk` has `chunk_type`, `label`, `order`, and `extraction_confidence` set.
- **Does not guarantee:** perfect extraction. Any implementation — including future ones — must report `extraction_confidence`; this is part of the contract, not an implementation detail of any single extractor.

**Fallback rule:** if no recognizable marker is found, the extractor falls back to paragraph-based chunking with `chunk_type = fragmento_sin_clasificar` and `extraction_confidence = baja`. The system never labels an unclassified fragment as "Artículo X" just to fit the schema — honesty about structural quality is treated with the same seriousness as honesty about generated answers.

**Extensibility point (not implemented in v1):** `StructuredSourceAdapter`, an alternate implementation of the same interface for cases where structure arrives pre-parsed (e.g. from a structured legal API or JSON/XML source). Kept as an interface only — no legal-data API is assumed to be reliably available across jurisdictions, so the regex-based extractor is the real, defensible solution to the actual problem: extracting structure from unstructured legal text.

### 3.1 Jurisdiction configuration (data, not code)

Patterns live in per-jurisdiction, per-document-type YAML files, not hardcoded in the extractor:

```yaml
jurisdiction: BO
document_type: contrato
markers:
  clausula:
    pattern: "Cl[aá]usula\\s+(Primera|Segunda|Tercera|...|[0-9]+)"
  antecedente:
    pattern: "(ANTECEDENTES|EXPOSITIVO)"
  referencia_cruzada:
    pattern: "conforme\\s+a\\s+la\\s+Cl[aá]usula\\s+\\w+"
sub_markers:
  inciso: "^[a-z]\\)"
  numeral: "^[0-9]+\\."
token_limit_before_split: 500
```

Adding a new jurisdiction (e.g. `ES` or `MX`) is a data change, not an engine change. This is the concrete proof behind the "multi-jurisdiction from day one" claim.

---

## 4. Anti-hallucination verification (two-tier, not single-pass)

**Rejected generic approach:** verifying an LLM's citation with another LLM prompt call. This is circular — the same class of model that hallucinates can also fail to catch its own hallucination during verification.

**Adopted design:**

- **Tier 1 — NLI entailment (cheap, deterministic):** every generated claim + its cited source chunk is passed through a dedicated NLI cross-encoder (e.g. `cross-encoder/nli-deberta-v3`), classified as `entailment`, `neutral`, or `contradiction`. This is a non-generative classifier with a different architecture from the generator — not the same model checking itself.
- **Tier 2 — LLM verification (only for ambiguous cases):** invoked only when Tier 1 returns `neutral` or confidence below a threshold. Prompt is constrained: "Does this fragment literally support this claim? Answer SUPPORTS / CONTRADICTS / NOT_APPLICABLE and quote the exact span." (Refined during implementation: a binary YES/NO cannot distinguish "does not mention it" from "contradicts it", which would inflate false contradictions.) **Tier 2 is optional**: without an LLM, ambiguous cases are reported as "not verified" (`inconclusive_reason=tier2_unavailable`) and a low-confidence NLI label is never promoted to a verified result." Reserved for the ambiguous fraction to control cost and latency.

```
EntailmentVerifier.verify(claim: str, source_chunk: Chunk | SubChunk) -> VerificationResult
VerificationResult = { label: entailment | neutral | contradiction, confidence: float, method: "nli" | "llm_fallback" }
```

**UI consequence:** the answer is never withheld pending verification. Every claim is shown with a visible status (`verificado` / `no_verificado` / `contradicción_detectada`). Hiding unverified claims would be paternalistic and would contradict the total-traceability differentiator.

---

## 5. Hybrid retrieval

**Rejected generic approach:** weighted sum of vector similarity and BM25 scores. Rejected because the two scores live on incomparable scales (BM25 is unbounded TF-IDF-based; cosine similarity is bounded), which makes the weighting fragile and corpus-dependent.

**Adopted design: Reciprocal Rank Fusion (RRF).** Uses only rank position from each list, not raw scores — scale-agnostic by construction.

```
RRF_score(d) = Σ 1 / (k + rank_i(d))     (k = 60, default)
```

**Pipeline (concrete numbers, not left open):**

1. Top-50 candidates from the vector branch + Top-50 from BM25
2. RRF fusion → Top-20
3. Cross-encoder re-ranking over those 20 (expensive step, deliberately bounded — never run over the full corpus)
4. Top-5 to Top-8 passed to the generator

```
HybridRetriever.retrieve(query: str, top_k_final: int = 8) -> list[RankedChunk]
```

Intermediate numbers (50, 20, k=60) are configuration, not part of the method's public contract.

**Why BM25 matters here, specifically:** a query referencing "Artículo 15" needs exact lexical recall, which a pure embedding can dilute if several articles are semantically similar. This is a domain-driven justification, not a generic RAG best practice.

---

## 6. Contradiction detection

**Rejected generic approach:** comparing all chunk pairs by embedding similarity and flagging high-similarity-but-divergent pairs. High false-positive rate, `O(n²)`, and no legal reasoning behind the flag.

**Adopted design — three layers:**

1. **Candidate grouping (`TopicGrouper`):** chunks are grouped by shared topic/regulated matter (e.g. "plazo de entrega", "forma del contrato") using lightweight NER/keyword matching at chunking time. Comparisons only happen within a topic group — never all-against-all.
2. **Comparison (`EntailmentVerifier`, reused):** the same NLI verifier from Section 4 is applied within each topic group. `contradiction` with high confidence is a candidate alert. Reusing this component is a deliberate architectural choice: anti-hallucination checking and contradiction detection are, structurally, the same problem — is text A consistent with text B.
3. **Legal resolution hint (not automated resolution):** a detected contradiction between two norms is not treated as a tie. Simple, explicit rules based on hierarchy (higher norm prevails), temporality (`effective_date`, later norm may derogate an earlier one), and speciality (special norm over general) generate a `resolution_hint` — the system surfaces the applicable legal criterion, it does not decide the outcome. Final judgment stays with the lawyer.

```
ContradictionDetector.find_contradictions(chunks, topic_grouper, verifier) -> list[Contradiction]
Contradiction = { chunk_a_id, chunk_b_id, topic, entailment_result, resolution_hint: str | null }

TopicGrouper.group(chunks: list[Chunk]) -> dict[str, list[Chunk]]
```

---

## 7. Full interface contracts

```python
class StructureExtractor(ABC):
    def extract(self, raw_text: str, jurisdiction: str, document_type: DocumentType) -> Document: ...

class Indexer(ABC):
    def index(self, leaf_units: list[Chunk | SubChunk]) -> None: ...

class HybridRetriever(ABC):
    def retrieve(self, query: str, top_k_final: int = 8) -> list[RankedChunk]: ...

class EntailmentVerifier(ABC):
    def verify(self, claim: str, source_chunk: Chunk | SubChunk) -> VerificationResult: ...

class TopicGrouper(ABC):
    def group(self, chunks: list[Chunk]) -> dict[str, list[Chunk]]: ...

class ContradictionDetector(ABC):
    def find_contradictions(
        self, chunks: list[Chunk], topic_grouper: TopicGrouper, verifier: EntailmentVerifier
    ) -> list[Contradiction]: ...

class AnswerGenerator(ABC):
    def generate(self, query: str, ranked_chunks: list[RankedChunk]) -> RawAnswer: ...
    # RawAnswer: raw text + LLM-proposed claim->chunk_id mapping, NOT yet verified

class ResponseAssembler(ABC):
    def assemble(
        self, raw_answer: RawAnswer, verifications: list[VerificationResult], contradictions: list[Contradiction]
    ) -> AnswerResponse: ...
```

Design principle across all seven: each interface owns exactly one responsibility, and no layer trusts another layer's output without an explicit contract (e.g. generation proposes citations, verification decides if they hold — never the same step).

---

## 8. API contract for the frontend

Answers are returned as **segmented claims**, not a single block of text with citations appended — this is what lets the Vue frontend paint per-segment highlighting without parsing free text.

```json
{
  "answer_segments": [
    {
      "text": "El plazo de entrega es de 30 días calendario",
      "verification_status": "verificado",
      "source_chunk_id": "uuid",
      "source_label": "Cláusula Quinta",
      "confidence": "alta"
    },
    {
      "text": "y puede extenderse en caso de fuerza mayor",
      "verification_status": "no_verificado",
      "source_chunk_id": null,
      "confidence": null
    }
  ],
  "contradictions_detected": [
    {
      "chunk_a": "uuid",
      "chunk_b": "uuid",
      "topic": "plazo de entrega",
      "note": "Cláusula Quinta (contrato) vs. Artículo 12 (norma) — verificar vigencia y jerarquía"
    }
  ]
}
```

Traceability logic lives in the backend, where the source of truth belongs — not inferred client-side from free text.

---

## 9. Demo dataset plan

**Chosen contract type: compraventa de bien inmueble.**

Rejected alternatives and why:
- *Reconocimiento de deuda*: tied to Bolivia-specific procedural mechanics (novation into an executive title), not comparable across jurisdictions without excessive procedural exposition.
- *Préstamo de dinero*: doctrinally flatter — does not exercise all four formation requirements with real tension.
- *Arrendamiento*: comparable across jurisdictions, but rarely engages the *forma* requirement, leaving one of the four formation elements untested.

**Why compraventa de inmueble activates all four formation requirements (Bolivian doctrine, per project owner's domain expertise):**

| Requirement | Where it shows up | System component it exercises |
|---|---|---|
| Consentimiento | Opening clause (party identification) + closing conformity clause + signatures | `chunk_type = clausula`, `references_to` (conformity clause implicitly references party identification) |
| Objeto | Transfer of property for a price | Risk mode — e.g. flagging an indeterminate price or an insufficiently individualized property description |
| Causa | Not explicit; presumed; in practice found in the antecedentes/recitals | Justifies `chunk_type = antecedente` as distinct from `clausula` |
| Forma | Not required for the contract's formation/validity, but required for registration in Derechos Reales and consequent *oponibilidad* against third parties | A genuine, non-forced risk-mode case: a private-document (non-notarized) compraventa de inmueble is valid between parties but not yet opponible/registrable — the system should flag this as a registration-risk note, not a validity defect |

**Dataset composition (minimum viable, deliberately narrow in scope):**

1. **Bolivian compraventa de inmueble contract** (project owner's own document, anonymized, with at least one intentional imprecision — e.g. missing notarization, or incomplete property description — to give risk mode something genuine to flag)
2. **Extract of the Bolivian Civil Code** — only the articles relevant to compraventa (contract formation, objeto, forma/registration, transfer of ownership, saneamiento por evicción y vicios ocultos) — not the full code. This is the real multi-article corpus that exercises hybrid retrieval and re-ranking meaningfully.
3. **A comparable compraventa de inmueble template from Spain or Mexico**, sourced from an open notarial/bar-association template, for genuine structural comparison across jurisdictions — not an unrelated document included only to claim "multi-jurisdiction."

**Explicitly excluded from v1** (kept as documented extensibility, not implemented): jurisprudence/court rulings, additional contract types "for variety." Both would exercise `chunk_type` values already reserved in the schema (`considerando`, `hecho_probado`, `por_tanto`) without adding a new differentiator — mentioned in the README as a designed extension point.

**Data hygiene:** the Bolivian contract must be anonymized/converted into a synthetic template before being committed to the public repository, consistent with the project's own rule of never publishing real client documents.

---

## 10. Build order (already agreed)

1. Metadata classes + `StructureExtractor` interface + `RegexStructureExtractor` implementation
2. Anti-hallucination verification flow (`EntailmentVerifier`, two-tier)
3. Remaining components: `Indexer`, `HybridRetriever`, `TopicGrouper`, `ContradictionDetector`, `AnswerGenerator`, `ResponseAssembler`
4. Frontend (Vue, side-by-side view with segment-level highlighting)

Build the full pipeline end-to-end with the simplest version of each layer first (basic NLI, RRF with fixed k, contradiction detection without the legal-resolution layer); deepen sophistication per layer afterward. A complete but simple pipeline demonstrates more than one perfected component in isolation.
