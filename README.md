# Bolivian Contracts RAG

**English** · [Español](README.es.md)

**Structure-aware retrieval and verification over Bolivian contracts and civil-code norms** — built so that every statement can be traced to the exact clause or article it came from, and so the system says *"not verified"* instead of guessing.

The legal corpus and worked examples are in Spanish; the code, tests and documentation are in English.

![tests](https://github.com/andresespinoza-inf/bolivian-contracts-rag/actions/workflows/tests.yml/badge.svg)

> **Status: working research prototype.** The extraction, retrieval, verification-orchestration and party-consistency layers are implemented and covered by 33 tests that run without any model, GPU or API key. Answer generation, contradiction detection and the UI are being built locally and are **not** in this repository yet — see [What works today](#what-works-today) for the exact boundary.

## Why this is not "chat with your PDF"

A tutorial RAG (fixed-size chunks, default embeddings, a one-line prompt) is a poor fit for law, for reasons that are specific to the domain:

| Generic RAG assumption | What goes wrong with contracts and codes | What this project does instead |
|---|---|---|
| Split text every N tokens | Cuts a rule away from its exception inside the same clause; breaks cross-references | Splits on the document's own structure (clause / article / recital), keeps the whole clause as parent, sub-splits long ones on their own numbered items |
| Semantic similarity is enough | "Artículo 1538" or "cláusula quinta" must match *exactly* | Hybrid retrieval: BM25 + vectors fused with Reciprocal Rank Fusion, then reranked |
| Trust the LLM's citation | A fabricated clause or article is a professional liability, not a typo | Every claim is checked against the chunk it cites; if it can't be backed, it is shown as **not verified** — never hidden, never promoted |
| One parser fits all | Bolivian contracts have no recitals section; Spanish ones do | Per-jurisdiction patterns live in YAML ([`configs/`](contracts_rag/configs)); adding a jurisdiction is a data change, not an engine change |
| Everything needs a model | Some legal checks are deterministic | Where the rule is mechanical (do the signatories match the parties named in clause one?), it is a plain function, with the evidence attached |

## What works today

| Component | Status | Notes |
|---|---|---|
| Structure extraction (`RegexStructureExtractor`) | ✅ implemented, tested | Clauses, articles, recitals, sections, sub-chunks, cross-references, confidence flags |
| Per-jurisdiction config (BO contract, ES contract, BO law) | ✅ implemented, tested | YAML, no code changes to add a jurisdiction |
| Deterministic party-consistency check | ✅ implemented, tested | Bolivian format only; flags, never concludes invalidity |
| Hybrid retrieval: BM25 + RRF fusion | ✅ implemented, tested | Runs anywhere, no downloads |
| Vector + rerank, classical stack (TF-IDF, token overlap) | ✅ implemented, tested | Deliberately weak stand-in so the pipeline runs with zero downloads |
| Vector + rerank, neural stack (bge-m3, bge-reranker) | 🧪 implemented, **not benchmarked yet** | `pip install -e ".[models]"` |
| Two-tier verifier — orchestration | ✅ implemented, tested with fakes | Tier 2 is optional; without it, ambiguous results are "not verified" |
| Verifier Tier 1 — NLI model | 🧪 implemented, **not benchmarked yet** | `cross-encoder/nli-deberta-v3-base` |
| Verifier Tier 2 — LLM | 🧪 interface + constrained prompt only | No LLM is connected or evaluated in this repo |
| Contradiction detection across documents | 🚧 in progress locally | Will reuse the verifier |
| Answer generation with per-claim verification status | 🚧 in progress locally | Planned with a local LLM (Ollama) |
| Source-highlighting UI | 🚧 planned | |
| Evaluation set (recall@k, citation precision) | 🚧 planned | Currently there are **no retrieval metrics** — see Limitations |

## Quickstart

```bash
git clone https://github.com/andresespinoza-inf/bolivian-contracts-rag.git
cd bolivian-contracts-rag
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pytest -q                                               # 33 tests, ~1 s, no models needed

python examples/01_extract_structure.py
python examples/02_party_consistency.py
python examples/03_hybrid_retrieval.py

# Optional — real neural models (downloads ~4 GB from Hugging Face on first run)
pip install -e ".[models]"
python examples/03_hybrid_retrieval.py --real-models
python examples/04_verify_claims.py
```

## Example: a consent-formation red flag, with no model at all

In Bolivian doctrine the four formation requirements of a contract are consent, object, cause and form. Consent shows up twice in a contract — the parties identified in clause one and the signatories at the end — and those must coincide. `examples/02_party_consistency.py` checks exactly that on the bundled synthetic contract, where the signature block deliberately names different people:

```text
status: INCONSISTENT
parties clause : ['ROBERTO CALLE MAMANI', 'SILVIA CALLE MAMANI']
closing clause : ['ELIAS FLORES CONDORI', 'TERESA ROJAS LIMACHI']

Names identified in the parties clause do not match the names in the closing/signature
clause. Requires lawyer review (possible consent-formation issue); this is a flag, not
a finding of invalidity.
evidence chunk ids (for highlighting): ['5f26…', 'ca86…']      # ids differ on every run
```

The report carries the ids of both source chunks so a UI can highlight them side by side. The legal framing (error, fraud, nullity vs. annulability) lives in the bundled Civil Code extract (Arts. 473–482, 549–559), so the system can *retrieve* the norm rather than assert it from memory.

## Example: hybrid retrieval over the bundled corpus

Output of `examples/03_hybrid_retrieval.py` with the **classical stack** (no downloads). Labels are truncated and the arrows are my own annotations; the ranking itself is unedited, weaker spots included:

```text
Q: ¿Qué pasa si el contrato de compraventa no consta en documento público?
  #1  CONTRATO DE COMPRAVENTA DE INMUEBLE Conste por el presente doc…   (the "private document" preamble)
  #2  SEXTA. - ESCRITURA PÚBLICA. La escritura pública de compravent…
  #3  Artículo 491°.- (Contratos y actos que deben hacerse por docum…
  #4  Artículo 569°.- (Cláusula resolutoria) …                           ← weak hit

Q: nulidad del contrato por vicios del consentimiento
  #1  Artículo 556°.- (Prescriptibilidad de la acción de anulación) …
  #2  Artículo 550°.- (Nulidad parcial del contrato) …
  #3  Artículo 549°.- (Casos de nulidad del contrato) …
  #4  Artículo 633°.- (Perecimiento de la cosa) …                        ← irrelevant hit
```

The right norms are found, but the tail is noisy. That is what a lexical/TF-IDF stack looks like; whether the neural stack fixes it is something I have **not measured yet** and do not want to claim.

## Design decisions

1. **Chunk on legal structure, index only leaves.** A clause or article is the atomic unit. If it exceeds the token threshold it is split on its own sub-markers (`2.1.`, `a)`, `I.`), with a plain token window only as a last resort that is logged as such. The index contains the leaf units only — never a parent *and* its children, which would double-count the same text.
2. **Never drop text silently.** Anything that doesn't match a known marker is kept as `fragmento_sin_clasificar` with `confidence=baja`. This is not cosmetic: the opening line of the bundled Bolivian contract ("documento privado, que podrá ser elevado a público…") sits *before* clause one, and is precisely the evidence for a form/registration risk.
3. **Configuration is data.** Marker patterns per jurisdiction and document type live in YAML. The Spanish template has numbered recitals (`EXPONEN I–III`) that the Bolivian one lacks; the engine doesn't care.
4. **Two-tier verification, second tier optional.** A dedicated NLI classifier checks each claim against its cited chunk first (cheap, deterministic, a different kind of model from whatever generated the claim). Only ambiguous cases go to an LLM, with a three-way constrained answer (*supports / contradicts / not applicable*) because a yes/no cannot tell "doesn't mention it" from "contradicts it". With no LLM configured, ambiguous cases are reported as *not verified*, and a low-confidence NLI label is never promoted to verified.
5. **Rank fusion, not score fusion.** BM25 scores are unbounded; cosine similarity is bounded. Weighted sums are fragile across corpora; RRF uses ranks only.
6. **Deterministic where the law is deterministic.** Signature/party consistency is a function, not a prompt.

## Engineering notes: what real PDFs taught me

The parser was developed against real, privately held contracts (not redistributed) and then re-verified on the synthetic fixtures in this repo, which reproduce the same quirks.

- **Headings are wrapped.** PDF extraction splits `SEXTA (OBLIGACIONES / DE LAS PARTES).-` across two lines; the Spanish roman numerals arrive alone on their own line. Markers therefore can't require their closing punctuation on the same line.
- **Case-insensitive matching creates false chunks.** A wrapped line starting with the word *segunda* in "…la cláusula / segunda…" must not open a new clause; marker matching is case-sensitive on purpose.
- **Silent loss is easy to write.** Two bugs of this kind surfaced while building test cases from real documents: a dropped preamble, and — when a long clause was split on its numbered items — the introductory sentence before the first item being discarded (in the Spanish fixture, the sentence that states the price). The second has a regression test that reconstructs every parent clause from its sub-chunks and fails if any word is missing — I checked that the test really fails when the fix is removed.
- **Term-frequency bias is real.** On the real documents, a long payment-schedule clause outranked a short, more directly relevant one under BM25/TF-IDF. This is the kind of case that motivates semantic retrieval; I have not yet quantified the improvement.

## Limitations (please read before judging the project)

- **Tiny corpus, synthetic fixtures.** Two synthetic purchase contracts (Bolivia, Spain) and a hand-compiled extract of the Bolivian Civil Code (122 articles, in force since 2 April 1976 per its Art. 1570; **verify against the official text before relying on it**). The Bolivian fixture is derived from a real contract with all identifiers replaced; the Spanish one is original text modelled on the structure of a notarial template.
- **No evaluation yet.** There is no labelled question set, so no recall@k or citation-precision numbers. Anything I say about retrieval quality is qualitative until that exists.
- **Neural components not benchmarked.** The bge-m3 / bge-reranker / NLI paths are implemented but I have not published results for them.
- **Party-consistency is a heuristic.** It reads UPPERCASE name runs, is calibrated on Bolivian contracts only, and returns `not_evaluable` elsewhere rather than guessing.
- **Regex extraction has limits.** It handles the formats configured in YAML. A new layout needs a new config, and scanned documents need OCR upstream.
- **Not legal advice.** This is a research prototype. Its flags are prompts for a lawyer's review, not conclusions.

## Roadmap (in progress locally)

- Contradiction detection between documents, reusing the verifier on topic-grouped chunks
- Answer generation where each sentence carries its source and verification status, using a local LLM
- A minimal UI that highlights the source clause next to the answer
- A small hand-labelled evaluation set and retrieval/citation metrics
- A second jurisdiction fully via configuration (Spain is wired for contracts only)

## Repository layout

```text
contracts_rag/
  core/          data model (Document, Section, Chunk, SubChunk) and interfaces
  extractors/    regex structure extractor
  configs/       per-jurisdiction patterns (YAML)
  indexing/      BM25, TF-IDF and sentence-transformers indexes
  retrieval/     RRF fusion, hybrid retriever, rerankers
  verifiers/     NLI tier, LLM tier (injectable client), two-tier orchestrator
  analysis/      deterministic party-consistency check
data/            synthetic contracts + Civil Code extract (see data/README.md)
examples/        runnable demos
tests/           33 tests
docs/            [architecture decision record](docs/architecture-decisions.md)
```

## About

Built by a Bolivian lawyer (law degree, 2022; master's in civil and family law in progress) who also studies computer science, with a focus on AI and data science. The domain knowledge is the point: the design follows how contracts are actually formed and challenged in Bolivian civil law, instead of treating legal text as generic prose.

## License

MIT — see [LICENSE](LICENSE). The bundled Civil Code extract is public legal text; fixtures contain no real personal data.
