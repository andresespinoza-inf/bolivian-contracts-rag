"""
Bolivian Contracts RAG — RegexStructureExtractor.

First (and, for v1, only) implementation of StructureExtractor. Structure
is inferred from raw text using per-jurisdiction/per-document-type regex
configuration (configs/<JURISDICTION>_<document_type>.yaml) — never
hardcoded in this file. Adding a new jurisdiction is a matter of adding a
YAML file, not touching this engine (ADR 3.1).

Algorithm, at a glance:
  1. Load the jurisdiction/document_type config.
  2. Walk the text line by line.
  3. A line matching a `section_markers` pattern starts a new Section.
  4. A line matching any `markers.<chunk_type>` pattern closes the current
     Chunk (if any) and starts a new one of that type.
  5. Anything before the first recognized marker is emitted as a single
     ChunkType.FRAGMENTO_SIN_CLASIFICAR chunk (fallback rule, ADR 3).
  6. Once a Chunk's full text is assembled, cross-references are scanned
     for and recorded, and the chunk is sub-split if it exceeds the
     configured token threshold (ADR 2.4 split-priority rule).
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml

from contracts_rag.core.models import (
    Chunk,
    ChunkType,
    Document,
    DocumentType,
    ExtractionConfidence,
    Section,
    SplitReason,
    SubChunk,
)
from contracts_rag.core.structure_extractor import StructureExtractor

CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"


class ConfigNotFoundError(RuntimeError):
    pass


def load_jurisdiction_config(jurisdiction: str, document_type: DocumentType) -> dict:
    config_path = CONFIG_DIR / f"{jurisdiction}_{document_type.value}.yaml"
    if not config_path.exists():
        raise ConfigNotFoundError(
            f"No regex config found for jurisdiction={jurisdiction!r}, "
            f"document_type={document_type.value!r} (looked in {config_path}). "
            f"Add one instead of hardcoding patterns in the extractor."
        )
    with config_path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


class RegexStructureExtractor(StructureExtractor):
    def extract(
        self,
        raw_text: str,
        jurisdiction: str,
        document_type: DocumentType,
        title: str,
        source: str,
        effective_date=None,
    ) -> Document:
        config = load_jurisdiction_config(jurisdiction, document_type)

        document = Document(
            jurisdiction=jurisdiction,
            document_type=document_type,
            title=title,
            source=source,
            raw_text=raw_text,
            effective_date=effective_date,
        )

        chunk_markers = self._compile_chunk_markers(config)
        section_markers = self._compile_section_markers(config)
        crossref_pattern = self._compile_optional(config.get("referencia_cruzada"))
        sub_marker_patterns = self._compile_sub_markers(config)
        token_limit = config.get("token_limit_before_split", 500)

        lines = raw_text.splitlines()

        current_section: Section | None = None
        current_chunk_type: ChunkType | None = None
        current_order = 0
        buffer: list[str] = []
        preamble_buffer: list[str] = []
        seen_any_marker = False

        def flush_chunk() -> None:
            nonlocal current_chunk_type, buffer
            if current_chunk_type is None or not buffer:
                buffer = []
                return
            full_text = "\n".join(buffer).strip()
            if full_text:
                chunk = self._build_chunk(
                    document_id=document.document_id,
                    section_id=current_section.section_id if current_section else None,
                    chunk_type=current_chunk_type,
                    label=self._derive_label(full_text),
                    order=current_order,
                    full_text=full_text,
                    extraction_confidence=ExtractionConfidence.ALTA,
                    crossref_pattern=crossref_pattern,
                    sub_marker_patterns=sub_marker_patterns,
                    token_limit=token_limit,
                )
                self._attach_chunk(document, current_section, chunk)
            buffer = []

        for raw_line in lines:
            line = raw_line.rstrip("\r")

            section_label = self._match_section(line, section_markers)
            if section_label is not None:
                flush_chunk()
                current_chunk_type = None
                current_label = None
                current_section = Section(
                    document_id=document.document_id,
                    label=section_label,
                    order=len(document.sections),
                )
                document.sections.append(current_section)
                continue

            matched_type = self._match_chunk_marker(line, chunk_markers)
            if matched_type is not None:
                flush_chunk()
                seen_any_marker = True
                current_order += 1
                current_chunk_type = matched_type
                buffer = [line]
                continue

            if current_chunk_type is not None:
                buffer.append(line)
            elif not seen_any_marker:
                # Text before the first recognized marker. NOT discarded:
                # preambles routinely carry legally meaningful content (e.g.
                # a contract's opening line stating it is a private — not
                # notarized — document). Collected here and emitted as a
                # low-confidence fragment below, same fallback philosophy
                # as the no-markers-at-all case (ADR 3): never silently
                # drop text just because it didn't fit a recognized marker.
                preamble_buffer.append(line)

        flush_chunk()

        # Fallback rule (ADR 3): text that never matched a recognized marker
        # — either because the whole document has none, or because it's a
        # preamble before the first one — becomes a low-confidence fragment
        # rather than being silently dropped or mislabeled.
        if not seen_any_marker:
            full_text = raw_text.strip()
            if full_text:
                document.chunks.append(
                    self._build_chunk(
                        document_id=document.document_id,
                        section_id=None,
                        chunk_type=ChunkType.FRAGMENTO_SIN_CLASIFICAR,
                        label="(sin clasificar)",
                        order=1,
                        full_text=full_text,
                        extraction_confidence=ExtractionConfidence.BAJA,
                        crossref_pattern=crossref_pattern,
                        sub_marker_patterns=sub_marker_patterns,
                        token_limit=token_limit,
                    )
                )
        else:
            preamble_text = "\n".join(preamble_buffer).strip()
            if preamble_text:
                preamble_chunk = self._build_chunk(
                    document_id=document.document_id,
                    section_id=None,
                    chunk_type=ChunkType.FRAGMENTO_SIN_CLASIFICAR,
                    label=self._derive_label(preamble_text),
                    order=0,
                    full_text=preamble_text,
                    extraction_confidence=ExtractionConfidence.BAJA,
                    crossref_pattern=crossref_pattern,
                    sub_marker_patterns=sub_marker_patterns,
                    token_limit=token_limit,
                )
                # Insert at the front, not appended, so document order
                # reflects reading order (preamble comes before clause 1).
                document.chunks.insert(0, preamble_chunk)

        return document

    # -- helpers -----------------------------------------------------------

    @staticmethod
    def _compile_chunk_markers(config: dict) -> list[tuple[ChunkType, re.Pattern]]:
        # Deliberately NOT case-insensitive: clause/article headers in these
        # documents are consistently uppercase ("SEXTA", "Artículo"), and a
        # lowercase in-sentence mention ("la cláusula segunda") must never be
        # allowed to trigger a false chunk boundary just because line-wrap
        # happened to put it at the start of a line.
        compiled = []
        for type_name, spec in (config.get("markers") or {}).items():
            try:
                chunk_type = ChunkType(type_name)
            except ValueError as exc:
                raise ValueError(
                    f"Unknown chunk type {type_name!r} in config markers. "
                    f"Valid types: {[t.value for t in ChunkType]}"
                ) from exc
            compiled.append((chunk_type, re.compile(spec["pattern"])))
        return compiled

    @staticmethod
    def _compile_section_markers(config: dict) -> list[re.Pattern]:
        return [
            re.compile(spec["pattern"], re.IGNORECASE)
            for spec in (config.get("section_markers") or [])
        ]

    @staticmethod
    def _compile_optional(spec: dict | None) -> re.Pattern | None:
        if not spec:
            return None
        return re.compile(spec["pattern"], re.IGNORECASE)

    @staticmethod
    def _compile_sub_markers(config: dict) -> list[tuple[SplitReason, re.Pattern]]:
        compiled = []
        for spec in config.get("sub_markers") or []:
            try:
                reason = SplitReason(spec["name"])
            except ValueError as exc:
                raise ValueError(
                    f"Unknown sub-marker name {spec['name']!r}. "
                    f"Valid: {[r.value for r in SplitReason]}"
                ) from exc
            compiled.append((reason, re.compile(spec["pattern"], re.IGNORECASE)))
        return compiled

    @staticmethod
    def _match_section(line: str, patterns: list[re.Pattern]) -> str | None:
        for pattern in patterns:
            if pattern.match(line):
                return line.strip().lstrip("#").strip()
        return None

    @staticmethod
    def _match_chunk_marker(
        line: str, markers: list[tuple[ChunkType, re.Pattern]]
    ) -> ChunkType | None:
        for chunk_type, pattern in markers:
            if pattern.match(line):
                return chunk_type
        return None

    @staticmethod
    def _derive_label(full_text: str, max_len: int = 90) -> str:
        """
        Build a readable label from the assembled chunk text rather than
        just the trigger line — real PDF extraction routinely wraps a
        heading like "SEXTA (OBLIGACIONES DE LAS PARTES).-" across two
        lines, and a label taken from only the first line would be cut
        mid-word. Collapsing whitespace and taking a prefix is simple and
        robust to that.
        """
        collapsed = re.sub(r"\s+", " ", full_text).strip()
        if len(collapsed) <= max_len:
            return collapsed
        return collapsed[:max_len].rsplit(" ", 1)[0] + "…"

    def _build_chunk(
        self,
        *,
        document_id: str,
        section_id: str | None,
        chunk_type: ChunkType,
        label: str,
        order: int,
        full_text: str,
        extraction_confidence: ExtractionConfidence,
        crossref_pattern: re.Pattern | None,
        sub_marker_patterns: list[tuple[SplitReason, re.Pattern]],
        token_limit: int,
    ) -> Chunk:
        references_to = []
        if crossref_pattern:
            references_to = sorted(
                {
                    re.sub(r"\s+", " ", m.group(0)).strip()
                    for m in crossref_pattern.finditer(full_text)
                }
            )

        chunk = Chunk(
            document_id=document_id,
            section_id=section_id,
            chunk_type=chunk_type,
            label=label,
            order=order,
            full_text=full_text,
            extraction_confidence=extraction_confidence,
            references_to=references_to,
        )

        if chunk.token_count > token_limit:
            chunk.sub_chunks = self._split_chunk(chunk, sub_marker_patterns)

        return chunk

    @staticmethod
    def _split_chunk(
        chunk: Chunk, sub_marker_patterns: list[tuple[SplitReason, re.Pattern]]
    ) -> list[SubChunk]:
        """
        Split priority (ADR 2.4): try each configured sub-marker in order.
        Only fall back to a forced token-window split if nothing matches —
        and log that with SplitReason.VENTANA_FORZADA so a high rate of it
        signals "this jurisdiction's config needs better sub-markers".
        """
        lines = chunk.full_text.splitlines()

        for reason, pattern in sub_marker_patterns:
            boundaries = [i for i, line in enumerate(lines) if pattern.match(line)]
            if len(boundaries) >= 2:  # need at least 2 to produce a real split
                sub_chunks = []
                # Text BEFORE the first sub-marker (e.g. the clause heading and the
                # sentence introducing the numbered items — often the one that states
                # the price or the main obligation) must not be dropped: it becomes
                # its own leading sub-chunk. Regression-tested: sub-chunk text must
                # always reconstruct the parent's full text.
                lead_text = "\n".join(lines[: boundaries[0]]).strip()
                if lead_text:
                    sub_chunks.append(
                        SubChunk(
                            parent_chunk_id=chunk.chunk_id,
                            position_in_parent=0,
                            text=lead_text,
                            split_reason=SplitReason.PARRAFO,
                        )
                    )
                for start_idx, start in enumerate(boundaries):
                    end = boundaries[start_idx + 1] if start_idx + 1 < len(boundaries) else len(lines)
                    text = "\n".join(lines[start:end]).strip()
                    if text:
                        sub_chunks.append(
                            SubChunk(
                                parent_chunk_id=chunk.chunk_id,
                                position_in_parent=len(sub_chunks),
                                text=text,
                                split_reason=reason,
                            )
                        )
                if sub_chunks:
                    return sub_chunks

        # Forced window split, last resort.
        words = chunk.full_text.split()
        window = 400  # words per forced sub-chunk; conservative vs. the 500-token limit
        sub_chunks = []
        for idx, start in enumerate(range(0, len(words), window)):
            text = " ".join(words[start : start + window])
            sub_chunks.append(
                SubChunk(
                    parent_chunk_id=chunk.chunk_id,
                    position_in_parent=idx,
                    text=text,
                    split_reason=SplitReason.VENTANA_FORZADA,
                )
            )
        return sub_chunks

    @staticmethod
    def _attach_chunk(document: Document, section: Section | None, chunk: Chunk) -> None:
        if section is not None:
            section.chunks.append(chunk)
        else:
            document.chunks.append(chunk)
