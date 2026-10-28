"""Stage 2 of ingestion: text lines -> chunks.

Two strategies, so they can be compared with the retrieval eval:

`chunk_by_structure` (used by the app) follows the document's own numbering:
    "4. Exclusions"                 -> a section  (parent chunk)
    "4.2 Racing and Speed Testing"  -> a clause   (child chunk)
Each clause becomes one chunk, keeping its section, page and a stable id
("AUTO-GOLD-V1-4.2") that the LLM can cite. Retrieval matches the small,
precise clause and hands the LLM the whole parent section as context
(parent-child / small-to-big retrieval). Over-long clauses are split on
sentence boundaries so no chunk exceeds `max_chars`.

`chunk_fixed_size` is the naive baseline: fixed-size character windows with
overlap, ignoring structure. Only the chunking experiment uses it.
"""

from __future__ import annotations

import hashlib
import re

from app.domain.models import PolicyClause, PolicySection
from app.ingestion.pdf_parser import PageLine

# A heading is a number, then a short title starting with a capital letter.
_SECTION = re.compile(r"^(\d+)\.\s+([A-Z].{0,80})$")  # "4. Exclusions"
_CLAUSE = re.compile(r"^(\d+)\.(\d+)\s+([A-Z].{0,80})$")  # "4.2 Racing and Speed Testing"


def embedding_text(section_title: str, title: str, text: str) -> str:
    """What gets embedded: the clause plus its section title as a contextual
    header, so "4.2 Racing" also carries the meaning "this is an exclusion"."""
    header = f"{section_title} > {title}" if section_title else title
    return f"{header}\n{text}"


def chunk_hash(section_title: str, title: str, text: str) -> str:
    """Hash of exactly what is embedded -- the key of the embedding cache."""
    return hashlib.sha256(embedding_text(section_title, title, text).encode()).hexdigest()[:16]


def chunk_by_structure(
    lines: list[PageLine], *, product_version_id: str, max_chars: int
) -> tuple[list[PolicySection], list[PolicyClause]]:
    sections: list[dict] = []  # {number, title, page, intro: [str], clauses: [{number, title, page, lines}]}

    for line in lines:
        if match := _CLAUSE.match(line.text):
            if sections:
                sections[-1]["clauses"].append(
                    {"number": f"{match[1]}.{match[2]}", "title": line.text, "page": line.page, "lines": []}
                )
                continue
        if match := _SECTION.match(line.text):
            sections.append({"number": match[1], "title": line.text, "page": line.page, "intro": [], "clauses": []})
            continue
        if not sections:
            continue  # cover page / preamble before the first numbered section
        current = sections[-1]
        (current["clauses"][-1]["lines"] if current["clauses"] else current["intro"]).append(line.text)

    parent_sections: list[PolicySection] = []
    clauses: list[PolicyClause] = []
    for section in sections:
        section_id = f"{product_version_id}-S{section['number']}"
        intro = " ".join(section["intro"])
        body = [intro] if intro else []
        body += [f"{clause['title']}: {' '.join(clause['lines'])}" for clause in section["clauses"]]
        parent_sections.append(
            PolicySection(
                section_id=section_id,
                product_version_id=product_version_id,
                title=section["title"],
                text="\n".join(body),
                page=section["page"],
            )
        )

        # A section with no numbered clauses is retrievable as one chunk on its own.
        children = section["clauses"] or [
            {"number": section["number"], "title": section["title"], "page": section["page"], "lines": [intro]}
        ]
        for clause in children:
            parts = _split_long_text(" ".join(clause["lines"]), max_chars)
            for index, part in enumerate(parts, start=1):
                suffix = f"-p{index}" if len(parts) > 1 else ""
                clauses.append(
                    PolicyClause(
                        clause_id=f"{product_version_id}-{clause['number']}{suffix}",
                        product_version_id=product_version_id,
                        section_id=section_id,
                        title=clause["title"],
                        text=part,
                        page=clause["page"],
                        chunk_hash=chunk_hash(section["title"], clause["title"], part),
                    )
                )
    return parent_sections, clauses


def chunk_fixed_size(
    lines: list[PageLine], *, product_version_id: str, size: int = 800, overlap: int = 150
) -> list[PolicyClause]:
    """Baseline: slide a fixed-size window over the full text, ignoring structure."""
    text, page_at = "", []  # page_at[i] = page of character i
    for line in lines:
        piece = line.text + " "
        text += piece
        page_at += [line.page] * len(piece)

    chunks: list[PolicyClause] = []
    start = 0
    while start < len(text):
        window = text[start : start + size].strip()
        title = f"Chunk {len(chunks) + 1}"
        chunks.append(
            PolicyClause(
                clause_id=f"{product_version_id}-F{len(chunks) + 1}",
                product_version_id=product_version_id,
                section_id="",
                title=title,
                text=window,
                page=page_at[start],
                chunk_hash=chunk_hash("", title, window),
            )
        )
        start += size - overlap
    return chunks


def _split_long_text(text: str, max_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    parts: list[str] = []
    current = ""
    for sentence in re.split(r"(?<=[.;])\s+", text):
        if current and len(current) + len(sentence) + 1 > max_chars:
            parts.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:
        parts.append(current)
    return parts
