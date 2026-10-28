"""Stage 1 of ingestion: PDF -> clean text lines with page numbers.

PyMuPDF extracts the text of each page. Cleaning then:
  - collapses whitespace,
  - drops page footers ("Page 3 of 8"),
  - drops running headers/footers, detected as lines that repeat on most
    pages -- generic, so it works for any policy document layout.
Page numbers are kept on every line so chunks can cite the page they came from.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pymupdf

_PAGE_FOOTER = re.compile(r"^page \d+( of \d+)?$", re.IGNORECASE)


@dataclass(frozen=True)
class PageLine:
    page: int
    text: str


def extract_lines(pdf_path: str | Path) -> list[PageLine]:
    with pymupdf.open(pdf_path) as document:
        pages = [page.get_text("text").splitlines() for page in document]

    cleaned_pages = [[" ".join(raw.split()) for raw in page] for page in pages]
    repeated = _repeated_lines(cleaned_pages)

    lines: list[PageLine] = []
    for page_number, page in enumerate(cleaned_pages, start=1):
        for text in page:
            if text and text not in repeated and not _PAGE_FOOTER.match(text):
                lines.append(PageLine(page=page_number, text=text))
    return lines


def _repeated_lines(pages: list[list[str]]) -> set[str]:
    """Lines present on more than half the pages are headers/footers, not content."""
    if len(pages) < 3:
        return set()
    counts = Counter(line for page in pages for line in set(page) if line)
    return {line for line, count in counts.items() if count > len(pages) / 2}
