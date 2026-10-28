"""Render the policy wording sources (data/policies/src/*.txt) into PDFs.

The PDFs are the input to the ingestion pipeline. Each page gets a running
header and a "Page n of N" footer, like a real policy document, so the
parser's cleaning step has real noise to remove.

Usage:
    python scripts/make_policy_pdfs.py
"""

import re
from pathlib import Path

import pymupdf

POLICY_DIR = Path(__file__).resolve().parent.parent / "data" / "policies"
HEADING = re.compile(r"^\d+(\.\d+)?\s+[A-Z]")

PAGE_WIDTH, PAGE_HEIGHT = pymupdf.paper_size("a4")
MARGIN = 85
FONT_SIZE = 12
LINE_HEIGHT = 21
TEXT_WIDTH = PAGE_WIDTH - 2 * MARGIN


def wrap(text: str, font: str) -> list[str]:
    lines, current = [], ""
    for word in text.split():
        candidate = f"{current} {word}".strip()
        if pymupdf.get_text_length(candidate, fontname=font, fontsize=FONT_SIZE) > TEXT_WIDTH and current:
            lines.append(current)
            current = word
        else:
            current = candidate
    return lines + [current] if current else lines


def layout(source: str) -> tuple[str, list[list[tuple[str, str]]]]:
    """Returns (document title, pages) where a page is a list of (font, line)."""
    paragraphs = source.strip().split("\n")
    title = paragraphs[0]
    rows: list[tuple[str, str]] = []
    for paragraph in paragraphs:
        if not paragraph.strip():
            rows.append(("helv", ""))
            continue
        font = "hebo" if HEADING.match(paragraph) or paragraph == title else "helv"
        rows += [(font, line) for line in wrap(paragraph, font)]

    rows_per_page = int((PAGE_HEIGHT - 2 * MARGIN) // LINE_HEIGHT)
    pages = [rows[i : i + rows_per_page] for i in range(0, len(rows), rows_per_page)]
    return title, pages


def render(source_path: Path) -> Path:
    title, pages = layout(source_path.read_text(encoding="utf-8"))
    document = pymupdf.open()
    for number, rows in enumerate(pages, start=1):
        page = document.new_page(width=PAGE_WIDTH, height=PAGE_HEIGHT)
        page.insert_text((MARGIN, MARGIN - 30), f"{title} - Policy Wording", fontname="helv", fontsize=8)
        page.insert_text((MARGIN, PAGE_HEIGHT - MARGIN + 40), f"Page {number} of {len(pages)}", fontsize=8)
        y = MARGIN
        for font, line in rows:
            page.insert_text((MARGIN, y), line, fontname=font, fontsize=FONT_SIZE)
            y += LINE_HEIGHT
    output = POLICY_DIR / f"{source_path.stem}.pdf"
    document.save(output)
    print(f"{output.name}: {len(pages)} pages")
    return output


if __name__ == "__main__":
    for source in sorted((POLICY_DIR / "src").glob("*.txt")):
        render(source)
