"""Parse corpus source files (PDF, HTML) into LlamaIndex Documents.

One source document becomes several Documents — one per PDF page or per HTML
section — so a page/section locator survives into every retrieval chunk and can
later be used as a citation target.
"""

from __future__ import annotations

import csv
from pathlib import Path

from bs4 import BeautifulSoup
from llama_index.core import Document
from pypdf import PdfReader

from clinrag.config import PATHS

# HTML elements that never carry guideline content.
_HTML_NOISE = ["script", "style", "nav", "header", "footer", "aside", "form", "noscript"]

# Sections/pages shorter than this are dropped as boilerplate.
_MIN_TEXT_LEN = 40

# Metadata kept out of the embedded text (noise) and the LLM prompt context.
_EXCLUDE_EMBED = ["doc_id", "url", "pub_date", "publisher"]
_EXCLUDE_LLM = ["url", "pub_date", "publisher"]


def _base_metadata(row: dict[str, str]) -> dict[str, str]:
    return {
        "doc_id": row["doc_id"],
        "title": row["title"],
        "source": row["source"],
        "publisher": row["publisher"],
        "pub_date": row["pub_date"],
        "url": row["url"],
    }


def _make_doc(text: str, metadata: dict[str, str]) -> Document:
    doc = Document(text=text, metadata=metadata)
    doc.excluded_embed_metadata_keys = _EXCLUDE_EMBED
    doc.excluded_llm_metadata_keys = _EXCLUDE_LLM
    return doc


def parse_pdf(path: Path, row: dict[str, str]) -> list[Document]:
    """One Document per page, with a 'p. N' locator."""
    reader = PdfReader(str(path))
    docs: list[Document] = []
    for page_num, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if len(text) < _MIN_TEXT_LEN:
            continue
        meta = _base_metadata(row) | {"location": f"p. {page_num}"}
        docs.append(_make_doc(text, meta))
    return docs


def parse_html(path: Path, row: dict[str, str]) -> list[Document]:
    """One Document per heading-delimited section."""
    html = path.read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(_HTML_NOISE):
        tag.decompose()
    root = soup.find("main") or soup.body or soup

    # Walk content in document order, opening a new section at each heading.
    sections: list[tuple[str, list[str]]] = [("Overview", [])]
    for el in root.find_all(["h1", "h2", "h3", "p", "li"]):
        text = el.get_text(" ", strip=True)
        if not text:
            continue
        if el.name in ("h1", "h2", "h3"):
            sections.append((text, []))
        else:
            sections[-1][1].append(text)

    docs: list[Document] = []
    for heading, body in sections:
        text = "\n".join(body).strip()
        if len(text) < _MIN_TEXT_LEN:
            continue
        meta = _base_metadata(row) | {"location": f"Section: {heading}"}
        docs.append(_make_doc(text, meta))
    return docs


def load_corpus() -> list[Document]:
    """Parse every available document listed in the corpus manifest."""
    with PATHS.manifest.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    documents: list[Document] = []
    for row in rows:
        path = PATHS.root / row["local_path"]
        if not path.exists():
            print(f"  skip {row['doc_id']}: file missing ({row['local_path']})")
            continue

        if row["format"] == "pdf":
            parsed = parse_pdf(path, row)
        elif row["format"] == "html":
            parsed = parse_html(path, row)
        else:
            print(f"  skip {row['doc_id']}: unsupported format '{row['format']}'")
            continue

        print(f"  {row['doc_id']}: {len(parsed)} sections/pages")
        documents.extend(parsed)
    return documents
