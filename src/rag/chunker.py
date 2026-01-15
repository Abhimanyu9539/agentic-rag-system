import json
import os
import re

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

# OpenDataLoader embeds this pattern in the markdown file (Java substitutes the page number).
_ODL_PAGE_PATTERN = re.compile(r"<<<ODL_PAGE_BREAK_(\d+)>>>")
# Working sentinel used inside this module during splitting; stripped before final output.
_SENTINEL_RE      = re.compile(r"<!-- page:(\d+) -->")


# ---------------------------------------------------------------------------
# Step 1 — Load markdown with page markers
# ---------------------------------------------------------------------------

def load_markdown_with_page_markers(md_path: str) -> str:
    """
    Read the .md file saved by parse_pdf and replace the OpenDataLoader page
    separators with <!-- page:N --> sentinels so that page numbers can be
    recovered from each chunk after splitting.
    """
    with open(md_path, "r", encoding="utf-8") as f:
        content = f.read()
    return _ODL_PAGE_PATTERN.sub(lambda m: f"<!-- page:{m.group(1)} -->", content)


# ---------------------------------------------------------------------------
# Step 2 — Header-aware split
# ---------------------------------------------------------------------------

def split_by_headers(markdown: str) -> list[Document]:
    """
    Split on # / ## / ### boundaries.  Each resulting Document carries heading
    context (h1, h2, h3) in its metadata — the only way to preserve section
    information without post-hoc parsing.

    If the markdown has no heading markers the splitter returns the whole text
    as a single Document; split_chunks will still subdivide it by size.
    """
    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")],
        strip_headers=False,
    )
    return splitter.split_text(markdown)


# ---------------------------------------------------------------------------
# Step 3 — Size-based split
# ---------------------------------------------------------------------------

def split_chunks(
    header_docs: list[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> list[Document]:
    """Subdivide any header-split Document that still exceeds chunk_size."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
    )
    return splitter.split_documents(header_docs)


# ---------------------------------------------------------------------------
# Step 4 — Page tracking helpers
# ---------------------------------------------------------------------------

def _extract_pages(chunk_text: str, last_page: int = 1) -> list[int]:
    """Return sorted, deduplicated page numbers from sentinels in chunk_text."""
    found = [int(m) for m in _SENTINEL_RE.findall(chunk_text)]
    return sorted(set(found)) if found else [last_page]


def _strip_sentinels(text: str) -> str:
    return re.sub(r"\n?<!-- page:\d+ -->\n?", "\n", text).strip()


# ---------------------------------------------------------------------------
# Step 5 — Table index
# ---------------------------------------------------------------------------

def build_page_table_index(table_metadata: list[dict]) -> dict[int, list[str]]:
    """
    Build {page_number: [image_path, ...]} from the list produced by
    fetch_table.extract_tables_as_images / extract_tables_from_folder.
    """
    index: dict[int, list[str]] = {}
    for entry in table_metadata:
        img = entry.get("image_path", "")
        for page in entry.get("pages", []):
            index.setdefault(page, []).append(img)
    return index


# ---------------------------------------------------------------------------
# Step 6 — Associate tables + finalise metadata
# ---------------------------------------------------------------------------

def associate_tables(
    chunks: list[Document],
    page_table_index: dict[int, list[str]],
    source_pdf: str,
) -> list[Document]:
    """
    Attach source_pdf, chunk_index, pages, section, and table_images to each
    chunk's metadata, then strip page sentinels from page_content.
    """
    last_page = 1
    result: list[Document] = []

    for i, chunk in enumerate(chunks):
        pages = _extract_pages(chunk.page_content, last_page)
        last_page = pages[-1]

        # Collect image paths for all pages (deduplicated, stable order)
        seen: set[str] = set()
        images: list[str] = []
        for p in pages:
            for img in page_table_index.get(p, []):
                if img not in seen:
                    seen.add(img)
                    images.append(img)

        # Deepest non-empty heading key becomes the section label
        hm = chunk.metadata or {}
        section = hm.get("h3") or hm.get("h2") or hm.get("h1") or ""

        result.append(Document(
            page_content=_strip_sentinels(chunk.page_content),
            metadata={
                "source_pdf":   source_pdf,
                "pages":        pages,
                "section":      section,
                "chunk_index":  i,
                "table_images": images,
            },
        ))

    return result


# ---------------------------------------------------------------------------
# Top-level orchestrator
# ---------------------------------------------------------------------------

def chunk_pdf(
    md_path: str,
    table_metadata: list[dict],
    source_pdf: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> list[Document]:
    """
    Full pipeline: markdown file → header split → size split → table association.

    Returns LangChain Documents ready for embedding and Pinecone ingestion.
    Each Document metadata has: source_pdf, pages, section, chunk_index, table_images.
    """
    markdown    = load_markdown_with_page_markers(md_path)
    header_docs = split_by_headers(markdown)
    chunks      = split_chunks(header_docs, chunk_size, chunk_overlap)
    page_index  = build_page_table_index(table_metadata)
    return associate_tables(chunks, page_index, source_pdf)


def save_chunks(chunks: list[Document], output_path: str) -> None:
    """Persist chunks as JSON for inspection before embedding."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    payload = [{"text": c.page_content, "metadata": c.metadata} for c in chunks]
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"📦 {len(chunks)} chunks saved → {output_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from collections import defaultdict as _dd

    parsed_dir    = r"E:\LLMOps\agentic-rag-system\data\output\parsed"
    metadata_path = r"E:\LLMOps\agentic-rag-system\data\output\tables\table_metadata.json"
    chunks_dir    = r"E:\LLMOps\agentic-rag-system\data\output\chunks"

    with open(metadata_path, "r", encoding="utf-8") as f:
        all_meta = json.load(f)

    # Group table metadata by source PDF
    by_pdf: dict = _dd(list)
    for entry in all_meta:
        by_pdf[entry["source_pdf"]].append(entry)

    for source_pdf, table_meta in by_pdf.items():
        stem    = os.path.splitext(source_pdf)[0]
        md_path = os.path.join(parsed_dir, stem, stem + ".md")
        if not os.path.exists(md_path):
            print(f"⚠️  Markdown not found for {source_pdf}, skipping")
            continue

        chunks = chunk_pdf(md_path, table_meta, source_pdf)
        save_chunks(chunks, os.path.join(chunks_dir, stem + "_chunks.json"))
