import json
import os
import re

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from src.common.logging import get_logger
from src.config.constants import CHUNK_OVERLAP, CHUNK_SIZE, ODL_PAGE_PATTERN, SENTINEL_RE

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Step 1 — Header-aware split
# ---------------------------------------------------------------------------

def split_by_headers(markdown: str) -> list[Document]:
    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")],
        strip_headers=False,
    )
    try:
        docs = splitter.split_text(markdown)
    except Exception as e:
        logger.error(f"Header splitting failed: {e}")
        raise
    logger.debug(f"Header split produced {len(docs)} section(s)")
    return docs


# ---------------------------------------------------------------------------
# Step 2 — Size-based split
# ---------------------------------------------------------------------------

def split_chunks(
    header_docs: list[Document],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
    )
    try:
        chunks = splitter.split_documents(header_docs)
    except Exception as e:
        logger.error(f"Size splitting failed: {e}")
        raise
    logger.debug(f"Size split produced {len(chunks)} chunk(s) (size={chunk_size}, overlap={chunk_overlap})")
    return chunks


# ---------------------------------------------------------------------------
# Step 3 — Page tracking helpers
# ---------------------------------------------------------------------------

def _extract_pages(chunk_text: str, last_page: int = 1) -> list[int]:
    """Return sorted, deduplicated page numbers from sentinels in chunk_text."""
    found = [int(m) for m in SENTINEL_RE.findall(chunk_text)]
    return sorted(set(found)) if found else [last_page]


def _strip_sentinels(text: str) -> str:
    return re.sub(r"\n?<!-- page:\d+ -->\n?", "\n", text).strip()


# ---------------------------------------------------------------------------
# Step 4 — Table index
# ---------------------------------------------------------------------------

def build_page_table_index(table_metadata: list[dict]) -> dict[int, list[str]]:
    index: dict[int, list[str]] = {}
    for entry in table_metadata:
        img = entry.get("image_path", "")
        for page in entry.get("pages", []):
            index.setdefault(page, []).append(img)
    logger.debug(f"Built page-table index for {len(index)} page(s)")
    return index


# ---------------------------------------------------------------------------
# Step 5 — Associate tables + finalise metadata
# ---------------------------------------------------------------------------

def associate_tables(
    chunks: list[Document],
    page_table_index: dict[int, list[str]],
    source_pdf: str,
) -> list[Document]:
    last_page = 1
    result: list[Document] = []

    for i, chunk in enumerate(chunks):
        pages = _extract_pages(chunk.page_content, last_page)
        last_page = pages[-1]

        seen: set[str] = set()
        images: list[str] = []
        for p in pages:
            for img in page_table_index.get(p, []):
                if img not in seen:
                    seen.add(img)
                    images.append(img)

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

    logger.debug(f"Associated tables for {len(result)} chunk(s) from {source_pdf}")
    return result


# ---------------------------------------------------------------------------
# Top-level orchestrator
# ---------------------------------------------------------------------------

def chunk_pdf(
    markdown: str,
    table_metadata: list[dict],
    source_pdf: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[Document]:
    logger.info(f"Chunking {source_pdf} (chunk_size={chunk_size}, overlap={chunk_overlap})")
    try:
        processed   = ODL_PAGE_PATTERN.sub(lambda m: f"<!-- page:{m.group(1)} -->", markdown)
        header_docs = split_by_headers(processed)
        chunks      = split_chunks(header_docs, chunk_size, chunk_overlap)
        page_index  = build_page_table_index(table_metadata)
        result      = associate_tables(chunks, page_index, source_pdf)
    except Exception as e:
        logger.error(f"Chunking failed for {source_pdf}: {e}")
        raise
    logger.info(f"Produced {len(result)} chunk(s) for {source_pdf}")
    return result


def save_chunks(chunks: list[Document], output_path: str) -> None:
    """Persist chunks to JSON. Available for debug/standalone use."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    payload = [{"text": c.page_content, "metadata": c.metadata} for c in chunks]
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Failed to save chunks to {output_path}: {e}")
        raise
    logger.info(f"{len(chunks)} chunk(s) saved -> {output_path}")
