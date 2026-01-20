import hashlib

from langchain_core.documents import Document

from src.common.logging import get_logger

logger = get_logger(__name__)


def _make_id(fhash: str, chunk_index: int) -> str:
    """Deterministic vector ID: file hash + chunk index."""
    return hashlib.sha256(f"{fhash}::{chunk_index}".encode()).hexdigest()[:32]


def _sanitize_metadata(doc: Document) -> Document:
    """Pinecone requires list-type metadata to be list[str] - convert pages."""
    meta = dict(doc.metadata)
    if "pages" in meta:
        meta["pages"] = [str(p) for p in meta["pages"]]
    return Document(page_content=doc.page_content, metadata=meta)


def prepare_chunks_for_pinecone(
    chunks: list[Document],
    fhash: str,
) -> tuple[list[Document], list[str]]:
    """
    Sanitize chunk metadata and assign deterministic vector IDs.
    Returns (chunks, ids) ready to be handed to the load stage.
    """
    chunks = [_sanitize_metadata(c) for c in chunks]
    ids = [_make_id(fhash, c.metadata.get("chunk_index", i)) for i, c in enumerate(chunks)]
    return chunks, ids
