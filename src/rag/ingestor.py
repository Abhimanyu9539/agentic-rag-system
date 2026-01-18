import hashlib
import json
from pathlib import Path

from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore

from src.common.logging import get_logger
from src.config.constants import BATCH_SIZE

logger = get_logger(__name__)


def load_chunks_from_json(chunks_path: str) -> list[Document]:
    with open(chunks_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return [Document(page_content=item["text"], metadata=item["metadata"]) for item in raw]


def file_hash(path: str) -> str:
    """SHA-256 (16-char prefix) of any file - used for change detection and ID derivation."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def _make_id(fhash: str, chunk_index: int) -> str:
    """Deterministic vector ID: file hash + chunk index."""
    return hashlib.sha256(f"{fhash}::{chunk_index}".encode()).hexdigest()[:32]


def _sanitize_metadata(doc: Document) -> Document:
    """Pinecone requires list-type metadata to be list[str] - convert pages."""
    meta = dict(doc.metadata)
    if "pages" in meta:
        meta["pages"] = [str(p) for p in meta["pages"]]
    return Document(page_content=doc.page_content, metadata=meta)


def embed_and_upsert(
    chunks: list[Document],
    vectorstore: PineconeVectorStore,
    fhash: str,
) -> int:
    chunks = [_sanitize_metadata(c) for c in chunks]
    ids = [_make_id(fhash, c.metadata.get("chunk_index", i)) for i, c in enumerate(chunks)]
    total_batches = max(1, -(-len(chunks) // BATCH_SIZE))

    for batch_num, i in enumerate(range(0, len(chunks), BATCH_SIZE), start=1):
        batch_chunks = chunks[i : i + BATCH_SIZE]
        batch_ids    = ids[i : i + BATCH_SIZE]
        vectorstore.add_documents(documents=batch_chunks, ids=batch_ids)
        logger.info(f"Batch {batch_num}/{total_batches} upserted ({len(batch_chunks)} chunks)")

    logger.info(f"Upserted {len(chunks)} chunks total")
    return len(chunks)
