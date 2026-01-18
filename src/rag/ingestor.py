import hashlib
import json
from pathlib import Path

from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore

from src.common.logging import get_logger
from src.config.constants import _BATCH_SIZE
from src.llm_adapters.embeddings.base import get_embeddings_model
from src.db.pinecone_client import delete_vectors_by_filter, get_pinecone_index

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
    total_batches = max(1, -(-len(chunks) // _BATCH_SIZE))

    for batch_num, i in enumerate(range(0, len(chunks), _BATCH_SIZE), start=1):
        batch_chunks = chunks[i : i + _BATCH_SIZE]
        batch_ids    = ids[i : i + _BATCH_SIZE]
        vectorstore.add_documents(documents=batch_chunks, ids=batch_ids)
        logger.info(f"Batch {batch_num}/{total_batches} upserted ({len(batch_chunks)} chunks)")

    logger.info(f"Upserted {len(chunks)} chunks total")
    return len(chunks)


def ingest_folder(chunks_dir: str, index_name: str, namespace: str = "") -> int:
    """Legacy entry point - no registry, no change detection. Use sync_folder for production."""
    chunk_files = list(Path(chunks_dir).glob("*_chunks.json"))
    logger.info(f"Found {len(chunk_files)} chunk file(s) in {chunks_dir}")

    if not chunk_files:
        logger.error(f"No chunk files found in {chunks_dir} - aborting")
        return 0

    index = get_pinecone_index(index_name)
    embeddings = get_embeddings_model(model="text-embedding-3-small", model_provider="openai")
    vectorstore = PineconeVectorStore(index=index, embedding=embeddings, namespace=namespace)

    total = 0
    for path in chunk_files:
        chunks = load_chunks_from_json(str(path))
        if not chunks:
            logger.warning(f"No chunks in {path.name}, skipping")
            continue
        source_pdf = chunks[0].metadata.get("source_pdf", path.name)
        fhash = file_hash(str(path))
        logger.info(f"Upserting {len(chunks)} chunk(s) from {path.name}")
        try:
            total += embed_and_upsert(chunks, vectorstore=vectorstore, fhash=fhash)
        except Exception:
            logger.exception(f"Ingestion failed for {source_pdf} - rolling back")
            try:
                delete_vectors_by_filter({"source_pdf": source_pdf}, index_name=index_name, namespace=namespace)
            except Exception:
                logger.exception(f"Rollback also failed for {source_pdf} - index may contain partial vectors")
    return total
