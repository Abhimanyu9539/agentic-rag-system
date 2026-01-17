import hashlib
import json
from pathlib import Path

from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore

from ..common.logging import get_logger
from ..config.constants import _BATCH_SIZE
from ..llm_adapters.embeddings.base import get_embeddings_model
from ..db.pinecone_client import delete_vectors_by_filter, get_pinecone_index

logger = get_logger(__name__)


def load_chunks_from_json(chunks_path: str) -> list[Document]:
    with open(chunks_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return [Document(page_content=item["text"], metadata=item["metadata"]) for item in raw]


def _file_hash(chunks_path: str) -> str:
    """SHA-256 of the chunk file's raw bytes — stable identity for its source PDF."""
    return hashlib.sha256(Path(chunks_path).read_bytes()).hexdigest()[:16]


def _make_id(file_hash: str, chunk_index: int) -> str:
    """Deterministic vector ID: file content hash + chunk index.
    Same file → same IDs (re-run overwrites). Different files, same name → different IDs.
    """
    key = f"{file_hash}::{chunk_index}"
    return hashlib.sha256(key.encode()).hexdigest()[:32]


def _sanitize_metadata(doc: Document) -> Document:
    """Pinecone requires list-type metadata to be list[str] — convert pages."""
    meta = dict(doc.metadata)
    if "pages" in meta:
        meta["pages"] = [str(p) for p in meta["pages"]]
    return Document(page_content=doc.page_content, metadata=meta)


def embed_and_upsert(
    chunks: list[Document],
    index,
    vectorstore: PineconeVectorStore,
    file_hash: str,
    namespace: str = "",
) -> int:
    chunks = [_sanitize_metadata(c) for c in chunks]
    total = skipped = 0

    for i in range(0, len(chunks), _BATCH_SIZE):
        batch = chunks[i : i + _BATCH_SIZE]
        ids   = [_make_id(file_hash, c.metadata.get("chunk_index", i + j)) for j, c in enumerate(batch)]

        existing = set(index.fetch(ids=ids, namespace=namespace).vectors.keys())
        new_pairs = [(c, id_) for c, id_ in zip(batch, ids) if id_ not in existing]

        if not new_pairs:
            skipped += len(batch)
            logger.info("Batch %d–%d already in Pinecone, skipping OpenAI call", i + 1, i + len(batch))
            continue

        new_docs, new_ids = zip(*new_pairs)
        vectorstore.add_documents(documents=list(new_docs), ids=list(new_ids))
        total   += len(new_docs)
        skipped += len(batch) - len(new_docs)
        logger.info("Upserted %d new, skipped %d existing (batch %d–%d)", len(new_docs), len(batch) - len(new_docs), i + 1, i + len(batch))

    logger.info("Done — %d upserted, %d skipped (already existed)", total, skipped)
    return total


def ingest_folder(chunks_dir: str, index_name: str, namespace: str = "") -> int:
    chunk_files = list(Path(chunks_dir).glob("*_chunks.json"))
    logger.info("Found %d chunk file(s) in %s", len(chunk_files), chunks_dir)

    index = get_pinecone_index(index_name)
    embeddings = get_embeddings_model(model="text-embedding-3-small", model_provider="openai")
    vectorstore = PineconeVectorStore(index=index, embedding=embeddings, namespace=namespace)

    total = 0
    for path in chunk_files:
        logger.info("Loading chunks from %s", path.name)
        chunks = load_chunks_from_json(str(path))
        if not chunks:
            logger.warning("No chunks in %s, skipping", path.name)
            continue
        fhash = _file_hash(str(path))
        source_pdf = chunks[0].metadata.get("source_pdf", path.name)
        logger.info("Embedding and upserting %d chunk(s) from %s (hash=%s)", len(chunks), path.name, fhash)
        try:
            total += embed_and_upsert(chunks, index=index, vectorstore=vectorstore, file_hash=fhash, namespace=namespace)
        except Exception:
            logger.exception("Ingestion failed for %s — rolling back all vectors for this file", source_pdf)
            try:
                delete_vectors_by_filter({"source_pdf": source_pdf}, index_name=index_name, namespace=namespace)
            except Exception:
                logger.exception("Rollback also failed for %s — index may contain partial vectors", source_pdf)
    return total
