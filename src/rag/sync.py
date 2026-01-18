from pathlib import Path

from langchain_pinecone import PineconeVectorStore

from src.common.logging import get_logger
from src.db.pinecone_client import delete_vectors_by_filter, get_pinecone_index
from src.db.supabase_client import (
    get_all_active_files,
    get_registry_entry,
    mark_file_inactive,
    upsert_registry_entry,
)
from src.llm_adapters.embeddings.base import get_embeddings_model
from src.rag.ingestor import embed_and_upsert, file_hash, load_chunks_from_json

logger = get_logger(__name__)


def _pdf_hash(chunk_path: Path, raw_dir: str) -> str:
    """Hash the source PDF. Falls back to the chunk file if the PDF is missing."""
    source_pdf = chunk_path.stem.replace("_chunks", "") + ".pdf"
    pdf_path = Path(raw_dir) / source_pdf
    if pdf_path.exists():
        return file_hash(str(pdf_path))
    logger.warning(f"Source PDF not found for {chunk_path.name} - hashing chunk file instead")
    return file_hash(str(chunk_path))


def sync_folder(chunks_dir: str, raw_dir: str, index_name: str, namespace: str = "") -> dict:
    """
    Sync chunk files against Pinecone + Supabase registry.

    States per file:
      NEW          - not in registry, ingest
      UNCHANGED    - hash matches active entry, skip
      CHANGED      - hash differs from active entry, delete old vectors + re-ingest
      IN-PROGRESS  - previous run crashed mid-file, delete partial + re-ingest
      RECOVERED    - previously inactive, back on disk, re-ingest
      DELETED      - active in registry but chunk file gone, delete vectors + mark inactive
    """
    chunk_files = list(Path(chunks_dir).glob("*_chunks.json"))

    if not chunk_files:
        logger.error(f"No chunk files found in {chunks_dir} - aborting to prevent accidental mass deletion")
        return {"new": 0, "changed": 0, "unchanged": 0, "deleted": 0, "errors": 0}

    index = get_pinecone_index(index_name)
    embeddings = get_embeddings_model(model="text-embedding-3-small", model_provider="openai")
    vectorstore = PineconeVectorStore(index=index, embedding=embeddings, namespace=namespace)

    active_registry = {e["file_name"]: e["file_hash"] for e in get_all_active_files()}
    processed: set[str] = set()
    summary = {"new": 0, "changed": 0, "unchanged": 0, "deleted": 0, "errors": 0}

    for path in chunk_files:
        chunks = load_chunks_from_json(str(path))
        if not chunks:
            logger.warning(f"No chunks in {path.name} - skipping")
            continue

        source_pdf = chunks[0].metadata.get("source_pdf", path.stem.replace("_chunks", "") + ".pdf")
        processed.add(source_pdf)
        current_hash = _pdf_hash(path, raw_dir)

        entry = get_registry_entry(source_pdf)
        status = entry.get("status") if entry else None
        stored_hash = entry.get("file_hash") if entry else None

        if status == "active" and stored_hash == current_hash:
            logger.info(f"UNCHANGED: {source_pdf} - skipping")
            summary["unchanged"] += 1
            continue

        if status in ("active", "in-progress"):
            logger.info(f"{status.upper()}: {source_pdf} - deleting old vectors before re-ingestion")
            try:
                delete_vectors_by_filter({"source_pdf": source_pdf}, index_name=index_name, namespace=namespace)
            except Exception:
                logger.exception(f"Failed to delete old vectors for {source_pdf}")
                summary["errors"] += 1
                continue

        action = "changed" if status == "active" else "new"
        logger.info(f"{action.upper()}: upserting {len(chunks)} chunk(s) from {source_pdf}")

        upsert_registry_entry({
            "file_name": source_pdf,
            "file_hash": current_hash,
            "status": "in-progress",
            "chunk_count": len(chunks),
        })

        try:
            vector_count = embed_and_upsert(chunks, vectorstore=vectorstore, fhash=current_hash)
            upsert_registry_entry({
                "file_name": source_pdf,
                "file_hash": current_hash,
                "status": "active",
                "chunk_count": len(chunks),
                "vector_count": vector_count,
            })
            summary[action] += 1
            logger.info(f"Done: {source_pdf} - {vector_count} vectors indexed")
        except Exception:
            logger.exception(f"Ingestion failed for {source_pdf} - rolling back")
            try:
                delete_vectors_by_filter({"source_pdf": source_pdf}, index_name=index_name, namespace=namespace)
            except Exception:
                logger.exception(f"Rollback failed for {source_pdf} - index may contain partial vectors")
            upsert_registry_entry({"file_name": source_pdf, "file_hash": current_hash, "status": "inactive"})
            summary["errors"] += 1

    for file_name in active_registry:
        if file_name not in processed:
            logger.info(f"DELETED: {file_name} - removing vectors")
            try:
                delete_vectors_by_filter({"source_pdf": file_name}, index_name=index_name, namespace=namespace)
                mark_file_inactive(file_name)
                summary["deleted"] += 1
            except Exception:
                logger.exception(f"Failed to clean up deleted file {file_name}")
                summary["errors"] += 1

    logger.info(
        f"Sync complete - new={summary['new']} changed={summary['changed']} "
        f"unchanged={summary['unchanged']} deleted={summary['deleted']} errors={summary['errors']}"
    )
    return summary
