from pathlib import Path

from src.common.logging import get_logger
from src.db_clients.pinecone_client import delete_vectors_by_filter
from src.db_clients.supabase_client import (
    delete_images_for_pdf,
    get_all_active_files,
    get_registry_entry,
    mark_file_inactive,
    upsert_registry_entry,
)
from src.common.hashing import file_hash
from src.pipeline.load.pinecone_loader import embed_and_upsert_chunks

logger = get_logger(__name__)


def pdf_hash(pdf_name: str, raw_dir: str) -> str:
    """Hash the source PDF on disk."""
    pdf_path = Path(raw_dir) / pdf_name
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    return file_hash(str(pdf_path))


def is_pdf_unchanged(pdf_name: str, raw_dir: str) -> bool:
    """True if the registry has an active entry whose hash matches the PDF on disk."""
    entry = get_registry_entry(pdf_name)
    if not entry or entry.get("status") != "active":
        return False
    return entry.get("file_hash") == pdf_hash(pdf_name, raw_dir)


def needs_reprocess_cleanup(source_pdf: str) -> bool:
    """
    True when a previous run already wrote artifacts for this PDF that must be
    cleared before re-ingesting (CHANGED file or crashed in-progress recovery).
    """
    entry = get_registry_entry(source_pdf)
    return bool(entry and entry.get("status") in ("active", "in-progress"))


def clear_pdf_artifacts(source_pdf: str) -> None:
    """Delete Pinecone vectors and Supabase images for a PDF being reprocessed."""
    delete_vectors_by_filter({"source_pdf": source_pdf})
    delete_images_for_pdf(source_pdf)


def mark_in_progress(source_pdf: str, current_hash: str, chunk_count: int | None = None) -> None:
    """
    Upsert the file_registry row as in-progress so image_registry inserts during
    extraction satisfy the foreign key.
    """
    payload = {"file_name": source_pdf, "file_hash": current_hash, "status": "in-progress"}
    if chunk_count is not None:
        payload["chunk_count"] = chunk_count
    upsert_registry_entry(payload)


def sync_file(
    source_pdf: str,
    chunks: list,
    current_hash: str,
    summary: dict,
    action: str = "new",
) -> None:
    """
    Embed chunks, upsert vectors to Pinecone, and flip file_registry to active.
    Caller must have already upserted file_registry as 'in-progress' and
    cleared any previous artifacts (vectors + images).
    """
    try:
        vector_count = embed_and_upsert_chunks(chunks, fhash=current_hash)
        upsert_registry_entry({
            "file_name":    source_pdf,
            "file_hash":    current_hash,
            "status":       "active",
            "chunk_count":  len(chunks),
            "vector_count": vector_count,
        })
        summary[action] += 1
        logger.info(f"Done: {source_pdf} - {vector_count} vectors indexed")
    except Exception:
        logger.exception(f"Ingestion failed for {source_pdf} - rolling back")
        rollback(source_pdf, current_hash)
        summary["errors"] += 1


def sweep_deleted(processed_pdfs: set[str], summary: dict) -> None:
    """Delete vectors + images and mark inactive for registry files no longer on disk."""
    active_registry = {e["file_name"]: e for e in get_all_active_files()}
    for file_name in active_registry:
        if file_name not in processed_pdfs:
            logger.info(f"DELETED (raw missing): {file_name}")
            _process_deleted_file(file_name, summary)


def _process_deleted_file(source_pdf: str, summary: dict) -> None:
    """Remove vectors + images and mark the registry entry inactive."""
    try:
        delete_vectors_by_filter({"source_pdf": source_pdf})
        delete_images_for_pdf(source_pdf)
        mark_file_inactive(source_pdf)
        summary["deleted"] += 1
    except Exception:
        logger.exception(f"Failed to clean up deleted file {source_pdf}")
        summary["errors"] += 1


def rollback(source_pdf: str, current_hash: str) -> None:
    """
    Best-effort cleanup after a failed pipeline run. Public so main.py can call it
    when extraction or chunking fails after the in-progress upsert.
    Leaves status=in-progress if delete fails so the next run can recover.
    """
    try:
        delete_vectors_by_filter({"source_pdf": source_pdf})
        delete_images_for_pdf(source_pdf)
        upsert_registry_entry({
            "file_name": source_pdf,
            "file_hash": current_hash,
            "status":    "inactive",
        })
    except Exception:
        logger.exception(
            f"Rollback failed for {source_pdf} - leaving status=in-progress for next-run recovery"
        )
