from pathlib import Path

from src.common.logging import get_logger
from src.db_clients.pinecone_client import delete_vectors_by_filter
from src.db_clients.supabase_client import (
    get_all_active_files,
    get_registry_entry,
    mark_file_inactive,
    upsert_registry_entry,
)
from src.rag.ingestor import embed_and_upsert_chunks, file_hash

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


def sync_file(source_pdf: str, chunks: list, current_hash: str, summary: dict) -> None:
    """
    Sync a single file's chunks into Pinecone with registry tracking.

    States handled:
      NEW         - not in registry or inactive, ingest
      CHANGED     - hash differs from active entry, delete old vectors + re-ingest
      IN-PROGRESS - previous run crashed mid-file, delete partial + re-ingest
      UNCHANGED   - hash matches active entry (guard; main.py pre-filters these)
    """
    entry       = get_registry_entry(source_pdf)
    status      = entry.get("status") if entry else None
    stored_hash = entry.get("file_hash") if entry else None

    if status == "active" and stored_hash == current_hash:
        logger.info(f"UNCHANGED: {source_pdf} - skipping")
        summary["unchanged"] += 1

    elif status in ("active", "in-progress"):
        logger.info(f"CHANGED: {source_pdf}")
        _process_changed_file(source_pdf, chunks, current_hash, summary)

    else:
        logger.info(f"NEW: {source_pdf}")
        _process_new_file(source_pdf, chunks, current_hash, summary)


def sweep_deleted(processed_pdfs: set[str], summary: dict) -> None:
    """Delete vectors and mark inactive for registry files no longer on disk."""
    active_registry = {e["file_name"]: e for e in get_all_active_files()}
    for file_name, entry in active_registry.items():
        if file_name not in processed_pdfs:
            logger.info(f"DELETED (raw missing): {file_name}")
            _process_deleted_file(file_name, entry, summary)


def _process_new_file(
    source_pdf: str,
    chunks: list,
    current_hash: str,
    summary: dict,
    action: str = "new",
) -> None:
    """Embed and register a new (or recovered) file. Uses two-phase commit for crash safety."""
    upsert_registry_entry({
        "file_name": source_pdf,
        "file_hash": current_hash,
        "status": "in-progress",
        "chunk_count": len(chunks),
    })
    try:
        vector_count = embed_and_upsert_chunks(chunks, fhash=current_hash)
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
        _rollback(source_pdf, current_hash)
        summary["errors"] += 1


def _process_changed_file(
    source_pdf: str,
    chunks: list,
    current_hash: str,
    summary: dict,
) -> None:
    """Delete stale vectors then re-ingest the updated file."""
    try:
        delete_vectors_by_filter({"source_pdf": source_pdf})
    except Exception:
        logger.exception(f"Failed to delete old vectors for {source_pdf}")
        summary["errors"] += 1
        return

    _process_new_file(source_pdf, chunks, current_hash, summary, action="changed")


def _process_deleted_file(
    source_pdf: str,
    entry: dict | None,
    summary: dict,
) -> None:
    """Remove vectors and mark the registry entry inactive."""
    try:
        delete_vectors_by_filter({"source_pdf": source_pdf})
        if entry:
            mark_file_inactive(source_pdf)
        summary["deleted"] += 1
    except Exception:
        logger.exception(f"Failed to clean up deleted file {source_pdf}")
        summary["errors"] += 1


def _rollback(source_pdf: str, current_hash: str) -> None:
    """Best-effort cleanup after a failed ingestion. Leaves status=in-progress if delete fails."""
    try:
        delete_vectors_by_filter({"source_pdf": source_pdf})
        upsert_registry_entry({"file_name": source_pdf, "file_hash": current_hash, "status": "inactive"})
    except Exception:
        logger.exception(
            f"Rollback failed for {source_pdf} - leaving status=in-progress for next-run recovery"
        )
