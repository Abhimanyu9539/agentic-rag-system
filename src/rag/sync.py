from pathlib import Path

from src.common.logging import get_logger
from src.db.pinecone_client import delete_vectors_by_filter
from src.db.supabase_client import (
    get_all_active_files,
    get_registry_entry,
    mark_file_inactive,
    upsert_registry_entry,
)
from src.rag.ingestor import embed_and_upsert_chunks, file_hash, load_chunks_from_json

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


def sync_folder(chunks_dir: str, raw_dir: str) -> dict:
    """
    Sync chunk files against Pinecone + Supabase registry.

    States per file:
      NEW          - not in registry, ingest
      UNCHANGED    - hash matches active entry, skip
      CHANGED      - hash differs from active entry, delete old vectors + re-ingest
      IN-PROGRESS  - previous run crashed mid-file, delete partial + re-ingest
      RECOVERED    - previously inactive, back on disk, re-ingest (counted as new)
      DELETED      - chunk file or raw PDF gone, delete vectors + mark inactive
    """
    chunk_files = list(Path(chunks_dir).glob("*_chunks.json"))

    if not chunk_files:
        logger.error(f"No chunk files found in {chunks_dir} - aborting to prevent accidental mass deletion")
        return {"new": 0, "changed": 0, "unchanged": 0, "deleted": 0, "errors": 0}

    active_registry = {e["file_name"]: e["file_hash"] for e in get_all_active_files()}
    processed: set[str] = set()
    summary = {"new": 0, "changed": 0, "unchanged": 0, "deleted": 0, "errors": 0}

    for path in chunk_files:
        chunks = load_chunks_from_json(str(path))
        if not chunks:
            logger.warning(f"No chunks in {path.name} - skipping")
            continue

        source_pdf = chunks[0].metadata.get("source_pdf")
        if not source_pdf:
            logger.error(f"{path.name} missing source_pdf metadata - skipping")
            summary["errors"] += 1
            continue

        processed.add(source_pdf)
        entry = get_registry_entry(source_pdf)
        current_hash = pdf_hash(source_pdf, raw_dir) if (Path(raw_dir) / source_pdf).exists() else None
        status = entry.get("status") if entry else None
        stored_hash = entry.get("file_hash") if entry else None

        if current_hash is None:
            logger.info(f"DELETED (raw missing): {source_pdf}")
            _process_deleted_file(source_pdf, entry, summary)

        elif status == "active" and stored_hash == current_hash:
            logger.info(f"UNCHANGED: {source_pdf} - skipping")
            summary["unchanged"] += 1

        elif status in ("active", "in-progress"):
            logger.info(f"CHANGED: {source_pdf}")
            _process_changed_file(source_pdf, chunks, current_hash, summary)

        else:
            logger.info(f"NEW: {source_pdf}")
            _process_new_file(source_pdf, chunks, current_hash, summary)

    for file_name in active_registry:
        if file_name not in processed:
            logger.info(f"DELETED: {file_name}")
            _process_deleted_file(file_name, entry=None, summary=summary)

    logger.info(
        f"Sync complete - new={summary['new']} changed={summary['changed']} "
        f"unchanged={summary['unchanged']} deleted={summary['deleted']} errors={summary['errors']}"
    )
    return summary


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
