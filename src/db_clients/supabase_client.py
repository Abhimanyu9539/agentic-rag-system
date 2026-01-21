from datetime import datetime, timezone
from functools import lru_cache

from supabase import Client, create_client

from src.common.logging import get_logger
from src.config import settings
from src.config.constants import (
    SUPABASE_FILE_REGISTRY_TABLE,
    SUPABASE_IMAGE_REGISTRY_TABLE,
    SUPABASE_IMAGES_BUCKET,
)

logger = get_logger(__name__)


@lru_cache(maxsize=1)
def get_supabase_client() -> Client:
    try:
        logger.info("Initializing Supabase client")
        return create_client(settings.supabase_url, settings.supabase_service_key)
    except Exception as e:
        raise RuntimeError(f"Failed to create Supabase client: {e}") from e


def get_registry_entry(file_name: str) -> dict | None:
    try:
        response = (
            get_supabase_client()
            .table(SUPABASE_FILE_REGISTRY_TABLE)
            .select("*")
            .eq("file_name", file_name)
            .limit(1)
            .execute()
        )
        return response.data[0] if response.data else None
    except Exception as e:
        logger.error(f"Failed to fetch registry entry for {file_name}: {e}")
        raise


def upsert_registry_entry(data: dict) -> None:
    payload = {**data, "updated_at": datetime.now(timezone.utc).isoformat()}
    try:
        get_supabase_client().table(SUPABASE_FILE_REGISTRY_TABLE).upsert(payload).execute()
        logger.info(f"Registry upserted: {data.get('file_name')} (status={data.get('status')})")
    except Exception as e:
        logger.error(f"Failed to upsert registry entry for {data.get('file_name')}: {e}")
        raise


def get_all_active_files() -> list[dict]:
    try:
        response = (
            get_supabase_client()
            .table(SUPABASE_FILE_REGISTRY_TABLE)
            .select("file_name, file_hash")
            .eq("status", "active")
            .execute()
        )
        return response.data or []
    except Exception as e:
        logger.error(f"Failed to fetch active files: {e}")
        raise


def mark_file_inactive(file_name: str) -> None:
    try:
        upsert_registry_entry({"file_name": file_name, "file_hash": "", "status": "inactive"})
        logger.info(f"Marked inactive: {file_name}")
    except Exception as e:
        logger.error(f"Failed to mark file inactive {file_name}: {e}")
        raise


# ---------------------------------------------------------------------------
# Image storage + image_registry
# ---------------------------------------------------------------------------

def upload_image_to_storage(image_bytes: bytes, storage_path: str) -> str:
    """Upload PNG bytes to the table-images bucket and return its public URL."""
    try:
        bucket = get_supabase_client().storage.from_(SUPABASE_IMAGES_BUCKET)
        bucket.upload(
            path=storage_path,
            file=image_bytes,
            file_options={"content-type": "image/png", "upsert": "true"},
        )
        return bucket.get_public_url(storage_path)
    except Exception as e:
        logger.error(f"Failed to upload image to {storage_path}: {e}")
        raise


def insert_image_registry_entry(row: dict) -> None:
    """Insert a single image_registry row. Caller supplies all required fields."""
    try:
        get_supabase_client().table(SUPABASE_IMAGE_REGISTRY_TABLE).insert(row).execute()
    except Exception as e:
        logger.error(f"Failed to insert image_registry row for {row.get('storage_path')}: {e}")
        raise


def delete_images_for_pdf(file_name: str) -> None:
    """Delete all storage objects and image_registry rows tied to a PDF."""
    try:
        response = (
            get_supabase_client()
            .table(SUPABASE_IMAGE_REGISTRY_TABLE)
            .select("storage_path")
            .eq("file_name", file_name)
            .execute()
        )
        paths = [r["storage_path"] for r in (response.data or [])]
        if not paths:
            logger.debug(f"No images registered for {file_name}; nothing to delete")
            return

        get_supabase_client().storage.from_(SUPABASE_IMAGES_BUCKET).remove(paths)
        (
            get_supabase_client()
            .table(SUPABASE_IMAGE_REGISTRY_TABLE)
            .delete()
            .eq("file_name", file_name)
            .execute()
        )
        logger.info(f"Deleted {len(paths)} image(s) for {file_name}")
    except Exception as e:
        logger.error(f"Failed to delete images for {file_name}: {e}")
        raise


if __name__ == "__main__":
    client = get_supabase_client()
    active_files = get_all_active_files()
    logger.info(f"Active files in registry: {len(active_files)}")
