import os
from datetime import datetime, timezone

from dotenv import load_dotenv
from supabase import Client, create_client

from src.common.logging import get_logger
from src.config.constants import SUPABASE_FILE_REGISTRY_TABLE

logger = get_logger(__name__)
load_dotenv()
_supabase: Client | None = None


def get_supabase_client() -> Client:
    global _supabase
    if _supabase is None:
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_SERVICE_KEY")
        if not url or not key:
            raise RuntimeError("SUPABASE_URL and SUPABASE_SERVICE_KEY must be set in environment")
        try:
            logger.info("Initializing Supabase client")
            _supabase = create_client(url, key)
        except Exception as e:
            raise RuntimeError(f"Failed to create Supabase client: {e}") from e
    return _supabase


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


if __name__ == "__main__":
    client = get_supabase_client()
    active_files = get_all_active_files()
    logger.info(f"Active files in registry: {len(active_files)}")
