import os
from typing import Any

from pinecone import Pinecone, ServerlessSpec

from ..common.logging import get_logger
from ..config.constants import _BATCH_SIZE

logger = get_logger(__name__)

# Defaults for index creation. 1536 matches OpenAI text-embedding-3-small.
_DEFAULT_DIMENSION = 1536
_DEFAULT_METRIC = "cosine"
_DEFAULT_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
_DEFAULT_REGION = os.getenv("PINECONE_REGION", "us-east-1")

_client: Pinecone | None = None
_ensured_indexes: set[str] = set()


def get_pinecone_client() -> Pinecone:
    """Cached Pinecone client — avoids rebuilding per call."""
    global _client
    if _client is None:
        logger.info("Initializing Pinecone client")
        _client = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    return _client


def ensure_index(
    index_name: str,
    dimension: int = _DEFAULT_DIMENSION,
    metric: str = _DEFAULT_METRIC,
) -> None:
    """Create the index if it doesn't exist. Memoized per process."""
    if index_name in _ensured_indexes:
        return
    pc = get_pinecone_client()
    existing = {idx.name for idx in pc.list_indexes()}
    if index_name not in existing:
        logger.info("Index '%s' not found — creating (dim=%d, metric=%s)", index_name, dimension, metric)
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric=metric,
            spec=ServerlessSpec(cloud=_DEFAULT_CLOUD, region=_DEFAULT_REGION),
        )
        logger.info("Created Pinecone index: %s", index_name)
    else:
        logger.info("Pinecone index '%s' confirmed active", index_name)
    _ensured_indexes.add(index_name)


def get_pinecone_index(index_name: str):
    """Return the named index, auto-creating it if missing."""
    ensure_index(index_name)
    return get_pinecone_client().Index(index_name)


def upsert_vectors(
    vectors: list[dict[str, Any]],
    index_name: str,
    namespace: str = "",
    batch_size: int = _BATCH_SIZE,
) -> int:
    """Upsert pre-embedded vectors. Each item: {"id", "values", "metadata"}."""
    index = get_pinecone_index(index_name)
    upserted = 0
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i : i + batch_size]
        index.upsert(vectors=batch, namespace=namespace)
        upserted += len(batch)
        logger.info("Upserted batch of %d vectors to '%s'", len(batch), index_name)
    return upserted


def delete_vectors_by_filter(
    filter_dict: dict,
    index_name: str,
    namespace: str = "",
) -> None:
    """Delete vectors matching a metadata filter (e.g. cleanup by source_file)."""
    index = get_pinecone_index(index_name)
    index.delete(filter=filter_dict, namespace=namespace)
    logger.info("Deleted vectors from '%s' matching filter: %s", index_name, filter_dict)
