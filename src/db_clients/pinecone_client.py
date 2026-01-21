from functools import lru_cache
from typing import Any

from pinecone import Pinecone, ServerlessSpec

from src.common.logging import get_logger
from src.config import settings
from src.config.constants import BATCH_SIZE, DEFAULT_PINECONE_INDEX, EMBEDDING_DIMENSION, PINECONE_METRIC

logger = get_logger(__name__)


@lru_cache(maxsize=1)
def get_pinecone_client() -> Pinecone:
    try:
        logger.info("Initializing Pinecone client")
        return Pinecone(api_key=settings.pinecone_api_key)
    except Exception as e:
        raise RuntimeError(f"Failed to initialize Pinecone client: {e}") from e


@lru_cache(maxsize=4)
def ensure_index(
    index_name: str,
    dimension: int = EMBEDDING_DIMENSION,
    metric: str = PINECONE_METRIC,
) -> None:
    try:
        pc = get_pinecone_client()
        existing = {idx.name for idx in pc.list_indexes()}
        if index_name not in existing:
            logger.info(f"Index '{index_name}' not found - creating (dim={dimension}, metric={metric})")
            pc.create_index(
                name=index_name,
                dimension=dimension,
                metric=metric,
                spec=ServerlessSpec(cloud=settings.pinecone_cloud, region=settings.pinecone_region),
            )
            logger.info(f"Created Pinecone index: {index_name}")
        else:
            logger.info(f"Pinecone index '{index_name}' confirmed active")
    except Exception as e:
        logger.error(f"Failed to ensure index '{index_name}': {e}")
        raise


def get_pinecone_index(index_name: str = DEFAULT_PINECONE_INDEX):
    ensure_index(index_name)
    return get_pinecone_client().Index(index_name)


def upsert_vectors(
    vectors: list[dict[str, Any]],
    index_name: str = DEFAULT_PINECONE_INDEX,
    namespace: str = "",
    batch_size: int = BATCH_SIZE,
) -> int:
    index = get_pinecone_index(index_name)
    upserted = 0
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i : i + batch_size]
        try:
            index.upsert(vectors=batch, namespace=namespace)
            upserted += len(batch)
            logger.info(f"Upserted batch of {len(batch)} vectors to '{index_name}'")
        except Exception as e:
            logger.error(f"Failed to upsert batch {i} through {i + len(batch)} to '{index_name}': {e}")
            raise
    return upserted


def delete_vectors_by_filter(
    filter_dict: dict,
    index_name: str = DEFAULT_PINECONE_INDEX,
    namespace: str = "",
) -> None:
    try:
        index = get_pinecone_index(index_name)
        index.delete(filter=filter_dict, namespace=namespace)
        logger.info(f"Deleted vectors from '{index_name}' matching filter: {filter_dict}")
    except Exception as e:
        logger.error(f"Failed to delete vectors from '{index_name}' with filter {filter_dict}: {e}")
        raise
