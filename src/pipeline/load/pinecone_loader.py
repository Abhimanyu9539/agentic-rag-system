import json
from functools import lru_cache

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_pinecone import PineconeVectorStore

from src.common.logging import get_logger
from src.config.constants import BATCH_SIZE, DEFAULT_PINECONE_INDEX, EMBEDDING_MODEL, EMBEDDING_PROVIDER
from src.db_clients.pinecone_client import get_pinecone_index
from src.llm_adapters.embeddings.base import get_embeddings_model
from src.pipeline.transform.embedder import prepare_chunks_for_pinecone

logger = get_logger(__name__)


@lru_cache(maxsize=4)
def _get_cached_embeddings_client(model: str, provider: str) -> Embeddings:
    return get_embeddings_model(model=model, model_provider=provider)


@lru_cache(maxsize=4)
def _get_cached_vectorstore_client(
    index_name: str,
    namespace: str,
    model: str,
    provider: str,
) -> PineconeVectorStore:
    return PineconeVectorStore(
        index=get_pinecone_index(index_name),
        embedding=_get_cached_embeddings_client(model, provider),
        namespace=namespace,
    )


def _upsert_in_batches(
    chunks: list[Document],
    ids: list[str],
    vectorstore: PineconeVectorStore,
) -> int:
    total_batches = max(1, -(-len(chunks) // BATCH_SIZE))

    for batch_num, i in enumerate(range(0, len(chunks), BATCH_SIZE), start=1):
        batch_chunks = chunks[i : i + BATCH_SIZE]
        batch_ids    = ids[i : i + BATCH_SIZE]
        vectorstore.add_documents(documents=batch_chunks, ids=batch_ids)
        logger.info(f"Batch {batch_num}/{total_batches} upserted ({len(batch_chunks)} chunks)")

    logger.info(f"Upserted {len(chunks)} chunks total")
    return len(chunks)


def embed_and_upsert_chunks(
    chunks: list[Document],
    fhash: str,
    index_name: str = DEFAULT_PINECONE_INDEX,
    namespace: str = "",
) -> int:
    """
    Orchestrate the transform → load handoff for a single file's chunks.
    Prepares (sanitize + IDs) via the transform stage, then runs the batched
    Pinecone upsert. Returns the number of vectors written.
    """
    chunks, ids = prepare_chunks_for_pinecone(chunks, fhash)
    vectorstore = _get_cached_vectorstore_client(
        index_name=index_name,
        namespace=namespace,
        model=EMBEDDING_MODEL,
        provider=EMBEDDING_PROVIDER,
    )
    return _upsert_in_batches(chunks, ids, vectorstore)


# ---------------------------------------------------------------------------
# Debug helper
# ---------------------------------------------------------------------------

def load_chunks_from_json(chunks_path: str) -> list[Document]:
    """Load chunks previously dumped via chunker.save_chunks(). Debug only."""
    with open(chunks_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return [Document(page_content=item["text"], metadata=item["metadata"]) for item in raw]
