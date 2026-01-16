import json
import os
from pathlib import Path

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

from ..common.logging import get_logger
from ..config.constants import _BATCH_SIZE 

logger = get_logger(__name__)


def load_chunks_from_json(chunks_path: str) -> list[Document]:
    with open(chunks_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return [Document(page_content=item["text"], metadata=item["metadata"]) for item in raw]


def _sanitize_metadata(doc: Document) -> Document:
    """Pinecone only allows list[str] for list-type metadata fields — convert pages to list[str]."""
    meta = dict(doc.metadata)
    if "pages" in meta:
        meta["pages"] = [str(p) for p in meta["pages"]]
    return Document(page_content=doc.page_content, metadata=meta)


def embed_and_upsert(chunks: list[Document], index_name: str, namespace: str = "") -> int:
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    chunks = [_sanitize_metadata(c) for c in chunks]
    total = 0
    for i in range(0, len(chunks), _BATCH_SIZE):
        batch = chunks[i : i + _BATCH_SIZE]
        PineconeVectorStore.from_documents(
            documents=batch,
            embedding=embeddings,
            index_name=index_name,
            namespace=namespace,
        )
        total += len(batch)
        logger.info("Upserted batch %d–%d (%d vectors so far)", i + 1, i + len(batch), total)
    return total


def ingest_folder(chunks_dir: str, index_name: str, namespace: str = "") -> int:
    chunk_files = list(Path(chunks_dir).glob("*_chunks.json"))
    logger.info("Found %d chunk file(s) in %s", len(chunk_files), chunks_dir)
    total = 0
    for path in chunk_files:
        logger.info("Loading chunks from %s", path.name)
        chunks = load_chunks_from_json(str(path))
        logger.info("Embedding and upserting %d chunk(s) from %s", len(chunks), path.name)
        total += embed_and_upsert(chunks, index_name=index_name, namespace=namespace)
    return total
