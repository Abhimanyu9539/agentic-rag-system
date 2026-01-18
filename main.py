import os
from collections import defaultdict

from dotenv import load_dotenv

from src.common.logging import get_logger
from src.config.constants import DEFAULT_PINECONE_INDEX
from src.rag.fetch_table import extract_tables_from_folder
from src.rag.chunker import chunk_pdf, save_chunks
from src.rag.sync import is_pdf_unchanged, sync_folder

load_dotenv()
logger = get_logger(__name__)

RAW_DIR       = os.path.join("data", "raw")
OUTPUT_DIR    = os.path.join("data", "output")
TABLES_DIR    = os.path.join(OUTPUT_DIR, "tables")
PARSED_DIR    = os.path.join(OUTPUT_DIR, "parsed")
CHUNKS_DIR    = os.path.join(OUTPUT_DIR, "chunks")
METADATA_PATH = os.path.join(TABLES_DIR, "table_metadata.json")


def main():
    logger.info("=== Agentic RAG Pipeline Start ===")

    # Pre-compute unchanged PDFs to skip parse+chunk entirely
    skip_pdfs = {
        pdf_name
        for pdf_name in os.listdir(RAW_DIR)
        if pdf_name.lower().endswith(".pdf") and is_pdf_unchanged(pdf_name, RAW_DIR)
    }
    if skip_pdfs:
        logger.info(f"Skipping {len(skip_pdfs)} unchanged PDF(s): {sorted(skip_pdfs)}")

    # Step 1: Extract tables and parse PDFs
    logger.info("Step 1: Extracting tables and parsing PDFs from %s", RAW_DIR)
    try:
        table_metadata = extract_tables_from_folder(
            folder_path=RAW_DIR,
            output_dir=TABLES_DIR,
            parsed_dir=PARSED_DIR,
            metadata_path=METADATA_PATH,
            skip_files=skip_pdfs,
        )
    except Exception as e:
        logger.error(f"Step 1 failed — table extraction aborted: {e}")
        raise

    if not table_metadata:
        logger.warning(f"No tables extracted. Check that PDFs exist in {RAW_DIR}")

    # Step 2: Chunk each PDF's markdown
    logger.info("Step 2: Chunking parsed markdown files")
    by_pdf: dict = defaultdict(list)
    for entry in table_metadata:
        by_pdf[entry["source_pdf"]].append(entry)

    total_chunks = 0
    for source_pdf, pdf_table_meta in by_pdf.items():
        stem    = os.path.splitext(source_pdf)[0]
        md_path = os.path.join(PARSED_DIR, stem, stem + ".md")

        if not os.path.exists(md_path):
            logger.warning(f"Markdown not found for {source_pdf}, skipping")
            continue

        try:
            chunks = chunk_pdf(md_path, pdf_table_meta, source_pdf)
            save_chunks(chunks, os.path.join(CHUNKS_DIR, stem + "_chunks.json"))
            total_chunks += len(chunks)
        except Exception as e:
            logger.error(f"Step 2 failed for {source_pdf}: {e}")
            raise

    logger.info(f"=== Chunking Complete: {total_chunks} chunk(s) across {len(by_pdf)} PDF(s) ===")

    # Step 3: Sync chunks into Pinecone via registry
    logger.info("Step 3: Syncing chunks to Pinecone")
    try:
        summary = sync_folder(
            chunks_dir=CHUNKS_DIR,
            raw_dir=RAW_DIR,
            index_name=os.getenv("PINECONE_INDEX_NAME", DEFAULT_PINECONE_INDEX),
        )
    except Exception as e:
        logger.error(f"Step 3 failed — sync aborted: {e}")
        raise
    logger.info("Sync result — %s", summary)

    logger.info("=== Pipeline Complete ===")


if __name__ == "__main__":
    main()
