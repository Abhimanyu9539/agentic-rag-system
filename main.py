import os

from dotenv import load_dotenv

from src.common.logging import get_logger
from src.rag.chunker import chunk_pdf
from src.rag.fetch_table import extract_tables_as_images
from src.rag.parser import parse_pdf
from src.rag.sync import is_pdf_unchanged, pdf_hash, sweep_deleted, sync_file

load_dotenv()
logger = get_logger(__name__)

RAW_DIR    = os.path.join("data", "raw")
OUTPUT_DIR = os.path.join("data", "output")
TABLES_DIR = os.path.join(OUTPUT_DIR, "tables")
PARSED_DIR = os.path.join(OUTPUT_DIR, "parsed")


def main():
    logger.info("=== Agentic RAG Pipeline Start ===")

    pdf_files = sorted(f for f in os.listdir(RAW_DIR) if f.lower().endswith(".pdf"))

    skip_pdfs = {f for f in pdf_files if is_pdf_unchanged(f, RAW_DIR)}
    if skip_pdfs:
        logger.info(f"Skipping {len(skip_pdfs)} unchanged PDF(s)")

    summary = {"new": 0, "changed": 0, "unchanged": len(skip_pdfs), "deleted": 0, "errors": 0}

    for pdf_name in pdf_files:
        if pdf_name in skip_pdfs:
            continue

        pdf_path = os.path.join(RAW_DIR, pdf_name)
        stem     = os.path.splitext(pdf_name)[0]

        logger.info(f"Processing: {pdf_name}")
        try:
            # Step 1: Parse PDF → structured data + markdown (in memory)
            parsed_data, markdown = parse_pdf(pdf_path, PARSED_DIR)

            # Step 2: Extract table images from the parsed data
            table_metadata = extract_tables_as_images(
                pdf_path,
                os.path.join(TABLES_DIR, stem),
                parsed_data,
            )

            # Step 3: Chunk markdown into LangChain Documents
            chunks = chunk_pdf(markdown, table_metadata, pdf_name)

            # Step 4: Embed + sync to Pinecone via registry
            current_hash = pdf_hash(pdf_name, RAW_DIR)
            sync_file(pdf_name, chunks, current_hash, summary)

        except Exception as e:
            logger.error(f"Pipeline failed for {pdf_name}: {e}")
            summary["errors"] += 1

    # Step 5: Mark deleted any PDF removed from raw_dir
    sweep_deleted(set(pdf_files), summary)

    logger.info(f"Result — {summary}")
    logger.info("=== Pipeline Complete ===")


if __name__ == "__main__":
    main()
