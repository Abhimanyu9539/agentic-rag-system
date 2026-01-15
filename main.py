import os

from src.common.logging import get_logger
from src.rag.fetch_table import extract_tables_from_folder
from src.rag.chunker import chunk_pdf, save_chunks
from collections import defaultdict

logger = get_logger(__name__)

RAW_DIR       = os.path.join("data", "raw")
OUTPUT_DIR    = os.path.join("data", "output")
TABLES_DIR    = os.path.join(OUTPUT_DIR, "tables")
PARSED_DIR    = os.path.join(OUTPUT_DIR, "parsed")
CHUNKS_DIR    = os.path.join(OUTPUT_DIR, "chunks")
METADATA_PATH = os.path.join(TABLES_DIR, "table_metadata.json")


def main():
    logger.info("=== Agentic RAG Pipeline Start ===")

    # Step 1: Extract tables and parse PDFs
    logger.info("Step 1: Extracting tables and parsing PDFs from %s", RAW_DIR)
    table_metadata = extract_tables_from_folder(
        folder_path=RAW_DIR,
        output_dir=TABLES_DIR,
        parsed_dir=PARSED_DIR,
        metadata_path=METADATA_PATH,
    )

    if not table_metadata:
        logger.warning("No tables extracted. Check that PDFs exist in %s", RAW_DIR)

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
            logger.warning("Markdown not found for %s, skipping", source_pdf)
            continue

        chunks = chunk_pdf(md_path, pdf_table_meta, source_pdf)
        save_chunks(chunks, os.path.join(CHUNKS_DIR, stem + "_chunks.json"))
        total_chunks += len(chunks)

    logger.info("=== Pipeline Complete: %d chunk(s) across %d PDF(s) ===", total_chunks, len(by_pdf))


if __name__ == "__main__":
    main()
