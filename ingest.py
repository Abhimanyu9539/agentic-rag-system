import os

from src.common.logging import get_logger
from src.config.constants import DEBUG_SAVE_IMAGES
from src.pipeline.extract.fetch_table import extract_tables_as_images
from src.pipeline.extract.parser import parse_pdf
from src.pipeline.load.sync import (
    clear_pdf_artifacts,
    is_pdf_unchanged,
    mark_in_progress,
    needs_reprocess_cleanup,
    pdf_hash,
    rollback,
    sweep_deleted,
    sync_file,
)
from src.pipeline.transform.chunker import chunk_pdf

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

    save_images_locally = DEBUG_SAVE_IMAGES

    for pdf_name in pdf_files:
        if pdf_name in skip_pdfs:
            continue

        pdf_path = os.path.join(RAW_DIR, pdf_name)
        stem     = os.path.splitext(pdf_name)[0]

        logger.info(f"Processing: {pdf_name}")

        current_hash       = pdf_hash(pdf_name, RAW_DIR)
        is_reprocess       = needs_reprocess_cleanup(pdf_name)
        in_progress_marked = False

        try:
            if is_reprocess:
                logger.info(f"CHANGED: clearing old vectors and images for {pdf_name}")
                clear_pdf_artifacts(pdf_name)
            else:
                logger.info(f"NEW: {pdf_name}")

            mark_in_progress(pdf_name, current_hash)
            in_progress_marked = True

            # Step 1: Parse PDF → structured data + markdown (in memory)
            parsed_data, markdown = parse_pdf(pdf_path, PARSED_DIR)

            # Step 2: Extract table images → upload to Supabase + register
            local_dir = os.path.join(TABLES_DIR, stem) if save_images_locally else None
            table_metadata = extract_tables_as_images(
                pdf_path,
                parsed_data,
                pdf_name=pdf_name,
                local_dir=local_dir,
            )

            # Step 3: Chunk markdown into LangChain Documents
            chunks = chunk_pdf(markdown, table_metadata, pdf_name)

            # Step 4: Embed + flip status to active (rollback on failure)
            action = "changed" if is_reprocess else "new"
            sync_file(pdf_name, chunks, current_hash, summary, action=action)

        except Exception as e:
            logger.error(f"Pipeline failed for {pdf_name}: {e}")
            summary["errors"] += 1
            if in_progress_marked:
                rollback(pdf_name, current_hash)

    # Step 5: Mark deleted any PDF removed from raw_dir
    sweep_deleted(set(pdf_files), summary)

    logger.info(f"Result — {summary}")
    logger.info("=== Pipeline Complete ===")


if __name__ == "__main__":
    main()
