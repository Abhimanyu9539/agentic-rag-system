import io
import json
import os

import fitz  # PyMuPDF
from PIL import Image

from src.common.logging import get_logger
from src.rag.parser import load_json_from_file, parse_pdf

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------

def pdf_bbox_to_fitz_rect(page, bbox):
    """Convert an OpenDataLoader bbox (origin at bottom-left) to a PyMuPDF Rect."""
    if not bbox or len(bbox) != 4:
        raise ValueError(f"Invalid bounding box: {bbox}")
    x0, y0, x1, y1 = bbox
    h = page.rect.height
    return fitz.Rect(x0, h - y1, x1, h - y0)


def _pix_to_pil(pix):
    return Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")


# ---------------------------------------------------------------------------
# Same-page context helper (PyMuPDF text blocks)
# ---------------------------------------------------------------------------

def _same_page_context_rect(page, table_rect, n_blocks=4, max_distance=250):
    """
    Return a Rect extending table_rect upward to include the nearest
    n_blocks non-empty text blocks that sit above it on the same page
    (within max_distance pts).  Returns None if nothing is found.
    """
    above = [
        b for b in page.get_text("blocks")
        if b[4].strip()
        and b[3] <= table_rect.y0 + 2
        and b[1] > 15
        and (table_rect.y0 - b[3]) <= max_distance
    ]
    if not above:
        return None
    above.sort(key=lambda b: b[1])
    ctx = above[-n_blocks:]
    y0 = min(b[1] for b in ctx)
    return fitz.Rect(table_rect.x0, y0, table_rect.x1, table_rect.y1) & page.rect


# ---------------------------------------------------------------------------
# Image merge helper
# ---------------------------------------------------------------------------

def _merge_images(parts):
    """
    Vertically stack a list of PIL Images into one.
    All parts are scaled to the same width before stitching.
    """
    if len(parts) == 1:
        return parts[0]
    w = max(img.width for img in parts)
    resized = []
    for img in parts:
        if img.width != w:
            img = img.resize(
                (w, round(img.height * w / img.width)), Image.LANCZOS)
        resized.append(img)
    total_h = sum(img.height for img in resized)
    merged = Image.new("RGB", (w, total_h), "white")
    y = 0
    for img in resized:
        merged.paste(img, (0, y))
        y += img.height
    return merged


# ---------------------------------------------------------------------------
# Main extraction
# ---------------------------------------------------------------------------

def extract_tables_as_images(
    pdf_path: str,
    output_dir: str,
    parsed_dir: str,
    metadata_path: str | None = None,
) -> tuple[list, str]:
    """
    Parse the PDF (JSON + markdown in one JVM call), extract all tables as PNG
    images, and return (table_metadata, md_path).

    Context strategy
    ----------------
    • First page of a table  – crop is extended upward via PyMuPDF text blocks
      to include the heading / title that appears above the table on the same page.
    • Continuation pages     – raw table crop only; context is already in the
      first page's image.

    Continuation detection
    ----------------------
    A table on page N is treated as a continuation of the active group when:
      1. Its top edge (fitz y0) is within 100 pt of the page top  → started at
         the very top, no room for a same-page heading.
      2. The last heading node seen was on a *different* page      → this page
         contains no heading of its own.
      3. The previous table was on page N-1                        → consecutive
         pages with no gap.
    """
    os.makedirs(output_dir, exist_ok=True)

    stale = [f for f in os.listdir(output_dir) if f.startswith("table_") and f.endswith(".png")]
    for f in stale:
        os.remove(os.path.join(output_dir, f))
    if stale:
        logger.debug(f"Removed {len(stale)} stale table image(s) from {output_dir}")

    json_path, md_path = parse_pdf(pdf_path, parsed_dir)
    data = load_json_from_file(json_path)

    table_count    = 0
    table_metadata = []

    fitz_doc = fitz.open(pdf_path)
    logger.info(f"Extracting tables from: {pdf_path}")

    try:
        state = {
            "heading_page":     None,
            "last_table_page":  None,
        }

        active = {
            "table_count": None,
            "first_page":  None,
            "all_pages":   [],
            "image_path":  None,
            "parts":       [],
        }

        def _finalize_group():
            if not active["parts"]:
                return

            merged = _merge_images(active["parts"])
            merged.save(active["image_path"])

            pages = active["all_pages"]
            span  = f"p{pages[0]}" if len(pages) == 1 else f"p{pages[0]}-{pages[-1]}"
            logger.info(f"Saved table image: {active['image_path']} ({span}, {len(pages)} page(s))")

            table_metadata.append({
                "table_id":   active["table_count"],
                "pages":      list(pages),
                "image_path": active["image_path"],
            })

            active["parts"]       = []
            active["all_pages"]   = []
            active["image_path"]  = None
            active["first_page"]  = None
            active["table_count"] = None

        def _process_table(node):
            nonlocal table_count

            table_page  = node.get("page number", 1)
            page_num    = table_page - 1
            table_bbox  = node.get("bounding box")

            if not table_bbox or not (0 <= page_num < len(fitz_doc)):
                return

            try:
                page       = fitz_doc[page_num]
                table_rect = pdf_bbox_to_fitz_rect(page, table_bbox) & page.rect

                if table_rect.is_empty or table_rect.width < 10 or table_rect.height < 10:
                    logger.debug(f"Skipping degenerate table rect on page {table_page}")
                    return

                is_continuation = (
                    active["parts"]
                    and table_rect.y0 < 100
                    and state["heading_page"] is not None
                    and state["heading_page"] != table_page
                    and state["last_table_page"] is not None
                    and table_page == state["last_table_page"] + 1
                )

                if not is_continuation:
                    _finalize_group()
                    table_count += 1
                    active["table_count"] = table_count
                    active["first_page"]  = table_page
                    active["image_path"]  = os.path.join(
                        output_dir, f"table_p{table_page}_{table_count}.png")
                    logger.debug(f"New table #{table_count} on page {table_page}")

                    ctx_rect = _same_page_context_rect(page, table_rect)
                    crop = (
                        fitz.Rect(table_rect.x0, ctx_rect.y0,
                                  table_rect.x1, table_rect.y1) & page.rect
                        if ctx_rect else table_rect
                    )
                    img = _pix_to_pil(page.get_pixmap(clip=crop, dpi=200))

                else:
                    logger.debug(f"Continuation of table #{active['table_count']} on page {table_page}")
                    img = _pix_to_pil(page.get_pixmap(clip=table_rect, dpi=200))

                active["parts"].append(img)
                active["all_pages"].append(table_page)
                state["last_table_page"] = table_page

            except Exception as e:
                logger.error(f"Error processing table on page {table_page}: {e}")

        def traverse(node):
            if isinstance(node, dict):
                node_type = node.get("type", "").lower()

                if node_type == "heading":
                    state["heading_page"] = node.get("page number")

                if node_type == "table":
                    _process_table(node)
                    return

                for value in node.values():
                    if isinstance(value, (dict, list)):
                        traverse(value)

            elif isinstance(node, list):
                for item in node:
                    traverse(item)

        traverse(data)
        _finalize_group()

    finally:
        fitz_doc.close()

    if metadata_path:
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(table_metadata, f, indent=2)
        logger.info(f"Metadata saved to {metadata_path}")

    logger.info(f"Total logical tables saved: {table_count}")
    return table_metadata, md_path


# ---------------------------------------------------------------------------
# Folder-level extraction
# ---------------------------------------------------------------------------

def extract_tables_from_folder(
    folder_path: str,
    output_dir: str,
    parsed_dir: str,
    metadata_path: str | None = None,
    skip_files: set[str] | None = None,
) -> list:
    """
    Process every PDF in *folder_path* and extract tables from each.

    Each PDF gets:
    - Its own sub-directory in *output_dir* for table PNG images
    - Its own sub-directory in *parsed_dir* for the .json and .md parse outputs

    All per-file metadata is merged into a single list; each entry gains a
    ``source_pdf`` key with the file name.

    Returns the combined metadata list.
    """
    pdf_files = sorted(
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.lower().endswith(".pdf")
    )

    if not pdf_files:
        logger.warning(f"No PDF files found in: {folder_path}")
        return []

    logger.info(f"Found {len(pdf_files)} PDF(s) in {folder_path}")

    combined_metadata = []

    for pdf_path in pdf_files:
        pdf_name   = os.path.basename(pdf_path)
        stem       = os.path.splitext(pdf_name)[0]
        pdf_outdir = os.path.join(output_dir, stem)

        if skip_files and pdf_name in skip_files:
            logger.info("Skipping unchanged (registry): %s", pdf_name)
            continue

        logger.info(f"Processing: {pdf_name}")
        per_file_meta, _ = extract_tables_as_images(pdf_path, pdf_outdir, parsed_dir)

        for entry in per_file_meta:
            entry["source_pdf"] = os.path.basename(pdf_path)

        combined_metadata.extend(per_file_meta)

    logger.info(f"Grand total logical tables: {len(combined_metadata)}")

    if metadata_path:
        os.makedirs(os.path.dirname(metadata_path) or ".", exist_ok=True)
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(combined_metadata, f, indent=2)
        logger.info(f"Combined metadata saved to {metadata_path}")

    return combined_metadata


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    folder_path   = r"E:\LLMOps\agentic-rag-system\data\raw"
    output_dir    = r"E:\LLMOps\agentic-rag-system\data\output\tables"
    parsed_dir    = r"E:\LLMOps\agentic-rag-system\data\output\parsed"
    metadata_path = os.path.join(output_dir, "table_metadata.json")

    extract_tables_from_folder(folder_path, output_dir, parsed_dir, metadata_path=metadata_path)
