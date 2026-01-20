import json
import os
import shutil
from collections import defaultdict

import opendataloader_pdf

from src.common.logging import get_logger
from src.config.constants import DEBUG_SAVE_PARSED, MD_PAGE_SEPARATOR

logger = get_logger(__name__)


def parse_pdf(pdf_path: str, parsed_dir: str) -> tuple[list, str]:
    """
    Convert a PDF to JSON and markdown using a single JVM call.

    Returns (parsed_data, markdown_str) in memory.
    Files are written to parsed_dir/{stem}/ as a JVM requirement.
    If DEBUG_SAVE_PARSED is True they are kept for inspection; otherwise deleted.
    """
    stem    = os.path.splitext(os.path.basename(pdf_path))[0]
    out_dir = os.path.join(parsed_dir, stem)
    os.makedirs(out_dir, exist_ok=True)

    logger.info(f"Parsing PDF: {pdf_path}")
    try:
        opendataloader_pdf.convert(
            input_path=pdf_path,
            output_dir=out_dir,
            format=["json", "markdown"],
            markdown_page_separator=MD_PAGE_SEPARATOR,
            quiet=True,
        )
    except Exception as e:
        logger.error(f"PDF conversion failed for {pdf_path}: {e}")
        raise

    json_path = os.path.join(out_dir, stem + ".json")
    md_path   = os.path.join(out_dir, stem + ".md")

    if not os.path.exists(json_path):
        raise RuntimeError(f"Expected JSON output not found: {json_path}")
    if not os.path.exists(md_path):
        raise RuntimeError(f"Expected markdown output not found: {md_path}")

    parsed_data = load_json_from_file(json_path)
    with open(md_path, "r", encoding="utf-8") as f:
        markdown = f.read()

    if DEBUG_SAVE_PARSED:
        logger.debug(f"Debug: parsed files kept in {out_dir}")
    else:
        shutil.rmtree(out_dir, ignore_errors=True)
        logger.debug(f"Cleaned up parsed output for {stem}")

    return parsed_data, markdown


def load_json_from_file(json_path: str) -> list:
    """
    Read the JSON file produced by parse_pdf and return a list of per-page dicts
    [{"page number": N, "kids": [...]}, ...].
    """
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load JSON from {json_path}: {e}")
        raise

    pages_map: dict = defaultdict(list)
    for element in data.get("kids", []):
        pages_map[element.get("page number", 1)].append(element)

    pages = [
        {"page number": pnum, "kids": kids}
        for pnum, kids in sorted(pages_map.items())
    ]

    logger.info(f"Loaded {len(pages)} page(s) from {json_path}")
    return pages
