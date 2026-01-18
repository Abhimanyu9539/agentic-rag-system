import json
import os
from collections import defaultdict

import opendataloader_pdf

from src.common.logging import get_logger
from src.config.constants import MD_PAGE_SEPARATOR

logger = get_logger(__name__)


def parse_pdf(pdf_path: str, parsed_dir: str) -> tuple[str, str]:
    """
    Convert a PDF to both JSON and markdown using a single JVM call.

    Outputs are written to *parsed_dir/{stem}/* so you can inspect them.
    Returns (json_path, md_path).
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

    logger.info(f"Parsed {stem}.json + {stem}.md -> {out_dir}")
    return json_path, md_path


def load_json_from_file(json_path: str) -> list:
    """
    Read the JSON file produced by parse_pdf and return a list of per-page dicts
    [{"page number": N, "kids": [...]}, ...].

    The raw JSON file has a flat {"kids": [all_elements]} structure where each
    element carries its own "page number" field.  We group by page here so the
    rest of the pipeline is unchanged.
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
