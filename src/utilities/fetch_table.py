import io
import json
import os
from collections import defaultdict

import fitz  # PyMuPDF
import opendataloader_pdf
from PIL import Image


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
# PDF parsing — both formats in a single JVM call
# ---------------------------------------------------------------------------

# Same separator the LangChain wrapper uses; Java substitutes the actual page number.
_MD_PAGE_SEPARATOR = "\n<<<ODL_PAGE_BREAK_%page-number%>>>\n"


def parse_pdf(pdf_path: str, parsed_dir: str) -> tuple[str, str]:
    """
    Convert a PDF to both JSON and markdown using a single JVM call.

    Outputs are written to *parsed_dir/{stem}/* so you can inspect them.
    Returns (json_path, md_path).
    """
    stem    = os.path.splitext(os.path.basename(pdf_path))[0]
    out_dir = os.path.join(parsed_dir, stem)
    os.makedirs(out_dir, exist_ok=True)

    opendataloader_pdf.convert(
        input_path=pdf_path,
        output_dir=out_dir,
        format=["json", "markdown"],
        markdown_page_separator=_MD_PAGE_SEPARATOR,
        quiet=True,
    )

    json_path = os.path.join(out_dir, stem + ".json")
    md_path   = os.path.join(out_dir, stem + ".md")

    if not os.path.exists(json_path):
        raise RuntimeError(f"Expected JSON output not found: {json_path}")
    if not os.path.exists(md_path):
        raise RuntimeError(f"Expected markdown output not found: {md_path}")

    print(f"✅ Parsed: {stem}.json + {stem}.md  →  {out_dir}")
    return json_path, md_path


def load_json_from_file(json_path: str) -> list:
    """
    Read the JSON file produced by parse_pdf and return a list of per-page dicts
    [{"page number": N, "kids": [...]}, ...] — the same structure the
    traverse / table-extraction logic expects.

    The raw JSON file has a flat {"kids": [all_elements]} structure where each
    element carries its own "page number" field.  We group by page here so the
    rest of the pipeline is unchanged.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    pages_map: dict = defaultdict(list)
    for element in data.get("kids", []):
        pages_map[element.get("page number", 1)].append(element)

    pages = [
        {"page number": pnum, "kids": kids}
        for pnum, kids in sorted(pages_map.items())
    ]

    print(f"✅ Loaded {len(pages)} page(s) from JSON")
    return pages


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
        and b[1] > 15                               # skip page-header margin
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

    # Remove any table PNGs left over from previous runs so stale
    # individual-page crops don't linger after tables are merged.
    for f in os.listdir(output_dir):
        if f.startswith("table_") and f.endswith(".png"):
            os.remove(os.path.join(output_dir, f))

    # Single JVM call → JSON (table detection) + markdown (chunking)
    json_path, md_path = parse_pdf(pdf_path, parsed_dir)
    data = load_json_from_file(json_path)

    table_count    = 0
    table_metadata = []

    fitz_doc = fitz.open(pdf_path)

    try:
        # ----- shared mutable state ----------------------------------------
        state = {
            "heading_page":     None,   # page of the most-recently-seen heading
            "last_table_page":  None,   # page of the most-recently-processed table
        }

        # ----- active group (tables being accumulated for the current image) -
        active = {
            "table_count": None,
            "first_page":  None,
            "all_pages":   [],
            "image_path":  None,
            "parts":       [],   # list[PIL.Image]
        }

        def _finalize_group():
            """Save the accumulated parts as one merged image and record metadata."""
            if not active["parts"]:
                return

            merged = _merge_images(active["parts"])
            merged.save(active["image_path"])

            pages = active["all_pages"]
            span  = f"p{pages[0]}" if len(pages) == 1 else f"p{pages[0]}-{pages[-1]}"
            print(f"✅ Saved: {active['image_path']}  ({span}, {len(pages)} page(s))")

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

            table_page  = node.get("page number", 1)   # 1-indexed (JSON)
            page_num    = table_page - 1                # 0-indexed (fitz)
            table_bbox  = node.get("bounding box")

            if not table_bbox or not (0 <= page_num < len(fitz_doc)):
                return

            try:
                page       = fitz_doc[page_num]
                table_rect = pdf_bbox_to_fitz_rect(page, table_bbox) & page.rect

                if table_rect.is_empty or table_rect.width < 10 or table_rect.height < 10:
                    return

                # ── Continuation detection ───────────────────────────────────
                is_continuation = (
                    active["parts"]                               # there is an active group
                    and table_rect.y0 < 100                       # table starts at top of page
                    and state["heading_page"] is not None
                    and state["heading_page"] != table_page        # no own heading on this page
                    and state["last_table_page"] is not None
                    and table_page == state["last_table_page"] + 1 # immediately follows
                )

                if not is_continuation:
                    # ── New table: finalise the previous group ────────────────
                    _finalize_group()
                    table_count += 1
                    active["table_count"] = table_count
                    active["first_page"]  = table_page
                    active["image_path"]  = os.path.join(
                        output_dir, f"table_p{table_page}_{table_count}.png")

                    # Extend crop upward to include same-page context
                    ctx_rect = _same_page_context_rect(page, table_rect)
                    crop = (
                        fitz.Rect(table_rect.x0, ctx_rect.y0,
                                  table_rect.x1, table_rect.y1) & page.rect
                        if ctx_rect else table_rect
                    )
                    img = _pix_to_pil(page.get_pixmap(clip=crop, dpi=200))

                else:
                    # ── Continuation: raw table crop, no repeated context ────
                    img = _pix_to_pil(page.get_pixmap(clip=table_rect, dpi=200))

                active["parts"].append(img)
                active["all_pages"].append(table_page)
                state["last_table_page"] = table_page

            except Exception as e:
                print(f"⚠️  Error on page {table_page}: {e}")

        # ----- JSON traversal -----------------------------------------------
        def traverse(node):
            if isinstance(node, dict):
                node_type = node.get("type", "").lower()

                if node_type == "heading":
                    state["heading_page"] = node.get("page number")

                if node_type == "table":
                    _process_table(node)
                    return  # don't descend into table cells

                for value in node.values():
                    if isinstance(value, (dict, list)):
                        traverse(value)

            elif isinstance(node, list):
                for item in node:
                    traverse(item)

        traverse(data)
        _finalize_group()   # flush the last group

    finally:
        fitz_doc.close()

    if metadata_path:
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(table_metadata, f, indent=2)
        print(f"📦 Metadata saved to {metadata_path}")

    print(f"\n🎯 Total logical tables saved: {table_count}")
    return table_metadata, md_path


# ---------------------------------------------------------------------------
# Folder-level extraction
# ---------------------------------------------------------------------------

def extract_tables_from_folder(
    folder_path: str,
    output_dir: str,
    parsed_dir: str,
    metadata_path: str | None = None,
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
        print(f"⚠️  No PDF files found in: {folder_path}")
        return []

    print(f"📂 Found {len(pdf_files)} PDF(s) in {folder_path}\n")

    combined_metadata = []

    for pdf_path in pdf_files:
        stem       = os.path.splitext(os.path.basename(pdf_path))[0]
        pdf_outdir = os.path.join(output_dir, stem)

        print(f"{'─' * 60}")
        print(f"📄 Processing: {os.path.basename(pdf_path)}")

        per_file_meta, _ = extract_tables_as_images(pdf_path, pdf_outdir, parsed_dir)

        for entry in per_file_meta:
            entry["source_pdf"] = os.path.basename(pdf_path)

        combined_metadata.extend(per_file_meta)
        print()

    print(f"{'=' * 60}")
    print(f"🎯 Grand total logical tables: {len(combined_metadata)}")

    if metadata_path:
        os.makedirs(os.path.dirname(metadata_path) or ".", exist_ok=True)
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(combined_metadata, f, indent=2)
        print(f"📦 Combined metadata saved to {metadata_path}")

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
