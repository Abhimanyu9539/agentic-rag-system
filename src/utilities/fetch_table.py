import io
import json
import os

import fitz  # PyMuPDF
from langchain_opendataloader_pdf import OpenDataLoaderPDFLoader
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
# OpenDataLoader JSON loader
# ---------------------------------------------------------------------------

def load_opendataloader_json(pdf_path):
    """Return a list of page-level dicts from OpenDataLoader."""
    loader = OpenDataLoaderPDFLoader(file_path=[pdf_path], format="json")
    documents = loader.load()

    if not documents:
        raise RuntimeError(f"OpenDataLoader returned no documents for: {pdf_path}")

    print(f"📄 OpenDataLoader returned {len(documents)} document(s)")

    pages = []
    for i, doc in enumerate(documents):
        content = doc.page_content
        if not content:
            continue
        if isinstance(content, str):
            try:
                content = json.loads(content)
            except json.JSONDecodeError as e:
                print(f"  ⚠️  Document {i} is not valid JSON: {e}")
                continue
        pages.append(content)

    print(f"✅ Parsed {len(pages)} page(s)")
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

def extract_tables_as_images(pdf_path, output_dir, metadata_path=None):
    """
    Extract all tables from a PDF, merging parts that span multiple pages,
    and save each logical table as a single PNG image.

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

    data = load_opendataloader_json(pdf_path)

    table_count   = 0
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

            active["parts"]     = []
            active["all_pages"] = []
            active["image_path"] = None
            active["first_page"] = None
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
                    # Track the most-recently-seen heading's page so continuation
                    # detection can check whether the current table has its own heading.
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
    return table_metadata


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    pdf_path      = r"E:\LLMOps\agentic-rag-system\data\raw\Employees' Pension Scheme, 1995.pdf"
    output_dir    = r"E:\LLMOps\agentic-rag-system\data\processed\tables"
    metadata_path = os.path.join(output_dir, "table_metadata.json")

    extract_tables_as_images(pdf_path, output_dir, metadata_path=metadata_path)
