import fitz  # PyMuPDF
import json


def draw_bbox_on_pdf(pdf_path, json_path, output_pdf):
    # Load PDF
    doc = fitz.open(pdf_path)

    # Load JSON
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Traverse elements recursively
    def process_element(element):
        if "bounding box" in element:
            page_num = element.get("page number", 1) - 1
            bbox = element["bounding box"]

            if page_num < len(doc):
                page = doc[page_num]

                rect = fitz.Rect(bbox)

                # Draw rectangle
                page.draw_rect(rect, color=(1, 0, 0), width=1)

                # Optional: label type
                label = element.get("type", "")
                if label:
                    page.insert_text(
                        (rect.x0, rect.y0 - 5),
                        label,
                        fontsize=6,
                        color=(0, 0, 1),
                    )

        # Recurse into kids
        if "kids" in element:
            for child in element["kids"]:
                process_element(child)

        # Handle tables (rows → cells → kids)
        if element.get("type") == "table":
            for row in element.get("rows", []):
                for cell in row.get("cells", []):
                    process_element(cell)

    # Start processing
    for element in data.get("kids", []):
        process_element(element)

    # Save output
    doc.save(output_pdf)
    doc.close()


# ----------------------------
# Example Usage
# ----------------------------
if __name__ == "__main__":
    pdf_path = r"E:\LLMOps\agentic-rag\data\raw\Employees' Provident Funds Scheme.1952.pdf"
    json_path = r"E:\LLMOps\agentic-rag\data\json\input.json"
    output_pdf = r"E:\LLMOps\agentic-rag\data\processed\annotated.pdf"

    draw_bbox_on_pdf(pdf_path, json_path, output_pdf)

    print(f"Saved annotated PDF → {output_pdf}")