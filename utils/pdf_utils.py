# utils/pdf_utils.py
import re
from typing import Any, Dict, List
import fitz  # PyMuPDF


def extract_text_from_pdf(uploaded_file) -> List[Dict[str, Any]]:
    """Extract text page-by-page from an uploaded PDF."""
    pdf_bytes = uploaded_file.read()
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages = []

    for i, page in enumerate(doc):
        text = page.get_text("text") or ""
        text = re.sub(r"\s+", " ", text).strip()
        pages.append({
            "page_number": i + 1,
            "text": text,
        })

    return pages