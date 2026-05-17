from pathlib import Path
from typing import Dict, Any
import logging

from app.interfaces.document_loader import IDocumentLoader

logger = logging.getLogger(__name__)


class PDFLoader(IDocumentLoader):
    def load(self, file_path: Path) -> Dict[str, Any]:
        try:
            import PyPDF2
            text = ""
            with open(file_path, 'rb') as f:
                pdf_reader = PyPDF2.PdfReader(f)
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
                    except Exception as e:
                        logger.warning(f"Failed to extract page {page_num}: {e}")
            if not text.strip():
                raise ValueError("PDF has no extractable text")
            return {'text': text}
        except ImportError:
            raise RuntimeError("PyPDF2 not installed. Install with: pip install PyPDF2")
