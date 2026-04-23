from pathlib import Path
from typing import Dict, Any
import logging

from app.interfaces.document_loader import IDocumentLoader

logger = logging.getLogger(__name__)


class TextLoader(IDocumentLoader):
    def load(self, file_path: Path) -> Dict[str, Any]:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            return {'text': text}
        except UnicodeDecodeError:
            with open(file_path, 'r', encoding='latin-1') as f:
                text = f.read()
            return {'text': text}
