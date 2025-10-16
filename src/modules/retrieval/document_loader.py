from pathlib import Path
from typing import List, Dict, Any
import logging
from abc import ABC, abstractmethod

class DocumentLoader(ABC):
    """Base class for document loaders"""
    
    @abstractmethod
    async def load(self, file_path: Path) -> Dict[str, Any]:
        pass

class PDFLoader(DocumentLoader):
    async def load(self, file_path: Path) -> Dict[str, Any]:
        import PyPDF2
        text = ""
        with open(file_path, 'rb') as f:
            pdf_reader = PyPDF2.PdfReader(f)
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
        return {'text': text}

class TextLoader(DocumentLoader):
    async def load(self, file_path: Path) -> Dict[str, Any]:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        return {'text': text}

class DOCXLoader(DocumentLoader):
    async def load(self, file_path: Path) -> Dict[str, Any]:
        from docx import Document
        doc = Document(file_path)
        text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
        return {'text': text}