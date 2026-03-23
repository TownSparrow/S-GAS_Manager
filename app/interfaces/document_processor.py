from pathlib import Path
from typing import Dict, Any, Optional

from app.models.document import DocumentProcessingResult
from app.models.session import SessionDocumentMetadata


class IDocumentProcessor:
    def process_document(self, file_path: Path, session_id: str, metadata: Optional[Dict[str, Any]] = None) -> DocumentProcessingResult:
        raise NotImplementedError

    def get_session_documents(self, session_id: str) -> SessionDocumentMetadata:
        raise NotImplementedError

    def get_session_state(self, session_id: str) -> Dict[str, Any]:
        raise NotImplementedError

    def clear_session(self, session_id: str) -> None:
        raise NotImplementedError
