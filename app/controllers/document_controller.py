import logging
from pathlib import Path
from typing import Dict, Any

import aiofiles
from fastapi import HTTPException, UploadFile

from app.interfaces.document_processor import IDocumentProcessor
from app.models.document import DocumentProcessingResult

logger = logging.getLogger(__name__)


class DocumentController:
    def __init__(self, document_processor: IDocumentProcessor, sessions: Dict[str, Any]):
        self._document_processor = document_processor
        self._sessions = sessions

    async def upload_document(self, session_id, file: UploadFile, document_type="general"):
        if session_id not in self._sessions:
            raise HTTPException(status_code=404, detail="Session not found")
        import uuid as _uuid
        safe_name = f"{_uuid.uuid4().hex[:8]}_{file.filename}"
        temp_path = Path(f"/tmp/{safe_name}")
        try:
            async with aiofiles.open(temp_path, 'wb') as f:
                await f.write(await file.read())
            result = self._document_processor.process_document(temp_path, session_id=session_id, metadata={"document_type": document_type})
            return result.to_dict()
        except Exception as e:
            return DocumentProcessingResult(status='error', document_uuid='', document_name=file.filename, session_id=session_id, chunks_created=0, error=str(e), error_details={'exception_type': type(e).__name__}).to_dict()
        finally:
            if temp_path.exists():
                temp_path.unlink()
