import uuid
import logging
from datetime import datetime, timezone
from typing import Dict, Any

from fastapi import HTTPException

from app.interfaces.document_processor import IDocumentProcessor
from app.interfaces.chunker import IChunker
from app.interfaces.vector_store import IVectorStore

logger = logging.getLogger(__name__)


class SessionController:
    def __init__(self, sessions: Dict[str, Any], document_processor: IDocumentProcessor, chunker: IChunker, vector_store: IVectorStore):
        self._sessions = sessions
        self._document_processor = document_processor
        self._chunker = chunker
        self._vector_store = vector_store

    def create_session(self, user_id=None):
        session_id = f"session_{uuid.uuid4().hex[:12]}"
        self._sessions[session_id] = {'iteration': 0, 'excluded_chunks': [], 'created_at': datetime.now(timezone.utc).isoformat(), 'user_id': user_id, 'swap_initialized': False}
        return {'session_id': session_id, 'created_at': datetime.now(timezone.utc).isoformat()}

    def clear_session(self, session_id):
        self._validate(session_id)
        result = self._vector_store.delete_session_chunks(session_id)
        self._chunker.reset_session_tracking(session_id)
        self._document_processor.clear_session(session_id)
        self._sessions.pop(session_id)
        return {"status": "success", "cleared_chunks": result['deleted'], "session_id": session_id, "message": "Session completely cleared"}

    def get_session_info(self, session_id):
        self._validate(session_id)
        return {"session_id": session_id, "documents": self._document_processor.get_session_state(session_id), "chunker_statistics": self._chunker.get_statistics(session_id), "session_data": self._sessions.get(session_id, {})}

    def list_session_documents(self, session_id):
        self._validate(session_id)
        docs = self._document_processor.get_session_documents(session_id).to_dict()
        active = docs.get('active_documents', [])
        result = [{'filename': d.get('document_name', 'unknown'), 'chunks': len(d.get('chunks', [])), 'size': d.get('file_size', 0), 'uuid': d.get('document_uuid', ''), 'version': d.get('version', 1)} for d in active if isinstance(d, dict)]
        return {"session_id": session_id, "total_chunks": sum(d.get('chunks', 0) for d in result), "documents": result}

    def _validate(self, session_id):
        if session_id not in self._sessions:
            raise HTTPException(status_code=404, detail="Session not found")
