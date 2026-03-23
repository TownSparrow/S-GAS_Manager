from typing import List, Dict, Any, Set

from app.models.chunk import Chunk
from app.models.document import DocumentHeader


class IChunker:
    def initialize_document(self, text: str, doc_header: DocumentHeader, session_id: str) -> List[Chunk]:
        raise NotImplementedError

    def mark_chunks_used(self, session_id: str, chunk_ids: List[str], iteration: int) -> None:
        raise NotImplementedError

    def get_excluded_chunk_ids(self, session_id: str) -> Set[str]:
        raise NotImplementedError

    def get_unused_chunks(self, session_id: str) -> List[Chunk]:
        raise NotImplementedError

    def get_statistics(self, session_id: str) -> Dict[str, Any]:
        raise NotImplementedError

    def reset_session_tracking(self, session_id: str) -> None:
        raise NotImplementedError

    def clear_session(self, session_id: str) -> None:
        raise NotImplementedError

    def get_all_chunks(self, session_id: str) -> List[Dict[str, Any]]:
        raise NotImplementedError
