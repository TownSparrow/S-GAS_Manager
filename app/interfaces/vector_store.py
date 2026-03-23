from typing import List, Dict, Any, Optional
import numpy as np


class IVectorStore:
    def add_chunks(self, chunks: List[Dict[str, Any]], embeddings: np.ndarray) -> Dict[str, Any]:
        raise NotImplementedError

    async def search(self, query_embedding: np.ndarray, session_id: str, top_k: int = 10, filter_metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        raise NotImplementedError

    async def search_exclude_chunks(self, query_embedding: np.ndarray, session_id: str, exclude_ids: List[str], top_k: int = 10) -> List[Dict[str, Any]]:
        raise NotImplementedError

    def delete_session_chunks(self, session_id: str) -> Dict[str, Any]:
        raise NotImplementedError

    def delete_chunks_by_document_uuid(self, session_id: str, document_uuid: str) -> int:
        raise NotImplementedError

    def delete_chunks_by_ids(self, chunk_ids: List[str]) -> int:
        raise NotImplementedError

    async def get_all_session_chunks(self, session_id: str) -> List[Dict[str, Any]]:
        raise NotImplementedError

    def get_chunks_by_document(self, session_id: str, document_uuid: str) -> List[Dict[str, Any]]:
        raise NotImplementedError

    def get_collection_info(self) -> Dict[str, Any]:
        raise NotImplementedError
