from typing import Dict, Any
from fastapi import HTTPException

from app.interfaces.embedding import IEmbeddingService
from app.interfaces.vector_store import IVectorStore


class SearchController:
    def __init__(self, embedding_service: IEmbeddingService, vector_store: IVectorStore, sessions: Dict[str, Any]):
        self._embedding = embedding_service
        self._vector_store = vector_store
        self._sessions = sessions

    async def search(self, session_id, query, top_k=10):
        if session_id not in self._sessions:
            raise HTTPException(status_code=404, detail="Session not found")
        q = self._embedding.get_embeddings([query])[0]
        results = await self._vector_store.search(q, session_id=session_id, top_k=top_k)
        return {"query": query, "session_id": session_id, "results": results}
