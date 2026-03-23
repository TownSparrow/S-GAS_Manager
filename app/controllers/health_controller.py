from datetime import datetime, timezone
from typing import Dict, Any

from app.interfaces.vllm_client import IVLLMClient
from app.interfaces.vector_store import IVectorStore
from app.interfaces.chunker import IChunker
from app.consts.defaults import API_VERSION


class HealthController:
    def __init__(self, vllm_client: IVLLMClient, vector_store: IVectorStore,
                 chunker: IChunker, model_name: str, sessions: Dict[str, Any]):
        self._vllm = vllm_client
        self._vector_store = vector_store
        self._chunker = chunker
        self._model_name = model_name
        self._sessions = sessions

    async def health_check(self) -> Dict[str, Any]:
        vllm_status = await self._vllm.check_health()
        vector_store_status = "ready" if self._vector_store is not None else "not_initialized"
        chunker_status = "ready" if self._chunker is not None else "not_initialized"
        return {
            "status": "healthy",
            "vllm_status": vllm_status,
            "vector_store_status": vector_store_status,
            "chunker_status": chunker_status,
            "model": self._model_name,
            "active_sessions": len(self._sessions),
            "time": datetime.now(timezone.utc).isoformat(),
            "api_version": API_VERSION,
        }
