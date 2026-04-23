from typing import Dict, Any
from fastapi import HTTPException

from app.services.chat_service import ChatService


class ChatController:
    def __init__(self, chat_service: ChatService, sessions: Dict[str, Any]):
        self._chat_service = chat_service
        self._sessions = sessions

    async def chat(self, session_id, message, use_rag=True, n_chunks=5, temperature=None, top_p=None, max_tokens=None):
        if session_id not in self._sessions:
            raise HTTPException(status_code=404, detail="Session not found")
        return await self._chat_service.process_chat(session_id=session_id, session_data=self._sessions[session_id], message=message, use_rag=use_rag, n_chunks=n_chunks, temperature=temperature, top_p=top_p, max_tokens=max_tokens)
