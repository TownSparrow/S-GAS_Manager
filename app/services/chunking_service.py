import logging
from typing import List, Dict, Any, Set

import spacy

from app.interfaces.chunker import IChunker
from app.models.chunk import Chunk
from app.models.document import DocumentHeader

logger = logging.getLogger(__name__)


class ChunkingService(IChunker):
    def __init__(self, max_chunk_size: int = 512, overlap_size: int = 50, spacy_model: str = "ru_core_news_md"):
        self._max_chunk_size = max_chunk_size
        self._overlap_size = overlap_size
        try:
            self._nlp = spacy.load(spacy_model)
        except OSError:
            try:
                self._nlp = spacy.load("en_core_web_sm")
            except OSError:
                raise RuntimeError("No spaCy model found.")
        self._session_chunks_pool: Dict[str, List[Chunk]] = {}
        self._session_used_chunk_ids: Dict[str, Set[str]] = {}
        self._session_usage_history: Dict[str, Dict[int, Set[str]]] = {}
        self._session_iteration: Dict[str, int] = {}

    def initialize_document(self, text: str, doc_header: DocumentHeader, session_id: str) -> List[Chunk]:
        if not session_id:
            raise ValueError("session_id is required")
        if not doc_header.document_uuid:
            raise ValueError("DocumentHeader must have document_uuid")
        all_chunks = self._chunk_document(text, doc_header, session_id)
        if session_id not in self._session_chunks_pool:
            self._session_chunks_pool[session_id] = []
            self._session_used_chunk_ids[session_id] = set()
            self._session_usage_history[session_id] = {}
            self._session_iteration[session_id] = 0
        self._session_chunks_pool[session_id].extend(all_chunks)
        return all_chunks

    def _chunk_document(self, text: str, doc_header: DocumentHeader, session_id: str) -> List[Chunk]:
        if not text or not text.strip():
            return []
        try:
            doc = self._nlp(text[:1000000])
            sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
        except Exception:
            sentences = [s.strip() for s in text.split('.') if s.strip()]
        if not sentences:
            return []
        chunks, current_chunk, current_size, chunk_index = [], [], 0, 0
        for sentence in sentences:
            sentence_size = len(sentence.split())
            if current_size + sentence_size > self._max_chunk_size and current_chunk:
                chunks.append(Chunk.create(text=" ".join(current_chunk), chunk_index=chunk_index, doc_header=doc_header, session_id=session_id))
                chunk_index += 1
                overlap_count = max(1, len(current_chunk) // 4)
                current_chunk = current_chunk[-overlap_count:] + [sentence]
                current_size = sum(len(s.split()) for s in current_chunk)
            else:
                current_chunk.append(sentence)
                current_size += sentence_size
        if current_chunk:
            chunks.append(Chunk.create(text=" ".join(current_chunk), chunk_index=chunk_index, doc_header=doc_header, session_id=session_id))
        return chunks

    def mark_chunks_used(self, session_id: str, chunk_ids: List[str], iteration: int) -> None:
        if session_id not in self._session_used_chunk_ids:
            return
        if iteration not in self._session_usage_history[session_id]:
            self._session_usage_history[session_id][iteration] = set()
        self._session_usage_history[session_id][iteration].update(chunk_ids)
        self._session_used_chunk_ids[session_id].update(chunk_ids)
        self._session_iteration[session_id] = iteration

    def get_excluded_chunk_ids(self, session_id: str) -> Set[str]:
        return self._session_used_chunk_ids.get(session_id, set()).copy()

    def get_unused_chunks(self, session_id: str) -> List[Chunk]:
        if session_id not in self._session_chunks_pool:
            return []
        excluded = self._session_used_chunk_ids.get(session_id, set())
        return [c for c in self._session_chunks_pool[session_id] if c.id not in excluded]

    def get_statistics(self, session_id: str) -> Dict[str, Any]:
        if session_id not in self._session_chunks_pool:
            return {'total_chunks_in_pool': 0, 'used_chunks': 0, 'unused_chunks': 0, 'coverage_percent': 0.0, 'current_iteration': 0, 'total_iterations': 0}
        total = len(self._session_chunks_pool[session_id])
        used = len(self._session_used_chunk_ids[session_id])
        return {'session_id': session_id, 'total_chunks_in_pool': total, 'used_chunks': used, 'unused_chunks': total - used, 'coverage_percent': 100 * used / total if total > 0 else 0, 'current_iteration': self._session_iteration.get(session_id, 0), 'total_iterations': len(self._session_usage_history.get(session_id, {}))}

    def reset_session_tracking(self, session_id: str) -> None:
        if session_id in self._session_used_chunk_ids:
            self._session_used_chunk_ids[session_id] = set()
        if session_id in self._session_usage_history:
            self._session_usage_history[session_id] = {}
        if session_id in self._session_iteration:
            self._session_iteration[session_id] = 0

    def clear_session(self, session_id: str) -> None:
        self._session_chunks_pool.pop(session_id, None)
        self._session_used_chunk_ids.pop(session_id, None)
        self._session_usage_history.pop(session_id, None)
        self._session_iteration.pop(session_id, None)

    def get_all_chunks(self, session_id: str) -> List[Dict[str, Any]]:
        return self._session_chunks_pool.get(session_id, []).copy()
