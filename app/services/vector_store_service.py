import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
import chromadb
import chromadb.errors
from chromadb.config import Settings as ChromaSettings

from app.interfaces.vector_store import IVectorStore

logger = logging.getLogger(__name__)


class ChromaVectorStoreService(IVectorStore):
    def __init__(self, persist_directory: str, collection_name: str):
        self._persist_directory = Path(persist_directory)
        self._persist_directory.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=str(self._persist_directory),
            settings=ChromaSettings(anonymized_telemetry=False, allow_reset=True),
        )
        self._collection_name = collection_name
        self._collection = self._get_or_create_collection()

    def _get_or_create_collection(self):
        try:
            return self._client.get_collection(name=self._collection_name)
        except chromadb.errors.NotFoundError:
            return self._client.create_collection(name=self._collection_name, metadata={"hnsw:space": "cosine"})

    def add_chunks(self, chunks: List[Dict[str, Any]], embeddings: np.ndarray) -> Dict[str, Any]:
        if len(chunks) != len(embeddings):
            raise ValueError("Number of chunks must match number of embeddings")
        ids = [c['id'] for c in chunks]
        documents = [c['text'] for c in chunks]
        metadatas = [c['metadata'] for c in chunks]

        for chunk_id, metadata in zip(ids, metadatas):
            if 'session_id' not in metadata or 'document_uuid' not in metadata:
                raise ValueError(f"Missing required metadata in chunk {chunk_id}")

        self._collection.add(ids=ids, embeddings=embeddings.tolist(), documents=documents, metadatas=metadatas)
        return {'status': 'success', 'chunks_added': len(chunks)}

    async def search(self, query_embedding: np.ndarray, session_id: str, top_k: int = 10, filter_metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        where_filter = {"session_id": {"$eq": session_id}}
        if filter_metadata:
            where_filter = {"$and": [where_filter, filter_metadata]}
        emb = query_embedding.tolist() if isinstance(query_embedding, np.ndarray) else query_embedding
        results = self._collection.query(query_embeddings=[emb], where=where_filter, n_results=top_k, include=['documents', 'metadatas', 'distances'])
        return [
            {'id': results['ids'][0][i], 'text': results['documents'][0][i], 'metadata': results['metadatas'][0][i], 'similarity_score': 1 - results['distances'][0][i]}
            for i in range(len(results['ids'][0]))
        ]

    async def search_exclude_chunks(self, query_embedding: np.ndarray, session_id: str, exclude_ids: List[str], top_k: int = 10) -> List[Dict[str, Any]]:
        where_filter = {"session_id": {"$eq": session_id}}
        emb = query_embedding.tolist() if isinstance(query_embedding, np.ndarray) else query_embedding
        exclude_set = set(exclude_ids)
        results = self._collection.query(query_embeddings=[emb], where=where_filter, n_results=top_k * 3, include=['documents', 'metadatas', 'distances'])
        formatted = []
        for i in range(len(results['ids'][0])):
            cid = results['ids'][0][i]
            if cid not in exclude_set:
                formatted.append({'id': cid, 'text': results['documents'][0][i], 'metadata': results['metadatas'][0][i], 'similarity_score': 1 - results['distances'][0][i]})
                if len(formatted) >= top_k:
                    break
        return formatted

    def delete_session_chunks(self, session_id: str) -> Dict[str, Any]:
        results = self._collection.get(where={"session_id": {"$eq": session_id}}, include=[])
        chunk_ids = results['ids']
        if chunk_ids:
            self._collection.delete(ids=chunk_ids)
        return {'deleted': len(chunk_ids), 'session_id': session_id}

    def delete_chunks_by_document_uuid(self, session_id: str, document_uuid: str) -> int:
        where = {"$and": [{"session_id": {"$eq": session_id}}, {"document_uuid": {"$eq": document_uuid}}]}
        results = self._collection.get(where=where, include=[])
        ids = results['ids']
        if ids:
            self._collection.delete(ids=ids)
        return len(ids)

    def delete_chunks_by_ids(self, chunk_ids: List[str]) -> int:
        if not chunk_ids:
            return 0
        self._collection.delete(ids=chunk_ids)
        return len(chunk_ids)

    async def get_all_session_chunks(self, session_id: str) -> List[Dict[str, Any]]:
        try:
            results = self._collection.get(where={"session_id": {"$eq": session_id}}, include=['documents', 'metadatas', 'embeddings'])
            if not results['ids']:
                return []
            chunks = []
            embeddings_list = results.get('embeddings') or []
            has_embeddings = isinstance(embeddings_list, list) and len(embeddings_list) > 0
            for i in range(len(results['ids'])):
                embedding = np.array(embeddings_list[i]) if has_embeddings and i < len(embeddings_list) else None
                chunks.append({'id': results['ids'][i], 'text': results['documents'][i], 'metadata': results['metadatas'][i], 'embedding': embedding})
            return chunks
        except Exception as e:
            logger.error(f"Failed to get all chunks: {e}")
            return []

    def get_chunks_by_document(self, session_id: str, document_uuid: str) -> List[Dict[str, Any]]:
        where = {"$and": [{"session_id": {"$eq": session_id}}, {"document_uuid": {"$eq": document_uuid}}]}
        results = self._collection.get(where=where, include=['documents', 'metadatas'])
        return [{'id': results['ids'][i], 'text': results['documents'][i], 'metadata': results['metadatas'][i]} for i in range(len(results['ids']))]

    def get_collection_info(self) -> Dict[str, Any]:
        return {'name': self._collection_name, 'count': self._collection.count(), 'persist_directory': str(self._persist_directory)}
