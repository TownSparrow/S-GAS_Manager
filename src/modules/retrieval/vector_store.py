import chromadb
from chromadb.config import Settings
import numpy as np
from typing import List, Dict, Any, Optional, Set
import logging
from pathlib import Path
import chromadb.errors

from .retrieval_models import SearchResult, Chunk

logger = logging.getLogger(__name__)


class ChromaVectorStore:
    """ChromaDB wrapper for document storage and retrieval"""
    
    def __init__(
        self,
        persist_directory: str = "data/chroma_db",
        collection_name: str = "documents"
    ):
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        self.collection_name = collection_name
        self.collection = self._get_or_create_collection()
        
        logger.info(f"✅ ChromaDB initialized at {self.persist_directory}")
    
    def _get_or_create_collection(self):
        """Getting existing collection or create new one"""

        try:
            collection = self.client.get_collection(name=self.collection_name)
            logger.info(f"✅ Loaded existing collection: {self.collection_name}")

        except chromadb.errors.NotFoundError:
            # If collection doesn't exist, create it
            collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info(f"✅ Created new collection: {self.collection_name}")
        
        return collection
    
    def add_chunks(self, chunks: List[Dict[str, Any]], embeddings: np.ndarray):
        """Adding document chunks to the vector store"""
        
        if len(chunks) != len(embeddings):
            raise ValueError("❌ Number of chunks must match number of embeddings")
        
        # Preparing data for ChromaDB
        ids = [chunk['id'] for chunk in chunks]
        documents = [chunk['text'] for chunk in chunks]
        metadatas = [chunk['metadata'] for chunk in chunks]
        embeddings_list = embeddings.tolist()
        
        # Checking structure
        for chunk_id, metadata in zip(ids, metadatas):
            if 'session_id' not in metadata or 'document_uuid' not in metadata:
                raise ValueError(f"❌ Missing required metadata in chunk {chunk_id}")

        # Adding to collection
        try:
            self.collection.add(
                ids=ids,
                embeddings=embeddings_list,
                documents=documents,
                metadatas=metadatas
            )
            
            logger.info(f"✅ Added {len(chunks)} chunks to ChromaDB")
            
            return {
                'status': 'success',
                'chunks_added': len(chunks)
            }
        
        except Exception as e:
            logger.error(f"❌ Failed to add chunks: {e}")
            raise
    
    async def search(
        self,
        query_embedding: np.ndarray,
        session_id: str,
        top_k: int = 10,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Searching in embeddings"""

        # Filter by session_id
        where_filter = {"session_id": {"$eq": session_id}}
    
        # Combine filters
        if filter_metadata:
            where_filter = {
                "$and": [
                    where_filter,
                    filter_metadata
                ]
            }
        
        logger.debug(f"Searching for top {top_k} chunks in session {session_id}")
    
        # Perform similarity search
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist() if isinstance(query_embedding, np.ndarray) else query_embedding],
            where=where_filter,
            n_results=top_k,
            include=['documents', 'metadatas', 'distances']
        )
    
        # Format results
        chunks = []

        for i in range(len(results['ids'][0])):
            chunk = {
                'id': results['ids'][0][i],
                'text': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'similarity_score': 1 - results['distances'][0][i]
            }
            chunks.append(chunk)
        
        logger.info(f"✅ Retrieved {len(chunks)} chunks")
        
        return chunks
    
    async def search_exclude_chunks(
        self,
        query_embedding: np.ndarray,
        session_id: str,
        exclude_ids: Set[str],
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Searching all chunks, but exclude already used ones.
        
        Used for S-GAS iterations:
        - Iteration 1: search without exclusions
        - Iteration 2: search excluding chunks from iteration 1
        - Iteration 3: search excluding chunks from iterations 1-2
        """

        # Filter via session ID
        where_filter = {"session_id": {"$eq": session_id}}
        
        logger.debug(f"Searching (excluding {len(exclude_ids)} chunks) for top {top_k} in session {session_id}")

        # Getting more results (for filtering)
        larger_k = top_k * 3
        
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist() if isinstance(query_embedding, np.ndarray) else query_embedding],
            where=where_filter,
            n_results=larger_k,
            include=['documents', 'metadatas', 'distances']
        )

        # Filter excluded IDs
        formatted_results = []

        for i in range(len(results['ids'][0])):
            chunk_id = results['ids'][0][i]
            
            if chunk_id not in exclude_ids:
                chunk = {
                    'id': chunk_id,
                    'text': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'similarity_score': 1 - results['distances'][0][i]
                }
                formatted_results.append(chunk)
                
                if len(formatted_results) >= top_k:
                    break
        
        logger.info(f"✅ Retrieved {len(formatted_results)} chunks (after exclusion)")
        
        return formatted_results

    def delete_session_chunks(self, session_id: str) -> Dict[str, Any]:
        """Deleting ALL chunks of the session"""

        where_filter = {"session_id": {"$eq": session_id}}
        
        # Getting all chunk IDs of the session
        results = self.collection.get(where=where_filter, include=[])
        chunk_ids = results['ids']
        
        logger.info(f"Deleting {len(chunk_ids)} chunks from session {session_id}")

        # Deleting
        if chunk_ids:
            self.collection.delete(ids=chunk_ids)
            logger.info(f"✅ Deleted {len(chunk_ids)} chunks")

        return {
            'deleted': len(chunk_ids),
            'session_id': session_id
        }

    def delete_chunks_by_document_uuid(
        self,
        session_id: str,
        document_uuid: str
    ) -> int:
        """Deleting all chuns of exact document file"""
        
        where_filter = {
            "$and": [
                {"session_id": {"$eq": session_id}},
                {"document_uuid": {"$eq": document_uuid}}
            ]
        }
        
        # Getting all IDs of chunks
        results = self.collection.get(where=where_filter, include=[])     
        chunk_ids = results['ids']
        
        logger.info(f"Deleting {len(chunk_ids)} chunks for document {document_uuid}")
        
        # Deleting
        if chunk_ids:
            self.collection.delete(ids=chunk_ids)
            logger.info(f"✅ Deleted {len(chunk_ids)} chunks for document {document_uuid}")
        
        return len(chunk_ids)

    def delete_chunks_by_ids(self, chunk_ids: List[str]) -> int:
        """Deleting exact chunks via their IDs"""
        
        if not chunk_ids:
            return 0
        
        logger.info(f"Deleting {len(chunk_ids)} specific chunks")
        
        self.collection.delete(ids=chunk_ids)
        
        logger.info(f"✅ Deleted {len(chunk_ids)} chunks")
        
        return len(chunk_ids)

    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the current collection"""

        count = self.collection.count()

        return {
            'name': self.collection_name,
            'count': count,
            'persist_directory': str(self.persist_directory)
        }
    
    def get_chunks_by_document(
        self,
        session_id: str,
        document_uuid: str
    ) -> List[Dict[str, Any]]:
        """Getting all document's chunks"""
        
        where_filter = {
            "$and": [
                {"session_id": {"$eq": session_id}},
                {"document_uuid": {"$eq": document_uuid}}
            ]
        }
        
        results = self.collection.get(
            where=where_filter,
            include=['documents', 'metadatas']
        )
        
        chunks = []
        
        for i in range(len(results['ids'])):
            chunk = {
                'id': results['ids'][i],
                'text': results['documents'][i],
                'metadata': results['metadatas'][i]
            }
            chunks.append(chunk)
        
        logger.info(f"✅ Retrieved {len(chunks)} chunks for document {document_uuid}")
        
        return chunks
    

    async def get_all_session_chunks(self, session_id: str) -> List[Dict[str, Any]]:
        """Getting all chunks from a session (no limit)"""
        try:
            where_filter = {"session_id": {"$eq": session_id}}
            logger.info(f"Retrieving ALL chunks for session {session_id}...")
        
            results = self.collection.get(
                where=where_filter,
                include=['documents', 'metadatas', 'embeddings']
            )
        
            if not results['ids']:
                logger.warning(f"⚠️ No chunks found for session {session_id}")
                return []
        
            chunks = []
            for i in range(len(results['ids'])):
                embedding = None
                if results['embeddings'] is not None and len(results['embeddings']) > i:
                    embedding = np.array(results['embeddings'][i])
            
                chunk = {
                    'id': results['ids'][i],
                    'text': results['documents'][i],
                    'metadata': results['metadatas'][i],
                    'embedding': embedding
                }
                chunks.append(chunk)
        
            logger.info(f"✅ Retrieved {len(chunks)} total chunks for session {session_id}")
        
            if chunks:
                avg_text_size = sum(len(c['text']) for c in chunks) / len(chunks)
                estimated_memory_mb = (len(chunks) * (384 * 4 + avg_text_size)) / (1024 * 1024)
                logger.info(f"   Estimated memory: ~{estimated_memory_mb:.2f} MB")
        
            return chunks
        
        except Exception as e:
            logger.error(f"❌ Failed to get all chunks: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return []
