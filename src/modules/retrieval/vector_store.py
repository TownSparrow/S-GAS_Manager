import chromadb
from chromadb.config import Settings
import numpy as np
from typing import List, Dict, Any, Optional
import logging
from pathlib import Path
import chromadb.errors

logger = logging.getLogger(__name__)

class ChromaVectorStore:
    """ChromaDB wrapper for document storage and retrieval"""
    
    def __init__(self, persist_directory: str = "data/chroma_db", collection_name: str = "documents"):
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
        
        logger.info(f"ChromaDB initialized at {self.persist_directory}")
    
    def _get_or_create_collection(self):
        """Get existing collection or create new one"""
        try:
            collection = self.client.get_collection(name=self.collection_name)
            logger.info(f"Loaded existing collection: {self.collection_name}")
        except chromadb.errors.NotFoundError:
            # Collection doesn't exist, create it
            collection = self.client.create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
            )
            logger.info(f"Created new collection: {self.collection_name}")
        
        return collection
    
    async def add_chunks(self, chunks: List[Dict[str, Any]], embeddings: np.ndarray):
        """Add document chunks to the vector store"""
        
        if len(chunks) != len(embeddings):
            raise ValueError("Number of chunks must match number of embeddings")
        
        # Prepare data for ChromaDB
        ids = [chunk['id'] for chunk in chunks]
        documents = [chunk['text'] for chunk in chunks]
        metadatas = [chunk['metadata'] for chunk in chunks]
        embeddings_list = embeddings.tolist()
        
        # Add to collection
        self.collection.add(
            ids=ids,
            embeddings=embeddings_list,
            documents=documents,
            metadatas=metadatas
        )
        
        logger.info(f"Added {len(chunks)} chunks to ChromaDB")
    
    async def search(self, 
                    query_embedding: np.ndarray, 
                    top_k: int = 10,
                    filter_metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Search for similar chunks"""
        
        # Perform similarity search
        results = self.collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=top_k,
            where=filter_metadata,
            include=['documents', 'metadatas', 'distances']
        )
        
        # Format results
        chunks = []
        for i in range(len(results['ids'][0])):
            chunk = {
                'id': results['ids'][0][i],
                'text': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'similarity_score': 1 - results['distances'][0][i]  # Convert distance to similarity
            }
            chunks.append(chunk)
        
        logger.info(f"Retrieved {len(chunks)} chunks for query")
        return chunks
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the current collection"""
        count = self.collection.count()
        return {
            'name': self.collection_name,
            'count': count,
            'persist_directory': str(self.persist_directory)
        }