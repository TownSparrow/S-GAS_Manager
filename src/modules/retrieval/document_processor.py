import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging

from .document_loader import DocumentLoader, PDFLoader, TextLoader, DOCXLoader
from .chunking import SemanticChunker
from .vector_store import ChromaVectorStore
from .embedder import get_embeddings

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Main document processing pipeline"""
    
    def __init__(self, vector_store: ChromaVectorStore, chunker: SemanticChunker):
        self.vector_store = vector_store
        self.chunker = chunker
        
        # Initialize loaders
        self.loaders = {
            '.pdf': PDFLoader(),
            '.txt': TextLoader(),
            '.docx': DOCXLoader(),
            '.doc': DOCXLoader(),
        }
    

    async def process_document(self, 
                          file_path: Path, 
                          session_id: str,
                          metadata: Optional[Dict[str, Any]] = None
                          ) -> Dict[str, Any]:
        """
        Process a single document with adaptive chunking and session-based tracking
        
        Args:
            file_path: Path to the document
            session_id: ID of the session
            metadata: Optional metadata for the document
            
        Returns:
            Dictionary containing processing results

        Process:
        1. Load document
        2. Prepare metadata without priorities
        3. Divide into chunks
        4. Generate embeddings
        5. Save in vector store
        """

        if not session_id:
            raise ValueError("⚠️ Warning: session_id is required!")
        
        if not file_path.exists():
            raise FileNotFoundError(f"Document not found: {file_path}")
        
        file_extension = file_path.suffix.lower()
        if file_extension not in self.loaders:
            raise ValueError(f"Unsupported file type: {file_extension}")
        
        logger.info(f"Processing document: {file_path.name}")
        
        # Step 1: Load document
        loader = self.loaders[file_extension]
        document_data = await loader.load(file_path)
        
        # Step 2: Prepare metadata
        doc_metadata = {
            'document_id': file_path.stem,
            'document_name': file_path.name,
            'file_path': str(file_path),
            'file_extension': file_extension,
            'file_size': file_path.stat().st_size,
            'uploaded_at': datetime.now(timezone.utc).isoformat(),
            'session_id': session_id,
            'document_type': metadata.get('document_type', 'general') if metadata else 'general',
            **(metadata or {})
        }
        
        # Step 3: Creating a full chunk pool
        all_chunks = self.chunker.initialize_document(
            document_data['text'],
            doc_metadata,
            session_id=session_id
        )
        
        # Step 4: Generate embeddings
        chunk_texts = [chunk['text'] for chunk in all_chunks]
        embeddings = await get_embeddings(chunk_texts)
        
        # Step 5: Store in vector database
        await self.vector_store.add_chunks(all_chunks, embeddings)
        
        result = await self.vector_store.add_chunks(
            all_chunks,
            embeddings,
            session_id=session_id
        )
        
        logger.info(f"✅ Successfully processed {file_path.name}: {len(all_chunks)} chunks created")
        
        return result
    

    async def process_directory(self, directory_path: Path, recursive: bool = True) -> List[Dict[str, Any]]:
        """Process all supported documents in a directory"""
        if not directory_path.exists() or not directory_path.is_dir():
            raise ValueError(f"Invalid directory: {directory_path}")
        
        # Find all supported files
        pattern = "**/*" if recursive else "*"
        supported_extensions = set(self.loaders.keys())
        files_to_process = [
            f for f in directory_path.glob(pattern)
            if f.is_file() and f.suffix.lower() in supported_extensions
        ]
        
        logger.info(f"Found {len(files_to_process)} documents to process")
        
        semaphore = asyncio.Semaphore(3)
        
        async def process_with_semaphore(file_path):
            async with semaphore:
                try:
                    return await self.process_document(file_path)
                except Exception as e:
                    logger.error(f"Failed to process {file_path.name}: {e}")
                    return {
                        'document_id': file_path.stem,
                        'filename': file_path.name,
                        'status': 'error',
                        'error': str(e)
                    }
        
        results = await asyncio.gather(*[
            process_with_semaphore(file_path) for file_path in files_to_process
        ])
        
        successful = sum(1 for r in results if r['status'] == 'success')
        logger.info(f"Document processing complete: {successful}/{len(results)} successful")
        
        return results