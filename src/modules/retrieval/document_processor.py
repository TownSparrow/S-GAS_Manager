import asyncio
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
import logging

from .document_loader import DocumentLoader, PDFLoader, TextLoader, DOCXLoader
from .chunking import SemanticChunker
from .vector_store import ChromaVectorStore
from .embedder import get_embeddings
from .retrieval_models import (
    DocumentHeader,
    Chunk,
    DocumentProcessingResult,
    SessionDocumentMetadata
)

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

        # Control of documents in sessions
        self.session_documents: Dict[str, SessionDocumentMetadata] = {}

        logger.info("✅ DocumentProcessor initialized")
    
    def get_session_documents(self, session_id: str) -> SessionDocumentMetadata:
        """
        Getting or creating SessionDocumentMetadata for session.
        """

        if session_id not in self.session_documents:
            self.session_documents[session_id] = SessionDocumentMetadata(session_id=session_id)
        return self.session_documents[session_id]

    def process_document(
        self,
        file_path: Path,
        session_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> DocumentProcessingResult:

        """Process a single document with adaptive chunking and session-based tracking"""

        if not session_id:
            raise ValueError("⚠️ Warning: session_id is required!")
        
        if not file_path.exists():
            raise FileNotFoundError(f"❌ Document not found: {file_path}")
        
        file_extension = file_path.suffix.lower()
        if file_extension not in self.loaders:
            raise ValueError(f"❌ Unsupported file type: {file_extension}")
        
        logger.info(f"Processing document: {file_path.name} for session {session_id}")
        
        try:
            # Step 1. Creating DocumentHeader with UUID
            doc_header = DocumentHeader.from_file_path(
                file_path=file_path,
                session_id=session_id,
                document_type=metadata.get('document_type', 'general') if metadata else 'general'
            )

            logger.info(f"✅ Created DocumentHeader with UUID: {doc_header.document_uuid}")

            # Step 2. Checking the older version of this document
            session_docs = self.get_session_documents(session_id)
            previous_uuid = None

            if session_docs.document_exists(file_path.name):
                old_doc = session_docs.get_document_by_name(file_path.name)
                previous_uuid = old_doc.document_uuid
                doc_header.previous_uuid = previous_uuid
                doc_header.version = old_doc.version + 1
                
                logger.info(f"Detected document update: v{doc_header.version} (replacing v{old_doc.version})")

            # Step 3. Loading document
            loader = self.loaders[file_extension]
            document_data = loader.load(file_path)

            logger.info(f"✅ Loaded document: {len(document_data['text'])} chars")

            # Step 4. Chunking the document
            all_chunks = self.chunker.initialize_document(
                text=document_data['text'],
                doc_header=doc_header,
                session_id=session_id
            )
            
            logger.info(f"✅ Created {len(all_chunks)} chunks")

            # Step 5. Saving metadata of document to session
            session_docs.add_document(doc_header)
            logger.info(f"✅ Added document metadata to session")

            # Step 6. If update, delete old chunks
            if previous_uuid:
                deleted_count = self.vector_store.delete_chunks_by_document_uuid(
                    session_id=session_id,
                    document_uuid=previous_uuid
                )
                logger.info(f"Deleted {deleted_count} old chunks from version {old_doc.version}")

            # Step 7. Generating embeddings
            chunk_texts = [chunk.text for chunk in all_chunks]
            logger.info(f"Getting embeddings for {len(chunk_texts)} chunks...")

            embeddings = get_embeddings(chunk_texts)
            logger.info(f"✅ Generated {len(embeddings)} embeddings")

            # Step 8. Adding chunks into vector store
            chunk_dicts = [chunk.to_vector_store_format() for chunk in all_chunks]
            self.vector_store.add_chunks(chunk_dicts, embeddings)

            logger.info(f"✅ Successfully added {len(all_chunks)} chunks to vector store")

            # Return result
            return DocumentProcessingResult(
                status='success',
                document_uuid=doc_header.document_uuid,
                document_name=file_path.name,
                session_id=session_id,
                chunks_created=len(all_chunks),
                previous_uuid=previous_uuid
            )
        
        except Exception as e:
            logger.error(f"❌ Failed to process document: {e}")
            return DocumentProcessingResult(
                status='error',
                document_uuid='',
                document_name=file_path.name,
                session_id=session_id,
                chunks_created=0,
                error=str(e),
                error_details={'exception_type': type(e).__name__}
            )

    async def process_directory(
        self,
        directory_path: Path,
        session_id: str,
        recursive: bool = True
    ) -> List[DocumentProcessingResult]:
        """Process all supported documents in a directory"""
        
        if not directory_path.exists() or not directory_path.is_dir():
            raise ValueError(f"❌ Invalid directory: {directory_path}")
        
        # Find all supported files
        pattern = "**/*" if recursive else "*"
        supported_extensions = set(self.loaders.keys())
        
        files_to_process = [
            f for f in directory_path.glob(pattern)
            if f.is_file() and f.suffix.lower() in supported_extensions
        ]
        
        logger.info(f"Found {len(files_to_process)} documents to process")
        
        # Process with parallelism limit
        semaphore = asyncio.Semaphore(3)
        
        async def process_with_semaphore(file_path):
            async with semaphore:
                try:
                    return await self.process_document(file_path, session_id)
                except Exception as e:
                    logger.error(f"❌ Failed to process {file_path.name}: {e}")
                    return DocumentProcessingResult(
                        status='error',
                        document_uuid='',
                        document_name=file_path.name,
                        session_id=session_id,
                        chunks_created=0,
                        error=str(e)
                    )
        
        results = await asyncio.gather(*[
            process_with_semaphore(file_path) for file_path in files_to_process
        ])
        
        successful = sum(1 for r in results if r.status == 'success')
        logger.info(f"✅ Document processing complete: {successful}/{len(results)} successful")
        
        return results
    
    def get_session_state(self, session_id: str) -> Dict[str, Any]:
        """Getting state of documents in session"""

        session_docs = self.get_session_documents(session_id)
        return session_docs.to_dict()

    def clear_session(self, session_id: str) -> None:
        """Clear documents of session"""
        
        if session_id in self.session_documents:
            del self.session_documents[session_id]
        logger.info(f"✅ Session {session_id} cleared from document processor")