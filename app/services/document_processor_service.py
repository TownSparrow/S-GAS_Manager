import logging
from pathlib import Path
from typing import Dict, Any, Optional

from app.interfaces.document_processor import IDocumentProcessor
from app.interfaces.vector_store import IVectorStore
from app.interfaces.chunker import IChunker
from app.interfaces.embedding import IEmbeddingService
from app.interfaces.document_loader import IDocumentLoader
from app.models.document import DocumentHeader, DocumentProcessingResult
from app.models.session import SessionDocumentMetadata

logger = logging.getLogger(__name__)


class DocumentProcessorService(IDocumentProcessor):
    def __init__(self, vector_store: IVectorStore, chunker: IChunker, embedding_service: IEmbeddingService, loaders: Dict[str, IDocumentLoader]):
        self._vector_store = vector_store
        self._chunker = chunker
        self._embedding_service = embedding_service
        self._loaders = loaders
        self._session_documents: Dict[str, SessionDocumentMetadata] = {}

    def get_session_documents(self, session_id: str) -> SessionDocumentMetadata:
        if session_id not in self._session_documents:
            self._session_documents[session_id] = SessionDocumentMetadata(session_id=session_id)
        return self._session_documents[session_id]

    def process_document(self, file_path: Path, session_id: str, metadata: Optional[Dict[str, Any]] = None) -> DocumentProcessingResult:
        if not session_id:
            raise ValueError("session_id is required")
        if not file_path.exists():
            raise FileNotFoundError(f"Document not found: {file_path}")
        ext = file_path.suffix.lower()
        if ext not in self._loaders:
            raise ValueError(f"Unsupported file type: {ext}")
        try:
            doc_header = DocumentHeader.from_file_path(file_path=file_path, session_id=session_id, document_type=metadata.get('document_type', 'general') if metadata else 'general')
            session_docs = self.get_session_documents(session_id)
            previous_uuid = None
            if session_docs.document_exists(file_path.name):
                old_doc = session_docs.get_document_by_name(file_path.name)
                previous_uuid = old_doc.document_uuid
                doc_header.previous_uuid = previous_uuid
                doc_header.version = old_doc.version + 1
            document_data = self._loaders[ext].load(file_path)
            all_chunks = self._chunker.initialize_document(text=document_data['text'], doc_header=doc_header, session_id=session_id)
            session_docs.add_document(doc_header)
            if previous_uuid:
                self._vector_store.delete_chunks_by_document_uuid(session_id=session_id, document_uuid=previous_uuid)
            chunk_texts = [chunk.text for chunk in all_chunks]
            embeddings = self._embedding_service.get_embeddings(chunk_texts)
            chunk_dicts = [chunk.to_vector_store_format() for chunk in all_chunks]
            self._vector_store.add_chunks(chunk_dicts, embeddings)
            return DocumentProcessingResult(status='success', document_uuid=doc_header.document_uuid, document_name=file_path.name, session_id=session_id, chunks_created=len(all_chunks), previous_uuid=previous_uuid)
        except Exception as e:
            logger.error(f"Failed to process document: {e}")
            return DocumentProcessingResult(status='error', document_uuid='', document_name=file_path.name, session_id=session_id, chunks_created=0, error=str(e), error_details={'exception_type': type(e).__name__})

    async def process_directory(self, directory_path: Path, session_id: str, recursive: bool = True) -> list:
        import asyncio
        if not directory_path.exists() or not directory_path.is_dir():
            raise ValueError(f"Invalid directory: {directory_path}")

        pattern = "**/*" if recursive else "*"
        supported_extensions = set(self._loaders.keys())
        files_to_process = [
            f for f in directory_path.glob(pattern)
            if f.is_file() and f.suffix.lower() in supported_extensions
        ]

        semaphore = asyncio.Semaphore(3)

        async def process_with_semaphore(file_path):
            async with semaphore:
                try:
                    return self.process_document(file_path, session_id)
                except Exception as e:
                    return DocumentProcessingResult(
                        status='error', document_uuid='', document_name=file_path.name,
                        session_id=session_id, chunks_created=0, error=str(e),
                    )

        results = await asyncio.gather(*[process_with_semaphore(f) for f in files_to_process])
        return list(results)

    def get_session_state(self, session_id: str) -> Dict[str, Any]:
        return self.get_session_documents(session_id).to_dict()

    def clear_session(self, session_id: str) -> None:
        self._session_documents.pop(session_id, None)
