import uuid
import hashlib
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Set
from datetime import datetime, timezone
from pathlib import Path


# ═══════════════════════════════════════════════════════════════════════════
# DOCUMENT'S METADATA
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class DocumentHeader:
    document_uuid: str
    document_name: str
    file_path: str
    file_size: int
    file_extension: str
    document_type: str = "general"
    version: int = 1 
    previous_uuid: Optional[str] = None
    uploaded_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    upload_sequence: int = 0
    
    @classmethod
    def from_file_path(
        cls,
        file_path: Path,
        session_id: str,
        document_type: str = "general"
    ) -> 'DocumentHeader':
        """
        A factory for creating a DocumentHeader from a file path.
        Automatically generates a UUID.
        """

        return cls(
            document_uuid=str(uuid.uuid4()),
            document_name=file_path.name,
            file_path=str(file_path),
            file_size=file_path.stat().st_size,
            file_extension=file_path.suffix.lower(),
            document_type=document_type,
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Converting to dictionary for saving/transferring"""

        return asdict(self)
    
    def mark_as_updated(self) -> None:
        """Updating timestamp"""

        self.updated_at = datetime.now(timezone.utc).isoformat()
    

# ═══════════════════════════════════════════════════════════════════════════
# CHUNK'S METADATA
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ChunkMetadata:
    session_id: str
    document_uuid: str
    chunk_index: int
    chunk_size: int
    chunk_text_hash: str
    document_name: str
    document_type: str 
    file_extension: str
    document_version: int = 1
    chunk_created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    document_uploaded_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for ChromaDB"""
        return asdict(self)
    
    @staticmethod
    def hash_text(text: str) -> str:
        """Generate a SHA-256 hash of text"""
        return hashlib.sha256(text.encode()).hexdigest()


# ═══════════════════════════════════════════════════════════════════════════
# CHUNK
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class Chunk:
    id: str
    text: str
    metadata: ChunkMetadata
    similarity_context: Optional[Dict[str, Any]] = None
    
    @staticmethod
    def create(
        text: str,
        chunk_index: int,
        doc_header: DocumentHeader,
        session_id: str,
    ) -> 'Chunk':
        """
        Factory for creating Chunks with the correct ID hierarchy.
        """

        # Creating a globally unique ID
        chunk_id = f"{session_id}:{doc_header.document_uuid}:{chunk_index}"
        
        # Creating metadata
        chunk_meta = ChunkMetadata(
            session_id=session_id,
            document_uuid=doc_header.document_uuid,
            chunk_index=chunk_index,
            chunk_size=len(text.split()),
            chunk_text_hash=ChunkMetadata.hash_text(text),
            document_name=doc_header.document_name,
            document_type=doc_header.document_type,
            file_extension=doc_header.file_extension,
            document_version=doc_header.version,
            document_uploaded_at=doc_header.uploaded_at,
        )
        
        return Chunk(
            id=chunk_id,
            text=text,
            metadata=chunk_meta
        )
    
    def to_vector_store_format(self) -> Dict[str, Any]:
        """
        Converting to ChromaDB/Pinecone/Weaviate format.
        """

        return {
            'id': self.id,
            'text': self.text,
            'metadata': self.metadata.to_dict()
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Converting to a regular dictionary"""
        
        return {
            'id': self.id,
            'text': self.text,
            'metadata': self.metadata.to_dict(),
            'similarity_context': self.similarity_context
        }
    

# ════════════════════════════════════════════════════════════════════════════
# SEARCH RESULT
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class SearchResult:
    id: str 
    text: str 
    metadata: Dict[str, Any]
    similarity_score: float
    document_info: Optional[Dict[str, Any]] = None
    related_chunks: Optional[List['SearchResult']] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """For JSON serialization"""
        return {
            'id': self.id,
            'text': self.text,
            'metadata': self.metadata,
            'similarity_score': self.similarity_score,
            'document_info': self.document_info,
            'related_chunks': [r.to_dict() for r in self.related_chunks] if self.related_chunks else None
        }
    

# ════════════════════════════════════════════════════════════════════════════
# PROCESSING RESULT
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class DocumentProcessingResult:
    status: str
    document_uuid: str
    document_name: str
    session_id: str
    chunks_created: int
    previous_uuid: Optional[str] = None
    error: Optional[str] = None
    error_details: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """For JSON serialization (API response)"""
        return {
            'status': self.status,
            'document_uuid': self.document_uuid,
            'document_name': self.document_name,
            'session_id': self.session_id,
            'chunks_created': self.chunks_created,
            'previous_uuid': self.previous_uuid,
            'error': self.error,
            'error_details': self.error_details
        }


# ════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class SessionDocumentMetadata:
    session_id: str
    documents: Dict[str, DocumentHeader] = field(default_factory=dict)
    document_upload_order: List[str] = field(default_factory=list)
    
    def add_document(self, doc_header: DocumentHeader) -> None:
        """
        Adding a new document to the session.
        Automatically sets the upload_sequence.
        """

        doc_header.upload_sequence = len(self.document_upload_order)
        self.documents[doc_header.document_uuid] = doc_header
        self.document_upload_order.append(doc_header.document_uuid)
    
    def get_active_documents(self) -> List[DocumentHeader]:
        """
        Getting only active documents (exclude old versions).
        """

        active_docs = {}
        
        for uuid, header in self.documents.items():
            # Check the version
            if header.previous_uuid and header.previous_uuid in self.documents:
                pass
            # Check if there is a ref of a new file to older one
            is_old_version = any(
                doc.previous_uuid == uuid
                for doc in self.documents.values()
            )
            # In other case, it is the new document or last version
            if not is_old_version:
                active_docs[uuid] = header
        
        return list(active_docs.values())
    
    def get_active_uuids(self) -> List[str]:
        """Getting only the UUIDs of active documents"""

        return [doc.document_uuid for doc in self.get_active_documents()]
    
    def get_document_by_name(self, document_name: str) -> Optional[DocumentHeader]:
        """Finding the LATEST version of a document by name"""
        
        matching_docs = [
            doc for doc in self.documents.values()
            if doc.document_name == document_name
        ]
        
        if not matching_docs:
            return None
        
        return max(matching_docs, key=lambda d: d.version)
    
    def document_exists(self, document_name: str) -> bool:
        """Checking if a document with this name is loaded"""

        return any(
            doc.document_name == document_name
            for doc in self.documents.values()
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Converting session state to dictionary"""

        return {
            'session_id': self.session_id,
            'documents': {
                uuid: doc.to_dict()
                for uuid, doc in self.documents.items()
            },
            'document_upload_order': self.document_upload_order,
            'active_documents': [doc.to_dict() for doc in self.get_active_documents()],
            'active_uuids': self.get_active_uuids(),
        }


# ════════════════════════════════════════════════════════════════════════════
# UTILITIES AND VALIDATION
# ════════════════════════════════════════════════════════════════════════════

def validate_chunk_structure(chunk_dict: Dict[str, Any]) -> bool:
    """
    Validating chunk structure for ChromaDB.
    """

    required_keys = {'id', 'text', 'metadata'}
    if not all(key in chunk_dict for key in required_keys):
        return False
    
    required_metadata_keys = {
        'session_id', 'document_uuid', 'chunk_index', 'chunk_size'
    }
    metadata = chunk_dict.get('metadata', {})
    if not all(key in metadata for key in required_metadata_keys):
        return False
    
    return True


def generate_chunk_id(
    session_id: str,
    document_uuid: str,
    chunk_index: int
) -> str:
    """
    Generating a globally unique chunk ID.
    """

    return f"{session_id}:{document_uuid}:{chunk_index}"


def parse_chunk_id(chunk_id: str) -> tuple[str, str, int]:
    """
    Parsing chunk_id back into components.
    """

    parts = chunk_id.split(':')
    if len(parts) < 3:
        raise ValueError(f"Invalid chunk_id format: {chunk_id}")
    
    session_id = parts[0]
    document_uuid = ':'.join(parts[1:-1])
    chunk_index = int(parts[-1].replace('chunk_', ''))
    
    return session_id, document_uuid, chunk_index


# ════════════════════════════════════════════════════════════════════════════
# EXPORT
# ════════════════════════════════════════════════════════════════════════════

__all__ = [
    'DocumentHeader',
    'ChunkMetadata',
    'Chunk',
    'SearchResult',
    'DocumentProcessingResult',
    'SessionDocumentMetadata',
    'validate_chunk_structure',
    'generate_chunk_id',
    'parse_chunk_id',
]