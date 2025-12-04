from typing import List, Dict, Any, Optional, Set
import spacy
import numpy as np
import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


class SemanticChunker:
    """
    Advanced chunking that preserves semantic meaning.
    
    Main features:
    ✅ Clean chunking (without priorities)
    ✅ Session-based tracking used chunks
    ✅ S-GAS iterative search supported
    ✅ Works for ANY documents
    """

    def __init__(self,
                 max_chunk_size: int = 512,
                 overlap_size: int = 50,
                 similarity_threshold: float = 0.7):
        """
        Initialize SemanticChunker.
        
        Args:
            max_chunk_size: Maximum chunk size (in words)
            overlap_size: Overlap size between chunks
            similarity_threshold: Semantic similarity threshold
        """
        self.max_chunk_size = max_chunk_size
        self.overlap_size = overlap_size
        self.similarity_threshold = similarity_threshold

        # Load spaCy model for sentence segmentation
        try:
            self.nlp = spacy.load("ru_core_news_md")  # Russian
            logger.info("✅ Loaded Russian spaCy model")
        except OSError:
            try:
                self.nlp = spacy.load("en_core_web_sm")  # English fallback
                logger.info("✅ Loaded English spaCy model")
            except OSError:
                logger.error("❌ No spaCy model found. Install with: python -m spacy download ru_core_news_md")
                raise RuntimeError("No spaCy model found.")

        # Tracking - session-based
        self.session_chunks_pool: Dict[str, List[Dict[str, Any]]] = {}
        self.session_used_chunk_ids: Dict[str, Set[str]] = {}
        self.session_usage_history: Dict[str, Dict[int, Set[str]]] = {}
        self.session_iteration: Dict[str, int] = {}

        logger.info("✅ SemanticChunker initialized (Universal Approach - No Priorities)")

    def initialize_document(self, 
                           text: str, 
                           metadata: Dict[str, Any],
                           session_id: str) -> List[Dict[str, Any]]:
        """
        Initialize document for session.
        
        ⚠️ Warning: session_id is required
        
        Args:
            text: Full text of the document
            metadata: Document metadata (filename, document_type, etc.)
            session_id: ID of the session
            
        Returns:
            List of chunks for this document
        """
        if not session_id:
            raise ValueError("⚠️ Warning: session_id is required!")

        logger.info(f"📄 Initializing document for session {session_id}...")

        # Create chunks
        all_chunks = self._chunk_document(text, metadata)

        # Add session_id to metadata of each chunk
        for chunk in all_chunks:
            chunk['metadata']['session_id'] = session_id

        # Initialize session tracking if it doesn't exist
        if session_id not in self.session_chunks_pool:
            self.session_chunks_pool[session_id] = []
            self.session_used_chunk_ids[session_id] = set()
            self.session_usage_history[session_id] = {}
            self.session_iteration[session_id] = 0

        # Add chunks to session pool
        self.session_chunks_pool[session_id].extend(all_chunks)

        logger.info(f"✅ Document initialized for {session_id}: {len(all_chunks)} chunks added")
        return all_chunks

    def _chunk_document(self, 
                       text: str, 
                       metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Divide document into semantic chunks.
        
        Process:
        1. Sentence segmentation
        2. Grouping sentences into chunks
        3. Adding metadata without priorities
        """
        if not text or not text.strip():
            logger.warning("⚠️ Empty text provided")
            return []

        # Step 1: Sentence segmentation
        try:
            doc = self.nlp(text[:1000000])
            sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
        except Exception as e:
            logger.error(f"❌ Error: spaCy processing failed: {e}")
            sentences = [s.strip() for s in text.split('.') if s.strip()]

        if not sentences:
            logger.warning("⚠️ Warning: No sentences found in text")
            return []

        # Step 2: Group sentences into chunks
        chunks = []
        current_chunk = []
        current_size = 0

        for sentence in sentences:
            sentence_size = len(sentence.split())

            if current_size + sentence_size > self.max_chunk_size and current_chunk:
                chunk_text = " ".join(current_chunk)
                chunk = {
                    'id': f"{metadata.get('document_id', 'unknown')}_chunk_{len(chunks)}",
                    'text': chunk_text,
                    'metadata': {
                        **metadata,
                        'chunk_index': len(chunks),
                        'chunk_size': len(chunk_text.split()),
                        'timestamp': datetime.now(timezone.utc).isoformat(),
                    }
                }
                chunks.append(chunk)

                overlap_count = max(1, len(current_chunk) // 4)
                current_chunk = current_chunk[-overlap_count:] + [sentence]
                current_size = sum(len(s.split()) for s in current_chunk)
            else:
                current_chunk.append(sentence)
                current_size += sentence_size

        # Add final chunk
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            chunk = {
                'id': f"{metadata.get('document_id', 'unknown')}_chunk_{len(chunks)}",
                'text': chunk_text,
                'metadata': {
                    **metadata,
                    'chunk_index': len(chunks),
                    'chunk_size': len(chunk_text.split()),
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                }
            }
            chunks.append(chunk)

        logger.info(f"✅ Created {len(chunks)} chunks from document")
        return chunks

    def mark_chunks_used(self, 
                        session_id: str,
                        chunk_ids: List[str], 
                        iteration: int) -> None:
        """
        Mark chunks as used in iteration.
        
        For S-GAS: each iteration excludes used chunks.
        
        Args:
            session_id: ID of the session
            chunk_ids: List of IDs of used chunks
            iteration: Number of iteration
        """
        if session_id not in self.session_used_chunk_ids:
            logger.warning(f"⚠️ Warning: Session {session_id} not initialized")
            return

        if iteration not in self.session_usage_history[session_id]:
            self.session_usage_history[session_id][iteration] = set()

        self.session_usage_history[session_id][iteration].update(chunk_ids)
        self.session_used_chunk_ids[session_id].update(chunk_ids)
        self.session_iteration[session_id] = iteration

        logger.debug(f"✅ Marked {len(chunk_ids)} chunks as used in iteration {iteration} ({session_id})")

    def get_excluded_chunk_ids(self, session_id: str) -> Set[str]:
        """
        Get IDs of used chunks for exclusion from search.
        
        Args:
            session_id: ID of the session
            
        Returns:
            Set of chunk IDs to exclude
        """
        if session_id not in self.session_used_chunk_ids:
            return set()
        return self.session_used_chunk_ids[session_id].copy()

    def get_unused_chunks(self, session_id: str) -> List[Dict[str, Any]]:
        """
        Get chunks that haven't been used yet.
        
        Args:
            session_id: ID of the session
            
        Returns:
            List of unused chunks
        """
        if session_id not in self.session_chunks_pool:
            return []

        unused = [
            chunk for chunk in self.session_chunks_pool[session_id]
            if chunk['id'] not in self.session_used_chunk_ids.get(session_id, set())
        ]
        return unused

    def get_statistics(self, session_id: str) -> Dict[str, Any]:
        """
        Get statistics for session.
        
        Args:
            session_id: ID of the session
            
        Returns:
            Statistics of chunks, usage and coverage
        """
        if session_id not in self.session_chunks_pool:
            return {
                'total_chunks_in_pool': 0,
                'used_chunks': 0,
                'unused_chunks': 0,
                'coverage_percent': 0.0,
                'current_iteration': 0,
                'total_iterations': 0
            }

        total_chunks = len(self.session_chunks_pool[session_id])
        used_chunks = len(self.session_used_chunk_ids[session_id])
        unused_chunks = total_chunks - used_chunks
        coverage_percent = 100 * used_chunks / total_chunks if total_chunks > 0 else 0

        return {
            'session_id': session_id,
            'total_chunks_in_pool': total_chunks,
            'used_chunks': used_chunks,
            'unused_chunks': unused_chunks,
            'coverage_percent': coverage_percent,
            'current_iteration': self.session_iteration.get(session_id, 0),
            'total_iterations': len(self.session_usage_history.get(session_id, {})),
        }

    def reset_session_tracking(self, session_id: str) -> None:
        """Reset tracking for session."""
        if session_id in self.session_used_chunk_ids:
            self.session_used_chunk_ids[session_id] = set()
        if session_id in self.session_usage_history:
            self.session_usage_history[session_id] = {}
        if session_id in self.session_iteration:
            self.session_iteration[session_id] = 0

        logger.info(f"✅ Session {session_id} tracking reset")

    def clear_session(self, session_id: str) -> None:
        """Fully clear session."""
        if session_id in self.session_chunks_pool:
            del self.session_chunks_pool[session_id]
        if session_id in self.session_used_chunk_ids:
            del self.session_used_chunk_ids[session_id]
        if session_id in self.session_usage_history:
            del self.session_usage_history[session_id]
        if session_id in self.session_iteration:
            del self.session_iteration[session_id]

        logger.info(f"✅ Session {session_id} completely cleared")

    def get_all_chunks(self, session_id: str) -> List[Dict[str, Any]]:
        """Get ALL chunks for session."""
        return self.session_chunks_pool.get(session_id, []).copy()