from typing import List, Dict, Any, Optional, Set
import spacy
from sentence_transformers import SentenceTransformer
import numpy as np
import logging

logger = logging.getLogger(__name__)


class SemanticChunker:
    """Advanced chunking that preserves semantic meaning"""
    
    def __init__(self,
                 max_chunk_size: int = 512,
                 overlap_size: int = 50,
                 similarity_threshold: float = 0.7):
        self.max_chunk_size = max_chunk_size
        self.overlap_size = overlap_size
        self.similarity_threshold = similarity_threshold
        
        # Load spaCy model for sentence segmentation
        try:
            self.nlp = spacy.load("ru_core_news_md")  # For Russian text
        except OSError:
            try:
                self.nlp = spacy.load("en_core_web_sm")  # Fallback to English
            except OSError:
                raise RuntimeError("No spaCy model found. Install with: python -m spacy download ru_core_news_md")
        
        # Tracking
        self.all_chunks_pool: List[Dict[str, Any]] = []
        self.used_chunk_ids: Set[str] = set()
        self.usage_history: Dict[int, Set[str]] = {}
        self.raw_document_text: str = ""
        self.current_iteration: int = 0
    

    def initialize_document(self, text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Initialization of document"""

        logger.info("📄 Initializing document for adaptive chunking...")
        
        self.raw_document_text = text
        self.all_chunks_pool = self.chunk_document(text, metadata)
        self.used_chunk_ids = set()
        self.usage_history = {}
        self.current_iteration = 0
        
        logger.info(f"✅ Document initialized: {len(self.all_chunks_pool)} chunks in pool")
        
        return self.all_chunks_pool
    

    def chunk_document(self, text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Split document into semantic chunks"""
        # Step 1: Sentence segmentation
        doc = self.nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
        
        if not sentences:
            logger.warning("⚠️ No sentences found in text")
            return []
        
        # Step 2: Group sentences into chunks
        chunks = []
        current_chunk = []
        current_size = 0
        
        for sentence in sentences:
            sentence_size = len(sentence.split())
            
            if current_size + sentence_size > self.max_chunk_size and current_chunk:
                chunk_text = " ".join(current_chunk)
                chunks.append({
                    'id': f"{metadata.get('document_id', 'unknown')}_chunk_{len(chunks)}",
                    'text': chunk_text,
                    'metadata': {
                        **metadata,
                        'chunk_id': len(chunks),
                        'chunk_size': len(chunk_text.split()),
                    }
                })
                
                # Calculating overlap
                overlap_count = max(1, len(current_chunk) // 4)  # ~25% overlap
                current_chunk = current_chunk[-overlap_count:] + [sentence]
                current_size = sum(len(s.split()) for s in current_chunk)
            else:
                current_chunk.append(sentence)
                current_size += sentence_size
        
        # Add final chunk
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            chunks.append({
                'id': f"{metadata.get('document_id', 'unknown')}_chunk_{len(chunks)}",
                'text': chunk_text,
                'metadata': {
                    **metadata,
                    'chunk_id': len(chunks),
                    'chunk_size': len(chunk_text.split()),
                }
            })
        
        return chunks
    

    def mark_chunks_used(self, chunk_ids: List[str], iteration: int) -> None:
        """Marking used chunks for current iteration"""
        if iteration not in self.usage_history:
            self.usage_history[iteration] = set()
        
        self.usage_history[iteration].update(chunk_ids)
        self.used_chunk_ids.update(chunk_ids)
        self.current_iteration = iteration
        
        logger.debug(f"📝 Marked {len(chunk_ids)} chunks as used in iteration {iteration}")
    

    def get_excluded_chunk_ids(self) -> Set[str]:
        """Exclude used chunks from retrieval"""
        return self.used_chunk_ids.copy()
    

    def get_unused_chunks(self) -> List[Dict[str, Any]]:
        """Return chunks that haven't been used yet"""
        unused = [
            chunk for chunk in self.all_chunks_pool
            if chunk['id'] not in self.used_chunk_ids
        ]
        return unused
    

    def get_recent_chunks(self, iterations_back: int = 2) -> List[Dict[str, Any]]:
        """Receive recent used chunks"""
        recent_chunks_ids = set()
        
        for iter_num in range(max(1, self.current_iteration - iterations_back), self.current_iteration + 1):
            if iter_num in self.usage_history:
                recent_chunks_ids.update(self.usage_history[iter_num])
        
        recent_chunks = [
            chunk for chunk in self.all_chunks_pool
            if chunk['id'] in recent_chunks_ids
        ]
        
        return recent_chunks
    

    def get_statistics(self) -> Dict[str, Any]:
        """Statistics of pool status and usage of chunks"""
        total_chunks = len(self.all_chunks_pool)
        used_chunks = len(self.used_chunk_ids)
        unused_chunks = total_chunks - used_chunks
        coverage_percent = 100 * used_chunks / total_chunks if total_chunks > 0 else 0
        
        return {
            'total_chunks_in_pool': total_chunks,
            'used_chunks': used_chunks,
            'unused_chunks': unused_chunks,
            'coverage_percent': coverage_percent,
            'current_iteration': self.current_iteration,
            'total_iterations': len(self.usage_history),
            'chunks_per_iteration': {
                it: len(ids) for it, ids in self.usage_history.items()
            }
        }
    

    def reset_usage_tracking(self) -> None:
        """Resetting the tracking"""
        self.used_chunk_ids = set()
        self.usage_history = {}
        self.current_iteration = 0
        logger.info("✅ Usage tracking reset is completed")
    

    def get_all_chunks(self) -> List[Dict[str, Any]]:
        """Receive all pool of chunks"""
        return self.all_chunks_pool.copy()
    
    
    def get_raw_text(self) -> str:
        """Return all original text"""
        return self.raw_document_text