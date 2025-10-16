from typing import List, Dict, Any, Optional
import spacy
from sentence_transformers import SentenceTransformer
import numpy as np


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
        
    def chunk_document(self, text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Split document into semantic chunks"""
        
        # Step 1: Sentence segmentation
        doc = self.nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
        
        # Step 2: Group sentences into chunks
        chunks = []
        current_chunk = []
        current_size = 0
        
        for sentence in sentences:
            sentence_size = len(sentence.split())
            
            # Check if adding this sentence exceeds max size
            if current_size + sentence_size > self.max_chunk_size and current_chunk:
                # Create chunk from current sentences
                chunk_text = " ".join(current_chunk)
                chunks.append(self._create_chunk(chunk_text, metadata, len(chunks)))
                
                # Start new chunk with overlap
                overlap_sentences = current_chunk[-self._calculate_overlap_sentences():]
                current_chunk = overlap_sentences + [sentence]
                current_size = sum(len(s.split()) for s in current_chunk)
            else:
                current_chunk.append(sentence)
                current_size += sentence_size
        
        # Add final chunk
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            chunks.append(self._create_chunk(chunk_text, metadata, len(chunks)))
            
        return chunks
    
    def _calculate_overlap_sentences(self) -> int:
        """Calculate how many sentences to overlap based on overlap_size"""
        # Simple heuristic: assume average sentence is ~15 words
        avg_sentence_length = 15
        overlap_sentences = max(1, self.overlap_size // avg_sentence_length)
        return min(overlap_sentences, 3)  # Max 3 sentences overlap
    
    def _create_chunk(self, text: str, metadata: Dict[str, Any], chunk_id: int) -> Dict[str, Any]:
        """Create standardized chunk object"""
        return {
            'id': f"{metadata.get('document_id', 'unknown')}_{chunk_id}",
            'text': text,
            'metadata': {
                **metadata,
                'chunk_id': chunk_id,
                'chunk_size': len(text.split()),
                'chunk_type': 'semantic'
            }
        }