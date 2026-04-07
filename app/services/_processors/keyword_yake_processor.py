import logging
import yake
from typing import List, Dict, Any, Optional
from app.services._processors.base_processor import ProcessorInterface

logger = logging.getLogger(__name__)

class KeywordYakeProcessor(ProcessorInterface):
    """A keyword extraction processor using YAKE."""
    def __init__(self, lang_code: str = 'en', num_keywords: int = 10):
        self.yake_extractor = yake.KeywordExtractor(
            lan=lang_code,
            n=2, # bigrams
            dedupLim=0.7,
            top=num_keywords,
            features=None
        )
        self.lang_code = lang_code
        logger.info(f"KeywordYakeProcessor initialized for language: {lang_code}")

    def process(self, text: str, morph_analyzer=None) -> List[Dict[str, Any]]:
        """Extracting keywords from text using YAKE."""
        keywords_raw = self.yake_extractor.extract_keywords(text)
        # keywords_raw = [(keyword, score), ...]
        keywords = []
        for keyword, score in keywords_raw:
            word_normalized = keyword.lower()
            if morph_analyzer and self.lang_code == 'ru':
                try:
                    parsed = morph_analyzer.parse(keyword)[0]
                    word_normalized = parsed.normal_form.lower()
                except Exception:
                    pass # If no lemm, just using original

            keywords.append({
                'word': word_normalized,
                'label': 'KEYWORD',
                'type': 'KEYWORD',
                'score': score,
            })
        return keywords