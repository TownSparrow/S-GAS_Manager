import logging
from typing import List, Dict, Any, Optional
from ..base_processor import ProcessorInterface
from natasha import Doc, Segmenter, MorphVocab, NewsNERTagger, NewsEmbedding

logger = logging.getLogger(__name__)

class NerNatashaProcessor(ProcessorInterface):
    """
    NER Extraction Processor using Natasha.
    """
    def __init__(self, morph_analyzer=None):
        try:
            self.segmenter = Segmenter()
            self.morph_vocab = MorphVocab()
            emb = NewsEmbedding()
            self.ner_tagger = NewsNERTagger(emb)
            self.morph = morph_analyzer
            logger.info(f"✅ NerNatashaProcessor initialized.")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Natasha components: {e}")
            raise


    def process(self, text: str) -> List[Dict[str, Any]]:
        """
        Extracts NER entities from text using Natasha.
        Returns a list of dictionaries with the keys 'word', 'label', 'type'.
        """
        entities = []
        try:
            doc_natasha = Doc(text)
            doc_natasha.segment(self.segmenter)
            doc_natasha.tag_ner(self.ner_tagger)

            for span in doc_natasha.spans:
                entity_text = span.text.strip()
                entity_type = span.type # e.g., 'PER', 'ORG', 'LOC'

                if len(entity_text) > 2:
                    word_normalized = entity_text
                    if self.morph:
                        try:
                            parsed = self.morph.parse(entity_text)[0]
                            word_normalized = parsed.normal_form.lower()
                        except:
                            pass # If no lemm, just using original

                    entities.append({
                        'word': word_normalized.lower(),
                        'label': entity_type,
                        'type': 'NER',
                        'start': span.start,
                        'end': span.stop
                    })
        except Exception as e:
            logger.warning(f"⚠️ Natasha NER failed: {e}. Returning empty list.")
        return entities