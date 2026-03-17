import spacy
import logging
from typing import List, Dict, Any, Optional
from ..base_processor import ProcessorInterface

logger = logging.getLogger(__name__)

class NerSpacyProcessor(ProcessorInterface):
    """
    Processor for NER extraction using spaCy.
    """
    def __init__(self, model_name: str, morph_analyzer=None):
        try:
            self.nlp = spacy.load(model_name)
            self.morph = morph_analyzer
            logger.info(f"✅ NerSpacyProcessor initialized with model: {model_name}")
        except OSError as e:
            logger.error(f"❌ Failed to load spaCy model {model_name}: {e}")
            raise


    def process(self, text: str) -> List[Dict[str, Any]]:
        """
        Extracts NER entities from text.
        Returns a list of dictionaries with the keys 'word', 'label', and 'type'.
        """
        entities = []
        if self.nlp:
             doc = self.nlp(text)
             for ent in doc.ents:
                 if len(ent.text.strip()) > 2:
                     word_normalized = ent.text.strip().lower()
                     if self.morph:
                         try:
                             parsed = self.morph.parse(word_normalized)[0]
                             word_normalized = parsed.normal_form.lower()
                         except:
                             pass # If no lemm, just using original
                     entities.append({
                         'word': word_normalized,
                         'label': ent.label_,
                         'type': 'NER',
                         'start': ent.start_char,
                         'end': ent.end_char
                     })
        return entities