from abc import ABC, abstractmethod
from typing import List, Dict, Any


class ProcessorInterface(ABC):
    """
    Base interface for all graph processors (NER, Keyword)
    """
    @abstractmethod
    def process(self, text: str) -> List[Dict[str, Any]]:
        """
        Abstract method for text processing.
        Must be redefined in all needes places.
        """
        pass