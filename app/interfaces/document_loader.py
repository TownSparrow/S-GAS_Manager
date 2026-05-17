from pathlib import Path
from typing import Dict, Any
from abc import ABC, abstractmethod


class IDocumentLoader(ABC):
    @abstractmethod
    def load(self, file_path: Path) -> Dict[str, Any]:
        pass
