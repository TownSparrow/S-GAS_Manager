from typing import List
import numpy as np


class IEmbeddingService:
    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        raise NotImplementedError
