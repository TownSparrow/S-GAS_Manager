from typing import List, Dict, Any, Tuple, Optional
import numpy as np


class IScorer:
    def compute_hybrid_scores(self, query_embedding: np.ndarray, chunks: List[Dict[str, Any]], chunk_embeddings: np.ndarray, graph_distances: Dict[str, float]) -> List[Tuple[Dict[str, Any], float]]:
        raise NotImplementedError

    def rerank_chunks(self, query_embedding: np.ndarray, chunks: List[Dict[str, Any]], chunk_embeddings: np.ndarray, graph_distances: Dict[str, float], top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        raise NotImplementedError

    def get_score_statistics(self, scored_chunks: List[Tuple[Dict[str, Any], float]]) -> Dict[str, Any]:
        raise NotImplementedError
