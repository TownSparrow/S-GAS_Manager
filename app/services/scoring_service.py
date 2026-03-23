import logging
from typing import List, Dict, Any, Tuple, Optional

import numpy as np

from app.interfaces.scorer import IScorer
from app.utils.serialization import serialize_chunk_safe

logger = logging.getLogger(__name__)


class ScoringService(IScorer):
    def __init__(self, alpha: float = 0.6, beta: float = 0.4):
        if not np.isclose(alpha + beta, 1.0):
            logger.warning(
                f"The sum of the weights alpha={alpha} and beta={beta} is not equal to 1.0. "
                "This may lead to incorrect results."
            )
        self._alpha = alpha
        self._beta = beta

    @property
    def alpha(self) -> float:
        return self._alpha

    @property
    def beta(self) -> float:
        return self._beta

    def compute_hybrid_scores(self, query_embedding: np.ndarray, chunks: List[Dict[str, Any]], chunk_embeddings: np.ndarray, graph_distances: Dict[str, float]) -> List[Tuple[Dict[str, Any], float]]:
        if not isinstance(query_embedding, np.ndarray):
            query_embedding = np.array(query_embedding)
        if not isinstance(chunk_embeddings, np.ndarray):
            chunk_embeddings = np.array(chunk_embeddings)
        chunk_ids = [c.get('id', f'chunk_{i}') for i, c in enumerate(chunks)]
        q_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        c_norms = chunk_embeddings / (np.linalg.norm(chunk_embeddings, axis=1, keepdims=True) + 1e-8)
        semantic = np.clip(np.dot(c_norms, q_norm), 0.0, 1.0)
        graph = np.array([1.0 / (1.0 + float(graph_distances.get(cid, 100.0))) for cid in chunk_ids])
        hybrid = self._alpha * semantic + self._beta * graph
        scored = []
        for i, chunk in enumerate(chunks):
            cs = serialize_chunk_safe(chunk)
            cs.update({'semantic_score': float(semantic[i]), 'graph_score': float(graph[i]), 'hybrid_score': float(hybrid[i])})
            scored.append((cs, float(hybrid[i])))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def rerank_chunks(self, query_embedding: np.ndarray, chunks: List[Dict[str, Any]], chunk_embeddings: np.ndarray, graph_distances: Dict[str, float], top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        scored = self.compute_hybrid_scores(query_embedding, chunks, chunk_embeddings, graph_distances)
        reranked = [c for c, _ in scored]
        if top_k and isinstance(top_k, int) and top_k > 0:
            reranked = reranked[:top_k]
        return reranked

    def get_score_statistics(self, scored_chunks: List[Tuple[Dict[str, Any], float]]) -> Dict[str, Any]:
        if not scored_chunks:
            return {}
        scores = [s for _, s in scored_chunks]
        sem = [c.get('semantic_score', 0.0) for c, _ in scored_chunks]
        gr = [c.get('graph_score', 0.0) for c, _ in scored_chunks]
        return {'count': len(scored_chunks), 'hybrid_score_mean': float(np.mean(scores)), 'hybrid_score_std': float(np.std(scores)), 'hybrid_score_min': float(np.min(scores)), 'hybrid_score_max': float(np.max(scores)), 'semantic_score_mean': float(np.mean(sem)), 'graph_score_mean': float(np.mean(gr))}
