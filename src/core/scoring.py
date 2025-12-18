import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class HybridScorer:
    def __init__(self, alpha: float = 0.6, beta: float = 0.4):

        if not np.isclose(alpha + beta, 1.0):
            logger.warning(
                f"The sum of the weights alpha={alpha} and beta={beta} is not equal to 1.0. "
                "This may lead to incorrect results."
            )
        
        self.alpha = alpha
        self.beta = beta
        logger.info(f"HybridScorer initialized: alpha={alpha}, beta={beta}")
    
    def compute_semantic_scores(
        self,
        query_embedding: np.ndarray,
        chunk_embeddings: np.ndarray
    ) -> np.ndarray:
        
        # Ensure inputs are numpy arrays
        if not isinstance(query_embedding, np.ndarray):
            query_embedding = np.array(query_embedding)
        if not isinstance(chunk_embeddings, np.ndarray):
            chunk_embeddings = np.array(chunk_embeddings)

        # Normalization of embeddings
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        chunk_norms = chunk_embeddings / (
            np.linalg.norm(chunk_embeddings, axis=1, keepdims=True) + 1e-8
        )
        
        # Cosine similarity via matrix multiplication
        semantic_scores = np.dot(chunk_norms, query_norm)
        
        # Trimming the values ​​to the range [0, 1]
        semantic_scores = np.clip(semantic_scores, 0.0, 1.0)
        
        return semantic_scores
    
    def compute_graph_scores(
        self,
        graph_distances: Dict[str, float],
        chunk_ids: List[str],
        max_distance: float = 100.0
    ) -> np.ndarray:
        graph_scores = []
        
        for chunk_id in chunk_ids:
            distance = graph_distances.get(chunk_id, max_distance)
            
            # Ensure distance is float
            distance = float(distance)

            # Converting distance into a score: score = 1 / (1 + distance)
            # The shorter the distance, the higher the score
            score = 1.0 / (1.0 + distance)
            graph_scores.append(score)
        
        return np.array(graph_scores)
    
    def _serialize_chunk_safe(self, chunk: Dict[str, Any]) -> Dict[str, Any]:
        result = {}
        for key, value in chunk.items():
            if key == 'embedding':
                continue

            # Converting NumPy to Python native types
            if isinstance(value, np.ndarray):
                result[key] = value.tolist()
            elif isinstance(value, (np.integer, np.floating)):
                result[key] = value.item()
            elif isinstance(value, dict):
                result[key] = self._serialize_chunk_safe(value)
            elif isinstance(value, (list, tuple)):
                result[key] = [
                    item.item() if isinstance(item, (np.integer, np.floating)) else item
                    for item in value
                ]
            else:
                result[key] = value
        
        return result

    def compute_hybrid_scores(
        self,
        query_embedding: np.ndarray,
        chunks: List[Dict[str, Any]],
        chunk_embeddings: np.ndarray,
        graph_distances: Dict[str, float]
    ) -> List[Tuple[Dict[str, Any], float]]:

        # Getting chunk IDs in the correct order
        chunk_ids = [chunk.get('id', f'chunk_{i}') for i, chunk in enumerate(chunks)]
        
        # Calculating semantic scores
        semantic_scores = self.compute_semantic_scores(
            query_embedding,
            chunk_embeddings
        )
        
        # Calculating graph estimates
        graph_scores = self.compute_graph_scores(
            graph_distances,
            chunk_ids
        )
        
        # Calculating final hybrid scores for chunks
        hybrid_scores = self.alpha * semantic_scores + self.beta * graph_scores
        
        # Convert numpy arrays to lists for safety
        semantic_scores_list = semantic_scores.tolist() if isinstance(semantic_scores, np.ndarray) else semantic_scores
        graph_scores_list = graph_scores.tolist() if isinstance(graph_scores, np.ndarray) else graph_scores
        hybrid_scores_list = hybrid_scores.tolist() if isinstance(hybrid_scores, np.ndarray) else hybrid_scores

        # Forming a list (chunk, score)
        scored_chunks = []
        for i, chunk in enumerate(chunks):
            chunk_serialized = self._serialize_chunk_safe(chunk)

            chunk_with_scores = {
                **chunk_serialized,
                'semantic_score': float(semantic_scores_list[i]),
                'graph_score': float(graph_scores_list[i]),
                'hybrid_score': float(hybrid_scores_list[i])
            }
            scored_chunks.append((chunk_with_scores, float(hybrid_scores_list[i])))
        
        # Sorting by descending hybrid rating
        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        
        if scored_chunks:
            logger.info(
                f"Hybrid scores for {len(scored_chunks)} chunks are calculated. "
                f"Top-1 score: {scored_chunks[0][1]:.4f}"
            )
        else:
            logger.info("No chunks scored")

        return scored_chunks
    
    def rerank_chunks(
        self,
        query_embedding: np.ndarray,
        chunks: List[Dict[str, Any]],
        chunk_embeddings: np.ndarray,
        graph_distances: Dict[str, float],
        top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        scored_chunks = self.compute_hybrid_scores(
            query_embedding,
            chunks,
            chunk_embeddings,
            graph_distances
        )
        
        # Return only chunks (without separate ratings)
        reranked = [chunk for chunk, score in scored_chunks]
        
        if top_k is not None and isinstance(top_k, int) and top_k > 0:
            reranked = reranked[:top_k]
            logger.info(f"Reranked {len(reranked)} chunks (top_k={top_k})")
        else:
            logger.info(f"⚠️ No reranked chunks ")
        return reranked
    
    def get_score_statistics(
        self,
        scored_chunks: List[Tuple[Dict[str, Any], float]]
    ) -> Dict[str, Any]:
        if not scored_chunks:
            return {}

        scores = [score for _, score in scored_chunks]
        semantic_scores = [
            chunk.get('semantic_score', 0.0)
            for chunk, _ in scored_chunks
        ]
        graph_scores = [
            chunk.get('graph_score', 0.0)
            for chunk, _ in scored_chunks
        ]
        
        return {
            'count': len(scored_chunks),
            'hybrid_score_mean': float(np.mean(scores)),
            'hybrid_score_std': float(np.std(scores)),
            'hybrid_score_min': float(np.min(scores)),
            'hybrid_score_max': float(np.max(scores)),
            'semantic_score_mean': float(np.mean(semantic_scores)),
            'graph_score_mean': float(np.mean(graph_scores))
        }