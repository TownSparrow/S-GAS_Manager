import re
import logging
from typing import List, Dict, Any, Tuple, Optional

import numpy as np

from app.interfaces.scorer import IScorer
from app.utils.serialization import serialize_chunk_safe
from app.consts.defaults import (
    SEMANTIC_ANCHOR_THRESHOLD,
    MIN_RETURN_CHUNKS,
    LOW_SEMANTIC_WARNING_THRESHOLD,
)

logger = logging.getLogger(__name__)

# ── Patterns for dynamic weight classification ────────────────────────
_FACTUAL_PATTERNS = re.compile(
    r'\b(кто|что|где|когда|сколько|какой|какая|какое|какие|название|организация'
    r'|who|what|where|when|how much|which|name|organisation|organization|general)\b',
    re.IGNORECASE,
)
_COMPLEX_PATTERNS = re.compile(
    r'\b(как|почему|зачем|влияние|связь|отличие|сравни|объясни'
    r'|how|why|impact|relation|relationship|connection|compare|explain)\b',
    re.IGNORECASE,
)


def classify_query_weights(
    query: str,
    default_alpha: float = 0.6,
    default_beta: float = 0.4,
) -> Tuple[float, float, str]:
    """Determine semantic/graph weights based on query type.

    Returns:
        (alpha, beta, query_type)
    """
    # Factual questions: semantic dominates
    if _FACTUAL_PATTERNS.search(query):
        return 0.85, 0.15, "factual"

    # Complex analytical questions: graph dominates
    if _COMPLEX_PATTERNS.search(query):
        return 0.40, 0.60, "complex"

    # Default mode
    return default_alpha, default_beta, "default"


class ScoringService(IScorer):
    def __init__(self, alpha: float = 0.6, beta: float = 0.4,
                 cross_encoder_model: str = "",
                 cross_encoder_top_n: int = 15,
                 cross_encoder_weight: float = 0.0,
                 semantic_anchor_threshold: float = SEMANTIC_ANCHOR_THRESHOLD,
                 min_return_chunks: int = MIN_RETURN_CHUNKS,
                 enable_dynamic_weights: bool = False,
                 low_semantic_warning_threshold: float = LOW_SEMANTIC_WARNING_THRESHOLD):
        if not np.isclose(alpha + beta, 1.0):
            logger.warning(
                f"The sum of the weights alpha={alpha} and beta={beta} is not equal to 1.0. "
                "This may lead to incorrect results."
            )
        self._alpha = alpha
        self._beta = beta
        self._cross_encoder = None
        self._cross_encoder_top_n = cross_encoder_top_n
        self._cross_encoder_weight = cross_encoder_weight
        self._semantic_anchor_threshold = semantic_anchor_threshold
        self._min_return_chunks = min_return_chunks
        self._enable_dynamic_weights = enable_dynamic_weights
        self._low_semantic_warning_threshold = low_semantic_warning_threshold

        if cross_encoder_model:
            try:
                from sentence_transformers import CrossEncoder
                self._cross_encoder = CrossEncoder(cross_encoder_model)
                logger.info(f"Cross-encoder loaded: {cross_encoder_model}")
            except Exception as e:
                logger.warning(f"Failed to load cross-encoder '{cross_encoder_model}': {e}")

    @property
    def alpha(self) -> float:
        return self._alpha

    @property
    def beta(self) -> float:
        return self._beta

    def compute_hybrid_scores(
        self,
        query_embedding: np.ndarray,
        chunks: List[Dict[str, Any]],
        chunk_embeddings: np.ndarray,
        graph_distances: Dict[str, float],
        query_text: str = "",
        enable_dynamic_weights: Optional[bool] = None,
        enable_semantic_anchor: bool = True,
        enable_keyword_boost: bool = True,
    ) -> List[Tuple[Dict[str, Any], float]]:
        if not isinstance(query_embedding, np.ndarray):
            query_embedding = np.array(query_embedding)
        if not isinstance(chunk_embeddings, np.ndarray):
            chunk_embeddings = np.array(chunk_embeddings)
        chunk_ids = [c.get('id', f'chunk_{i}') for i, c in enumerate(chunks)]
        q_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        c_norms = chunk_embeddings / (np.linalg.norm(chunk_embeddings, axis=1, keepdims=True) + 1e-8)
        semantic = np.clip(np.dot(c_norms, q_norm), 0.0, 1.0)
        graph = np.array([1.0 / (1.0 + float(graph_distances.get(cid, 100.0))) for cid in chunk_ids])

        # Dynamic weight classification based on query type
        alpha, beta = self._alpha, self._beta
        query_type = "default"
        use_dynamic_weights = (
            self._enable_dynamic_weights
            if enable_dynamic_weights is None else enable_dynamic_weights
        )
        if use_dynamic_weights and query_text:
            alpha, beta, query_type = classify_query_weights(
                query_text, self._alpha, self._beta
            )
            if query_type != "default":
                logger.info(
                    f"Dynamic weights: query_type={query_type}, "
                    f"alpha={alpha:.2f}, beta={beta:.2f}"
                )

        # Empty graph fallback: if no graph distances were provided (e.g.
        # graph computation failed or no concepts extracted), the graph
        # signal is unavailable — fall back to pure semantic ranking.
        graph_unavailable = (
            not graph_distances
            or all(graph_distances.get(cid) is None for cid in chunk_ids)
        )
        if graph_unavailable:
            alpha, beta = 1.0, 0.0
            logger.info(
                "Graph signal unavailable (empty graph_distances): "
                "falling back to semantic-only ranking (alpha=1.0, beta=0.0)"
            )

        # Semantic fallback: if the best semantic score is weak, graph signal
        # is unreliable — fall back to pure semantic ranking.
        best_semantic = float(np.max(semantic)) if len(semantic) > 0 else 0.0
        if best_semantic < 0.40 and not graph_unavailable:
            alpha, beta = 1.0, 0.0
            logger.info(
                f"Semantic fallback: best_sem={best_semantic:.3f} < 0.40, "
                f"ignoring graph (alpha=1.0, beta=0.0)"
            )

        hybrid = alpha * semantic + beta * graph

        # Keyword boost: if the query contains content words that appear
        # in a chunk's text, boost its hybrid score.  This rescues chunks
        # that are lexically relevant but have weak embedding similarity.
        if enable_keyword_boost and query_text:
            _stop = {'the','a','an','is','are','was','were','in','on','at','to',
                     'for','of','and','or','not','by','it','its','this','that',
                     'with','from','as','do','does','did','has','have','had',
                     'what','who','where','when','how','why','which','can','will'}
            q_words = {w.lower().strip('?.,!') for w in query_text.split()
                       if len(w) > 2 and w.lower() not in _stop}
            if q_words:
                for i, chunk in enumerate(chunks):
                    text = (chunk.get('text') or chunk.get('document') or '').lower()
                    hits = sum(1 for w in q_words if w in text)
                    if hits >= 2:
                        boost = min(0.15, hits * 0.03)
                        hybrid[i] = min(1.0, float(hybrid[i]) + boost)

        # Semantic anchor: chunks with high semantic similarity get a guaranteed
        # minimum hybrid score (0.80) so they are not dropped by graph penalty.
        if enable_semantic_anchor:
            anchor_threshold = self._semantic_anchor_threshold
            for i in range(len(chunks)):
                if semantic[i] >= anchor_threshold:
                    anchored = max(float(hybrid[i]), 0.80)
                    if anchored > hybrid[i]:
                        logger.info(
                            f"Semantic anchor: chunk {chunk_ids[i][-20:]} "
                            f"sem={semantic[i]:.3f} >= {anchor_threshold}, "
                            f"hybrid {hybrid[i]:.3f} -> {anchored:.3f}"
                        )
                        hybrid[i] = anchored

        scored = []
        for i, chunk in enumerate(chunks):
            cs = serialize_chunk_safe(chunk)
            cs.update({
                'semantic_score': float(semantic[i]),
                'graph_score': float(graph[i]),
                'hybrid_score': float(hybrid[i]),
                'query_type': query_type,
            })
            scored.append((cs, float(hybrid[i])))
        scored.sort(key=lambda x: x[1], reverse=True)

        # Log top-3 chunks with their scores
        for rank, (c, s) in enumerate(scored[:3]):
            logger.info(
                f"Top-{rank+1}: {c.get('id', '?')[-20:]} "
                f"sem={c['semantic_score']:.3f} "
                f"graph={c['graph_score']:.3f} "
                f"hybrid={c['hybrid_score']:.3f}"
            )

        # Low semantic warning: flag cases where even the best chunk
        # has poor semantic similarity — likely a total retrieval miss.
        if scored:
            best_semantic = scored[0][0].get('semantic_score', 0.0)
            if best_semantic < self._low_semantic_warning_threshold:
                logger.warning(
                    f"LOW SEMANTIC QUALITY: best chunk semantic_score={best_semantic:.3f} "
                    f"< {self._low_semantic_warning_threshold}. "
                    f"Retrieval may have failed for this query."
                )

        return scored

    def _apply_cross_encoder(self, query_text: str, scored: List[Tuple[Dict[str, Any], float]]) -> List[Tuple[Dict[str, Any], float]]:
        """Re-score top-N candidates using the cross-encoder.

        The cross-encoder evaluates (query, chunk_text) pairs directly,
        producing a much more accurate relevance signal than bi-encoder
        cosine similarity alone.
        """
        if not self._cross_encoder or not query_text or not scored:
            return scored

        n = min(self._cross_encoder_top_n, len(scored))
        top_candidates = scored[:n]
        rest = scored[n:]

        pairs = [
            (query_text, c.get('text', '') or c.get('document', '') or '')
            for c, _ in top_candidates
        ]

        try:
            ce_scores_raw = self._cross_encoder.predict(pairs)
            # Normalize cross-encoder scores to [0, 1] via min-max within batch
            ce_min = float(np.min(ce_scores_raw))
            ce_max = float(np.max(ce_scores_raw))
            if ce_max - ce_min > 1e-8:
                ce_scores = (ce_scores_raw - ce_min) / (ce_max - ce_min)
            else:
                ce_scores = np.ones(len(ce_scores_raw))

            w = self._cross_encoder_weight
            reranked = []
            for i, (chunk, hybrid_score) in enumerate(top_candidates):
                ce_score = float(ce_scores[i])
                final_score = (1.0 - w) * hybrid_score + w * ce_score
                chunk['cross_encoder_score'] = ce_score
                chunk['final_score'] = final_score
                reranked.append((chunk, final_score))

            reranked.sort(key=lambda x: x[1], reverse=True)
            return reranked + rest

        except Exception as e:
            logger.warning(f"Cross-encoder reranking failed: {e}")
            return scored

    def rerank_chunks(
        self,
        query_embedding: np.ndarray,
        chunks: List[Dict[str, Any]],
        chunk_embeddings: np.ndarray,
        graph_distances: Dict[str, float],
        top_k: Optional[int] = None,
        min_score: float = 0.0,
        query_text: str = "",
        enable_adaptive_k: bool = True,
        enable_cross_encoder: bool = True,
        enable_dynamic_weights: Optional[bool] = None,
        enable_semantic_anchor: bool = True,
        enable_keyword_boost: bool = True,
    ) -> List[Dict[str, Any]]:
        scored = self.compute_hybrid_scores(
            query_embedding, chunks, chunk_embeddings, graph_distances,
            query_text=query_text,
            enable_dynamic_weights=enable_dynamic_weights,
            enable_semantic_anchor=enable_semantic_anchor,
            enable_keyword_boost=enable_keyword_boost,
        )

        # Soft filtering: demote low-semantic chunks to the end instead of
        # hard-dropping them.  This preserves the minimum return guarantee.
        if min_score > 0:
            above = [(c, s) for c, s in scored if c.get('semantic_score', 0.0) >= min_score]
            below = [(c, s) for c, s in scored if c.get('semantic_score', 0.0) < min_score]
            scored = above + below

        # Cross-encoder reranking FIRST on up to cross_encoder_top_n candidates.
        # This ensures the cross-encoder sees the full candidate pool (e.g. 20)
        # before adaptive K or top_k trim it down to the final set (e.g. 5).
        if enable_cross_encoder and self._cross_encoder and query_text:
            scored = self._apply_cross_encoder(query_text, scored)

        # Adaptive K: trim low-confidence tail AFTER cross-encoder.
        if enable_adaptive_k:
            scored = self._adaptive_k_filter(scored, top_k)

        reranked = [c for c, _ in scored]
        if top_k and isinstance(top_k, int) and top_k > 0:
            reranked = reranked[:top_k]

        # Empty-result protection: always return at least min_return_chunks,
        # even if their scores are low.  Let the LLM decide relevance.
        min_return = self._min_return_chunks
        if len(reranked) < min_return and chunks:
            # Re-score all chunks to get a full sorted list as fallback
            all_scored = self.compute_hybrid_scores(
                query_embedding, chunks, chunk_embeddings, graph_distances,
                query_text=query_text,
                enable_dynamic_weights=enable_dynamic_weights,
                enable_semantic_anchor=enable_semantic_anchor,
                enable_keyword_boost=enable_keyword_boost,
            )
            existing_ids = {c.get('id') for c in reranked}
            for c, _ in all_scored:
                if len(reranked) >= min_return:
                    break
                if c.get('id') not in existing_ids:
                    reranked.append(c)
                    existing_ids.add(c.get('id'))
            if len(reranked) < min_return:
                logger.warning(
                    f"Could only return {len(reranked)} chunks "
                    f"(min_return={min_return}), not enough candidates."
                )

        logger.info(f"Returning {len(reranked)} chunks (min: {min_return})")
        return reranked

    @staticmethod
    def _adaptive_k_filter(
        scored: List[Tuple[Dict[str, Any], float]],
        top_k: Optional[int],
        drop_ratio: float = 0.3,
        min_keep: int = 5,
    ) -> List[Tuple[Dict[str, Any], float]]:
        """Remove trailing chunks whose score drops sharply from the leader.

        After sorting, if a chunk's hybrid score is less than ``drop_ratio``
        of the best chunk's score, it is removed.  This implements "adaptive K":
        when all 5 chunks are close in score they are all kept; when only 2 are
        relevant the remaining 3 are pruned, directly improving precision.

        Additionally, detects large score gaps between consecutive chunks.
        If the gap exceeds 50% of the top score range, everything below is cut.
        """
        if not scored or len(scored) <= min_keep:
            return scored

        limit = top_k if (top_k and top_k > 0) else len(scored)
        candidates = scored[:limit]

        if not candidates:
            return scored

        top_score = candidates[0][1]
        if top_score <= 0:
            return candidates

        # 1. Absolute threshold: drop chunks below drop_ratio * top_score
        abs_threshold = top_score * drop_ratio

        # 2. Gap detection: find the largest relative gap in the top-K
        kept = [candidates[0]]
        for i in range(1, len(candidates)):
            curr_score = candidates[i][1]
            prev_score = candidates[i - 1][1]

            # If score dropped below absolute threshold, stop
            if curr_score < abs_threshold:
                break

            # If gap between consecutive scores > 50% of top score, stop
            gap = prev_score - curr_score
            if gap > top_score * 0.5 and len(kept) >= min_keep:
                break

            kept.append(candidates[i])

        # Ensure we keep at least min_keep
        if len(kept) < min_keep:
            kept = candidates[:min_keep]

        return kept

    def get_score_statistics(self, scored_chunks: List[Tuple[Dict[str, Any], float]]) -> Dict[str, Any]:
        if not scored_chunks:
            return {}
        scores = [s for _, s in scored_chunks]
        sem = [c.get('semantic_score', 0.0) for c, _ in scored_chunks]
        gr = [c.get('graph_score', 0.0) for c, _ in scored_chunks]
        stats = {'count': len(scored_chunks), 'hybrid_score_mean': float(np.mean(scores)), 'hybrid_score_std': float(np.std(scores)), 'hybrid_score_min': float(np.min(scores)), 'hybrid_score_max': float(np.max(scores)), 'semantic_score_mean': float(np.mean(sem)), 'graph_score_mean': float(np.mean(gr))}
        ce = [c.get('cross_encoder_score', -1.0) for c, _ in scored_chunks]
        ce_valid = [s for s in ce if s >= 0]
        if ce_valid:
            stats['cross_encoder_score_mean'] = float(np.mean(ce_valid))
        return stats
