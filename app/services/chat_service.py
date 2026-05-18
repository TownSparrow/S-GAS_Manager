import time
import json
import logging
import asyncio
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone

import torch
import numpy as np
import aiofiles

from app.interfaces.embedding import IEmbeddingService
from app.interfaces.vector_store import IVectorStore
from app.interfaces.chunker import IChunker
from app.interfaces.graph_builder import IGraphBuilder
from app.interfaces.scorer import IScorer
from app.interfaces.swap_manager import ISwapManager
from app.interfaces.vllm_client import IVLLMClient
from app.services.bm25_service import BM25Service
from app.consts.defaults import MAX_CHUNK_TEXT_LEN, EMBEDDING_DIM, METRICS_FILE
from app.consts.prompts import PROMPT_WITH_CONTEXT, PROMPT_WITHOUT_CONTEXT
from app.utils.gpu import (
    log_gpu_memory_detailed,
    get_gpu_memory_stats,
    get_nvidia_smi_stats,
    summarize_nvidia_smi_samples,
)
from app.utils.system_resources import get_system_resources

logger = logging.getLogger(__name__)


def _get_vram_gb() -> Dict[str, float]:
    """Gets current VRAM usage in GB."""
    if not torch.cuda.is_available():
        return {'allocated_gb': 0.0, 'reserved_gb': 0.0, 'peak_gb': 0.0}
    return {
        'allocated_gb': round(torch.cuda.memory_allocated() / (1024 ** 3), 4),
        'reserved_gb': round(torch.cuda.memory_reserved() / (1024 ** 3), 4),
        'peak_gb': round(torch.cuda.max_memory_allocated() / (1024 ** 3), 4),
    }


class ChatService:
    def __init__(self, embedding_service: IEmbeddingService, vector_store: IVectorStore,
                 chunker: IChunker, graph_builder: IGraphBuilder, scorer: IScorer,
                 swap_manager: ISwapManager, vllm_client: IVLLMClient,
                 prompt_config: Dict[str, Any], model_name: str = "", kv_monitor=None,
                 min_similarity_score: float = 0.0,
                 bm25_weight: float = 0.0, rrf_k: int = 60,
                 enable_sgas_filtering: bool = True):
        self._embedding = embedding_service
        self._vector_store = vector_store
        self._chunker = chunker
        self._graph = graph_builder
        self._scorer = scorer
        self._swap = swap_manager
        self._vllm = vllm_client
        self._prompt_config = prompt_config
        self._model_name = model_name
        self._kv_monitor = kv_monitor
        self._min_similarity_score = min_similarity_score
        self._bm25 = BM25Service()
        self._bm25_weight = bm25_weight  # 0 = disabled, >0 = hybrid
        self._rrf_k = rrf_k  # RRF constant (default 60)
        self._enable_sgas_filtering = enable_sgas_filtering
        self._enable_graph_expansion_filter = enable_sgas_filtering
        self._enable_cross_encoder_rerank = enable_sgas_filtering
        self._enable_dynamic_scoring_weights = enable_sgas_filtering
        self._enable_semantic_anchor = enable_sgas_filtering
        self._enable_keyword_boost = enable_sgas_filtering
        self._exclude_used_chunks = False  # default; overridden by config

    async def process_chat(self, session_id: str, session_data: Dict[str, Any], message: str,
                           use_rag: bool = True, n_chunks: int = 5,
                           temperature: Optional[float] = None, top_p: Optional[float] = None,
                           max_tokens: Optional[int] = None,
                           baseline_mode: bool = False,
                           run_id: Optional[str] = None) -> Dict[str, Any]:
        """Runs S-GAS pipeline."""

        session_data['iteration'] += 1
        iteration = session_data['iteration']
        # When exclude_used_chunks is False (default), S-GAS can re-use
        # previously selected chunks — the algorithm re-ranks all candidates
        # each turn, allowing the best chunks to appear again.
        exclude_used = session_data.get('exclude_used_chunks', self._exclude_used_chunks)
        if baseline_mode or not exclude_used:
            excluded_chunks = []
        else:
            excluded_chunks = session_data.get('excluded_chunks', [])

        # Reset peak VRAM tracker for this iteration
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        step_timings = {}
        vram_per_step = {}

        # ── STEP 1: Search ────────────────────────────────────────────
        t_search = time.time()
        context_chunks, query_vec = [], None
        if use_rag:
            context_chunks, query_vec = await self._search_chunks(session_id, message, excluded_chunks, n_chunks)
        step_timings['search_ms'] = round((time.time() - t_search) * 1000, 2)
        vram_per_step['after_search'] = _get_vram_gb()

        # Pre-build graph from all session chunks (first turn only)
        if not baseline_mode and iteration == 1 and not session_data.get('graph_initialized', False):
            t_graph_init = time.time()
            all_session_chunks = await self._vector_store.get_all_session_chunks(session_id)
            if all_session_chunks:
                all_embeddings = self._get_safe_embeddings(all_session_chunks)
                self._graph.build_graph(chunks=all_session_chunks, embeddings=all_embeddings)
                session_data['graph_initialized'] = True
                # Update swap service with high-centrality chunks for soft eviction
                try:
                    top_ids = self._graph.get_high_centrality_chunk_ids(
                        self._swap._retain_top_centrality_pct
                    )
                    self._swap.update_protected_chunks(top_ids)
                except Exception:
                    pass
                logger.info(
                    f"Graph pre-built from {len(all_session_chunks)} chunks "
                    f"in {round((time.time() - t_graph_init) * 1000)}ms"
                )

        # ── STEP 2: Graph + Reranking ─────────────────────────────────
        t_rerank = time.time()
        chunk_embeddings, graph_distances = np.array([]), {}
        graph_stats = {}
        final_context = context_chunks

        if use_rag and context_chunks and query_vec is not None:
            if baseline_mode:
                # Baseline: semantic-only ranking (no graph, no hybrid scoring)
                candidates = context_chunks[:n_chunks]
                # Filter out low-similarity chunks to improve precision.
                # Chunks with similarity below threshold are noise that dilutes
                # the context without contributing useful information.
                if self._min_similarity_score > 0:
                    candidates = [
                        c for c in candidates
                        if c.get('similarity_score', 1.0) >= self._min_similarity_score
                    ]
                final_context = candidates if candidates else context_chunks[:1]
            else:
                # S-GAS: full graph + hybrid reranking
                final_context, chunk_embeddings, graph_distances = self._rerank_chunks(
                    context_chunks, query_vec, message, n_chunks,
                    excluded_chunks=set(excluded_chunks),
                )
                # Ensure at least min_return_chunks survive score filtering
                if len(final_context) < self._scorer._min_return_chunks and context_chunks:
                    final_context = context_chunks[:self._scorer._min_return_chunks]
                try:
                    graph_stats = self._graph.get_graph_statistics()
                except Exception:
                    graph_stats = {}
        else:
            final_context = context_chunks[:n_chunks] if context_chunks else []

        step_timings['rerank_ms'] = round((time.time() - t_rerank) * 1000, 2)
        vram_per_step['after_rerank'] = _get_vram_gb()

        # Collect retrieved chunk IDs, texts, and scores
        retrieved_chunk_ids = [c.get('id', '') for c in final_context if 'id' in c]
        retrieved_chunk_texts = [c.get('text', '') for c in final_context]
        chunk_scores = []
        for rank, c in enumerate(final_context):
            entry = {
                'id': c.get('id', ''),
                'rank': rank + 1,
                'semantic_score': round(c.get('semantic_score', 0.0), 4),
                'graph_score': round(c.get('graph_score', 0.0), 4),
                'hybrid_score': round(c.get('hybrid_score', 0.0), 4),
                'cross_encoder_score': round(c.get('cross_encoder_score', -1.0), 4),
                'final_score': round(c.get('final_score', c.get('hybrid_score', 0.0)), 4),
            }
            chunk_scores.append(entry)

        # ── Log chunk selection reasons ──────────────────────────────
        if final_context and not baseline_mode:
            self._log_chunk_selection(
                session_id=session_id, iteration=iteration,
                query=message, chunk_scores=chunk_scores,
            )

        # ── STEP 3: Mark used chunks ──────────────────────────────────
        if final_context:
            used_ids = [c['id'] for c in final_context if 'id' in c]
            session_data['excluded_chunks'].extend(used_ids)
            self._chunker.mark_chunks_used(session_id, used_ids, iteration)

        # ── STEP 4: Optional chunk embedding hot/cold management ───────
        swap_enabled = getattr(self._swap, "enabled", True)
        # 4a. Initialize swap state on the first turn only when the optional
        #     embedding swap experiment is enabled.
        if swap_enabled and not baseline_mode and iteration == 1 and not session_data.get('swap_initialized', False):
            all_chunks = await self._vector_store.get_all_session_chunks(session_id)
            if all_chunks:
                self._swap.initialize_chunks(all_chunks)
                session_data['swap_initialized'] = True

        # 4b. Check cache BEFORE the swap so we measure whether the PREVIOUS
        #     turn's predictive prefetch placed these chunks in GPU already.
        cache_check = (
            self._swap.check_cache_hits(retrieved_chunk_ids)
            if swap_enabled else {'hits': 0, 'misses': 0, 'hit_rate': 0.0}
        )

        t_swap = time.time()
        if baseline_mode or not swap_enabled:
            swap_action = 'disabled'
        else:
            swap_action = await self._manage_swap(session_id, session_data, final_context, iteration)
        step_timings['swap_ms'] = round((time.time() - t_swap) * 1000, 2)
        # Soft eviction: delete marked candidates when RAM exceeds threshold
        if swap_enabled and not baseline_mode:
            self._swap.evict_candidates_if_needed()
        vram_per_step['after_swap'] = _get_vram_gb()

        # ── STEP 5: Prompt + Inference ────────────────────────────────
        prompt = self._build_prompt(final_context, message, run_id=run_id)
        prompt_length_tokens = len(prompt) // 4
        n_selected_chunks = len(final_context)
        n_archived_chunks_this_iter = len([c['id'] for c in final_context if 'id' in c])

        if self._kv_monitor:
            self._kv_monitor.log_iteration_start(
                iteration=iteration, prompt_length_tokens=prompt_length_tokens,
                n_selected_chunks=n_selected_chunks,
                n_archived_chunks_this_iter=n_archived_chunks_this_iter,
            )

        t_observability = time.time()
        vllm_metrics_before = await self._safe_get_vllm_metrics()
        observability_time = time.time() - t_observability

        t_inference = time.time()
        gpu_samples, stop_gpu_sampling = [], asyncio.Event()
        gpu_sampler = asyncio.create_task(
            self._sample_gpu_during_inference(gpu_samples, stop_gpu_sampling)
        )
        try:
            llm_response = await self._vllm.chat(prompt, {
                "temperature": temperature, "top_p": top_p, "max_tokens": max_tokens,
            })
            inference_time = time.time() - t_inference
        finally:
            stop_gpu_sampling.set()
            await gpu_sampler
        step_timings['inference_ms'] = round(inference_time * 1000, 2)

        t_observability = time.time()
        vllm_metrics_after = await self._safe_get_vllm_metrics()
        observability_time += time.time() - t_observability
        step_timings['observability_ms'] = round(observability_time * 1000, 2)
        self._swap.record_compute_time(inference_time * 1000)

        if self._kv_monitor:
            self._kv_monitor.log_iteration_end(
                iteration=iteration, generation_time_ms=inference_time * 1000,
                prompt_length_tokens=prompt_length_tokens,
                n_selected_chunks=n_selected_chunks,
                n_archived_chunks_this_iter=n_archived_chunks_this_iter,
            )

        vram_per_step['after_inference'] = _get_vram_gb()

        # System resources (RAM, disk, process memory)
        system_resources = get_system_resources()

        response_text = llm_response['choices'][0]['message']['content']
        usage = llm_response.get('usage', {})
        vllm_observability = self._build_vllm_observability(
            before=vllm_metrics_before,
            after=vllm_metrics_after,
            response_metrics=llm_response.get('sgas_observability', {}).get('vllm_metrics', {}),
            gpu_samples=gpu_samples,
            inference_time_sec=inference_time,
            usage=usage,
        )

        # ── Collect statistics ────────────────────────────────────────
        chunking_stats = self._chunker.get_statistics(session_id)
        swap_stats = self._swap.get_statistics()
        new_count = (
            len([c for c in final_context if c.get('id') not in excluded_chunks])
            if excluded_chunks else len(final_context)
        )

        total_latency_ms = sum(step_timings.values())

        await self._log_metrics(
            session_id, iteration, final_context, excluded_chunks,
            chunking_stats, inference_time, swap_action, usage,
        )

        return {
            'response': response_text,
            'metadata': {
                'model_used': self._model_name,
                'session_id': session_id,
                'iteration': iteration,
                'use_rag': use_rag,
                'baseline_mode': baseline_mode,

                # Chunk statistics
                'context_chunks_used': len(final_context),
                'new_chunks_in_iteration': new_count,
                'chunks_excluded_from_previous': len(excluded_chunks),
                'total_chunks_explored': chunking_stats['used_chunks'],
                'total_chunks_available': chunking_stats['total_chunks_in_pool'],
                'coverage_percent': chunking_stats['coverage_percent'],
                'retrieved_chunk_ids': retrieved_chunk_ids,
                'retrieved_chunk_texts': retrieved_chunk_texts,
                'chunk_scores': chunk_scores,

                # Latency breakdown (per step)
                'latency': {
                    'total_ms': round(total_latency_ms, 2),
                    'search_ms': step_timings.get('search_ms', 0),
                    'rerank_ms': step_timings.get('rerank_ms', 0),
                    'swap_ms': step_timings.get('swap_ms', 0),
                    'inference_ms': step_timings.get('inference_ms', 0),
                    'observability_ms': step_timings.get('observability_ms', 0),
                },
                'inference_time_sec': inference_time,
                'context_tokens': sum(
                    c.get('metadata', {}).get('chunk_size', 500) for c in final_context
                ),

                # VRAM per step
                'vram': vram_per_step,

                # System resources (RAM, disk, process memory)
                'system_resources': system_resources,

                # Cache / Swap
                'cache_check': cache_check,
                'swap_action': swap_action,
                'swap_statistics': swap_stats,

                # Graph
                'graph_statistics': graph_stats,

                # Hybrid reranking
                'hybrid_reranking': {
                    'chunks_before_reranking': len(context_chunks),
                    'chunks_after_reranking': len(final_context),
                    'graph_distances_used': len(graph_distances),
                    'semantic_embeddings_used': len(chunk_embeddings),
                },

                # LLM
                'tokens_generated': usage.get('completion_tokens', 0),
                'tokens_in_prompt': usage.get('prompt_tokens', 0),
                'total_tokens': usage.get('total_tokens', 0),
                'vllm_observability': vllm_observability,

                'timestamp': datetime.now(timezone.utc).isoformat(),
            },
        }

    async def _ensure_bm25_index(self, session_id: str) -> None:
        """Lazily build BM25 index on first search for a session."""
        if self._bm25_weight <= 0:
            return
        if self._bm25._indices.get(session_id) is not None:
            return
        all_chunks = await self._vector_store.get_all_session_chunks(session_id)
        if all_chunks:
            self._bm25.build_index(session_id, all_chunks)

    @staticmethod
    def _reciprocal_rank_fusion(
        ranked_lists: List[List[Dict[str, Any]]],
        weights: List[float],
        k: int = 60,
    ) -> List[Dict[str, Any]]:
        """Merge multiple ranked lists via weighted Reciprocal Rank Fusion."""
        scores: Dict[str, float] = {}
        chunk_map: Dict[str, Dict[str, Any]] = {}

        for ranked, w in zip(ranked_lists, weights):
            for rank, chunk in enumerate(ranked, start=1):
                cid = chunk.get("id", "")
                if not cid:
                    continue
                scores[cid] = scores.get(cid, 0.0) + w / (k + rank)
                if cid not in chunk_map:
                    chunk_map[cid] = chunk

        fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [chunk_map[cid] for cid, _ in fused if cid in chunk_map]

    async def _search_chunks(self, session_id, message, excluded_chunks, n_chunks):
        """Searches for relevant chunks using hybrid vector + BM25 retrieval."""
        try:
            q = self._embedding.get_embeddings([message])[0]
            top_k = max(20, n_chunks * 3)
            search_k = top_k * 2 if excluded_chunks else top_k

            # Vector search
            vector_candidates = await self._vector_store.search(
                query_embedding=q, session_id=session_id, top_k=search_k,
            )

            # Hybrid: combine vector + BM25 via RRF
            if self._bm25_weight > 0:
                await self._ensure_bm25_index(session_id)
                bm25_candidates = self._bm25.search(session_id, message, top_k=search_k)

                if bm25_candidates:
                    vec_w = 1.0 - self._bm25_weight
                    bm25_w = self._bm25_weight
                    all_candidates = self._reciprocal_rank_fusion(
                        [vector_candidates, bm25_candidates],
                        [vec_w, bm25_w],
                        k=self._rrf_k,
                    )
                    logger.debug(
                        f"Hybrid search: {len(vector_candidates)} vector + "
                        f"{len(bm25_candidates)} BM25 → {len(all_candidates)} fused"
                    )
                else:
                    all_candidates = vector_candidates
            else:
                all_candidates = vector_candidates

            if excluded_chunks:
                excluded_set = set(excluded_chunks)
                new_chunks = [c for c in all_candidates if c.get('id') not in excluded_set]

                if len(new_chunks) >= n_chunks:
                    return new_chunks[:top_k], q

                # Not enough new chunks — backfill with the most relevant previously-used chunks to guarantee at least n_chunks candidates.
                used_chunks = [c for c in all_candidates if c.get('id') in excluded_set]
                backfill_needed = n_chunks - len(new_chunks)
                combined = new_chunks + used_chunks[:backfill_needed]
                logger.info(
                    f"Backfilled {min(backfill_needed, len(used_chunks))} used chunks "
                    f"({len(new_chunks)} new + {min(backfill_needed, len(used_chunks))} reused "
                    f"= {len(combined)} total)"
                )
                return combined[:top_k], q

            return all_candidates[:top_k], q
        except Exception as e:
            logger.warning(f"RAG search failed: {e}")
            return [], None

    def _get_safe_embeddings(self, chunks):
        texts = [(c.get('text') or '').strip()[:MAX_CHUNK_TEXT_LEN] for c in chunks]
        if not any(texts):
            return np.zeros((len(chunks), EMBEDDING_DIM))
        try:
            emb = self._embedding.get_embeddings(texts)
            if isinstance(emb, np.ndarray) and np.isnan(emb).sum() > 0:
                emb = np.nan_to_num(emb, nan=0.0)
            return emb
        except Exception:
            return np.zeros((len(chunks), EMBEDDING_DIM))

    def _rerank_chunks(self, context_chunks, query_vec, message, n_chunks, excluded_chunks=None):
        chunk_embeddings = np.array([])
        graph_distances = {}
        excluded_set = excluded_chunks or set()
        try:
            chunk_embeddings = self._get_safe_embeddings(context_chunks)
            self._graph.update_graph(new_chunks=context_chunks, new_embeddings=chunk_embeddings)

            # Graph-neighbor expansion with optional semantic quality gate.
            try:
                semantic_ids = {c.get('id') for c in context_chunks if c.get('id')}
                graph_neighbors = self._graph.get_neighboring_chunks_data(
                    list(semantic_ids), top_k=5
                )
                new_neighbors = [
                    c for c in graph_neighbors
                    if c.get('id') not in semantic_ids and c.get('id') not in excluded_set
                ]
                if new_neighbors and query_vec is not None and self._enable_graph_expansion_filter:
                    neighbor_embs = self._get_safe_embeddings(new_neighbors)
                    q_norm = query_vec / (np.linalg.norm(query_vec) + 1e-8)
                    n_norms = neighbor_embs / (np.linalg.norm(neighbor_embs, axis=1, keepdims=True) + 1e-8)
                    sims = np.dot(n_norms, q_norm)
                    if context_chunks:
                        best_sim = max(
                            (c.get('similarity_score', 0.0) for c in context_chunks),
                            default=0.3,
                        )
                        sim_threshold = max(0.1, best_sim * 0.4)
                    else:
                        sim_threshold = 0.1
                    qualified = []
                    qualified_embs = []
                    for idx, neighbor in enumerate(new_neighbors):
                        if float(sims[idx]) >= sim_threshold:
                            qualified.append(neighbor)
                            qualified_embs.append(neighbor_embs[idx])
                    if qualified:
                        context_chunks = list(context_chunks) + qualified
                        chunk_embeddings = np.vstack([chunk_embeddings, np.array(qualified_embs)])
                        logger.debug(
                            f"Graph expansion: +{len(qualified)} neighbor chunks "
                            f"(filtered from {len(new_neighbors)}, threshold={sim_threshold:.3f})"
                        )
                    elif new_neighbors:
                        logger.debug(
                            f"Graph expansion: all {len(new_neighbors)} neighbors below "
                            f"semantic threshold {sim_threshold:.3f}"
                        )
                elif new_neighbors:
                    neighbor_embs = self._get_safe_embeddings(new_neighbors)
                    context_chunks = list(context_chunks) + new_neighbors
                    chunk_embeddings = np.vstack([chunk_embeddings, neighbor_embs])
                    logger.debug(
                        f"Graph expansion: +{len(new_neighbors)} neighbor chunks "
                        "without semantic filtering"
                    )
            except Exception as e:
                logger.debug(f"Graph expansion skipped: {e}")
            # ─────────────────────────────────────────────────────────────────

            chunk_ids = [c.get('id', f'chunk_{i}') for i, c in enumerate(context_chunks)]
            graph_distances = self._graph.compute_graph_distances(
                query_text=message, chunk_ids=chunk_ids,
                embedding_service=self._embedding,
            )
        except Exception as e:
            # Graph computation failed — log full traceback and signal the
            # scorer to ignore the graph signal entirely (empty dict).
            # The scorer will fall back to semantic-only ranking instead of
            # using a fake constant fallback distance for every chunk.
            logger.warning(
                f"Graph distance computation failed: {e}. "
                f"Falling back to semantic-only ranking.",
                exc_info=True,
            )
            graph_distances = {}
            if len(chunk_embeddings) == 0:
                chunk_embeddings = self._get_safe_embeddings(context_chunks)
        try:
            if isinstance(chunk_embeddings, list):
                chunk_embeddings = np.array(chunk_embeddings)
            if isinstance(query_vec, list):
                query_vec = np.array(query_vec)
            return (
                self._scorer.rerank_chunks(
                    query_embedding=query_vec, chunks=context_chunks,
                    chunk_embeddings=chunk_embeddings, graph_distances=graph_distances,
                    top_k=n_chunks,
                    min_score=self._min_similarity_score if self._enable_sgas_filtering else 0.0,
                    query_text=message,
                    enable_adaptive_k=self._enable_sgas_filtering,
                    enable_cross_encoder=self._enable_cross_encoder_rerank,
                    enable_dynamic_weights=self._enable_dynamic_scoring_weights,
                    enable_semantic_anchor=self._enable_semantic_anchor,
                    enable_keyword_boost=self._enable_keyword_boost,
                ),
                chunk_embeddings,
                graph_distances,
            )
        except Exception as e:
            logger.error(f"Reranking failed entirely: {e}", exc_info=True)
            return context_chunks[:n_chunks], chunk_embeddings, graph_distances

    async def _manage_swap(self, session_id, session_data, final_context, iteration):
        try:
            # Predictive prefetching
            current_ids = [c.get('id') for c in final_context if c.get('id')]
            next_chunk_ids = []
            try:
                next_chunk_ids = self._graph.get_neighboring_chunk_ids(
                    current_ids, top_k=self._swap._prefetch_count
                )
            except Exception:
                pass

            # Fall back to the next N sequential unloaded chunks from the full document pool
            if not next_chunk_ids:
                loaded_ids = set(self._swap.gpu_chunks.keys())
                current_ids_set = set(current_ids)
                next_chunk_ids = [
                    cid for cid in self._swap.cpu_chunks.keys()
                    if cid not in loaded_ids and cid not in current_ids_set
                ][:self._swap._prefetch_count]

            prefetch_targets = [{'id': cid} for cid in next_chunk_ids] if next_chunk_ids else final_context
            self._swap.update_prefetch_buffer(prefetch_targets)
            tokens = sum(c.get('metadata', {}).get('chunk_size', 500) for c in final_context)
            decision = self._swap.decide_swap_action(final_context, tokens, iteration=iteration)
            self._swap.execute_swap_decision(decision)
            return decision.get('action', 'none')
        except Exception as e:
            logger.warning(f"Swap management skipped: {e}")
            return 'skipped'

    def _build_prompt(self, context_chunks, message, run_id: Optional[str] = None):
        enable_limit = self._prompt_config.get('enable_context_limit', True)
        max_tokens = self._prompt_config.get('max_context_tokens', 5000)
        prefix = f"[run:{run_id}]\n" if run_id else ""
        if context_chunks:
            ctx, count = "", 0
            for chunk in context_chunks:
                text = chunk.get("text") or chunk.get("document") or ""
                t = len(text) // 4
                if enable_limit and count + t > max_tokens:
                    break
                ctx += text + "\n\n"
                count += t
            return prefix + PROMPT_WITH_CONTEXT.format(context=ctx, message=message)
        return prefix + PROMPT_WITHOUT_CONTEXT.format(message=message)

    async def flush_vllm_cache(self) -> bool:
        """Flush vLLM's prefix KV-cache before a benchmark run."""
        try:
            return await self._vllm.flush_cache()
        except Exception:
            return False

    async def _safe_get_vllm_metrics(self) -> Dict[str, Any]:
        try:
            return await self._vllm.get_runtime_metrics()
        except Exception as e:
            logger.debug(f"vLLM runtime metrics unavailable: {e}")
            return {"available": False, "error": str(e)}

    async def _sample_gpu_during_inference(
        self,
        samples: List[Dict[str, Any]],
        stop_event: asyncio.Event,
        interval_sec: float = 1.0,
    ) -> None:
        """Sample physical GPU stats while the external vLLM process is generating."""
        first = get_nvidia_smi_stats()
        if first:
            samples.append(first)
        while not stop_event.is_set():
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=interval_sec)
            except asyncio.TimeoutError:
                sample = get_nvidia_smi_stats()
                if sample:
                    samples.append(sample)
        last = get_nvidia_smi_stats()
        if last:
            samples.append(last)

    @staticmethod
    def _metric_delta(after: Dict[str, Any], before: Dict[str, Any], key: str) -> float:
        after_val = after.get(key)
        before_val = before.get(key)
        if after_val is None or before_val is None:
            return 0.0
        try:
            return max(0.0, float(after_val) - float(before_val))
        except (TypeError, ValueError):
            return 0.0

    def _build_vllm_observability(
        self,
        before: Dict[str, Any],
        after: Dict[str, Any],
        response_metrics: Dict[str, Any],
        gpu_samples: List[Dict[str, Any]],
        inference_time_sec: float,
        usage: Dict[str, Any],
    ) -> Dict[str, Any]:
        gpu_summary = summarize_nvidia_smi_samples(gpu_samples)
        prompt_delta = self._metric_delta(after, before, "prompt_tokens_total")
        generation_delta = self._metric_delta(after, before, "generation_tokens_total")
        prefix_query_delta = self._metric_delta(after, before, "prefix_cache_queries_total")
        prefix_hit_delta = self._metric_delta(after, before, "prefix_cache_hits_total")
        request_success_delta = self._metric_delta(after, before, "request_success_total")
        preemption_delta = self._metric_delta(after, before, "num_preemptions_total")
        prefix_hit_rate_delta = (
            prefix_hit_delta / prefix_query_delta if prefix_query_delta > 0 else 0.0
        )

        generated_tokens = usage.get('completion_tokens') or generation_delta
        prompt_tokens = usage.get('prompt_tokens') or prompt_delta
        total_tokens = usage.get('total_tokens') or (generated_tokens + prompt_tokens)

        return {
            "available": bool(after.get("available") or response_metrics.get("available")),
            "before": before,
            "after": after,
            "response_metrics": response_metrics,
            "gpu_inference_window": gpu_summary,
            "prompt_tokens_delta": prompt_delta,
            "generation_tokens_delta": generation_delta,
            "request_success_delta": request_success_delta,
            "prefix_cache_queries_delta": prefix_query_delta,
            "prefix_cache_hits_delta": prefix_hit_delta,
            "prefix_cache_hit_rate_delta": prefix_hit_rate_delta,
            "num_preemptions_delta": preemption_delta,
            "kv_cache_usage_before": before.get("kv_cache_usage_perc"),
            "kv_cache_usage_after": after.get("kv_cache_usage_perc"),
            "kv_cache_usage_delta": (
                float(after.get("kv_cache_usage_perc", 0) or 0)
                - float(before.get("kv_cache_usage_perc", 0) or 0)
            ),
            "tokens_per_second": (
                round(generated_tokens / inference_time_sec, 4)
                if inference_time_sec > 0 and generated_tokens else 0.0
            ),
            "total_tokens_per_second": (
                round(total_tokens / inference_time_sec, 4)
                if inference_time_sec > 0 and total_tokens else 0.0
            ),
        }

    def get_graph_snapshot(self) -> Dict[str, Any]:
        """Export of current graph state for visualization / archival."""
        try:
            return self._graph.export_graph_info()
        except Exception as e:
            logger.warning(f"Graph snapshot failed: {e}")
            return {"nodes": [], "edges": [], "statistics": {}}

    def reset_for_benchmark(self) -> None:
        """Reset of all stateful services before a new benchmark mode run."""
        try:
            self._graph.reset_state()
        except Exception as e:
            logger.warning(f"Graph reset skipped: {e}")
        try:
            self._swap.reset_state()
        except Exception as e:
            logger.warning(f"Swap reset skipped: {e}")
        try:
            self._bm25.reset_state()
        except Exception:
            pass
        try:
            self._embedding.clear_cache()
        except Exception:
            pass  # embedding service may not have a cache — ignore silently

        # Free GPU memory so the next mode starts with a clean VRAM baseline
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.reset_accumulated_memory_stats()
            logger.info(
                f"GPU memory cleared: allocated={torch.cuda.memory_allocated()/1024**3:.3f} GB, "
                f"reserved={torch.cuda.memory_reserved()/1024**3:.3f} GB"
            )

    def _log_chunk_selection(self, session_id: str, iteration: int,
                             query: str, chunk_scores: list):
        """Logging why each chunk was selected, with per-component scores."""
        try:
            log_path = Path("logs/chunk_selection.jsonl")
            log_path.parent.mkdir(parents=True, exist_ok=True)
            record = {
                "session_id": session_id,
                "iteration": iteration,
                "query": query[:120],
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "selected_chunks": chunk_scores,
            }
            with open(log_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

            # Console summary
            for cs in chunk_scores:
                logger.info(
                    f"    [Chunk {cs['rank']}] {cs['id'][-20:]}  "
                    f"sem={cs['semantic_score']:.3f}  "
                    f"graph={cs['graph_score']:.3f}  "
                    f"hybrid={cs['hybrid_score']:.3f}  "
                    f"ce={cs['cross_encoder_score']:.3f}  "
                    f"final={cs['final_score']:.3f}"
                )
        except Exception as e:
            logger.debug(f"Chunk selection logging failed: {e}")

    async def _log_metrics(self, session_id, iteration, final_context, excluded_chunks,
                           chunking_stats, inference_time, swap_action, usage):
        try:
            metrics = {
                "session_id": session_id,
                "iteration": iteration,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "chunks_used": len(final_context),
                "excluded_from_previous": len(excluded_chunks),
                "coverage_percent": chunking_stats['coverage_percent'],
                "inference_time_sec": round(inference_time, 3),
                "swap_action": swap_action,
                "gpu_memory_mb": get_gpu_memory_stats(),
                "tokens_generated": usage.get('completion_tokens', 0),
                "tokens_in_prompt": usage.get('prompt_tokens', 0),
            }
            METRICS_FILE.parent.mkdir(parents=True, exist_ok=True)
            async with aiofiles.open(METRICS_FILE, 'a') as f:
                await f.write(json.dumps(metrics) + "\n")
        except Exception as e:
            logger.debug(f"Failed to log metrics: {e}")
