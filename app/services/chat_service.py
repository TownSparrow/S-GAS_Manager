import time
import json
import logging
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
from app.consts.defaults import MAX_CHUNK_TEXT_LEN, GRAPH_FALLBACK_DISTANCE, EMBEDDING_DIM, METRICS_FILE
from app.consts.prompts import PROMPT_WITH_CONTEXT, PROMPT_WITHOUT_CONTEXT
from app.utils.gpu import log_gpu_memory_detailed, get_gpu_memory_stats

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
                 prompt_config: Dict[str, Any], model_name: str = "", kv_monitor=None):
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

    async def process_chat(self, session_id: str, session_data: Dict[str, Any], message: str,
                           use_rag: bool = True, n_chunks: int = 5,
                           temperature: Optional[float] = None, top_p: Optional[float] = None,
                           max_tokens: Optional[int] = None,
                           baseline_mode: bool = False,
                           run_id: Optional[str] = None) -> Dict[str, Any]:
        """Runs S-GAS pipeline.

        Args:
            baseline_mode: If True, skips graph scoring and swap management.
                          Uses only semantic vector search (vanilla RAG).
                          Used for comparison in benchmarks.
        """

        session_data['iteration'] += 1
        iteration = session_data['iteration']
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
                final_context = context_chunks[:n_chunks]
            else:
                # S-GAS: full graph + hybrid reranking
                final_context, chunk_embeddings, graph_distances = self._rerank_chunks(
                    context_chunks, query_vec, message, n_chunks,
                )
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
        chunk_scores = [
            {
                'id': c.get('id', ''),
                'hybrid_score': c.get('hybrid_score', 0.0),
                'semantic_score': c.get('semantic_score', 0.0),
                'graph_score': c.get('graph_score', 0.0),
            }
            for c in final_context
        ]

        # ── STEP 3: Mark used chunks ──────────────────────────────────
        if final_context:
            used_ids = [c['id'] for c in final_context if 'id' in c]
            session_data['excluded_chunks'].extend(used_ids)
            self._chunker.mark_chunks_used(session_id, used_ids, iteration)

        # ── STEP 4: Cache check + Swap management ─────────────────────
        # 4a. Initialize swap state on the first turn so that pre-loaded GPU
        #     chunks are visible to the cache-hit check below.
        if not baseline_mode and iteration == 1 and not session_data.get('swap_initialized', False):
            all_chunks = await self._vector_store.get_all_session_chunks(session_id)
            if all_chunks:
                self._swap.initialize_chunks(all_chunks)
                session_data['swap_initialized'] = True

        # 4b. Check cache BEFORE the swap so we measure whether the PREVIOUS
        #     turn's predictive prefetch placed these chunks in GPU already.
        cache_check = self._swap.check_cache_hits(retrieved_chunk_ids)

        t_swap = time.time()
        if baseline_mode:
            swap_action = 'disabled'
        else:
            swap_action = await self._manage_swap(session_id, session_data, final_context, iteration)
        step_timings['swap_ms'] = round((time.time() - t_swap) * 1000, 2)
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

        t_inference = time.time()
        llm_response = await self._vllm.chat(prompt, {
            "temperature": temperature, "top_p": top_p, "max_tokens": max_tokens,
        })
        inference_time = time.time() - t_inference
        step_timings['inference_ms'] = round(inference_time * 1000, 2)
        self._swap.record_compute_time(inference_time * 1000)

        if self._kv_monitor:
            self._kv_monitor.log_iteration_end(
                iteration=iteration, generation_time_ms=inference_time * 1000,
                prompt_length_tokens=prompt_length_tokens,
                n_selected_chunks=n_selected_chunks,
                n_archived_chunks_this_iter=n_archived_chunks_this_iter,
            )

        vram_per_step['after_inference'] = _get_vram_gb()

        response_text = llm_response['choices'][0]['message']['content']
        usage = llm_response.get('usage', {})

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
                },
                'inference_time_sec': inference_time,
                'context_tokens': sum(
                    c.get('metadata', {}).get('chunk_size', 500) for c in final_context
                ),

                # VRAM per step
                'vram': vram_per_step,

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

                'timestamp': datetime.now(timezone.utc).isoformat(),
            },
        }

    async def _search_chunks(self, session_id, message, excluded_chunks, n_chunks):
        """
        Searches for relevant chunks. If no new chunks found, fall back to best
        previously-used chunks (re-rank by current query relevance).
        """
        try:
            q = self._embedding.get_embeddings([message])[0]
            top_k = max(20, n_chunks * 3)
            search_k = top_k * 2 if excluded_chunks else top_k

            all_candidates = await self._vector_store.search(
                query_embedding=q, session_id=session_id, top_k=search_k,
            )

            if excluded_chunks:
                excluded_set = set(excluded_chunks)
                new_chunks = [c for c in all_candidates if c.get('id') not in excluded_set]

                if new_chunks:
                    return new_chunks[:top_k], q

                # Fallback: no new chunks available — return best matches from all chunks
                logger.info(
                    f"No new chunks for session {session_id} (all {len(excluded_set)} excluded). "
                    f"Falling back to top-{n_chunks} most relevant from all chunks."
                )
                return all_candidates[:n_chunks], q

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

    def _rerank_chunks(self, context_chunks, query_vec, message, n_chunks):
        chunk_embeddings = np.array([])
        graph_distances = {}
        try:
            chunk_embeddings = self._get_safe_embeddings(context_chunks)
            self._graph.update_graph(new_chunks=context_chunks, new_embeddings=chunk_embeddings)

            # Graph-neighbor expansion
            try:
                semantic_ids = {c.get('id') for c in context_chunks if c.get('id')}
                graph_neighbors = self._graph.get_neighboring_chunks_data(
                    list(semantic_ids), top_k=10
                )
                new_neighbors = [c for c in graph_neighbors if c.get('id') not in semantic_ids]
                if new_neighbors:
                    context_chunks = list(context_chunks) + new_neighbors
                    extra_emb = self._get_safe_embeddings(new_neighbors)
                    chunk_embeddings = np.vstack([chunk_embeddings, extra_emb])
                    logger.debug(f"Graph expansion: +{len(new_neighbors)} neighbor chunks")
            except Exception as e:
                logger.debug(f"Graph expansion skipped: {e}")
            # ─────────────────────────────────────────────────────────────────

            chunk_ids = [c.get('id', f'chunk_{i}') for i, c in enumerate(context_chunks)]
            graph_distances = self._graph.compute_graph_distances(query_text=message, chunk_ids=chunk_ids)
        except Exception:
            graph_distances = {
                c.get('id', f'chunk_{i}'): GRAPH_FALLBACK_DISTANCE
                for i, c in enumerate(context_chunks)
            }
            if len(chunk_embeddings) == 0:
                chunk_embeddings = self._get_safe_embeddings(context_chunks)
        try:
            if isinstance(chunk_embeddings, list):
                chunk_embeddings = np.array(chunk_embeddings)
            if isinstance(query_vec, list):
                query_vec = np.array(query_vec)
            if not graph_distances:
                graph_distances = {
                    c.get('id', f'chunk_{i}'): GRAPH_FALLBACK_DISTANCE
                    for i, c in enumerate(context_chunks)
                }
            return (
                self._scorer.rerank_chunks(
                    query_embedding=query_vec, chunks=context_chunks,
                    chunk_embeddings=chunk_embeddings, graph_distances=graph_distances,
                    top_k=n_chunks,
                ),
                chunk_embeddings,
                graph_distances,
            )
        except Exception:
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
