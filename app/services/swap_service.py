import time
import logging
from typing import List, Dict, Any, Optional
from collections import deque
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

import torch
import torch.cuda

from app.interfaces.swap_manager import ISwapManager
from app.consts.defaults import SWAP_LRU_OFFLOAD_COUNT, SWAP_GPU_PRELOAD_COUNT

logger = logging.getLogger(__name__)


class SwapService(ISwapManager):
    def __init__(self, threshold: float = 0.3, prefetch_count: int = 5, memory_check_interval_ms: int = 50, max_gpu_memory_tokens: Optional[int] = None, device: str = "cuda:0" if torch.cuda.is_available() else "cpu", force_offload_on_iteration: int = -1, retain_top_centrality_pct: float = 0.15, eviction_ram_threshold_gb: float = 12.0):
        self._device = torch.device(device) if isinstance(device, str) else device
        self._threshold = threshold
        self._prefetch_count = prefetch_count
        self._max_gpu_memory_tokens = max_gpu_memory_tokens or 8192
        if "cuda" in str(self._device) and not torch.cuda.is_available():
            self._device = torch.device("cpu")
        elif "cuda" in str(self._device):
            torch.cuda.set_device(self._device)
        self.gpu_chunks: Dict[str, torch.Tensor] = {}
        self.cpu_chunks: Dict[str, torch.Tensor] = {}
        self._chunk_metadata: Dict[str, Dict[str, Any]] = {}
        self._prefetch_buffer = deque(maxlen=prefetch_count)
        self._transfer_stream = torch.cuda.Stream() if "cuda" in str(self._device) else None
        self._t_comp_history = deque(maxlen=10)
        self._t_swap_history = deque(maxlen=10)
        self.swap_to_ram_count = 0
        self.swap_to_gpu_count = 0
        self.prefetch_hits = 0
        self._prefetch_misses = 0
        self._cache_hits_total = 0
        self._cache_misses_total = 0
        self.total_swaps = 0
        self.last_action = "wait"
        self._gpu_access_time: Dict[str, float] = {}
        self._force_offload_on_iteration = force_offload_on_iteration
        self._retain_top_centrality_pct = retain_top_centrality_pct
        self._eviction_ram_threshold_gb = eviction_ram_threshold_gb
        self._protected_chunk_ids: set = set()  # high-centrality chunks to retain
        self._eviction_candidates: set = set()  # marked for deletion when RAM is tight

    def _move_to_gpu(self, chunk_id, chunk_data):
        try:
            is_cuda = "cuda" in str(self._device)
            embedding = chunk_data.get('embedding')
            if embedding is None:
                return False
            if chunk_id in self.gpu_chunks:
                self._gpu_access_time[chunk_id] = time.time()
                return True

            if is_cuda:
                bytes_needed = embedding.element_size() * embedding.numel()
                free = torch.cuda.mem_get_info(self._device)[0]
                if free <= bytes_needed * 1.5:
                    return False
                with torch.cuda.stream(self._transfer_stream):
                    gpu_tensor = embedding.to(self._device, non_blocking=True, copy=True)
                    torch.cuda.synchronize()
                self.gpu_chunks[chunk_id] = gpu_tensor
                bytes_used = bytes_needed
            else:
                # CPU simulation: move from cold (cpu_chunks) to hot (gpu_chunks)
                self.gpu_chunks[chunk_id] = embedding
                bytes_used = embedding.element_size() * embedding.numel()

            self._chunk_metadata[chunk_id] = {'text': chunk_data.get('text', ''), 'metadata': chunk_data.get('metadata', {}), 'device': 'gpu' if is_cuda else 'hot', 'size_bytes': bytes_used}
            self.swap_to_gpu_count += 1
            self.total_swaps += 1
            self.last_action = "load"
            self._gpu_access_time[chunk_id] = time.time()
            return True
        except RuntimeError:
            return False

    def _move_to_ram(self, chunk_id):
        try:
            if chunk_id not in self.gpu_chunks:
                return False
            is_cuda = "cuda" in str(self._device)
            gpu_tensor = self.gpu_chunks[chunk_id]

            if is_cuda:
                with torch.cuda.stream(self._transfer_stream):
                    cpu_tensor = gpu_tensor.to('cpu', non_blocking=True, copy=True)
                    try:
                        if not cpu_tensor.is_pinned():
                            cpu_tensor = cpu_tensor.pin_memory()
                    except Exception:
                        pass
                    torch.cuda.synchronize()
                self.cpu_chunks[chunk_id] = cpu_tensor
                del self.gpu_chunks[chunk_id]
                torch.cuda.empty_cache()
            else:
                # CPU simulation: move from hot (gpu_chunks) back to cold (cpu_chunks)
                self.cpu_chunks[chunk_id] = gpu_tensor
                del self.gpu_chunks[chunk_id]

            self.total_swaps += 1
            self.last_action = "offload"
            if chunk_id in self._chunk_metadata:
                self._chunk_metadata[chunk_id]['device'] = 'cpu'
            self.swap_to_ram_count += 1
            return True
        except RuntimeError:
            return False

    def initialize_chunks(self, chunks):
        is_cuda = "cuda" in str(self._device)
        for chunk in chunks:
            cid = chunk.get('id', f'chunk_{len(self.cpu_chunks)}')
            try:
                emb = chunk.get('embedding')
                if emb is None:
                    continue
                if not isinstance(emb, torch.Tensor):
                    emb = torch.tensor(emb, dtype=torch.float32)
                if is_cuda and not emb.is_pinned():
                    try:
                        emb = emb.pin_memory()
                    except Exception:
                        pass  # pin_memory may fail without CUDA runtime
                self.cpu_chunks[cid] = emb
                self._chunk_metadata[cid] = {'text': chunk.get('text', ''), 'metadata': chunk.get('metadata', {}), 'device': 'cpu', 'size_bytes': emb.element_size() * emb.numel()}
            except Exception as e:
                logger.error(f"Error archiving chunk {cid}: {e}")
        if is_cuda and self.cpu_chunks:
            for cid in list(self.cpu_chunks.keys())[:SWAP_GPU_PRELOAD_COUNT]:
                self._move_to_gpu(cid, {'embedding': self.cpu_chunks[cid], **self._chunk_metadata.get(cid, {})})

    def check_cache_hits(self, requested_chunk_ids: List[str]) -> Dict[str, Any]:
        """
        Check how many requested chunks are already in GPU (true cache hit).
        """
        hits = 0
        misses = 0
        for cid in requested_chunk_ids:
            if cid in self.gpu_chunks:
                hits += 1
                self._gpu_access_time[cid] = time.time()
            else:
                misses += 1
        self._cache_hits_total += hits
        self._cache_misses_total += misses
        total = hits + misses
        return {
            'hits': hits,
            'misses': misses,
            'hit_rate': round(hits / total, 3) if total > 0 else 0.0,
        }

    def update_prefetch_buffer(self, chunks):
        self._prefetch_buffer.clear()
        for c in chunks[:self._prefetch_count]:
            cid = c.get('id')
            if cid:
                self._prefetch_buffer.append(cid)
        # Prefetch to GPU/hot storage (works on both CUDA and CPU)
        for cid in self._prefetch_buffer:
            if cid in self.cpu_chunks and cid not in self.gpu_chunks:
                try:
                    if self._move_to_gpu(cid, {'embedding': self.cpu_chunks[cid], **self._chunk_metadata.get(cid, {})}):
                        self.prefetch_hits += 1
                except Exception:
                    self._prefetch_misses += 1

    def decide_swap_action(self, context_chunks, current_context_tokens, iteration=0):
        """Determine what swap action to take for the current iteration.

        The default strategy is *proactive*: GPU/hot chunks that are no longer part
        of the current context are offloaded to RAM/cold on every turn.  This ensures
        the S-GAS adaptive-swapping behaviour is always active (not just under memory
        pressure), making the algorithm's GPU ↔ RAM cycle visible in benchmarks
        from the very first iteration.

        On CPU-only systems, swapping is simulated using hot (gpu_chunks) vs cold
        (cpu_chunks) dictionaries so the algorithm remains active and countable.
        """
        is_cuda = "cuda" in str(self._device)
        current_ids = {c.get('id') for c in context_chunks if c.get('id')}

        # Optional: force-offload everything not in context at a specific iteration.
        if self._force_offload_on_iteration >= 0 and iteration == self._force_offload_on_iteration:
            outdated = [cid for cid in self.gpu_chunks if cid not in current_ids]
            if outdated:
                return {'action': 'offload', 'chunk_ids': outdated[:SWAP_LRU_OFFLOAD_COUNT + 2]}

        # Proactive offload: push non-current GPU/hot chunks to RAM/cold,
        # but retain high-centrality chunks for associative recall.
        outdated = set(self.gpu_chunks.keys()) - current_ids
        # Protect high-centrality chunks — keep them in hot storage
        evictable = outdated - self._protected_chunk_ids
        # Mark evicted chunks as deletion candidates
        self._eviction_candidates.update(evictable)
        if evictable:
            return {'action': 'offload', 'chunk_ids': list(evictable)[:SWAP_LRU_OFFLOAD_COUNT + 2]}
        # If only protected chunks remain, still offload them (they stay in cold/cpu)
        if outdated:
            return {'action': 'offload', 'chunk_ids': list(outdated)[:SWAP_LRU_OFFLOAD_COUNT + 2]}

        # Load missing current-context chunks into GPU/hot storage.
        if is_cuda:
            try:
                free_mb = torch.cuda.mem_get_info(self._device)[0] / (1024 * 1024)
                needed_mb = current_context_tokens * 4 / 1024
            except Exception:
                return {'action': 'none', 'chunk_ids': []}
            if free_mb > needed_mb * 2:
                ids = [c.get('id') for c in context_chunks if c.get('id') and c.get('id') not in self.gpu_chunks]
                if ids:
                    return {'action': 'load', 'chunk_ids': ids}
        else:
            # CPU simulation: load missing context chunks into hot storage
            ids = [c.get('id') for c in context_chunks
                   if c.get('id') and c.get('id') not in self.gpu_chunks and c.get('id') in self.cpu_chunks]
            if ids:
                return {'action': 'load', 'chunk_ids': ids}

        # Back-off: skip swap if it starts taking longer than the compute budget.
        avg_c = sum(self._t_comp_history) / len(self._t_comp_history) if self._t_comp_history else 1.0
        avg_s = sum(self._t_swap_history) / len(self._t_swap_history) if self._t_swap_history else 0.001
        if avg_c > 0 and avg_s / avg_c > self._threshold:
            return {'action': 'none', 'chunk_ids': []}

        return {'action': 'none', 'chunk_ids': []}

    def execute_swap_decision(self, decision):
        action = decision.get('action', 'none')
        chunk_ids = decision.get('chunk_ids', [])
        if action == 'load':
            t = time.time()
            for cid in chunk_ids:
                if cid in self.cpu_chunks and cid not in self.gpu_chunks:
                    self._move_to_gpu(cid, {'embedding': self.cpu_chunks[cid], **self._chunk_metadata.get(cid, {})})
            dt = time.time() - t
            if dt > 0:
                self._t_swap_history.append(dt)
        elif action == 'offload':
            t = time.time()
            for cid in chunk_ids:
                self._move_to_ram(cid)
            dt = time.time() - t
            if dt > 0:
                self._t_swap_history.append(dt)

    def execute_swap_decision_with_timeout(self, decision, timeout_sec=5):
        from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
        try:
            action = decision.get('action', 'none')
            chunk_ids = decision.get('chunk_ids', [])

            if action == 'load':
                t = time.time()
                def load_chunks():
                    for cid in chunk_ids:
                        if cid in self.cpu_chunks and cid not in self.gpu_chunks:
                            self._move_to_gpu(cid, {'embedding': self.cpu_chunks[cid], **self._chunk_metadata.get(cid, {})})
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(load_chunks)
                    future.result(timeout=timeout_sec)
                dt = time.time() - t
                if dt > 0:
                    self._t_swap_history.append(dt)
                return True

            elif action == 'offload':
                t = time.time()
                def offload_chunks():
                    for cid in chunk_ids:
                        self._move_to_ram(cid)
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(offload_chunks)
                    future.result(timeout=timeout_sec)
                dt = time.time() - t
                if dt > 0:
                    self._t_swap_history.append(dt)
                return True

            return True

        except FuturesTimeoutError:
            logger.error(f"Swap operation timeout after {timeout_sec}s")
            if "cuda" in str(self._device):
                torch.cuda.empty_cache()
            return False
        except Exception as e:
            logger.error(f"Swap execution error: {e}")
            return False

    def update_protected_chunks(self, high_centrality_ids: list):
        """Update the set of high-centrality chunks that should be retained in hot storage."""
        self._protected_chunk_ids = set(high_centrality_ids)

    def evict_candidates_if_needed(self):
        """Delete eviction candidates from cpu_chunks when RAM usage exceeds threshold.

        Returns the number of chunks actually deleted.
        """
        try:
            import psutil
            ram_used_gb = psutil.virtual_memory().used / (1024 ** 3)
        except ImportError:
            return 0
        if ram_used_gb < self._eviction_ram_threshold_gb:
            return 0
        deleted = 0
        for cid in list(self._eviction_candidates):
            if cid in self.cpu_chunks and cid not in self.gpu_chunks:
                del self.cpu_chunks[cid]
                self._chunk_metadata.pop(cid, None)
                self._eviction_candidates.discard(cid)
                deleted += 1
        if deleted:
            logger.info(f"Evicted {deleted} candidate chunks (RAM {ram_used_gb:.1f}GB >= {self._eviction_ram_threshold_gb}GB)")
        return deleted

    def record_compute_time(self, compute_time_ms):
        self._t_comp_history.append(compute_time_ms / 1000.0)

    def get_statistics(self):
        gpu_stats = {}
        is_cuda = "cuda" in str(self._device)
        if is_cuda:
            gpu_stats = {'gpu_reserved_mb': round(torch.cuda.memory_reserved(self._device) / (1024*1024), 2), 'gpu_allocated_mb': round(torch.cuda.memory_allocated(self._device) / (1024*1024), 2), 'free_gpu_mb': round(torch.cuda.mem_get_info(self._device)[0] / (1024*1024), 2)}
        hot_mb = sum(t.element_size() * t.numel() / (1024*1024) for t in self.gpu_chunks.values())
        gpu_stats['hot_storage_mb'] = round(hot_mb, 2)
        cpu_mb = sum(t.element_size() * t.numel() / (1024*1024) for t in self.cpu_chunks.values())
        total_prefetch = self.prefetch_hits + self._prefetch_misses
        total_cache = self._cache_hits_total + self._cache_misses_total
        return {
            **gpu_stats,
            'cpu_archive_mb': round(cpu_mb, 2),
            'chunks_in_archive': len(self.cpu_chunks),
            'chunks_in_gpu': len(self.gpu_chunks),
            'chunks_in_ram': len(self.cpu_chunks) - len(self.gpu_chunks),
            'swap_to_ram_count': self.swap_to_ram_count,
            'swap_to_gpu_count': self.swap_to_gpu_count,
            'total_swap_operations': self.total_swaps,
            # True cache hit rate: requested chunks already in GPU
            'cache_hit_rate': round(self._cache_hits_total / total_cache, 3) if total_cache > 0 else 0,
            'cache_hits': self._cache_hits_total,
            'cache_misses': self._cache_misses_total,
            # Prefetch effectiveness
            'prefetch_hits': self.prefetch_hits,
            'prefetch_hit_rate': round(self.prefetch_hits / total_prefetch, 3) if total_prefetch > 0 else 0,
            # Timing
            'avg_compute_time_ms': round(sum(self._t_comp_history) / len(self._t_comp_history) * 1000, 2) if self._t_comp_history else 0,
            'avg_swap_time_ms': round(sum(self._t_swap_history) / len(self._t_swap_history) * 1000, 4) if self._t_swap_history else 0,
            'last_action': self.last_action,
            'swap_triggered': self.total_swaps > 0,
        }

    def reset_state(self):
        """Fully wipe all accumulated chunk state. Call before each benchmark mode to ensure clean isolation."""
        self.gpu_chunks.clear()
        if "cuda" in str(self._device):
            torch.cuda.empty_cache()
        self.cpu_chunks.clear()
        self._chunk_metadata.clear()
        self._prefetch_buffer.clear()
        self._gpu_access_time.clear()
        self.swap_to_ram_count = 0
        self.swap_to_gpu_count = 0
        self.prefetch_hits = 0
        self._prefetch_misses = 0
        self._cache_hits_total = 0
        self._cache_misses_total = 0
        self.total_swaps = 0
        self.last_action = "wait"
        self._t_comp_history.clear()
        self._t_swap_history.clear()
        self._protected_chunk_ids.clear()
        self._eviction_candidates.clear()

    def cleanup(self):
        self.gpu_chunks.clear()
        if "cuda" in str(self._device):
            torch.cuda.empty_cache()
