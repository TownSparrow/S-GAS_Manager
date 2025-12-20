import time
import logging
import torch
import torch.cuda
from typing import List, Dict, Any, Optional
from collections import deque
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

logger = logging.getLogger(__name__)


class SwapManager:
    """
    GPU ↔ CPU swap manager aligned with S-GAS adaptive chunking philosophy.
    
    Core principle:
    - All chunks live permanently in pinned CPU RAM (never deleted)
    - GPU VRAM holds only currently relevant chunks (prefetch buffer)
    - Swapping is memory optimization, NOT data deletion
    - Chunks remain available across ALL iterations for future relevance
    
    Based on PyTorch best practices:
    - torch.cuda for real memory management
    - Pinned memory for fast CPU ↔ GPU transfers
    - Asynchronous non_blocking transfers
    - Explicit cache cleanup via torch.cuda.empty_cache()
    - CUDA streams for parallel operations
    """
    
    def __init__(
        self,
        threshold: float = 0.3,
        prefetch_count: int = 5,
        memory_check_interval_ms: int = 50,
        max_gpu_memory_tokens: Optional[int] = None,
        device: str = "cuda:0" if torch.cuda.is_available() else "cpu",
        debug_mode: bool = False,
        force_offload_on_iteration: int = -1
    ):
        """
        Initialize SwapManager with GPU/CPU orchestration.
        
        Args:
            threshold: T_swap/T_compute ratio for swap decision (0.3 = swap if overhead < 30%)
            prefetch_count: number of chunks to prefetch into GPU
            memory_check_interval_ms: GPU memory check interval
            max_gpu_memory_tokens: maximum tokens allowed in GPU VRAM
            device: CUDA device to use (e.g., "cuda:0")
        """

        self.device = torch.device(device) if isinstance(device, str) else device
        self.threshold = threshold
        self.prefetch_count = prefetch_count
        self.memory_check_interval_ms = memory_check_interval_ms
        self.max_gpu_memory_tokens = max_gpu_memory_tokens or 8192
        
        # Verify CUDA availability
        if "cuda" in str(self.device):
            if not torch.cuda.is_available():
                logger.warning("CUDA not available, falling back to CPU")
                self.device = torch.device("cpu")
            else:
                torch.cuda.set_device(self.device)
                logger.info(f"Using GPU device: {torch.cuda.get_device_name(self.device)}")
        
        # Storage layers (philosophy: CPU RAM is permanent, GPU is temporary)
        self.gpu_chunks: Dict[str, torch.Tensor] = {}      # Current working set in GPU VRAM
        self.cpu_chunks: Dict[str, torch.Tensor] = {}      # Permanent archive in pinned RAM
        self.chunk_metadata: Dict[str, Dict[str, Any]] = {}  # Metadata for all chunks
        
        # Prefetch strategy
        self.prefetch_buffer: deque = deque(maxlen=prefetch_count)
        
        # Async transfer stream for non-blocking GPU operations
        self.transfer_stream = torch.cuda.Stream() if "cuda" in str(self.device) else None
        
        # Performance tracking
        self.t_comp_history = deque(maxlen=10)
        self.t_swap_history = deque(maxlen=10)
        self.swap_to_ram_count = 0
        self.swap_to_gpu_count = 0
        self.prefetch_hits = 0
        self.prefetch_misses = 0
        self.total_swaps = 0
        self.last_action = "wait"
        
        # Track GPU residency timing (for prefetch optimization, NOT deletion)
        self.gpu_access_time: Dict[str, float] = {}

        self.debug_mode = debug_mode
        self.force_offload_on_iteration = force_offload_on_iteration
        self.current_iteration = 0
        
        logger.info("SwapManager initialized with permanent CPU RAM archive")
    
    def move_to_gpu(self, chunk_id: str, chunk_data: Dict[str, Any]) -> bool:
        """
        Load chunk from CPU RAM into GPU VRAM for inference acceleration.
        Chunk remains in CPU RAM for future iterations.
        
        Args:
            chunk_id: Unique chunk identifier
            chunk_data: Dictionary with 'embedding' tensor and metadata
        
        Returns:
            True if successful, False if insufficient GPU memory
        """
        try:
            if "cuda" not in str(self.device):
                # CPU mode: store in CPU archive
                self.cpu_chunks[chunk_id] = chunk_data
                return True
            
            # Extract embedding tensor
            embedding = chunk_data.get('embedding')
            if embedding is None:
                logger.warning(f"⚠️ No embedding for chunk {chunk_id}")
                return False
            
            # Skip if already in GPU (avoid duplicate transfers)
            if chunk_id in self.gpu_chunks:
                self.gpu_access_time[chunk_id] = time.time()
                return True
            
            # Check if we have enough GPU memory
            bytes_needed = embedding.element_size() * embedding.numel()
            if not self._check_gpu_memory(bytes_needed):
                logger.debug(f"⚠️ Insufficient GPU memory for chunk {chunk_id}")
                return False
            
            # Using asynchronous stream for fast GPU loading
            with torch.cuda.stream(self.transfer_stream):
                # Non-blocking copy to GPU (transfer happens in parallel with compute)
                gpu_tensor = embedding.to(
                    self.device,
                    non_blocking=True,
                    copy=True  # Explicit copy (not just reference)
                )
               
                # Synchronize to ensure transfer completion
                torch.cuda.synchronize()
            
            # Store in GPU working set
            self.gpu_chunks[chunk_id] = gpu_tensor
            
            # Update metadata
            self.chunk_metadata[chunk_id] = {
                'text': chunk_data.get('text', ''),
                'metadata': chunk_data.get('metadata', {}),
                'device': 'gpu',
                'size_bytes': bytes_needed
            }
            
            self.swap_to_gpu_count += 1
            self.total_swaps += 1
            self.last_action = "load"
            self.gpu_access_time[chunk_id] = time.time()
            
            logger.debug(f"✅ Chunk {chunk_id} loaded to GPU ({embedding.numel()} elements)")
            return True
            
        except RuntimeError as e:
            logger.error(f"❌ CUDA error loading chunk to GPU: {e}")
            return False
    
    def move_to_ram(self, chunk_id: str) -> bool:
        """
        Unload chunk from GPU VRAM back to CPU RAM for memory management.
        Chunk remains in permanent CPU archive - NEVER deleted.
        
        Args:
            chunk_id: Unique chunk identifier
        
        Returns:
            True if successful, False if chunk not in GPU
        """
        try:
            if chunk_id not in self.gpu_chunks:
                logger.warning(f"⚠️ Chunk {chunk_id} not in GPU")
                return False
            
            # Get tensor from GPU
            gpu_tensor = self.gpu_chunks[chunk_id]
            
            # Using asynchronous stream for CPU unload
            with torch.cuda.stream(self.transfer_stream):
                # Non-blocking copy back to CPU
                cpu_tensor = gpu_tensor.to(
                    'cpu',
                    non_blocking=True,
                    copy=True
                )
                
                # Pin memory for future fast GPU transfers
                if not cpu_tensor.is_pinned():
                    cpu_tensor = cpu_tensor.pin_memory()

                # Synchronize
                torch.cuda.synchronize()
            
            # Store in permanent CPU archive (never lose data!)
            self.cpu_chunks[chunk_id] = cpu_tensor
            self.total_swaps += 1
            self.last_action = "offload"
            
            # Remove from GPU working set only
            del self.gpu_chunks[chunk_id]
            
            # Explicitly free GPU memory
            torch.cuda.empty_cache()
            
            # Update metadata: chunk still exists, just on CPU now
            if chunk_id in self.chunk_metadata:
                self.chunk_metadata[chunk_id]['device'] = 'cpu'
            
            self.swap_to_ram_count += 1
            logger.debug(f"✅ Chunk {chunk_id} unloaded to CPU RAM (archived)")
            return True
            
        except RuntimeError as e:
            logger.error(f"❌ CUDA error unloading chunk to RAM: {e}")
            return False
    
    # =====================================================
    # INITIALIZATION (CPU ARCHIVE BUILDING)
    # =====================================================
    
    def initialize_chunks(self, chunks: List[Dict[str, Any]]) -> None:
        """
        Build permanent CPU RAM archive of all chunks.
        These chunks persist across ALL iterations - foundation of S-GAS philosophy.
        
        Args:
            chunks: List of chunks with embeddings and metadata
        """
        logger.info(f"Building CPU RAM archive for {len(chunks)} chunks...")
        
        # Step 1: Archive ALL chunks in CPU RAM (permanent)
        for chunk in chunks:
            chunk_id = chunk.get('id', f'chunk_{len(self.cpu_chunks)}')
            
            try:
                # Get embedding tensor
                embedding = chunk.get('embedding')
                
                if embedding is None:
                    logger.warning(f"⚠️ No embedding for chunk {chunk_id}, skipping")
                    continue
                
                # Convert to PyTorch tensor if needed
                if not isinstance(embedding, torch.Tensor):
                    embedding = torch.tensor(embedding, dtype=torch.float32)
                
                # Pin memory for fast GPU access later
                if not embedding.is_pinned():
                    embedding = embedding.pin_memory()
                
                # Store in CPU archive
                self.cpu_chunks[chunk_id] = embedding
                
                # Store metadata
                self.chunk_metadata[chunk_id] = {
                    'text': chunk.get('text', ''),
                    'metadata': chunk.get('metadata', {}),
                    'device': 'cpu',
                    'size_bytes': embedding.element_size() * embedding.numel()
                }
                
                logger.debug(f"Chunk {chunk_id} archived in CPU RAM ({embedding.numel()} floats)")
                
            except Exception as e:
                logger.error(f"❌ Error archiving chunk {chunk_id}: {e}")

        logger.info(f"✅ CPU archive complete: {len(self.cpu_chunks)} chunks (permanent storage)")

        # Step 2: Preload top-5 to GPU for acceleration
        if "cuda" in str(self.device) and len(self.cpu_chunks) > 0:
            logger.info("Preloading top chunks to GPU...")

            top_k = min(5, len(self.cpu_chunks))
            preload_ids = list(self.cpu_chunks.keys())[:top_k]

            for chunk_id in preload_ids:
                try:
                    chunk_data = {
                        'embedding': self.cpu_chunks[chunk_id],
                        **self.chunk_metadata.get(chunk_id, {})
                    }
                    if self.move_to_gpu(chunk_id, chunk_data):
                        logger.debug(f"✅ Preloaded {chunk_id} to GPU")
                    else:
                        logger.warning(f"⚠️ Failed to preload {chunk_id} to GPU")
                except Exception as e:
                    logger.error(f"❌ Preload error for {chunk_id}: {e}")
        
            logger.info(f"✅ Preloaded {len(self.gpu_chunks)} chunks to GPU") 
        
    def update_prefetch_buffer(self, chunks: List[Dict[str, Any]]) -> None:
        """
        Update prefetch buffer with next likely-needed chunks.
        These will be preloaded into GPU asynchronously.
        
        Args:
            chunks: List of chunks expected in next iteration
        """
        self.prefetch_buffer.clear()
        
        for chunk in chunks[:self.prefetch_count]:
            chunk_id = chunk.get('id')
            if chunk_id:
                self.prefetch_buffer.append(chunk_id)
        
        # Asynchronously preload into GPU
        self._prefetch_async()
    
    def _prefetch_async(self) -> None:
        """
        Asynchronously load prefetch buffer chunks into GPU.
        This hides GPU loading latency behind compute time.
        """
        if "cuda" not in str(self.device):
            return
        
        for chunk_id in self.prefetch_buffer:
            # Only prefetch if in CPU archive and not already in GPU
            if chunk_id in self.cpu_chunks and chunk_id not in self.gpu_chunks:
                try:
                    chunk_data = {
                        'embedding': self.cpu_chunks[chunk_id],
                        **self.chunk_metadata.get(chunk_id, {})
                    }
                    if self.move_to_gpu(chunk_id, chunk_data):
                        self.prefetch_hits += 1
                except Exception as e:
                    logger.debug(f"❌ Prefetch failed for {chunk_id}: {e}")
                    self.prefetch_misses += 1
    
    # =====================================================
    # SWAP DECISION ALGORITHM
    # =====================================================
    
    def decide_swap_action(
        self,
        context_chunks: List[Dict[str, Any]],
        current_context_tokens: int,
        iteration: int = 0
    ) -> Dict[str, Any]:
        """
        Decide GPU memory optimization strategy based on performance metrics.
        
        Philosophy:
        - If compute time >> swap time: LOAD chunks to GPU (acceleration worth it)
        - If compute time ~= swap time: Don't swap (overhead too high)
        - If GPU full: UNLOAD least-recently-used chunk to RAM (keep in archive!)
        
        Args:
            context_chunks: Current iteration's relevant chunks
            current_context_tokens: Total tokens in current context
        
        Returns:
            Decision dict with 'action' (load/offload/none) and 'chunk_ids'
        """

        # No GPU = no swapping
        if "cuda" not in str(self.device):
            return {'action': 'none', 'chunk_ids': []}
        
        # =====================================================
        # DEBUG MODE:
        # =====================================================

        if self.debug_mode:
            # Mode 1: Force offload on specific iteration
            if self.force_offload_on_iteration >= 0 and iteration == self.force_offload_on_iteration:
                logger.warning(f"🔴 DEBUG MODE: Force offload on iteration {iteration}")
                outdated = [c for c in self.gpu_chunks.keys() if c not in [ch.get('id') for ch in context_chunks]]
                if outdated:
                    logger.info(f"OFFLOADING {len(outdated)} chunks for testing")
                    return {'action': 'offload', 'chunk_ids': outdated[:5]}
        
            # Mode 2: Always check for outdated chunks
            current_ids = set(c.get('id') for c in context_chunks if c.get('id'))
            gpu_ids = set(self.gpu_chunks.keys())
            outdated_chunks = gpu_ids - current_ids
        
            if outdated_chunks:
                logger.warning(f"🔴 DEBUG MODE: Found {len(outdated_chunks)} outdated chunks")
                return {'action': 'offload', 'chunk_ids': list(outdated_chunks)[:5]}

        # ════════════════════════════════════════════════════════════════
        # NORMAL MODE: Intelligent swap decisions
        # ════════════════════════════════════════════════════════════════

        try:
            gpu_utilization = torch.cuda.utilization(self.device)
            gpu_allocated = torch.cuda.memory_allocated(self.device)
            gpu_mb = gpu_allocated / 1024 / 1024
        except Exception as e:
            logger.warning(f"❌ Failed to get GPU stats: {e}")
            gpu_utilization = 0
            gpu_mb = 0
        
        free_memory_mb = self._get_free_gpu_memory_mb()
        needed_memory_mb = current_context_tokens * 4 / 1024
    
        # Decision 1: If GPU memory very LOW, unload oldest chunks to RAM
        if free_memory_mb < needed_memory_mb * 0.2:
            logger.warning(f"⚠️ LOW GPU memory ({free_memory_mb:.1f} MB), offloading LRU chunks")
        
            candidates = [
                cid for cid in self.gpu_chunks.keys()
                if cid not in [c.get('id') for c in context_chunks]
            ]
        
            if candidates:
                lru_chunks = sorted(
                    candidates,
                    key=lambda x: self.gpu_access_time.get(x, 0)
                )[:3]
            
                logger.info(f"Will offload {len(lru_chunks)} LRU chunks: {lru_chunks}")
                return {'action': 'offload', 'chunk_ids': lru_chunks}
    
        # Decision 2: If GPU memory plenty, LOAD current chunks for acceleration
        if free_memory_mb > needed_memory_mb * 2:
            chunk_ids = [
                c.get('id') for c in context_chunks 
                if c.get('id') and c.get('id') not in self.gpu_chunks
            ]
        
            if chunk_ids:
                logger.info(f"✅ Plenty of GPU memory, preloading {len(chunk_ids)} chunks")
                return {'action': 'load', 'chunk_ids': chunk_ids}

        # Analyze performance history
        avg_compute_time = (
            sum(self.t_comp_history) / len(self.t_comp_history) 
            if self.t_comp_history else 1.0
        )
        avg_swap_time = (
            sum(self.t_swap_history) / len(self.t_swap_history) 
            if self.t_swap_history else 0.001
        )
        
        # Calculate swap overhead ratio
        if avg_compute_time > 0:
            swap_ratio = avg_swap_time / avg_compute_time
        else:
            swap_ratio = 0
        
        # If swapping is too slow, don't do it
        if swap_ratio > self.threshold:
            logger.debug(
                f"Swap overhead {swap_ratio:.4f} exceeds threshold {self.threshold}"
            )
            return {'action': 'none', 'chunk_ids': []}
        
        # No action needed
        return {'action': 'none', 'chunk_ids': []}

    # =====================================================
    # SWAP DECISION ALGORITHM (WITH LIMITER)
    # =====================================================

    # def decide_swap_action(
    #     self,
    #     context_chunks: List[Dict[str, Any]],
    #     current_context_tokens: int
    # ) -> Dict[str, Any]:
    #     """
    #     Decide GPU memory optimization strategy based on performance metrics.
    #     """
    #     # No GPU = no swapping
    #     if "cuda" not in str(self.device):
    #         return {'action': 'none', 'chunk_ids': []}

    #     try:
    #         gpu_utilization = torch.cuda.utilization(self.device)
    #         gpu_allocated = torch.cuda.memory_allocated(self.device)
    #         gpu_mb = gpu_allocated / 1024 / 1024
    #     except Exception as e:
    #         logger.warning(f"❌ Failed to get GPU stats: {e}")
    #         gpu_utilization = 0
    #         gpu_mb = 0

    #     free_memory_mb = self._get_free_gpu_memory_mb()
    #     needed_memory_mb = current_context_tokens * 4 / 1024

    
    #     current_chunk_ids = set(c.get('id') for c in context_chunks if c.get('id'))
    #     gpu_chunk_ids = set(self.gpu_chunks.keys())
    
    #     # Chunks that are in GPU but NOT in current context → OFFLOAD them!
    #     outdated_chunks = gpu_chunk_ids - current_chunk_ids
    
    #     if len(outdated_chunks) > 0:
    #         logger.warning(f"⚠️ OFFLOAD DECISION: {len(outdated_chunks)} chunks in GPU are not needed")
    #         logger.info(f"   Outdated chunks: {list(outdated_chunks)[:5]}")
        
    #         # Force offload
    #         for chunk_id in list(outdated_chunks)[:3]:  # Offload top 3
    #             logger.info(f"   Scheduling OFFLOAD: {chunk_id}")
        
    #         return {'action': 'offload', 'chunk_ids': list(outdated_chunks)[:5]}

    #     # Decision 1: If GPU memory very LOW, unload oldest chunks to RAM
    #     if free_memory_mb < needed_memory_mb * 0.2:
    #         logger.warning(f"⚠️ LOW GPU memory ({free_memory_mb:.1f} MB), offloading LRU chunks")

    #         candidates = [
    #             cid for cid in self.gpu_chunks.keys()
    #             if cid not in [c.get('id') for c in context_chunks]
    #         ]

    #         if candidates:
    #             lru_chunks = sorted(
    #                 candidates,
    #                 key=lambda x: self.gpu_access_time.get(x, 0)
    #             )[:3]

    #             logger.info(f"Will offload {len(lru_chunks)} LRU chunks: {lru_chunks}")
    #             return {'action': 'offload', 'chunk_ids': lru_chunks}

    #     # Decision 2: If GPU memory plenty, LOAD current chunks for acceleration
    #     if free_memory_mb > needed_memory_mb * 2:
    #         chunk_ids = [
    #             c.get('id') for c in context_chunks
    #             if c.get('id') and c.get('id') not in self.gpu_chunks
    #         ]

    #         if chunk_ids:
    #             logger.info(f"✅ Plenty of GPU memory, preloading {len(chunk_ids)} chunks")
    #             return {'action': 'load', 'chunk_ids': chunk_ids}

    #     # Analyze performance history
    #     avg_compute_time = (
    #         sum(self.t_comp_history) / len(self.t_comp_history)
    #         if self.t_comp_history else 1.0
    #     )

    #     avg_swap_time = (
    #         sum(self.t_swap_history) / len(self.t_swap_history)
    #         if self.t_swap_history else 0.001
    #     )

    #     # Calculate swap overhead ratio
    #     if avg_compute_time > 0:
    #         swap_ratio = avg_swap_time / avg_compute_time
    #     else:
    #         swap_ratio = 0

    #     # If swapping is too slow, don't do it
    #     if swap_ratio > self.threshold:
    #         logger.debug(
    #             f"Swap overhead {swap_ratio:.4f} exceeds threshold {self.threshold}"
    #         )

    #         return {'action': 'none', 'chunk_ids': []}

    #     # No action needed
    #     return {'action': 'none', 'chunk_ids': []}
    
    def execute_swap_decision(self, decision: Dict[str, Any]) -> None:
        """
        Execute swap decision by moving chunks between GPU and CPU RAM.
        Note: Chunks NEVER leave the system, only move between storage layers.
        
        Args:
            decision: Output from decide_swap_action()
        """
        action = decision.get('action', 'none')
        chunk_ids = decision.get('chunk_ids', [])
        
        if action == 'load':
            # Load chunks from CPU archive to GPU
            swap_start = time.time()
            for chunk_id in chunk_ids:
                if chunk_id in self.cpu_chunks and chunk_id not in self.gpu_chunks:
                    chunk_data = {
                        'embedding': self.cpu_chunks[chunk_id],
                        **self.chunk_metadata.get(chunk_id, {})
                    }
                    self.move_to_gpu(chunk_id, chunk_data)
            swap_time = time.time() - swap_start
            if swap_time > 0:
                self.t_swap_history.append(swap_time)
            
        elif action == 'offload':
            # Unload chunks from GPU back to CPU archive
            swap_start = time.time()
            for chunk_id in chunk_ids:
                self.move_to_ram(chunk_id)
            swap_time = time.time() - swap_start
            if swap_time > 0:
                self.t_swap_history.append(swap_time)
            
            logger.info(f"Offloaded {len(chunk_ids)} chunks to RAM archive (all data preserved)")
    
    def execute_swap_decision_with_timeout(
        self, 
        decision: Dict[str, Any], 
        timeout_sec: int = 5
    ) -> bool:
        """Execute swap with timeout protection"""
        try:
            action = decision.get('action', 'none')
            chunk_ids = decision.get('chunk_ids', [])
        
            if action == 'load':
                swap_start = time.time()
            
                def load_chunks():
                    for chunk_id in chunk_ids:
                        if chunk_id in self.cpu_chunks and chunk_id not in self.gpu_chunks:
                            chunk_data = {
                                'embedding': self.cpu_chunks[chunk_id],
                                **self.chunk_metadata.get(chunk_id, {})
                            }
                            self.move_to_gpu(chunk_id, chunk_data)
            
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(load_chunks)
                    future.result(timeout=timeout_sec)
            
                swap_time = time.time() - swap_start
            
                if swap_time > 0:
                    self.t_swap_history.append(swap_time)
            
                return True
            
            elif action == 'offload':
                swap_start = time.time()
            
                def offload_chunks():
                    for chunk_id in chunk_ids:
                        self.move_to_ram(chunk_id)
            
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(offload_chunks)
                    future.result(timeout=timeout_sec)

                swap_time = time.time() - swap_start
            
                if swap_time > 0:
                    self.t_swap_history.append(swap_time)
            
                return True
        
            return True
        
        except FuturesTimeoutError:
            logger.error(f"❌ Swap operation timeout after {timeout_sec}s")
            torch.cuda.empty_cache()  # Forcefully clear on timeout
            return False
        
        except Exception as e:
            logger.error(f"❌ Swap execution error: {e}")
            return False

    def record_compute_time(self, compute_time_ms: float) -> None:
        """Record inference time for swap decision optimization."""
        self.t_comp_history.append(compute_time_ms / 1000.0)
    
    # =====================================================
    # UTILITIES
    # =====================================================
    
    def _check_gpu_memory(self, bytes_needed: int) -> bool:
        """Check if sufficient GPU memory available with safety margin."""
        if "cuda" not in str(self.device):
            return True
        
        free_memory = torch.cuda.mem_get_info(self.device)[0]
        return free_memory > bytes_needed * 1.5  # 1.5x safety margin
    
    def _get_free_gpu_memory_mb(self) -> float:
        """Get free GPU memory in megabytes."""
        if "cuda" not in str(self.device):
            return 1000000.0  # Unlimited on CPU
        
        free_bytes = torch.cuda.mem_get_info(self.device)[0]
        return free_bytes / (1024 * 1024)
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive swap manager statistics.
        Shows both GPU working set and CPU archive status.
        """
        gpu_stats = {}
        if "cuda" in str(self.device):
            gpu_reserved_mb = torch.cuda.memory_reserved(self.device) / (1024 * 1024)
            gpu_allocated_mb = torch.cuda.memory_allocated(self.device) / (1024 * 1024)
            gpu_stats = {
                'gpu_reserved_mb': round(gpu_reserved_mb, 2),
                'gpu_allocated_mb': round(gpu_allocated_mb, 2),
                'free_gpu_mb': round(self._get_free_gpu_memory_mb(), 2),
            }
        
        # Calculate CPU archive size
        cpu_allocated_mb = sum(
            t.element_size() * t.numel() / (1024 * 1024)
            for t in self.cpu_chunks.values()
        )
        
        total_hits = self.prefetch_hits + self.prefetch_misses
        hit_rate = (
            round(self.prefetch_hits / total_hits, 3)
            if total_hits > 0 else 0
        )
        
        avg_compute_ms = (
            round(sum(self.t_comp_history) / len(self.t_comp_history) * 1000, 2)
            if self.t_comp_history else 0
        )
        
        avg_swap_ms = (
            round(sum(self.t_swap_history) / len(self.t_swap_history) * 1000, 4)
            if self.t_swap_history else 0
        )
        
        return {
            **gpu_stats,
            'cpu_archive_mb': round(cpu_allocated_mb, 2),
            'chunks_in_archive': len(self.cpu_chunks),
            'chunks_in_gpu': len(self.gpu_chunks),
            'chunks_in_ram': len(self.cpu_chunks) - len(self.gpu_chunks),
            'swap_to_ram_count': self.swap_to_ram_count,
            'swap_to_gpu_count': self.swap_to_gpu_count,
            'total_swap_operations': self.total_swaps,
            'prefetch_hits': self.prefetch_hits,
            'prefetch_hit_rate': hit_rate,
            'avg_compute_time_ms': avg_compute_ms,
            'avg_swap_time_ms': avg_swap_ms,
            'last_action': self.last_action,
            'swap_triggered': self.total_swaps > 0,
        }
    
    def cleanup(self) -> None:
        """Clean up GPU resources before shutdown."""
        if "cuda" in str(self.device):
            self.gpu_chunks.clear()
            torch.cuda.empty_cache()
        # CPU archive remains (data persistence)
        logger.info("SwapManager cleanup completed (CPU archive preserved)")
