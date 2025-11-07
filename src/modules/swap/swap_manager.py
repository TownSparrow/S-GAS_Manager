import time
import logging
from typing import List, Dict, Any, Optional
from collections import deque
import numpy as np

logger = logging.getLogger(__name__)


class SwapManager:  
    def __init__(
        self,
        threshold: float = 0.3,
        prefetch_count: int = 5,
        memory_check_interval_ms: int = 50,
        max_gpu_memory_tokens: Optional[int] = None
    ):
        self.threshold = threshold # T_swap/T_comp ratio threshold for swap decisions
        self.prefetch_count = prefetch_count # Number of chunks to prefetch
        self.memory_check_interval_ms = memory_check_interval_ms # GPU Memory Check Interval (ms)
        self.max_gpu_memory_tokens = max_gpu_memory_tokens # Maximum context size in tokens
        
        # Performance metrics
        self.t_comp_history = deque(maxlen=10)
        self.t_swap_history = deque(maxlen=10)
        
        # GPU chunk cache (simplified simulation)
        self.gpu_chunks = {}  # {chunk_id: chunk_data}
        self.ram_chunks = {}  # {chunk_id: chunk_data}
        
        # Prefetch buffer (queue of chunks for prefetching)
        self.prefetch_buffer = deque(maxlen=prefetch_count)
        
        # Counters for statistics
        self.swap_to_ram_count = 0
        self.swap_to_gpu_count = 0
        self.prefetch_hits = 0
        
        logger.info(
            f"SwapManager is initialized: threshold={threshold}, "
            f"prefetch_count={prefetch_count}"
        )
    
    def record_compute_time(self, t_comp: float):
        self.t_comp_history.append(t_comp)
    
    def record_swap_time(self, t_swap: float):
        self.t_swap_history.append(t_swap)
    
    def get_avg_compute_time(self) -> float:
        if not self.t_comp_history:
            return 0.0
        return sum(self.t_comp_history) / len(self.t_comp_history)
    
    def get_avg_swap_time(self) -> float:
        if not self.t_swap_history:
            return 0.0
        return sum(self.t_swap_history) / len(self.t_swap_history)
    
    def should_swap_to_ram(self) -> bool:
        avg_t_comp = self.get_avg_compute_time()
        avg_t_swap = self.get_avg_swap_time()
        
        if avg_t_comp == 0.0:
            return False
        
        ratio = avg_t_swap / avg_t_comp
        should_swap = ratio > self.threshold
        
        if should_swap:
            logger.debug(
                f"Swap decision: T_swap/T_comp = {ratio:.3f} > {self.threshold}"
            )
        
        return should_swap
    
    def decide_swap_action(
        self,
        ranked_chunks: List[Dict[str, Any]],
        current_context_tokens: int
    ) -> Dict[str, Any]:
        decision = {
            'action': 'none',
            'chunks_to_load': [],
            'chunks_to_offload': []
        }
        
        # Checking whether chunks need to be unloaded
        if self.should_swap_to_ram():
            # Offloading the least relevant chunks from the GPU to RAM
            chunks_in_gpu = list(self.gpu_chunks.keys())
            
            if len(chunks_in_gpu) > 0:
                # Finding chunks that aren't at the top ranked_chunks
                top_chunk_ids = {
                    chunk.get('id', f"chunk_{i}") 
                    for i, chunk in enumerate(ranked_chunks[:10])
                }
                
                chunks_to_offload = [
                    cid for cid in chunks_in_gpu 
                    if cid not in top_chunk_ids
                ]
                
                if chunks_to_offload:
                    decision['action'] = 'offload'
                    decision['chunks_to_offload'] = chunks_to_offload[:5]  # Max 5
                    
                    logger.info(f"Solution: Offload {len(decision['chunks_to_offload'])} chunks to RAM")
        
        # Checking if chunks can be loaded
        elif self.max_gpu_memory_tokens is not None:
            available_tokens = self.max_gpu_memory_tokens - current_context_tokens
            
            if available_tokens > 1000:  # There is free memory
                # Loading top chunks that aren't available on the GPU
                chunks_to_load = []
                
                for chunk in ranked_chunks[:self.prefetch_count]:
                    chunk_id = chunk.get('id', '')
                    
                    if chunk_id not in self.gpu_chunks and chunk_id in self.ram_chunks:
                        chunk_size = chunk.get('metadata', {}).get('chunk_size', 500)
                        
                        if available_tokens >= chunk_size:
                            chunks_to_load.append(chunk_id)
                            available_tokens -= chunk_size
                
                if chunks_to_load:
                    decision['action'] = 'load'
                    decision['chunks_to_load'] = chunks_to_load
                    
                    logger.info(f"Solution: Load {len(chunks_to_load)} chunks into the GPU")
        
        return decision
    
    def swap_to_ram(self, chunk_ids: List[str]):
        start_time = time.time()
        
        for chunk_id in chunk_ids:
            if chunk_id in self.gpu_chunks:
                # Unloading from GPU to RAM
                self.ram_chunks[chunk_id] = self.gpu_chunks.pop(chunk_id)
                self.swap_to_ram_count += 1
        
        elapsed = time.time() - start_time
        self.record_swap_time(elapsed)
        
        logger.debug(f"Unloaded {len(chunk_ids)} chunks into RAM in {elapsed:.4f}s")
    
    def swap_to_gpu(self, chunk_ids: List[str]):
        start_time = time.time()
        
        for chunk_id in chunk_ids:
            if chunk_id in self.ram_chunks:
                # Moving from RAM to GPU
                self.gpu_chunks[chunk_id] = self.ram_chunks.pop(chunk_id)
                self.swap_to_gpu_count += 1
                
                # Checking the prefetch hit
                if chunk_id in self.prefetch_buffer:
                    self.prefetch_hits += 1
        
        elapsed = time.time() - start_time
        self.record_swap_time(elapsed)
        
        logger.debug(f"Loaded {len(chunk_ids)} chunks into GPU in {elapsed:.4f}s")
    
    def execute_swap_decision(self, decision: Dict[str, Any]):
        action = decision.get('action', 'none')
        
        if action == 'offload':
            chunks_to_offload = decision.get('chunks_to_offload', [])
            if chunks_to_offload:
                self.swap_to_ram(chunks_to_offload)
        
        elif action == 'load':
            chunks_to_load = decision.get('chunks_to_load', [])
            if chunks_to_load:
                self.swap_to_gpu(chunks_to_load)
    
    def update_prefetch_buffer(self, ranked_chunks: List[Dict[str, Any]]):
        self.prefetch_buffer.clear()
        
        for chunk in ranked_chunks[:self.prefetch_count]:
            chunk_id = chunk.get('id', '')
            if chunk_id:
                self.prefetch_buffer.append(chunk_id)
        
        logger.debug(f"Prefetch buffer updated: {len(self.prefetch_buffer)} chunks")
    
    def initialize_chunks(self, chunks: List[Dict[str, Any]]):
        # Placing all chunks in RAM by the default
        for i, chunk in enumerate(chunks):
            chunk_id = chunk.get('id', f'chunk_{i}')
            self.ram_chunks[chunk_id] = chunk
        
        logger.info(f"Initialized {len(chunks)} chunks in RAM")
    
    def get_statistics(self) -> Dict[str, Any]:
        total_swaps = self.swap_to_ram_count + self.swap_to_gpu_count
        prefetch_hit_rate = (
            self.prefetch_hits / total_swaps if total_swaps > 0 else 0.0
        )
        
        return {
            'swap_to_ram_count': self.swap_to_ram_count,
            'swap_to_gpu_count': self.swap_to_gpu_count,
            'total_swap_operations': total_swaps,
            'prefetch_hits': self.prefetch_hits,
            'prefetch_hit_rate': prefetch_hit_rate,
            'avg_compute_time_ms': self.get_avg_compute_time() * 1000,
            'avg_swap_time_ms': self.get_avg_swap_time() * 1000,
            'chunks_in_gpu': len(self.gpu_chunks),
            'chunks_in_ram': len(self.ram_chunks),
            'prefetch_buffer_size': len(self.prefetch_buffer)
        }
    
    def reset_statistics(self):
        self.swap_to_ram_count = 0
        self.swap_to_gpu_count = 0
        self.prefetch_hits = 0
        self.t_comp_history.clear()
        self.t_swap_history.clear()
        
        logger.info("SwapManager statistics reset")