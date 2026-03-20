import time
from typing import Dict, List
import logging


logger = logging.getLogger(__name__)


class PerformanceMonitor:
    """ Monitor performance metrics during benchmark. Tracks latency, VRAM usage, and KV-cache statistics """
    
    def __init__(self):
        self.latency_history = []
        self.vram_history = []
        self.cache_hit_history = []
        self.swap_operations_history = []
        
        self.start_times = {}
        self.current_iteration = 0
    
    def start_timing(self, operation: str = "inference"):
        """ Starting timing an operation """
        self.start_times[operation] = time.perf_counter()
    
    def end_timing(self, operation: str = "inference") -> float:
        """ End timing an operation and return elapsed time """
        if operation not in self.start_times:
            logger.warning(f"Operation {operation} not started")
            return 0.0
        
        elapsed = time.perf_counter() - self.start_times[operation]
        del self.start_times[operation]
        
        return elapsed
    
    def record_metrics(
        self,
        latency_ms: float,
        vram_allocated_gb: float,
        vram_reserved_gb: float,
        cache_hit_rate: float,
        swap_operations: int
    ):
        """ Recording performance metrics for current iteration """
        self.latency_history.append(latency_ms)
        self.vram_history.append({
            'allocated': vram_allocated_gb,
            'reserved': vram_reserved_gb
        })
        self.cache_hit_history.append(cache_hit_rate)
        self.swap_operations_history.append(swap_operations)      
        self.current_iteration += 1
    
    def record_swap_operation(self, chunks_loaded: int, chunks_unloaded: int):
        """ Recording swap operations """
        self.swap_operations_history.append({
            'loaded': chunks_loaded,
            'unloaded': chunks_unloaded,
            'timestamp': time.time()
        })

    def calculate_summary(self) -> Dict:
        """ Calculating summary statistics for all perfomance metrics """
        if not self.latency_history:
            return {}
        
        # Latency statistics
        latencies = self.latency_history
        avg_latency = sum(latencies) / len(latencies)
        max_latency = max(latencies)
        min_latency = min(latencies)
        p95_latency = sorted(latencies)[int(len(latencies) * 0.95)]
        
        # VRAM statistics
        allocated_vram = [v['allocated'] for v in self.vram_history]
        reserved_vram = [v['reserved'] for v in self.vram_history]
        avg_allocated_vram = sum(allocated_vram) / len(allocated_vram)
        peak_allocated_vram = max(allocated_vram)
        avg_reserved_vram = sum(reserved_vram) / len(reserved_vram)
        peak_reserved_vram = max(reserved_vram)
        
        # Cache statistics
        cache_hits = self.cache_hit_history
        avg_cache_hit = sum(cache_hits) / len(cache_hits) if cache_hits else 0.0
        final_cache_hit = cache_hits[-1] if cache_hits else 0.0
        
        # Swap operations
        total_swaps = sum(self.swap_operations_history)
        avg_swaps_per_iter = total_swaps / len(self.swap_operations_history) if self.swap_operations_history else 0.0
        
        return {
            'latency': {
                'avg_ms': round(avg_latency, 2),
                'max_ms': round(max_latency, 2),
                'min_ms': round(min_latency, 2),
                'p95_ms': round(p95_latency, 2),
                'total_iterations': len(latencies)
            },
            'vram': {
                'avg_allocated_gb': round(avg_allocated_vram, 2),
                'peak_allocated_gb': round(peak_allocated_vram, 2),
                'avg_reserved_gb': round(avg_reserved_vram, 2),
                'peak_reserved_gb': round(peak_reserved_vram, 2)
            },
            'cache': {
                'avg_hit_rate': round(avg_cache_hit, 3),
                'final_hit_rate': round(final_cache_hit, 3),
                'total_iterations': len(cache_hits)
            },
            'swap_operations': {
                'total': total_swaps,
                'avg_per_iteration': round(avg_swaps_per_iter, 2)
            }
        }
    
    def reset(self):
        """ Resetting all statistics """
        self.latency_history = []
        self.vram_history = []
        self.cache_hit_history = []
        self.swap_operations_history = []
        self.start_times = {}
        self.current_iteration = 0