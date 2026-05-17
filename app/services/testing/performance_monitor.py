import time
from typing import Dict, List
import logging


logger = logging.getLogger(__name__)


class PerformanceMonitor:
    """Monitor performance metrics during benchmark. Tracks latency, VRAM usage, and KV-cache statistics."""

    def __init__(self):
        self.latency_history: List[float] = []
        self.vram_history: List[Dict[str, float]] = []
        self.cache_hit_history: List[float] = []
        self.swap_operations_count: int = 0

        self.start_times: Dict[str, float] = {}
        self.current_iteration: int = 0

    def start_timing(self, operation: str = "inference"):
        self.start_times[operation] = time.perf_counter()

    def end_timing(self, operation: str = "inference") -> float:
        if operation not in self.start_times:
            logger.warning(f"Operation {operation} not started")
            return 0.0
        elapsed = time.perf_counter() - self.start_times.pop(operation)
        return elapsed

    def record_metrics(
        self,
        latency_ms: float,
        vram_allocated_gb: float,
        vram_reserved_gb: float,
        cache_hit_rate: float,
        swap_operations: int,
    ):
        self.latency_history.append(latency_ms)
        self.vram_history.append({'allocated': vram_allocated_gb, 'reserved': vram_reserved_gb})
        self.cache_hit_history.append(cache_hit_rate)
        self.swap_operations_count += swap_operations
        self.current_iteration += 1

    def record_vram(self, allocated_gb: float, reserved_gb: float = 0.0):
        """Record VRAM usage for current iteration."""
        self.vram_history.append({'allocated': allocated_gb, 'reserved': reserved_gb})

    def record_swap_operation(self, chunks_loaded: int, chunks_unloaded: int):
        self.swap_operations_count += chunks_loaded + chunks_unloaded

    def calculate_summary(self) -> Dict:
        if not self.latency_history and not self.vram_history:
            return {}

        result = {}

        # Latency statistics
        if self.latency_history:
            latencies = self.latency_history
            sorted_lat = sorted(latencies)
            result['latency'] = {
                'avg_ms': round(sum(latencies) / len(latencies), 2),
                'max_ms': round(max(latencies), 2),
                'min_ms': round(min(latencies), 2),
                'p95_ms': round(sorted_lat[int(len(sorted_lat) * 0.95)], 2),
                'total_iterations': len(latencies),
            }

        # VRAM statistics
        if self.vram_history:
            allocated = [v['allocated'] for v in self.vram_history]
            reserved = [v['reserved'] for v in self.vram_history]
            result['vram'] = {
                'avg_allocated_gb': round(sum(allocated) / len(allocated), 2),
                'peak_allocated_gb': round(max(allocated), 2),
                'avg_reserved_gb': round(sum(reserved) / len(reserved), 2),
                'peak_reserved_gb': round(max(reserved), 2),
            }

        # Cache statistics
        if self.cache_hit_history:
            result['cache'] = {
                'avg_hit_rate': round(sum(self.cache_hit_history) / len(self.cache_hit_history), 3),
                'final_hit_rate': round(self.cache_hit_history[-1], 3),
                'total_iterations': len(self.cache_hit_history),
            }

        # Swap operations
        iterations = max(self.current_iteration, len(self.latency_history), 1)
        result['swap_operations'] = {
            'total': self.swap_operations_count,
            'avg_per_iteration': round(self.swap_operations_count / iterations, 2),
        }

        return result

    def reset(self):
        self.latency_history = []
        self.vram_history = []
        self.cache_hit_history = []
        self.swap_operations_count = 0
        self.start_times = {}
        self.current_iteration = 0
