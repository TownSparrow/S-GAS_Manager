from typing import Dict, Optional
import logging


logger = logging.getLogger(__name__)


class CacheAnalyzer:
    """ Cache status analyzer for benchmark. Based on KVCacheMonitor but adapted for metrics in testing module """
    
    def __init__(self):
        self.total_cache_hits = 0
        self.total_cache_misses = 0
        self.total_swap_operations = 0
        self.iteration_stats = []
    
    def record_cache_access(self, cache_hit: bool):
        """ Record of cahce hit per system processing """
        if cache_hit:
            self.total_cache_hits += 1
        else:
            self.total_cache_misses += 1
    
    def record_swap_operation(self, chunks_loaded: int, chunks_unloaded: int):
        """ Record of swap operation """
        self.total_swap_operations += 1
        self.iteration_stats.append({
            'swap_ops': self.total_swap_operations,
            'chunks_loaded': chunks_loaded,
            'chunks_unloaded': chunks_unloaded
        })
    
    def calculate_cache_hit_rate(self) -> float:
        """ Calculation of the cache hit rate """
        total = self.total_cache_hits + self.total_cache_misses
        if total == 0:
            return 0.0
        return round(self.total_cache_hits / total * 100, 2)
    
    def get_swap_statistics(self) -> Dict:
        """ Receiveing the swapping statistics """
        if not self.iteration_stats:
            return {
                'total_swaps': 0,
                'avg_chunks_loaded': 0,
                'avg_chunks_unloaded': 0
            }
        
        total_loaded = sum(s['chunks_loaded'] for s in self.iteration_stats)
        total_unloaded = sum(s['chunks_unloaded'] for s in self.iteration_stats)
        count = len(self.iteration_stats)
        
        return {
            'total_swaps': self.total_swap_operations,
            'avg_chunks_loaded': round(total_loaded / count, 2),
            'avg_chunks_unloaded': round(total_unloaded / count, 2)
        }
    
    def reset(self):
        """ Resetting the statistics """
        self.total_cache_hits = 0
        self.total_cache_misses = 0
        self.total_swap_operations = 0
        self.iteration_stats = []