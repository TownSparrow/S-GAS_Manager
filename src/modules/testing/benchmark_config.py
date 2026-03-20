from dataclasses import dataclass
from typing import List, Dict


@dataclass
class BenchmarkConfig:
    """ Configuration for benchmark runner """
    
    # API settings
    api_base: str = "http://localhost:8080"
    
    # Directories
    scenarios_dir: str = "data/scenarios"
    results_dir: str = "logs/benchmarks"
    documents_dir: str = "data/documents"
    
    # System settings
    default_n_chunks: int = 5
    default_temperature: float = 0.1
    max_iterations: int = 10
    
    # Target metrics (from NIRS/PTP)
    target_vram_gb: float = 7.5
    target_latency_ms: float = 200.0
    target_coverage: float = 0.8
    target_cache_hit_rate: float = 0.75
    target_recall_at_k: float = 0.70
    
    # Retrieval evaluation
    top_k_values: List[int] = None
    use_ground_truth: bool = True
    
    # Generation evaluation
    use_bertscore: bool = True
    use_rouge: bool = True
    bertscore_threshold: float = 0.85
    
    # Performance monitoring
    track_kv_cache: bool = True
    track_vram_usage: bool = True
    track_latency: bool = True
    
    def __post_init__(self):
        """ Initializing default values """
        if self.top_k_values is None:
            self.top_k_values = [5, 10, 20]