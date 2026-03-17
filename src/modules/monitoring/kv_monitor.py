import torch
import time
from typing import Dict, List, Optional
import logging


logger = logging.getLogger(__name__)


class KVCacheMonitor:
    """
    A monitoring module for tracking VRAM usage and indirectyl assessing the impact
    of the S-GAS Algorithm on context efficiency.
    """

    def __init__(self):
        self.metrics_history: List[Dict] = []

    def get_vram_usage(self) -> Dict[str, float]:
        """
        Recieving VRAM usage via PyTorch.
        """
        if torch.cuda.is_available():
            return {
                'allocated_gb': torch.cuda.memory_allocated() / (1024**3),
                'reserved_gb': torch.cuda.memory_reserved() / (1024**3),
                'max_allocated_gb': torch.cuda.max_memory_allocated() / (1024**3),
                'timestamp': time.time()
            }
        logging.warning("⚠️ [KV-Monitor] CUDA is unavailable. VRAM monitoring is disabled.")
        return {'allocated_gb': 0.0, 'reserved_gb': 0.0, 'max_allocated_gb': 0.0, 'timestamp': time.time()}

    def log_iteration_start(self, iteration: int, prompt_length_tokens: int, n_selected_chunks: int, n_archived_chunks_this_iter: int) -> Dict:
        """
        Logs the system state at the beginning of the iteration (before generation).
        """
        vram_before = self.get_vram_usage()

        metrics_entry = {
            'iteration': iteration,
            'event': 'start',
            'prompt_length_tokens': prompt_length_tokens,
            'n_selected_chunks': n_selected_chunks,
            'n_archived_chunks_this_iter': n_archived_chunks_this_iter,
            'vram_before_gen': vram_before,
        }

        self.metrics_history.append(metrics_entry)
        logging.info(f"[KV Monitor] Iteration {iteration} Start - VRAM: {vram_before['allocated_gb']:.2f}GB, Prompt Tokens: {prompt_length_tokens}")
        return metrics_entry

    def log_iteration_end(self, iteration: int, generation_time_ms: float, prompt_length_tokens: int, n_selected_chunks: int, n_archived_chunks_this_iter: int) -> Dict:
        """
        Logs the system state at the end of an iteration (after generation).
        Now accepts external parameters for more precise binding.
        """
        vram_after = self.get_vram_usage()

        # Finding the start entry for deltas calculation
        start_entry = next((m for m in self.metrics_history if m.get('iteration') == iteration and m.get('event') == 'start'), None)
        if start_entry:
            # VRAM delta calculation
            delta_allocated = vram_after['allocated_gb'] - start_entry['vram_before_gen']['allocated_gb']
            # Peak delta calculation
            peak_delta = vram_after['max_allocated_gb'] - start_entry['vram_before_gen']['max_allocated_gb']

            end_metrics_entry = {
                'iteration': iteration,
                'event': 'end',
                'generation_time_ms': generation_time_ms,
                'prompt_length_tokens': prompt_length_tokens,
                'n_selected_chunks': n_selected_chunks,
                'n_archived_chunks_this_iter': n_archived_chunks_this_iter,
                'vram_after_gen': vram_after,
                'delta_vram_allocated_gb': delta_allocated,
                'delta_peak_vram_gb': peak_delta
            }
            self.metrics_history.append(end_metrics_entry)
            logging.info(f"[KV Monitor] Iteration {iteration} End - VRAM: {vram_after['allocated_gb']:.2f}GB, Delta: {delta_allocated:+.2f}GB, Gen Time: {generation_time_ms}ms")
            return end_metrics_entry
        else:
            logging.warning(f"⚠️ [KV-Monitor] Start entry for iteration {iteration} not found for end logging.")
            end_metrics_entry = {
                'iteration': iteration,
                'event': 'end_no_start',
                'generation_time_ms': generation_time_ms,
                'prompt_length_tokens': prompt_length_tokens,
                'n_selected_chunks': n_selected_chunks,
                'n_archived_chunks_this_iter': n_archived_chunks_this_iter,
                'vram_after_gen': vram_after,
                'delta_vram_allocated_gb': None,
                'delta_peak_vram_gb': None
            }
            self.metrics_history.append(end_metrics_entry)            
            return end_metrics_entry


    def get_latest_metrics(self) -> Optional[Dict]:
        """
        Recieving latest metrics.
        """
        if self.metrics_history:
            return self.metrics_history[-1]
        return None

    def get_summary_stats(self) -> Dict[str, float]:
        """
        Recieving summary statistics for the all process time.
        """
        if not self.metrics_history:
            return {}

        # Filtering only 'end' entries for calculating VRAM after generation
        allocated_values = [m['vram_after_gen']['allocated_gb'] for m in self.metrics_history if m.get('event') == 'end' and 'vram_after_gen' in m]
        
        if allocated_values:
            avg_vram = sum(allocated_values) / len(allocated_values)
            max_vram = max(allocated_values)
            min_vram = min(allocated_values)
        else:
            avg_vram = max_vram = min_vram = 0.0

        # Calculating the count of all iterations
        iterations_count = len([m for m in self.metrics_history if m.get('event') == 'start'])

        return {
            'avg_vram_gb': avg_vram,
            'max_vram_gb': max_vram,
            'min_vram_gb': min_vram,
            'total_iterations_logged': iterations_count
        }
