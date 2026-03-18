import time
import json
from typing import Dict, List, Optional
from pathlib import Path


class MetricsCollector:
    """ Metrics collector for benchmark """
    
    def __init__(self):
        self.metrics_history = []
        self.start_time = None
        self.coverage_history = []
    
    def start_session(self):
        """ Starting the testing session """
        self.start_time = time.time()
        self.metrics_history = []
        self.coverage_history = []
    
    def record_turn(self, turn_data: Dict):
        """ Recording the metrics for single turn """
        metric_entry = {
            # Base metrics
            'iteration': len(self.metrics_history) + 1,
            'query': turn_data.get('query', '')[:60],
            'response': turn_data.get('response', '')[:100],
            'latency_ms': round(turn_data.get('latency_ms', 0), 2),
            
            # VRAM metrics
            'vram_allocated_gb': round(turn_data.get('vram_allocated_gb', 0), 2),
            'vram_reserved_gb': round(turn_data.get('vram_reserved_gb', 0), 2),
            'vram_peak_gb': round(turn_data.get('vram_peak_gb', 0), 2),
            
            # Chunks
            'active_chunks': turn_data.get('active_chunks', 0),
            'total_chunks': turn_data.get('total_chunks', 0),
            'coverage_ratio': round(turn_data.get('coverage_ratio', 0), 3),
            
            # KV-cache
            'cache_hit_rate': round(turn_data.get('cache_hit_rate', 0), 3),
            'swap_operations': turn_data.get('swap_operations', 0),
            
            # Graph
            'graph_nodes': turn_data.get('graph_nodes', 0),
            'graph_edges': turn_data.get('graph_edges', 0),
            
            # Semantic metrics
            'recall_at_k': round(turn_data.get('recall_at_k', 0), 3),
            'precision': round(turn_data.get('precision', 0), 3),
            
            'timestamp': time.time()
        }
        
        self.metrics_history.append(metric_entry)
        self.coverage_history.append(metric_entry['coverage_ratio'])
    
    def calculate_summary(self) -> Dict:
        """ Calculating summary metrics """
        if not self.metrics_history:
            return {}
        
        metrics = self.metrics_history
        
        # VRAM
        vrms_allocated = [m['vram_allocated_gb'] for m in metrics]
        vrms_peak = [m['vram_peak_gb'] for m in metrics]
        
        # Latency
        latencies = [m['latency_ms'] for m in metrics]
        
        # Chunks
        active_chunks = [m['active_chunks'] for m in metrics]
        
        # KV-cache
        cache_hits = [m['cache_hit_rate'] for m in metrics]
        
        # Coverage
        coverages = [m['coverage_ratio'] for m in metrics]
        
        return {
            # Perfomance
            'total_turns': len(metrics),
            'session_duration_s': round(time.time() - self.start_time, 2),
            
            # VRAM
            'avg_vram_gb': round(sum(vrms_allocated) / len(vrms_allocated), 2),
            'peak_vram_gb': max(vrms_peak),
            'min_vram_gb': min(vrms_allocated),
            
            # Latency
            'avg_latency_ms': round(sum(latencies) / len(latencies), 2),
            'max_latency_ms': max(latencies),
            'min_latency_ms': min(latencies),
            
            # Chunks
            'avg_active_chunks': round(sum(active_chunks) / len(active_chunks), 2),
            'final_coverage': coverages[-1] if coverages else 0,
            'coverage_progression': self.coverage_history,
            
            # KV-cache
            'avg_cache_hit_rate': round(sum(cache_hits) / len(cache_hits), 2),
            'final_cache_hit_rate': cache_hits[-1] if cache_hits else 0,
            
            # Comparison
            'target_comparison': {
                'vram_target': 7.5,
                'vram_achieved': round(max(vrms_peak), 2),
                'vram_efficiency': round((7.5 / max(vrms_peak)) * 100, 1) if vrms_peak else 0,
                
                'latency_target': 200,
                'latency_achieved': round(max(latencies), 2),
                'latency_efficiency': round((200 / max(latencies)) * 100, 1) if latencies else 0,
                
                'coverage_target': 0.8,
                'coverage_achieved': coverages[-1] if coverages else 0,
                'coverage_efficiency': round((coverages[-1] / 0.8) * 100, 1) if coverages else 0
            }
        }
    
    def export_to_csv(self, filepath: str):
        """ Exporting as CSV """
        import csv
        
        if not self.metrics_history:
            return
        
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=self.metrics_history[0].keys())
            writer.writeheader()
            writer.writerows(self.metrics_history)
    
    def export_to_json(self, filepath: str):
        """ Exporting as JSON """
        data = {
            'summary': self.calculate_summary(),
            'detailed_metrics': self.metrics_history
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)