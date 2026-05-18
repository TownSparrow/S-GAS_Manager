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

    @staticmethod
    def _as_percent(value: float) -> float:
        return round(float(value or 0) * 100, 2)
    
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
            'status': turn_data.get('status', 'ok'),
            'error': turn_data.get('error', ''),
            'query': turn_data.get('query', ''),
            'response': turn_data.get('response', ''),
            'latency_ms': round(turn_data.get('latency_ms', 0), 2),

            # Latency breakdown
            'latency_search_ms': round(turn_data.get('latency_search_ms', 0), 2),
            'latency_rerank_ms': round(turn_data.get('latency_rerank_ms', 0), 2),
            'latency_swap_ms': round(turn_data.get('latency_swap_ms', 0), 2),
            'latency_inference_ms': round(turn_data.get('latency_inference_ms', 0), 2),
            'latency_observability_ms': round(turn_data.get('latency_observability_ms', 0), 2),

            # VRAM metrics
            'vram_allocated_gb': round(turn_data.get('vram_allocated_gb', 0), 4),
            'vram_reserved_gb': round(turn_data.get('vram_reserved_gb', 0), 4),
            'vram_peak_gb': round(turn_data.get('vram_peak_gb', 0), 4),

            # Chunks
            'active_chunks': turn_data.get('active_chunks', 0),
            'total_chunks': turn_data.get('total_chunks', 0),
            'coverage_ratio': round(turn_data.get('coverage_ratio', 0), 3),
            'retrieval_recall_at_5': round(turn_data.get('retrieval_recall_at_5', 0), 4),
            'retrieval_precision_at_5': round(turn_data.get('retrieval_precision_at_5', 0), 4),
            'retrieval_f1_at_5': round(turn_data.get('retrieval_f1_at_5', 0), 4),
            'retrieval_hit_at_5': round(turn_data.get('retrieval_hit_at_5', 0), 4),
            'retrieval_mrr': round(turn_data.get('retrieval_mrr', 0), 4),
            'retrieval_ndcg_at_5': round(turn_data.get('retrieval_ndcg_at_5', 0), 4),
            'retrieval_map_at_5': round(turn_data.get('retrieval_map_at_5', 0), 4),
            'retrieval_relevant_count': turn_data.get('retrieval_relevant_count', 0),
            'retrieval_retrieved_count': turn_data.get('retrieval_retrieved_count', 0),
            'evidence_recall_at_5': round(turn_data.get('evidence_recall_at_5', 0), 4),
            'evidence_hit_at_5': round(turn_data.get('evidence_hit_at_5', 0), 4),
            'evidence_mrr': round(turn_data.get('evidence_mrr', 0), 4),
            'evidence_ndcg_at_5': round(turn_data.get('evidence_ndcg_at_5', 0), 4),
            'evidence_map_at_5': round(turn_data.get('evidence_map_at_5', 0), 4),
            'evidence_token_f1_at_5': round(turn_data.get('evidence_token_f1_at_5', 0), 4),
            'evidence_count': turn_data.get('evidence_count', 0),

            # KV-cache / swap
            'cache_hit_rate': round(turn_data.get('cache_hit_rate', 0), 3),
            'swap_operations': turn_data.get('swap_operations', 0),

            # Graph
            'graph_nodes': turn_data.get('graph_nodes', 0),
            'graph_edges': turn_data.get('graph_edges', 0),

            # System resources
            'ram_used_gb': round(turn_data.get('ram_used_gb', 0), 2),
            'ram_percent': round(turn_data.get('ram_percent', 0), 1),
            'disk_used_gb': round(turn_data.get('disk_used_gb', 0), 1),
            'disk_percent': round(turn_data.get('disk_percent', 0), 1),
            'process_rss_mb': round(turn_data.get('process_rss_mb', 0), 1),

            # vLLM /metrics and physical GPU inference-window metrics
            'vllm_metrics_available': turn_data.get('vllm_metrics_available', False),
            'vllm_kv_cache_usage_before': round(turn_data.get('vllm_kv_cache_usage_before', 0), 4),
            'vllm_kv_cache_usage_after': round(turn_data.get('vllm_kv_cache_usage_after', 0), 4),
            'vllm_kv_cache_usage_delta': round(turn_data.get('vllm_kv_cache_usage_delta', 0), 4),
            'vllm_prefix_cache_hit_rate_delta': round(turn_data.get('vllm_prefix_cache_hit_rate_delta', 0), 4),
            'vllm_prefix_cache_queries_delta': round(turn_data.get('vllm_prefix_cache_queries_delta', 0), 4),
            'vllm_prefix_cache_hits_delta': round(turn_data.get('vllm_prefix_cache_hits_delta', 0), 4),
            'vllm_preemptions_delta': round(turn_data.get('vllm_preemptions_delta', 0), 4),
            'vllm_request_success_delta': round(turn_data.get('vllm_request_success_delta', 0), 4),
            'vllm_generation_tokens_delta': round(turn_data.get('vllm_generation_tokens_delta', 0), 4),
            'vllm_prompt_tokens_delta': round(turn_data.get('vllm_prompt_tokens_delta', 0), 4),
            'vllm_tokens_per_second': round(turn_data.get('vllm_tokens_per_second', 0), 4),
            'vllm_total_tokens_per_second': round(turn_data.get('vllm_total_tokens_per_second', 0), 4),
            'vllm_ttft_avg_s': round(turn_data.get('vllm_ttft_avg_s', 0), 4),
            'vllm_inter_token_latency_avg_s': round(turn_data.get('vllm_inter_token_latency_avg_s', 0), 4),
            'vllm_prefill_time_avg_s': round(turn_data.get('vllm_prefill_time_avg_s', 0), 4),
            'vllm_decode_time_avg_s': round(turn_data.get('vllm_decode_time_avg_s', 0), 4),
            'vllm_queue_time_avg_s': round(turn_data.get('vllm_queue_time_avg_s', 0), 4),
            'gpu_sample_count': turn_data.get('gpu_sample_count', 0),
            'gpu_utilization_avg_pct': round(turn_data.get('gpu_utilization_avg_pct', 0), 2),
            'gpu_utilization_peak_pct': round(turn_data.get('gpu_utilization_peak_pct', 0), 2),
            'gpu_memory_used_peak_mb': round(turn_data.get('gpu_memory_used_peak_mb', 0), 2),
            'gpu_memory_used_peak_pct': round(turn_data.get('gpu_memory_used_peak_pct', 0), 2),
            'gpu_power_avg_w': round(turn_data.get('gpu_power_avg_w', 0), 2),
            'gpu_power_peak_w': round(turn_data.get('gpu_power_peak_w', 0), 2),

            # Semantic metrics
            'recall_at_k': round(turn_data.get('recall_at_k', 0), 3),
            'precision': round(turn_data.get('precision', 0), 3),
            'answer_semantic_similarity': round(turn_data.get('answer_semantic_similarity', 0), 4),
            'answer_token_f1': round(turn_data.get('answer_token_f1', 0), 4),
            'answer_exact_match': round(turn_data.get('answer_exact_match', 0), 4),
            'answer_rouge1': round(turn_data.get('answer_rouge1', 0), 4),
            'answer_rouge2': round(turn_data.get('answer_rouge2', 0), 4),
            'answer_rougeL': round(turn_data.get('answer_rougeL', 0), 4),

            'timestamp': time.time(),
        }
        percent_fields = [
            'coverage_ratio',
            'retrieval_recall_at_5',
            'retrieval_precision_at_5',
            'retrieval_f1_at_5',
            'retrieval_hit_at_5',
            'retrieval_mrr',
            'retrieval_ndcg_at_5',
            'retrieval_map_at_5',
            'evidence_recall_at_5',
            'evidence_hit_at_5',
            'evidence_mrr',
            'evidence_ndcg_at_5',
            'evidence_map_at_5',
            'evidence_token_f1_at_5',
            'cache_hit_rate',
            'vllm_kv_cache_usage_before',
            'vllm_kv_cache_usage_after',
            'vllm_kv_cache_usage_delta',
            'vllm_prefix_cache_hit_rate_delta',
            'recall_at_k',
            'precision',
            'answer_semantic_similarity',
            'answer_token_f1',
            'answer_exact_match',
            'answer_rouge1',
            'answer_rouge2',
            'answer_rougeL',
        ]
        for field in percent_fields:
            metric_entry[f'{field}_pct'] = self._as_percent(metric_entry.get(field, 0))

        self.metrics_history.append(metric_entry)
        self.coverage_history.append(metric_entry['coverage_ratio'])
    
    def calculate_summary(self) -> Dict:
        """ Calculating summary metrics """
        if not self.metrics_history:
            return {}

        all_metrics = self.metrics_history
        metrics = [m for m in all_metrics if m.get('status', 'ok') == 'ok']
        failed_metrics = [m for m in all_metrics if m.get('status') == 'failed']

        if not metrics:
            first_error = failed_metrics[0].get('error', '') if failed_metrics else ''
            return {
                'status': 'failed',
                'attempted_turns': len(all_metrics),
                'total_turns': 0,
                'failed_turns': len(failed_metrics),
                'first_error': first_error,
                'session_duration_s': round(time.time() - self.start_time, 2) if self.start_time else 0,
                'avg_latency_ms': 0,
                'avg_latency_excl_first_ms': 0,
                'avg_latency_inference_ms': 0,
                'avg_latency_observability_ms': 0,
                'max_latency_ms': 0,
                'min_latency_ms': 0,
                'avg_vram_gb': 0,
                'peak_vram_gb': 0,
                'min_vram_gb': 0,
                'avg_active_chunks': 0,
                'final_coverage': 0,
                'coverage_progression': [],
                'avg_cache_hit_rate': 0,
                'final_cache_hit_rate': 0,
                'total_swap_operations': 0,
            }
        
        # VRAM
        vrms_allocated = [m['vram_allocated_gb'] for m in metrics]
        vrms_peak = [m['vram_peak_gb'] for m in metrics]
        
        # Latency
        latencies = [m['latency_ms'] for m in metrics]
        inference_latencies = [m.get('latency_inference_ms', 0) for m in metrics]
        observability_latencies = [m.get('latency_observability_ms', 0) for m in metrics]
        
        # Chunks
        active_chunks = [m['active_chunks'] for m in metrics]
        
        # KV-cache
        cache_hits = [m['cache_hit_rate'] for m in metrics]
        
        # Coverage
        coverages = [m['coverage_ratio'] for m in metrics]

        # Retrieval quality
        retrieval_recall_at_5 = [m.get('retrieval_recall_at_5', 0) for m in metrics]
        retrieval_precision_at_5 = [m.get('retrieval_precision_at_5', 0) for m in metrics]
        retrieval_f1_at_5 = [m.get('retrieval_f1_at_5', 0) for m in metrics]
        retrieval_hit_at_5 = [m.get('retrieval_hit_at_5', 0) for m in metrics]
        retrieval_mrr = [m.get('retrieval_mrr', 0) for m in metrics]
        retrieval_ndcg_at_5 = [m.get('retrieval_ndcg_at_5', 0) for m in metrics]
        retrieval_map_at_5 = [m.get('retrieval_map_at_5', 0) for m in metrics]
        evidence_recall_at_5 = [m.get('evidence_recall_at_5', 0) for m in metrics if m.get('evidence_count', 0) > 0]
        evidence_hit_at_5 = [m.get('evidence_hit_at_5', 0) for m in metrics if m.get('evidence_count', 0) > 0]
        evidence_mrr = [m.get('evidence_mrr', 0) for m in metrics if m.get('evidence_count', 0) > 0]
        evidence_ndcg_at_5 = [m.get('evidence_ndcg_at_5', 0) for m in metrics if m.get('evidence_count', 0) > 0]
        evidence_map_at_5 = [m.get('evidence_map_at_5', 0) for m in metrics if m.get('evidence_count', 0) > 0]
        evidence_token_f1_at_5 = [m.get('evidence_token_f1_at_5', 0) for m in metrics if m.get('evidence_count', 0) > 0]

        # Generation quality
        answer_semantic_similarity = [m.get('answer_semantic_similarity', 0) for m in metrics]
        answer_token_f1 = [m.get('answer_token_f1', 0) for m in metrics]
        answer_exact_match = [m.get('answer_exact_match', 0) for m in metrics]
        answer_rougeL = [m.get('answer_rougeL', 0) for m in metrics]
        
        avg_vram = sum(vrms_allocated) / len(vrms_allocated) if vrms_allocated else 0
        peak_vram = max(vrms_peak) if vrms_peak else 0
        min_vram = min(vrms_allocated) if vrms_allocated else 0
        
        avg_latency = sum(latencies) / len(latencies) if latencies else 0
        max_latency = max(latencies) if latencies else 0
        min_latency = min(latencies) if latencies else 0
        avg_inference_latency = (
            sum(inference_latencies) / len(inference_latencies)
            if inference_latencies else 0
        )
        avg_observability_latency = (
            sum(observability_latencies) / len(observability_latencies)
            if observability_latencies else 0
        )
        # Exclude the first turn (model warmup + graph init inflate it significantly)
        latencies_excl_first = latencies[1:] if len(latencies) > 1 else latencies
        avg_latency_excl_first = sum(latencies_excl_first) / len(latencies_excl_first) if latencies_excl_first else 0
        
        avg_chunks = sum(active_chunks) / len(active_chunks) if active_chunks else 0
        avg_cache_hit = sum(cache_hits) / len(cache_hits) if cache_hits else 0

        # Graph metrics
        graph_nodes = [m.get('graph_nodes', 0) for m in metrics]
        graph_edges = [m.get('graph_edges', 0) for m in metrics]
        avg_graph_nodes = sum(graph_nodes) / len(graph_nodes) if graph_nodes else 0
        avg_graph_edges = sum(graph_edges) / len(graph_edges) if graph_edges else 0

        # Swap metrics
        swap_ops = [m.get('swap_operations', 0) for m in metrics]
        total_swap_ops = sum(swap_ops)

        # vLLM/GPU metrics
        kv_after = [m.get('vllm_kv_cache_usage_after', 0) for m in metrics]
        kv_delta = [m.get('vllm_kv_cache_usage_delta', 0) for m in metrics]
        prefix_hit_rates = [m.get('vllm_prefix_cache_hit_rate_delta', 0) for m in metrics]
        gpu_avg_util = [m.get('gpu_utilization_avg_pct', 0) for m in metrics if m.get('gpu_sample_count', 0) > 0]
        gpu_peak_util = [m.get('gpu_utilization_peak_pct', 0) for m in metrics if m.get('gpu_sample_count', 0) > 0]
        gpu_peak_memory = [m.get('gpu_memory_used_peak_mb', 0) for m in metrics if m.get('gpu_sample_count', 0) > 0]
        token_throughput = [m.get('vllm_tokens_per_second', 0) for m in metrics if m.get('vllm_tokens_per_second', 0) > 0]
        total_token_throughput = [m.get('vllm_total_tokens_per_second', 0) for m in metrics if m.get('vllm_total_tokens_per_second', 0) > 0]
        preemptions = [m.get('vllm_preemptions_delta', 0) for m in metrics]

        summary = {
            'status': 'partial_failed' if failed_metrics else 'completed',
            'attempted_turns': len(all_metrics),
            'failed_turns': len(failed_metrics),
            'first_error': failed_metrics[0].get('error', '') if failed_metrics else '',
            # Performance
            'total_turns': len(metrics),
            'session_duration_s': round(time.time() - self.start_time, 2),
            
            # VRAM
            'avg_vram_gb': round(avg_vram, 2),
            'peak_vram_gb': round(peak_vram, 2),
            'min_vram_gb': round(min_vram, 2),
            
            # Latency
            'avg_latency_ms': round(avg_latency, 2),
            'avg_latency_excl_first_ms': round(avg_latency_excl_first, 2),
            'avg_latency_inference_ms': round(avg_inference_latency, 2),
            'avg_latency_observability_ms': round(avg_observability_latency, 2),
            'max_latency_ms': round(max_latency, 2),
            'min_latency_ms': round(min_latency, 2),
            
            # Chunks
            'avg_active_chunks': round(avg_chunks, 2),
            'final_coverage': coverages[-1] if coverages else 0,
            'coverage_progression': coverages,

            # Retrieval quality
            'avg_retrieval_recall_at_5': round(sum(retrieval_recall_at_5) / len(retrieval_recall_at_5), 4) if retrieval_recall_at_5 else 0,
            'avg_retrieval_precision_at_5': round(sum(retrieval_precision_at_5) / len(retrieval_precision_at_5), 4) if retrieval_precision_at_5 else 0,
            'avg_retrieval_f1_at_5': round(sum(retrieval_f1_at_5) / len(retrieval_f1_at_5), 4) if retrieval_f1_at_5 else 0,
            'avg_retrieval_hit_at_5': round(sum(retrieval_hit_at_5) / len(retrieval_hit_at_5), 4) if retrieval_hit_at_5 else 0,
            'avg_retrieval_mrr': round(sum(retrieval_mrr) / len(retrieval_mrr), 4) if retrieval_mrr else 0,
            'avg_retrieval_ndcg_at_5': round(sum(retrieval_ndcg_at_5) / len(retrieval_ndcg_at_5), 4) if retrieval_ndcg_at_5 else 0,
            'avg_retrieval_map_at_5': round(sum(retrieval_map_at_5) / len(retrieval_map_at_5), 4) if retrieval_map_at_5 else 0,
            'avg_evidence_recall_at_5': round(sum(evidence_recall_at_5) / len(evidence_recall_at_5), 4) if evidence_recall_at_5 else 0,
            'avg_evidence_hit_at_5': round(sum(evidence_hit_at_5) / len(evidence_hit_at_5), 4) if evidence_hit_at_5 else 0,
            'avg_evidence_mrr': round(sum(evidence_mrr) / len(evidence_mrr), 4) if evidence_mrr else 0,
            'avg_evidence_ndcg_at_5': round(sum(evidence_ndcg_at_5) / len(evidence_ndcg_at_5), 4) if evidence_ndcg_at_5 else 0,
            'avg_evidence_map_at_5': round(sum(evidence_map_at_5) / len(evidence_map_at_5), 4) if evidence_map_at_5 else 0,
            'avg_evidence_token_f1_at_5': round(sum(evidence_token_f1_at_5) / len(evidence_token_f1_at_5), 4) if evidence_token_f1_at_5 else 0,

            # Generation quality
            'avg_answer_semantic_similarity': round(sum(answer_semantic_similarity) / len(answer_semantic_similarity), 4) if answer_semantic_similarity else 0,
            'avg_answer_token_f1': round(sum(answer_token_f1) / len(answer_token_f1), 4) if answer_token_f1 else 0,
            'avg_answer_exact_match': round(sum(answer_exact_match) / len(answer_exact_match), 4) if answer_exact_match else 0,
            'avg_answer_rougeL': round(sum(answer_rougeL) / len(answer_rougeL), 4) if answer_rougeL else 0,
            
            # KV-cache / swap
            'avg_cache_hit_rate': round(avg_cache_hit, 2),
            'final_cache_hit_rate': cache_hits[-1] if cache_hits else 0,
            'total_swap_operations': total_swap_ops,
            'avg_vllm_kv_cache_usage_after': round(sum(kv_after) / len(kv_after), 4) if kv_after else 0,
            'peak_vllm_kv_cache_usage_after': round(max(kv_after), 4) if kv_after else 0,
            'avg_vllm_kv_cache_usage_delta': round(sum(kv_delta) / len(kv_delta), 4) if kv_delta else 0,
            'avg_vllm_prefix_cache_hit_rate_delta': round(sum(prefix_hit_rates) / len(prefix_hit_rates), 4) if prefix_hit_rates else 0,
            'total_vllm_preemptions': round(sum(preemptions), 4),
            'avg_vllm_tokens_per_second': round(sum(token_throughput) / len(token_throughput), 4) if token_throughput else 0,
            'avg_vllm_total_tokens_per_second': round(sum(total_token_throughput) / len(total_token_throughput), 4) if total_token_throughput else 0,
            'avg_gpu_utilization_pct': round(sum(gpu_avg_util) / len(gpu_avg_util), 2) if gpu_avg_util else 0,
            'peak_gpu_utilization_pct': round(max(gpu_peak_util), 2) if gpu_peak_util else 0,
            'peak_gpu_memory_used_mb': round(max(gpu_peak_memory), 2) if gpu_peak_memory else 0,

            # Graph
            'avg_graph_nodes': round(avg_graph_nodes, 1),
            'avg_graph_edges': round(avg_graph_edges, 1),

            # System resources
            'peak_ram_used_gb': round(max((m.get('ram_used_gb', 0) for m in metrics), default=0), 2),
            'avg_ram_percent': round(sum(m.get('ram_percent', 0) for m in metrics) / len(metrics), 1),
            'peak_process_rss_mb': round(max((m.get('process_rss_mb', 0) for m in metrics), default=0), 1),
            'avg_disk_percent': round(sum(m.get('disk_percent', 0) for m in metrics) / len(metrics), 1),
            
            # Comparison with targets (защита от деления на ноль)
            'target_comparison': {
                'vram_target': 7.5,
                'vram_achieved': round(peak_vram, 2),
                'vram_efficiency': round((7.5 / peak_vram) * 100, 1) if peak_vram > 0 else 0,
                
                'latency_target': 200,
                'latency_achieved': round(max_latency, 2),
                'latency_efficiency': round((200 / max_latency) * 100, 1) if max_latency > 0 else 0,
                
                'coverage_target': 0.8,
                'coverage_achieved': coverages[-1] if coverages else 0,
                'coverage_efficiency': round((coverages[-1] / 0.8) * 100, 1) if coverages and coverages[-1] > 0 else 0
            }
        }
        summary.update({
            'final_coverage_pct': self._as_percent(summary.get('final_coverage', 0)),
            'avg_retrieval_recall_at_5_pct': self._as_percent(summary.get('avg_retrieval_recall_at_5', 0)),
            'avg_retrieval_precision_at_5_pct': self._as_percent(summary.get('avg_retrieval_precision_at_5', 0)),
            'avg_retrieval_f1_at_5_pct': self._as_percent(summary.get('avg_retrieval_f1_at_5', 0)),
            'avg_retrieval_hit_at_5_pct': self._as_percent(summary.get('avg_retrieval_hit_at_5', 0)),
            'avg_retrieval_mrr_pct': self._as_percent(summary.get('avg_retrieval_mrr', 0)),
            'avg_retrieval_ndcg_at_5_pct': self._as_percent(summary.get('avg_retrieval_ndcg_at_5', 0)),
            'avg_retrieval_map_at_5_pct': self._as_percent(summary.get('avg_retrieval_map_at_5', 0)),
            'avg_evidence_recall_at_5_pct': self._as_percent(summary.get('avg_evidence_recall_at_5', 0)),
            'avg_evidence_hit_at_5_pct': self._as_percent(summary.get('avg_evidence_hit_at_5', 0)),
            'avg_evidence_mrr_pct': self._as_percent(summary.get('avg_evidence_mrr', 0)),
            'avg_evidence_ndcg_at_5_pct': self._as_percent(summary.get('avg_evidence_ndcg_at_5', 0)),
            'avg_evidence_map_at_5_pct': self._as_percent(summary.get('avg_evidence_map_at_5', 0)),
            'avg_evidence_token_f1_at_5_pct': self._as_percent(summary.get('avg_evidence_token_f1_at_5', 0)),
            'avg_answer_semantic_similarity_pct': self._as_percent(summary.get('avg_answer_semantic_similarity', 0)),
            'avg_answer_token_f1_pct': self._as_percent(summary.get('avg_answer_token_f1', 0)),
            'avg_answer_exact_match_pct': self._as_percent(summary.get('avg_answer_exact_match', 0)),
            'avg_answer_rougeL_pct': self._as_percent(summary.get('avg_answer_rougeL', 0)),
            'avg_cache_hit_rate_pct': self._as_percent(summary.get('avg_cache_hit_rate', 0)),
            'final_cache_hit_rate_pct': self._as_percent(summary.get('final_cache_hit_rate', 0)),
            'avg_vllm_kv_cache_usage_after_pct': self._as_percent(summary.get('avg_vllm_kv_cache_usage_after', 0)),
            'peak_vllm_kv_cache_usage_after_pct': self._as_percent(summary.get('peak_vllm_kv_cache_usage_after', 0)),
            'avg_vllm_prefix_cache_hit_rate_delta_pct': self._as_percent(summary.get('avg_vllm_prefix_cache_hit_rate_delta', 0)),
        })
        return summary
    
    def export_to_csv(self, filepath: str):
        """ Exporting as CSV """
        import csv
        
        if not self.metrics_history:
            return
        
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            fieldnames = []
            for entry in self.metrics_history:
                for key in entry.keys():
                    if key not in fieldnames:
                        fieldnames.append(key)
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
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
