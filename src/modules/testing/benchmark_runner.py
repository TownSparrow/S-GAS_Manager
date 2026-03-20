import requests
import json
import time
from pathlib import Path
from typing import Dict, Optional, List
import logging


logger = logging.getLogger(__name__)


class BenchmarkRunner:
    """ Main benchmark runner for S-GAS system """
    
    def __init__(
        self,
        api_base: str = "http://localhost:8080",
        scenarios_dir: str = "tests/scenarios",
        results_dir: str = "logs/benchmarks",
        documents_dir: str = "tests/documents"
    ):
        self.api_base = api_base
        self.scenarios_dir = Path(scenarios_dir)
        self.results_dir = Path(results_dir)
        self.documents_dir = Path(documents_dir)
        
        # Creating directories
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.documents_dir.mkdir(parents=True, exist_ok=True)
        
        # Initializing components
        from .scenario_loader import ScenarioLoader
        from .metrics_collector import MetricsCollector
        from .retrieval_evaluator import RetrievalEvaluator
        from .generation_evaluator import GenerationEvaluator
        from .performance_monitor import PerformanceMonitor
        from .cache_analyzer import CacheAnalyzer
        
        self.scenario_loader = ScenarioLoader(scenarios_dir)
        self.metrics_collector = MetricsCollector()
        self.retrieval_evaluator = RetrievalEvaluator()
        self.generation_evaluator = GenerationEvaluator()
        self.performance_monitor = PerformanceMonitor()
        self.cache_analyzer = CacheAnalyzer()
        
        logger.info("Benchmark runner initialized")
    
    def run_scenario(
        self,
        scenario_name: str,
        session_id: Optional[str] = None
    ) -> Dict:
        """ Running benchmark scenario """
        # Loading scenario
        scenario = self.scenario_loader.load_scenario(scenario_name)
        
        logger.info(f"Running scenario: {scenario.name}")
        logger.info(f"Description: {scenario.description}")
        logger.info(f"Document: {scenario.document}")
        
        # Creating session if not provided
        if session_id is None:
            session_id = self._create_session()
            logger.info(f"Created new session: {session_id}")
        
        # Uploading document
        doc_path = self.documents_dir / scenario.document
        if not doc_path.exists():
            raise FileNotFoundError(
                f"Document {doc_path} not found. Place it in {self.documents_dir}"
            )
        
        self._upload_document(session_id, str(doc_path))
        logger.info(f"Document uploaded: {scenario.document}")
        
        # Initializing metrics collection
        self.metrics_collector.start_session()
        self.retrieval_evaluator.reset()
        self.performance_monitor.reset()
        self.cache_analyzer.reset()
        
        # Running all turns
        questions = []
        answers = []
        retrieved_chunks_list = []
        ground_truth_chunks_list = []
        ground_truth_answers = []
        
        for i, turn in enumerate(scenario.turns):
            logger.info(f"\nTurn {i+1}/{len(scenario.turns)}")
            logger.info(f"Query: {turn.query[:60]}...")
            
            # Running turn
            result = self._run_turn(
                session_id=session_id,
                query=turn.query,
                ground_truth_chunks=turn.ground_truth_chunks if hasattr(turn, 'ground_truth_chunks') else [],
                ground_truth_answer=turn.ground_truth_answer if hasattr(turn, 'ground_truth_answer') else ""
            )
            
            # Recording metrics
            self.metrics_collector.record_turn(result)
            
            # Collecting data for evaluation
            questions.append(turn.query)
            answers.append(result['response'])
            retrieved_chunks_list.append(result.get('retrieved_chunks', []))
            ground_truth_chunks_list.append(turn.ground_truth_chunks if hasattr(turn, 'ground_truth_chunks') else [])
            ground_truth_answers.append(turn.ground_truth_answer if hasattr(turn, 'ground_truth_answer') else "")
            
            # Logging results
            logger.info(f"  Coverage: {result['coverage_ratio']*100:.1f}%")
            logger.info(f"  Latency: {result['latency_ms']:.1f} ms")
            logger.info(f"  Cache Hit Rate: {result['cache_hit_rate']*100:.1f}%")
        
        # Evaluating retrieval
        retrieval_results = self._evaluate_retrieval(
            retrieved_chunks_list=retrieved_chunks_list,
            ground_truth_chunks_list=ground_truth_chunks_list
        )
        
        # Evaluateing generation (if ground truth answers available)
        generation_results = {}
        if any(ground_truth_answers):
            generation_results = self._evaluate_generation(
                generated_answers=answers,
                reference_answers=ground_truth_answers
            )
        
        # Calculating performance summary
        performance_summary = self.performance_monitor.calculate_summary()
        
        # Calculating overall summary
        summary = self.metrics_collector.calculate_summary()
        summary.update({
            'retrieval_metrics': retrieval_results,
            'generation_metrics': generation_results,
            'performance_metrics': performance_summary
        })
        
        # Saving results
        timestamp = self._get_timestamp()
        csv_file = self.results_dir / f"{scenario_name}_{timestamp}.csv"
        json_file = self.results_dir / f"{scenario_name}_{timestamp}.json"
        md_file = self.results_dir / f"{scenario_name}_{timestamp}.md"
        
        self.metrics_collector.export_to_csv(str(csv_file))
        self.metrics_collector.export_to_json(str(json_file))
        self.metrics_collector.export_to_markdown(str(md_file))
        
        logger.info(f"✅ Scenario completed!")
        logger.info(f"Results saved to:")
        logger.info(f"  CSV: {csv_file}")
        logger.info(f"  JSON: {json_file}")
        logger.info(f"  Markdown: {md_file}")
        
        return {
            'scenario': scenario.name,
            'session_id': session_id,
            'summary': summary,
            'csv_file': str(csv_file),
            'json_file': str(json_file),
            'md_file': str(md_file)
        }
    
    def _run_turn(
        self,
        session_id: str,
        query: str,
        ground_truth_chunks: List[str],
        ground_truth_answer: str
    ) -> Dict:
        """ Runninh single turn """
        import time
        
        # Starting timing
        self.performance_monitor.start_timing("inference")
        
        # Sending query
        resp = requests.post(
            f"{self.api_base}/api/session/{session_id}/chat",
            json={
                'message': query,
                'use_rag': True,
                'n_chunks': 5,
                'temperature': 0.1
            }
        )
        resp.raise_for_status()
        
        # Ending timing
        latency_seconds = self.performance_monitor.end_timing("inference")
        latency_ms = latency_seconds * 1000
        
        # Getting response
        response_data = resp.json()
        
        # Getting system statistics
        stats = requests.get(f"{self.api_base}/api/sgas-statistics").json()
        
        # Building result
        result = {
            'query': query,
            'response': response_data['response'],
            'latency_ms': latency_ms,
            
            # VRAM metrics
            'vram_allocated_gb': stats.get('gpu_memory_allocated_gb', 0),
            'vram_reserved_gb': stats.get('gpu_memory_reserved_gb', 0),
            'vram_peak_gb': stats.get('gpu_memory_peak_gb', 0),
            
            # Chunks metrics
            'active_chunks': stats.get('active_chunks', 0),
            'total_chunks': stats.get('total_chunks', 0),
            'coverage_ratio': stats.get('coverage_ratio', 0),
            'retrieved_chunks': stats.get('retrieved_chunks', []),
            
            # Cache metrics
            'cache_hit_rate': stats.get('cache_hit_rate', 0),
            'swap_operations': stats.get('swap_operations', 0),
            
            # Ground truth (for evaluation)
            'ground_truth_chunks': ground_truth_chunks,
            'ground_truth_answer': ground_truth_answer
        }
        
        # Recording performance metrics
        self.performance_monitor.record_metrics(
            latency_ms=latency_ms,
            vram_allocated_gb=stats.get('gpu_memory_allocated_gb', 0),
            vram_reserved_gb=stats.get('gpu_memory_reserved_gb', 0),
            cache_hit_rate=stats.get('cache_hit_rate', 0),
            swap_operations=stats.get('swap_operations', 0)
        )
        
        return result
    
    def _evaluate_retrieval(
        self,
        retrieved_chunks_list: List[List[str]],
        ground_truth_chunks_list: List[List[str]]
    ) -> Dict:
        """ Evaluating retrieval performance """
        all_results = []
        
        for i, (retrieved, ground_truth) in enumerate(zip(retrieved_chunks_list, ground_truth_chunks_list)):
            result = self.retrieval_evaluator.evaluate_retrieval(
                retrieved_chunks=retrieved,
                relevant_chunks=ground_truth,
                used_chunks=set(),  # TODO: Track used chunks
                total_chunks=20,  # TODO: Get actual total
                top_k_values=[5, 10, 20]
            )
            all_results.append(result)
        
        # Calculating averages
        avg_recall = sum(r['avg_recall'] for r in all_results) / len(all_results) if all_results else 0.0
        avg_precision = sum(r['avg_precision'] for r in all_results) / len(all_results) if all_results else 0.0
        avg_coverage = sum(r['coverage'] for r in all_results) / len(all_results) if all_results else 0.0
        
        return {
            'avg_recall': avg_recall,
            'avg_precision': avg_precision,
            'avg_coverage': avg_coverage,
            'per_turn_results': all_results,
            'total_turns': len(all_results)
        }
    
    def _evaluate_generation(
        self,
        generated_answers: List[str],
        reference_answers: List[str]
    ) -> Dict:
        """ Evaluating generation quality """
        return self.generation_evaluator.evaluate_multi_turn(
            generated_answers=generated_answers,
            reference_answers=reference_answers
        )
    
    def _create_session(self) -> str:
        """ Creating new session """
        resp = requests.post(f"{self.api_base}/api/session/new")
        resp.raise_for_status()
        return resp.json()['session_id']
    
    def _upload_document(self, session_id: str, doc_path: str):
        """ Uploading document to session """
        with open(doc_path, 'rb') as f:
            files = {'file': (Path(doc_path).name, f, 'text/plain')}
            resp = requests.post(
                f"{self.api_base}/api/session/{session_id}/upload-document",
                files=files,
                params={'document_type': 'general'}
            )
        resp.raise_for_status()
    
    def _get_timestamp(self) -> str:
        """ Getting timestamp string """
        from datetime import datetime
        return datetime.now().strftime("%Y%m%d_%H%M%S")

    def _get_vllm_cache_stats(self) -> Dict:
        """ Receving KV-cache statistics from vLLM """
        # Applying KVCacheMonitor
        stats = self._get_cache_stats()
    
        # Calculating cache-hit rate by checking the status changes
        # If no changes in active chunks amount then cache-hit
        active_chunks = stats.get('active_chunks', 0)
        total_chunks = stats.get('total_chunks', 1)
    
        # If uses more than 70% of active chunks then cache-hit
        cache_hit = active_chunks / total_chunks > 0.7 if total_chunks > 0 else False
    
        return {
            'cache_hit': cache_hit,
            'hit_rate': stats.get('cache_hit_rate', 0),
            'swap_count': stats.get('swap_operations', 0),
            'active_chunks': active_chunks,
            'total_chunks': total_chunks
        }