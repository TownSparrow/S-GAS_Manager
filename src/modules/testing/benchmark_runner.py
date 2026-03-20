import json
from pathlib import Path
from typing import Dict, Optional, List
import logging
from datetime import datetime


logger = logging.getLogger(__name__)


class BenchmarkRunner:
    """Benchmark runner for S-GAS system"""
    
    def __init__(
        self,
        scenarios_dir: str = "tests/scenarios",
        results_dir: str = "logs/benchmarks",
        documents_dir: str = "tests/documents"
    ):
        """Initializing benchmark runner"""
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
        
        self.scenario_loader = ScenarioLoader(scenarios_dir)
        self.metrics_collector = MetricsCollector()
        self.retrieval_evaluator = RetrievalEvaluator()
        self.generation_evaluator = GenerationEvaluator()
        self.performance_monitor = PerformanceMonitor()
        
        logger.info("✅ Benchmark runner initialized")
    
    def run_scenario(
        self,
        scenario_name: str,
        sessions: dict,
        document_processor,
        session_id: Optional[str] = None
    ) -> Dict:
        """
        Running benchmark scenario using provided components
        
        Args:
            scenario_name: Name of scenario
            sessions: Dictionary of sessions (app.state.sessions)
            document_processor: Document processor (app.state.document_processor)
            session_id: Existing session ID or None to create new
        """
        
        # Loading scenario
        scenario = self.scenario_loader.load_scenario(scenario_name)
        
        logger.info(f"Starting benchmark: {scenario.name}")
        logger.info(f"Document: {scenario.document}")
        logger.info(f"Total turns: {len(scenario.turns)}")
        
        # Creating session if not provided
        if session_id is None:
            session_id = self._create_session(sessions)
            logger.info(f"✅ Created session: {session_id}")
        else:
            logger.info(f"Using existing session: {session_id}")
        
        # Uploading document
        doc_path = self.documents_dir / scenario.document
        if not doc_path.exists():
            raise FileNotFoundError(f"❌ Document not found: {doc_path}")
        
        self._upload_document(session_id, doc_path, document_processor)
        logger.info(f"✅ Document uploaded: {scenario.document}")
        
        # Initializing metrics collection
        self.metrics_collector.start_session()
        self.retrieval_evaluator.reset()
        self.performance_monitor.reset()
        
        # Running all turns
        questions = []
        answers = []
        retrieved_chunks_list = []
        ground_truth_chunks_list = []
        ground_truth_answers = []
        
        for i, turn in enumerate(scenario.turns):
            logger.info(f"Turn {i+1}/{len(scenario.turns)}")
            logger.info(f"Query: {turn.query[:60]}...")
            
            # Running turn directly
            result = self._run_turn(
                session_id=session_id,
                sessions=sessions,
                query=turn.query,
                ground_truth_chunks=getattr(turn, 'ground_truth_chunks', []),
                ground_truth_answer=getattr(turn, 'ground_truth_answer', "")
            )
            
            # Recording metrics
            self.metrics_collector.record_turn(result)
            
            # Collecting data
            questions.append(turn.query)
            answers.append(result['response'])
            retrieved_chunks_list.append(result.get('retrieved_chunks', []))
            ground_truth_chunks_list.append(getattr(turn, 'ground_truth_chunks', []))
            ground_truth_answers.append(getattr(turn, 'ground_truth_answer', ""))
            
            logger.info(f"Coverage: {result['coverage_ratio']*100:.1f}%")
            logger.info(f"Latency: {result['latency_ms']:.1f} ms")
        
        # Calculating summary
        summary = self.metrics_collector.calculate_summary()
        
        # Evaluating retrieval
        retrieval_results = self._evaluate_retrieval(
            retrieved_chunks_list=retrieved_chunks_list,
            ground_truth_chunks_list=ground_truth_chunks_list
        )
        summary['retrieval_metrics'] = retrieval_results
        
        # Evaluating generation via ground truth
        if any(ground_truth_answers):
            generation_results = self._evaluate_generation(
                generated_answers=answers,
                reference_answers=ground_truth_answers
            )
            summary['generation_metrics'] = generation_results
        
        # Saving results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_file = self.results_dir / f"{scenario_name}_{timestamp}.csv"
        json_file = self.results_dir / f"{scenario_name}_{timestamp}.json"
        
        self.metrics_collector.export_to_csv(str(csv_file))
        self.metrics_collector.export_to_json(str(json_file))
        
        logger.info(f"✅ Benchmark completed!")
        logger.info(f"Results: {csv_file}")
        
        return {
            'scenario': scenario.name,
            'session_id': session_id,
            'summary': summary,
            'csv_file': str(csv_file),
            'json_file': str(json_file)
        }
    
    def _create_session(self, sessions: dict) -> str:
        """Creating session directly in sessions dict"""
        import uuid
        from datetime import timezone
        
        session_id = f"session_{uuid.uuid4().hex[:12]}"
        sessions[session_id] = {
            'iteration': 0,
            'excluded_chunks': [],
            'created_at': datetime.now(timezone.utc).isoformat(),
            'swap_initialized': False
        }
        return session_id
    
    def _upload_document(self, session_id: str, doc_path: Path, document_processor):
        """Uploading document using provided document processor"""
        result = document_processor.process_document(
            doc_path,
            session_id=session_id,
            metadata={"document_type": "general"}
        )
        
        if result.status != 'success':
            raise Exception(f"❌ Document upload failed: {result.error}")
    
    def _run_turn(
        self,
        session_id: str,
        sessions: dict,
        query: str,
        ground_truth_chunks: List[str],
        ground_truth_answer: str
    ) -> Dict:
        """Running single turn - mock implementation for now"""
        import time
        
        start_time = time.time()
        
        session_data = sessions[session_id]
        session_data['iteration'] += 1
        iteration = session_data['iteration']
        
        result = {
            'query': query,
            'response': f"Mock response for: {query[:50]}...",
            'latency_ms': (time.time() - start_time) * 1000,
            'vram_allocated_gb': 4.5,
            'active_chunks': 5,
            'total_chunks': 20,
            'coverage_ratio': min(1.0, iteration * 0.25),
            'cache_hit_rate': 0.75,
            'retrieved_chunks': ['chunk_1', 'chunk_2', 'chunk_3'],
            'ground_truth_chunks': ground_truth_chunks,
            'ground_truth_answer': ground_truth_answer
        }
        
        return result
    
    def _evaluate_retrieval(
        self,
        retrieved_chunks_list: List[List[str]],
        ground_truth_chunks_list: List[List[str]]
    ) -> Dict:
        """Evaluating retrieval performance"""
        all_results = []
        
        for retrieved, ground_truth in zip(retrieved_chunks_list, ground_truth_chunks_list):
            result = self.retrieval_evaluator.evaluate_retrieval(
                retrieved_chunks=retrieved,
                relevant_chunks=ground_truth,
                used_chunks=set(),
                total_chunks=20,
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
        """Evaluating generation quality"""
        return self.generation_evaluator.evaluate_multi_turn(
            generated_answers=generated_answers,
            reference_answers=reference_answers
        )