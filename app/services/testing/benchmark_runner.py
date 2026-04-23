import json
import time
import uuid
from pathlib import Path
from typing import Dict, Optional, List
import logging
from datetime import datetime, timezone

import torch

logger = logging.getLogger(__name__)


class BenchmarkRunner:
    """Benchmark runner for S-GAS system.

    Runs each scenario in two modes:
      1. S-GAS (full algorithm: semantic + graph + swap)
      2. Baseline (vanilla RAG: semantic only, no graph, no swap)

    Compares results and generates a comparative report.
    """

    def __init__(
        self,
        chat_service=None,
        document_processor=None,
        scenarios_dir: str = "tests/scenarios",
        results_dir: str = "logs/benchmarks",
        documents_dir: str = "tests/documents",
    ):
        self._chat_service = chat_service
        self._document_processor = document_processor
        self.scenarios_dir = Path(scenarios_dir)
        self.results_dir = Path(results_dir)
        self.documents_dir = Path(documents_dir)

        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.documents_dir.mkdir(parents=True, exist_ok=True)

        from .scenario_loader import ScenarioLoader
        from .metrics_collector import MetricsCollector
        from .retrieval_evaluator import RetrievalEvaluator
        from .generation_evaluator import GenerationEvaluator
        from .performance_monitor import PerformanceMonitor

        self.scenario_loader = ScenarioLoader(scenarios_dir)
        self.generation_evaluator = GenerationEvaluator()

        # Separate collectors for each mode
        self._sgas_collector = MetricsCollector()
        self._baseline_collector = MetricsCollector()
        self._sgas_retrieval = RetrievalEvaluator()
        self._baseline_retrieval = RetrievalEvaluator()
        self._sgas_perf = PerformanceMonitor()
        self._baseline_perf = PerformanceMonitor()

        logger.info("BenchmarkRunner initialized")

    def set_services(self, chat_service, document_processor):
        self._chat_service = chat_service
        self._document_processor = document_processor

    # ── Public API ─────────────────────────────────────────────────────

    async def _reset_and_flush(self, label: str, fresh_server: bool = False):
        """Reset of stateful services and optionally restart vLLM."""
        if self._chat_service is None:
            return
        self._chat_service.reset_for_benchmark()
        if fresh_server:
            restarted = await self._chat_service._vllm.force_restart()
            if restarted:
                logger.info(f"vLLM server restarted before {label}")
            else:
                logger.warning(f"vLLM restart failed before {label}; continuing with cache flush")
        flushed = await self._chat_service.flush_vllm_cache()
        if flushed:
            logger.info(f"vLLM cache flushed before {label}")

    async def run_scenario(
        self,
        scenario_name: str,
        sessions: dict,
        document_processor=None,
        session_id: Optional[str] = None,
        fresh_server: bool = False,
    ) -> Dict:
        """Running scenario in both S-GAS and baseline modes, compare results."""
        doc_processor = document_processor or self._document_processor
        if doc_processor is None:
            raise ValueError("document_processor is required")

        scenario = self.scenario_loader.load_scenario(scenario_name)
        logger.info(f"=== Benchmark: {scenario.name} ({len(scenario.turns)} turns) ===")
        logger.info(f"Document: {scenario.document}")

        doc_path = self.documents_dir / scenario.document
        if not doc_path.exists():
            raise FileNotFoundError(f"Document not found: {doc_path}")

        sgas_run_id = uuid.uuid4().hex
        baseline_run_id = uuid.uuid4().hex

        # ── Run S-GAS mode ────────────────────────────────────────────
        await self._reset_and_flush("S-GAS run", fresh_server=fresh_server)
        logger.info("── Mode: S-GAS (semantic + graph + swap) ──")
        sgas_session = self._create_session(sessions)
        self._upload_document(sgas_session, doc_path, doc_processor,
                              chunk_size=scenario.chunk_size,
                              chunk_overlap=scenario.chunk_overlap)
        sgas_result = await self._run_mode(
            scenario=scenario, sessions=sessions, session_id=sgas_session,
            baseline_mode=False, collector=self._sgas_collector,
            retrieval_eval=self._sgas_retrieval, perf_monitor=self._sgas_perf,
            run_id=sgas_run_id,
        )
        self._cleanup_session(sgas_session)
        self._save_graph_snapshot(scenario_name, timestamp=datetime.now().strftime("%Y%m%d_%H%M%S"))

        # ── Run Baseline mode ─────────────────────────────────────────
        await self._reset_and_flush("Baseline run", fresh_server=fresh_server)
        logger.info("── Mode: Baseline (semantic only) ──")
        baseline_session = self._create_session(sessions)
        self._upload_document(baseline_session, doc_path, doc_processor,
                              chunk_size=scenario.chunk_size,
                              chunk_overlap=scenario.chunk_overlap)
        baseline_result = await self._run_mode(
            scenario=scenario, sessions=sessions, session_id=baseline_session,
            baseline_mode=True, collector=self._baseline_collector,
            retrieval_eval=self._baseline_retrieval, perf_monitor=self._baseline_perf,
            run_id=baseline_run_id,
        )
        self._cleanup_session(baseline_session)

        # ── Compare ───────────────────────────────────────────────────
        comparison = self._compare_results(sgas_result['summary'], baseline_result['summary'])

        # ── Export ────────────────────────────────────────────────────
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        sgas_csv = self.results_dir / f"{scenario_name}_sgas_{timestamp}.csv"
        sgas_json = self.results_dir / f"{scenario_name}_sgas_{timestamp}.json"
        baseline_csv = self.results_dir / f"{scenario_name}_baseline_{timestamp}.csv"
        baseline_json = self.results_dir / f"{scenario_name}_baseline_{timestamp}.json"
        comparison_json = self.results_dir / f"{scenario_name}_comparison_{timestamp}.json"

        self._sgas_collector.export_to_csv(str(sgas_csv))
        self._sgas_collector.export_to_json(str(sgas_json))
        self._baseline_collector.export_to_csv(str(baseline_csv))
        self._baseline_collector.export_to_json(str(baseline_json))

        with open(comparison_json, 'w', encoding='utf-8') as f:
            json.dump(comparison, f, indent=2, ensure_ascii=False, default=str)

        logger.info(f"Benchmark completed! Comparison: {comparison_json}")
        self._log_comparison(comparison)

        return {
            'scenario': scenario.name,
            'sgas': {**sgas_result, 'csv_file': str(sgas_csv), 'json_file': str(sgas_json)},
            'baseline': {**baseline_result, 'csv_file': str(baseline_csv), 'json_file': str(baseline_json)},
            'comparison': comparison,
            'comparison_file': str(comparison_json),
        }

    async def run_single_mode(
        self,
        scenario_name: str,
        sessions: dict,
        mode: str = "sgas",
        document_processor=None,
        fresh_server: bool = False,
    ) -> Dict:
        """Running scenario in a single mode only (sgas or baseline)."""
        doc_processor = document_processor or self._document_processor
        if doc_processor is None:
            raise ValueError("document_processor is required")

        baseline_mode = (mode == "baseline")
        scenario = self.scenario_loader.load_scenario(scenario_name)
        logger.info(f"=== Single-mode Benchmark: {scenario.name} ({mode}) ===")

        doc_path = self.documents_dir / scenario.document
        if not doc_path.exists():
            raise FileNotFoundError(f"Document not found: {doc_path}")

        await self._reset_and_flush(f"{mode} run", fresh_server=fresh_server)

        run_id = uuid.uuid4().hex
        collector = self._sgas_collector if not baseline_mode else self._baseline_collector
        retrieval_eval = self._sgas_retrieval if not baseline_mode else self._baseline_retrieval
        perf_monitor = self._sgas_perf if not baseline_mode else self._baseline_perf

        session_id = self._create_session(sessions)
        self._upload_document(session_id, doc_path, doc_processor,
                              chunk_size=scenario.chunk_size,
                              chunk_overlap=scenario.chunk_overlap)
        result = await self._run_mode(
            scenario=scenario, sessions=sessions, session_id=session_id,
            baseline_mode=baseline_mode, collector=collector,
            retrieval_eval=retrieval_eval, perf_monitor=perf_monitor,
            run_id=run_id,
        )
        self._cleanup_session(session_id)

        if not baseline_mode:
            self._save_graph_snapshot(scenario_name, timestamp=datetime.now().strftime("%Y%m%d_%H%M%S"))

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = self.results_dir / f"{scenario_name}_{mode}_{timestamp}.csv"
        json_path = self.results_dir / f"{scenario_name}_{mode}_{timestamp}.json"
        collector.export_to_csv(str(csv_path))
        collector.export_to_json(str(json_path))

        return {
            'scenario': scenario.name,
            'mode': mode,
            'session_id': session_id,
            'summary': result['summary'],
            'csv_file': str(csv_path),
            'json_file': str(json_path),
        }

    # ── Run single mode ────────────────────────────────────────────────

    async def _run_mode(
        self, scenario, sessions, session_id, baseline_mode,
        collector, retrieval_eval, perf_monitor,
        run_id: Optional[str] = None,
    ) -> Dict:
        mode_name = "baseline" if baseline_mode else "sgas"

        collector.start_session()
        retrieval_eval.reset()
        perf_monitor.reset()

        # Measuring VRAM baseline before any work
        vram_baseline_gb = 0.0
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            vram_baseline_gb = torch.cuda.memory_allocated() / (1024 ** 3)
            logger.info(f"  [{mode_name}] VRAM baseline: {vram_baseline_gb:.4f} GB (subtracted from measurements)")

        questions, answers = [], []
        retrieved_chunks_list, ground_truth_chunks_list, ground_truth_answers = [], [], []
        retrieved_texts_list, ground_truth_texts_list = [], []
        total_chunks_per_turn = []

        for i, turn in enumerate(scenario.turns):
            logger.info(f"  [{mode_name}] Turn {i + 1}/{len(scenario.turns)}: {turn.query[:60]}...")

            gt_chunks = turn.ground_truth if turn.ground_truth else []
            gt_answer = turn.ground_truth_answer if turn.ground_truth_answer else ""
            gt_text = turn.ground_truth_text if turn.ground_truth_text else ""

            result = await self._run_turn(
                session_id=session_id, sessions=sessions, query=turn.query,
                ground_truth_chunks=gt_chunks, ground_truth_answer=gt_answer,
                baseline_mode=baseline_mode, run_id=run_id,
            )

            # Subtracting VRAM baseline
            result['vram_allocated_gb'] = max(0.0, result['vram_allocated_gb'] - vram_baseline_gb)
            result['vram_reserved_gb'] = max(0.0, result['vram_reserved_gb'] - vram_baseline_gb)
            result['vram_peak_gb'] = max(0.0, result['vram_peak_gb'] - vram_baseline_gb)

            collector.record_turn(result)
            questions.append(turn.query)
            answers.append(result['response'])
            retrieved_chunks_list.append(result.get('retrieved_chunks', []))
            ground_truth_chunks_list.append(gt_chunks)
            ground_truth_answers.append(gt_answer)
            retrieved_texts_list.append(result.get('retrieved_chunk_texts', []))
            ground_truth_texts_list.append(gt_text)
            total_chunks_per_turn.append(result.get('total_chunks', 20))

            logger.info(f"    Coverage: {result['coverage_ratio'] * 100:.1f}%")
            logger.info(f"    Latency: {result['latency_ms']:.1f} ms "
                        f"(search={result.get('latency_search_ms', 0):.0f}, "
                        f"rerank={result.get('latency_rerank_ms', 0):.0f}, "
                        f"swap={result.get('latency_swap_ms', 0):.0f}, "
                        f"inference={result.get('latency_inference_ms', 0):.0f})")
            logger.info(f"    VRAM: alloc={result['vram_allocated_gb']:.3f} GB, peak={result['vram_peak_gb']:.3f} GB")
            logger.info(f"    RAM: {result.get('ram_used_gb', 0):.2f} GB ({result.get('ram_percent', 0):.1f}%), "
                        f"Disk: {result.get('disk_used_gb', 0):.1f} GB ({result.get('disk_percent', 0):.1f}%), "
                        f"Process RSS: {result.get('process_rss_mb', 0):.1f} MB")
            logger.info(f"    Graph: {result['graph_nodes']} nodes, {result['graph_edges']} edges")
            logger.info(f"    Swap: {result['swap_operations']} ops, Cache hit: {result['cache_hit_rate']:.3f}")

            perf_monitor.end_timing('turn')
            perf_monitor.record_vram(result['vram_allocated_gb'], result['vram_reserved_gb'])

        summary = collector.calculate_summary()
        summary['mode'] = mode_name

        # Retrieval evaluation
        retrieval_results = self._evaluate_retrieval(
            retrieved_chunks_list, ground_truth_chunks_list, retrieval_eval,
            total_chunks_per_turn=total_chunks_per_turn,
            retrieved_texts_list=retrieved_texts_list,
            ground_truth_texts_list=ground_truth_texts_list,
        )
        summary['retrieval_metrics'] = retrieval_results

        # Generation evaluation
        if any(ground_truth_answers):
            generation_results = self.generation_evaluator.evaluate_multi_turn(
                generated_answers=answers, reference_answers=ground_truth_answers,
            )
            summary['generation_metrics'] = generation_results

        return {'session_id': session_id, 'summary': summary}

    # ── Turn execution ─────────────────────────────────────────────────

    async def _run_turn(
        self, session_id: str, sessions: dict, query: str,
        ground_truth_chunks: List[str], ground_truth_answer: str,
        baseline_mode: bool = False, run_id: Optional[str] = None,
    ) -> Dict:
        """Run single turn through real S-GAS pipeline via ChatService."""
        session_data = sessions[session_id]

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        start_time = time.time()

        if self._chat_service is not None:
            chat_result = await self._chat_service.process_chat(
                session_id=session_id, session_data=session_data,
                message=query, use_rag=True, n_chunks=5, temperature=0.1,
                baseline_mode=baseline_mode, run_id=run_id,
            )

            metadata = chat_result.get('metadata', {})
            swap_stats = metadata.get('swap_statistics', {})
            latency_info = metadata.get('latency', {})
            vram_info = metadata.get('vram', {})
            graph_stats = metadata.get('graph_statistics', {})
            cache_check = metadata.get('cache_check', {})

            vram_after = vram_info.get('after_inference', {})

            sys_res = metadata.get('system_resources', {})

            return {
                'query': query,
                'response': chat_result.get('response', ''),
                # Latency breakdown
                'latency_ms': latency_info.get('total_ms', (time.time() - start_time) * 1000),
                'latency_search_ms': latency_info.get('search_ms', 0),
                'latency_rerank_ms': latency_info.get('rerank_ms', 0),
                'latency_swap_ms': latency_info.get('swap_ms', 0),
                'latency_inference_ms': latency_info.get('inference_ms', 0),
                # VRAM
                'vram_allocated_gb': vram_after.get('allocated_gb', 0),
                'vram_reserved_gb': vram_after.get('reserved_gb', 0),
                'vram_peak_gb': vram_after.get('peak_gb', 0),
                # System resources
                'ram_used_gb': sys_res.get('ram', {}).get('used_gb', 0),
                'ram_percent': sys_res.get('ram', {}).get('percent', 0),
                'disk_used_gb': sys_res.get('disk', {}).get('used_gb', 0),
                'disk_percent': sys_res.get('disk', {}).get('percent', 0),
                'process_rss_mb': sys_res.get('process', {}).get('rss_mb', 0),
                # Chunks
                'active_chunks': metadata.get('context_chunks_used', 0),
                'total_chunks': metadata.get('total_chunks_available', 0),
                'coverage_ratio': metadata.get('coverage_percent', 0) / 100.0,
                # Cache / Swap
                'cache_hit_rate': cache_check.get('hit_rate', 0),
                'cache_hits': cache_check.get('hits', 0),
                'cache_misses': cache_check.get('misses', 0),
                'swap_operations': swap_stats.get('total_swap_operations', 0),
                # Graph
                'graph_nodes': graph_stats.get('total_nodes', 0),
                'graph_edges': graph_stats.get('total_edges', 0),
                # Retrieval
                'retrieved_chunks': metadata.get('retrieved_chunk_ids', []),
                'retrieved_chunk_texts': metadata.get('retrieved_chunk_texts', []),
                'chunk_scores': metadata.get('chunk_scores', []),
                # Ground truth
                'ground_truth_chunks': ground_truth_chunks,
                'ground_truth_answer': ground_truth_answer,
            }
        else:
            raise RuntimeError(
                "BenchmarkRunner requires a real chat_service. "
                "Mock responses are forbidden in benchmarks because they would "
                "produce fake metrics. Initialize BenchmarkRunner with a valid "
                "ChatService instance before running scenarios."
            )

    # ── Evaluation helpers ─────────────────────────────────────────────

    @staticmethod
    def _normalize_chunk_ids(chunk_ids: List[str]) -> List[str]:
        """Normalizing chunk IDs to comparable chunk-index form."""
        normalized = []
        for cid in chunk_ids:
            # Runtime format: "session_abc:doc-uuid:3"
            if ':' in cid:
                normalized.append(cid.rsplit(':', 1)[-1])
            # Scenario format: "chunk_001" → "0", "chunk_023" → "22"
            elif cid.startswith('chunk_'):
                try:
                    normalized.append(str(int(cid.split('_', 1)[1]) - 1))
                except ValueError:
                    normalized.append(cid)
            else:
                normalized.append(cid)
        return normalized

    def _evaluate_retrieval(
        self, retrieved_list: List[List[str]], gt_list: List[List[str]],
        evaluator, total_chunks_per_turn: List[int] = None,
        retrieved_texts_list: List[List[str]] = None,
        ground_truth_texts_list: List[str] = None,
    ) -> Dict:
        all_results = []
        for i, (retrieved, ground_truth) in enumerate(zip(retrieved_list, gt_list)):
            norm_retrieved = self._normalize_chunk_ids(retrieved)
            norm_gt = self._normalize_chunk_ids(ground_truth)
            total = (total_chunks_per_turn[i] if total_chunks_per_turn else None) or 20
            result = evaluator.evaluate_retrieval(
                retrieved_chunks=norm_retrieved, relevant_chunks=norm_gt,
                used_chunks=set(), total_chunks=total, top_k_values=[5, 10, 20],
            )
            all_results.append(result)

        valid_results = [r for r in all_results if r['relevant_count'] > 0]
        n = len(valid_results) if valid_results else 1

        # ── Text Recall: chunk-ID-independent ────────────────────────
        # Binary per-turn: 1.0 if ANY retrieved chunk contains the ground_truth_text phrase, else 0.0.
        text_recall_scores = []
        if retrieved_texts_list and ground_truth_texts_list:
            for turn_texts, gt_phrase in zip(retrieved_texts_list, ground_truth_texts_list):
                if not gt_phrase:
                    continue
                phrase_lower = gt_phrase.lower()
                combined = ' '.join(turn_texts).lower()
                text_recall_scores.append(1.0 if phrase_lower in combined else 0.0)

        avg_text_recall = (
            sum(text_recall_scores) / len(text_recall_scores)
            if text_recall_scores else 0.0
        )

        # ── Semantic Similarity: embedding-based relevance ───────────
        # For each turn, computing cosine similarity between the ground_truth_text and each retrieved chunk, then take the max.
        semantic_sim_scores = []
        if retrieved_texts_list and ground_truth_texts_list:
            try:
                from sentence_transformers import SentenceTransformer, util
                _st_model = SentenceTransformer('all-MiniLM-L6-v2')

                for turn_texts, gt_phrase in zip(retrieved_texts_list, ground_truth_texts_list):
                    if not gt_phrase or not turn_texts:
                        continue
                    gt_emb = _st_model.encode(gt_phrase, convert_to_tensor=True)
                    chunk_embs = _st_model.encode(turn_texts, convert_to_tensor=True)
                    sims = util.cos_sim(gt_emb, chunk_embs)[0]
                    semantic_sim_scores.append(float(sims.max()))
            except Exception as e:
                logger.warning(f"Semantic similarity computation failed: {e}")

        avg_semantic_similarity = (
            sum(semantic_sim_scores) / len(semantic_sim_scores)
            if semantic_sim_scores else 0.0
        )

        return {
            'avg_coverage': sum(r['coverage'] for r in valid_results) / n,
            'avg_text_recall': round(avg_text_recall, 4),
            'text_recall_turns': len(text_recall_scores),
            'text_recall_hits': sum(text_recall_scores),
            'avg_semantic_similarity': round(avg_semantic_similarity, 4),
            'semantic_similarity_turns': len(semantic_sim_scores),
            'total_turns': len(valid_results),
        }

    # ── Comparison ─────────────────────────────────────────────────────

    def _compare_results(self, sgas_summary: Dict, baseline_summary: Dict) -> Dict:
        """Compare S-GAS vs Baseline summaries and compute deltas."""

        def _delta(sgas_val, baseline_val):
            """Positive delta = S-GAS is better."""
            if baseline_val == 0:
                return {'sgas': sgas_val, 'baseline': baseline_val, 'delta': sgas_val, 'improvement_pct': 0}
            delta = sgas_val - baseline_val
            improvement = (delta / abs(baseline_val)) * 100
            return {
                'sgas': round(sgas_val, 4),
                'baseline': round(baseline_val, 4),
                'delta': round(delta, 4),
                'improvement_pct': round(improvement, 2),
            }

        def _delta_lower_better(sgas_val, baseline_val):
            """For metrics where lower is better (latency, VRAM)."""
            if baseline_val == 0:
                return {'sgas': sgas_val, 'baseline': baseline_val, 'delta': -sgas_val, 'improvement_pct': 0}
            delta = baseline_val - sgas_val  # positive = S-GAS uses less
            improvement = (delta / abs(baseline_val)) * 100
            return {
                'sgas': round(sgas_val, 4),
                'baseline': round(baseline_val, 4),
                'delta': round(delta, 4),
                'improvement_pct': round(improvement, 2),
            }

        # Retrieval quality
        sgas_ret = sgas_summary.get('retrieval_metrics', {})
        base_ret = baseline_summary.get('retrieval_metrics', {})

        # Generation quality
        sgas_gen = sgas_summary.get('generation_metrics', {})
        base_gen = baseline_summary.get('generation_metrics', {})

        comparison = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'total_turns': sgas_summary.get('total_turns', 0),
            'quality_metrics': {
                'text_recall': _delta(sgas_ret.get('avg_text_recall', 0), base_ret.get('avg_text_recall', 0)),
                'semantic_similarity': _delta(sgas_ret.get('avg_semantic_similarity', 0), base_ret.get('avg_semantic_similarity', 0)),
                'coverage': _delta(sgas_summary.get('final_coverage', 0), baseline_summary.get('final_coverage', 0)),
                'multi_turn_accuracy': _delta(
                    sgas_gen.get('multi_turn_accuracy', 0),
                    base_gen.get('multi_turn_accuracy', 0),
                ),
            },
            'performance_metrics': {
                'avg_latency_ms': _delta_lower_better(
                    sgas_summary.get('avg_latency_ms', 0),
                    baseline_summary.get('avg_latency_ms', 0),
                ),
                'avg_latency_excl_first_ms': _delta_lower_better(
                    sgas_summary.get('avg_latency_excl_first_ms', 0),
                    baseline_summary.get('avg_latency_excl_first_ms', 0),
                ),
                'avg_vram_gb': _delta_lower_better(
                    sgas_summary.get('avg_vram_gb', 0),
                    baseline_summary.get('avg_vram_gb', 0),
                ),
                'avg_cache_hit_rate': _delta(
                    sgas_summary.get('avg_cache_hit_rate', 0),
                    baseline_summary.get('avg_cache_hit_rate', 0),
                ),
                'total_swap_operations': {
                    'sgas': sgas_summary.get('total_swap_operations', 0),
                    'baseline': baseline_summary.get('total_swap_operations', 0),
                },
            },
            'graph_usage': {
                'sgas_has_graph': True,
                'baseline_has_graph': False,
                'avg_graph_nodes_sgas': sgas_summary.get('avg_graph_nodes', 0),
                'avg_graph_edges_sgas': sgas_summary.get('avg_graph_edges', 0),
            },
            'verdict': self._generate_verdict(sgas_summary, baseline_summary, sgas_ret, base_ret, sgas_gen, base_gen),
        }

        return comparison

    def _generate_verdict(self, sgas, baseline, sgas_ret, base_ret, sgas_gen, base_gen) -> Dict:
        """Generate a human-readable verdict of which approach wins per metric."""
        verdicts = {}

        # Text-based recall
        s_recall = sgas_ret.get('avg_text_recall') or sgas_ret.get('avg_recall', 0)
        b_recall = base_ret.get('avg_text_recall') or base_ret.get('avg_recall', 0)
        if s_recall > b_recall:
            verdicts['recall'] = f"S-GAS wins ({s_recall:.3f} vs {b_recall:.3f})"
        elif b_recall > s_recall:
            verdicts['recall'] = f"Baseline wins ({b_recall:.3f} vs {s_recall:.3f})"
        else:
            verdicts['recall'] = f"Tie ({s_recall:.3f})"

        # Coverage
        s_cov = sgas.get('final_coverage', 0)
        b_cov = baseline.get('final_coverage', 0)
        if s_cov > b_cov:
            verdicts['coverage'] = f"S-GAS wins ({s_cov:.3f} vs {b_cov:.3f})"
        elif b_cov > s_cov:
            verdicts['coverage'] = f"Baseline wins ({b_cov:.3f} vs {s_cov:.3f})"
        else:
            verdicts['coverage'] = f"Tie ({s_cov:.3f})"

        # Accuracy
        s_acc = sgas_gen.get('multi_turn_accuracy', 0)
        b_acc = base_gen.get('multi_turn_accuracy', 0)
        if s_acc > b_acc:
            verdicts['accuracy'] = f"S-GAS wins ({s_acc * 100:.1f}% vs {b_acc * 100:.1f}%)"
        elif b_acc > s_acc:
            verdicts['accuracy'] = f"Baseline wins ({b_acc * 100:.1f}% vs {s_acc * 100:.1f}%)"
        else:
            verdicts['accuracy'] = f"Tie ({s_acc * 100:.1f}%)"

        # Latency (lower is better; using steady-state avg to exclude warmup turn)
        s_lat = sgas.get('avg_latency_excl_first_ms') or sgas.get('avg_latency_ms', 0)
        b_lat = baseline.get('avg_latency_excl_first_ms') or baseline.get('avg_latency_ms', 0)
        if s_lat < b_lat:
            verdicts['latency'] = f"S-GAS wins ({s_lat:.0f}ms vs {b_lat:.0f}ms)"
        elif b_lat < s_lat:
            verdicts['latency'] = f"Baseline wins ({b_lat:.0f}ms vs {s_lat:.0f}ms)"
        else:
            verdicts['latency'] = f"Tie ({s_lat:.0f}ms)"

        # Overall of GPU
        sgas_wins = sum(1 for v in verdicts.values() if 'S-GAS wins' in v)
        baseline_wins = sum(1 for v in verdicts.values() if 'Baseline wins' in v)
        if sgas_wins > baseline_wins:
            verdicts['overall'] = f"S-GAS algorithm is better ({sgas_wins}/{len(verdicts) - 1} metrics)"
        elif baseline_wins > sgas_wins:
            verdicts['overall'] = f"Baseline is better ({baseline_wins}/{len(verdicts) - 1} metrics)"
        else:
            verdicts['overall'] = "Tie — no clear winner"

        return verdicts

    def _log_comparison(self, comparison: Dict):
        """Logging comparison results to console."""
        logger.info("=" * 60)
        logger.info("BENCHMARK COMPARISON: S-GAS vs Baseline")
        logger.info("=" * 60)

        for cat_name, cat_data in comparison.get('quality_metrics', {}).items():
            if isinstance(cat_data, dict) and 'improvement_pct' in cat_data:
                sign = "+" if cat_data['improvement_pct'] >= 0 else ""
                logger.info(f"  {cat_name}: S-GAS={cat_data['sgas']}, Baseline={cat_data['baseline']} "
                            f"({sign}{cat_data['improvement_pct']}%)")

        for cat_name, cat_data in comparison.get('performance_metrics', {}).items():
            if isinstance(cat_data, dict) and 'improvement_pct' in cat_data:
                sign = "+" if cat_data['improvement_pct'] >= 0 else ""
                logger.info(f"  {cat_name}: S-GAS={cat_data['sgas']}, Baseline={cat_data['baseline']} "
                            f"({sign}{cat_data['improvement_pct']}%)")

        verdict = comparison.get('verdict', {})
        logger.info("-" * 60)
        for k, v in verdict.items():
            logger.info(f"  {k}: {v}")
        logger.info("=" * 60)

    # ── Session helpers ────────────────────────────────────────────────

    def _create_session(self, sessions: dict) -> str:
        session_id = f"session_{uuid.uuid4().hex[:12]}"
        sessions[session_id] = {
            'iteration': 0,
            'excluded_chunks': [],
            'created_at': datetime.now(timezone.utc).isoformat(),
            'swap_initialized': False,
            'graph_initialized': False,
        }
        return session_id

    def _cleanup_session(self, session_id: str) -> None:
        """Removing a finished benchmark session's vector store chunks and in-memory state."""
        try:
            if self._chat_service is not None:
                self._chat_service._vector_store.delete_session_chunks(session_id)
                logger.info(f"Cleaned up vector store for session {session_id}")
        except Exception as e:
            logger.warning(f"Could not clean up session {session_id}: {e}")

    def _save_graph_snapshot(self, scenario_name: str, timestamp: str) -> None:
        """Persisting the current S-GAS knowledge graph to disk before it is reset."""
        if self._chat_service is None:
            return
        try:
            snapshot = self._chat_service.get_graph_snapshot()
            node_count = len(snapshot.get('nodes', []))
            if node_count == 0:
                logger.warning("Graph snapshot skipped: graph is empty after S-GAS run")
                return

            # Always-current alias (used by the web endpoint as fallback)
            latest_path = self.results_dir / "latest_sgas_graph.json"
            with open(latest_path, 'w', encoding='utf-8') as f:
                json.dump(snapshot, f, ensure_ascii=False, default=str)

            # Timestamped archive copy
            archive_path = self.results_dir / f"{scenario_name}_graph_{timestamp}.json"
            with open(archive_path, 'w', encoding='utf-8') as f:
                json.dump(snapshot, f, ensure_ascii=False, default=str)

            logger.info(
                f"Graph snapshot saved: {node_count} nodes, "
                f"{len(snapshot.get('edges', []))} edges → {latest_path}"
            )
        except Exception as e:
            logger.warning(f"Graph snapshot save failed: {e}")

    def _upload_document(self, session_id: str, doc_path: Path, document_processor,
                         chunk_size: Optional[int] = None, chunk_overlap: Optional[int] = None):
        """Uploading a document for benchmarking."""
        chunker = getattr(document_processor, '_chunker', None)
        orig_size = orig_overlap = None
        if chunker is not None and chunk_size is not None:
            orig_size = getattr(chunker, '_max_chunk_size', None)
            orig_overlap = getattr(chunker, '_overlap_size', None)
            chunker._max_chunk_size = chunk_size
            chunker._overlap_size = chunk_overlap if chunk_overlap is not None else max(1, chunk_size // 8)
            logger.info(f"Benchmark chunk size patched: {orig_size}→{chunk_size} words")
        try:
            result = document_processor.process_document(
                doc_path, session_id=session_id, metadata={"document_type": "general"},
            )
        finally:
            if chunker is not None and orig_size is not None:
                chunker._max_chunk_size = orig_size
                chunker._overlap_size = orig_overlap
        if result.status != 'success':
            raise RuntimeError(f"Document upload failed: {result.error}")
