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

    async def run_scenario(
        self,
        scenario_name: str,
        sessions: dict,
        document_processor=None,
        session_id: Optional[str] = None,
    ) -> Dict:
        """Run scenario in both S-GAS and baseline modes, compare results."""
        doc_processor = document_processor or self._document_processor
        if doc_processor is None:
            raise ValueError("document_processor is required")

        scenario = self.scenario_loader.load_scenario(scenario_name)
        logger.info(f"=== Benchmark: {scenario.name} ({len(scenario.turns)} turns) ===")
        logger.info(f"Document: {scenario.document}")

        doc_path = self.documents_dir / scenario.document
        if not doc_path.exists():
            raise FileNotFoundError(f"Document not found: {doc_path}")

        # Unique IDs so each mode gets a distinct vLLM prompt prefix —
        # this prevents the GPU KV-cache from bleeding across modes.
        sgas_run_id = uuid.uuid4().hex
        baseline_run_id = uuid.uuid4().hex

        # Flush vLLM's prefix KV-cache before each mode so latency
        # measurements are not distorted by previous runs.
        if self._chat_service is not None:
            flushed = await self._chat_service.flush_vllm_cache()
            if flushed:
                logger.info("vLLM cache flushed before S-GAS run")
            else:
                logger.debug("vLLM cache flush skipped (no reset endpoint); run_id prefix prevents cross-run reuse")

        # ── Run S-GAS mode ────────────────────────────────────────────
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

        if self._chat_service is not None:
            flushed = await self._chat_service.flush_vllm_cache()
            if flushed:
                logger.info("vLLM cache flushed before Baseline run")
            else:
                logger.debug("vLLM cache flush skipped (no reset endpoint); run_id prefix prevents cross-run reuse")

        # ── Run Baseline mode ─────────────────────────────────────────
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
            # Fallback: mock
            session_data['iteration'] += 1
            iteration = session_data['iteration']
            return {
                'query': query,
                'response': f"Mock response for: {query[:50]}...",
                'latency_ms': (time.time() - start_time) * 1000,
                'latency_search_ms': 0, 'latency_rerank_ms': 0,
                'latency_swap_ms': 0, 'latency_inference_ms': 0,
                'vram_allocated_gb': 0.0, 'vram_reserved_gb': 0.0, 'vram_peak_gb': 0.0,
                'active_chunks': 5, 'total_chunks': 20,
                'coverage_ratio': min(1.0, iteration * 0.25),
                'cache_hit_rate': 0.0, 'cache_hits': 0, 'cache_misses': 0,
                'swap_operations': 0, 'graph_nodes': 0, 'graph_edges': 0,
                'retrieved_chunks': [],
                'ground_truth_chunks': ground_truth_chunks,
                'ground_truth_answer': ground_truth_answer,
            }

    # ── Evaluation helpers ─────────────────────────────────────────────

    @staticmethod
    def _normalize_chunk_ids(chunk_ids: List[str]) -> List[str]:
        """Normalize chunk IDs to comparable chunk-index form.

        Scenario ground-truth uses labels like "chunk_001", "chunk_002".
        Runtime IDs have the form "session_id:document_uuid:index".
        We extract the trailing integer from both formats so they can be
        compared on equal footing, e.g. both become "0", "1", "2", …
        """
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

        # ── Text-based recall & precision: chunk-ID-independent ──────
        # Both metrics use the scenario's `ground_truth_text` key phrase as
        # the operationalization of "relevance":
        #
        #   Text Recall  = 1.0 if ANY retrieved chunk contains the phrase, else 0.0
        #                  (binary: did the system find the relevant passage?)
        #
        #   Text Precision@K = (# retrieved chunks containing the phrase) / K
        #                  (fraction of the retrieved set that is on-topic)
        #
        # These are standard IR definitions applied at the passage level.
        # They are robust to chunking granularity and overlap because they
        # match on surface text rather than on positional chunk IDs.
        text_recall_scores = []
        text_precision_scores = []
        if retrieved_texts_list and ground_truth_texts_list:
            for turn_texts, gt_phrase in zip(retrieved_texts_list, ground_truth_texts_list):
                if not gt_phrase:
                    continue
                phrase_lower = gt_phrase.lower()
                # Recall: at least one retrieved chunk contains the phrase
                combined = ' '.join(turn_texts).lower()
                text_recall_scores.append(1.0 if phrase_lower in combined else 0.0)
                # Precision: fraction of retrieved chunks containing the phrase
                if turn_texts:
                    hits = sum(1 for t in turn_texts if phrase_lower in t.lower())
                    text_precision_scores.append(hits / len(turn_texts))

        avg_text_recall = (
            sum(text_recall_scores) / len(text_recall_scores)
            if text_recall_scores else 0.0
        )
        avg_text_precision = (
            sum(text_precision_scores) / len(text_precision_scores)
            if text_precision_scores else 0.0
        )

        return {
            'avg_recall': sum(r['avg_recall'] for r in valid_results) / n if valid_results else 0.0,
            'avg_precision': sum(r['avg_precision'] for r in valid_results) / n if valid_results else 0.0,
            'avg_coverage': sum(r['coverage'] for r in valid_results) / n,
            # Text-based metrics are the authoritative retrieval metrics when
            # chunk-ID alignment with ground truth cannot be guaranteed.
            'avg_text_recall': round(avg_text_recall, 4),
            'text_recall_turns': len(text_recall_scores),
            'text_recall_hits': sum(text_recall_scores),
            'avg_text_precision': round(avg_text_precision, 4),
            'text_precision_turns': len(text_precision_scores),
            'per_turn_results': valid_results,
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
                'recall_at_k': _delta(sgas_ret.get('avg_recall', 0), base_ret.get('avg_recall', 0)),
                'text_recall': _delta(sgas_ret.get('avg_text_recall', 0), base_ret.get('avg_text_recall', 0)),
                # Text-based Precision@K: fraction of retrieved chunks containing the
                # ground_truth_text key phrase.  Replaces the broken chunk-ID version.
                'precision': _delta(sgas_ret.get('avg_text_precision', 0), base_ret.get('avg_text_precision', 0)),
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
                'peak_vram_gb': _delta_lower_better(
                    sgas_summary.get('peak_vram_gb', 0),
                    baseline_summary.get('peak_vram_gb', 0),
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

        # Text-based recall (authoritative — chunk-ID recall is unreliable when
        # chunking granularity does not exactly match the ground-truth labels)
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

        # Latency (lower is better; use steady-state avg to exclude warmup turn)
        s_lat = sgas.get('avg_latency_excl_first_ms') or sgas.get('avg_latency_ms', 0)
        b_lat = baseline.get('avg_latency_excl_first_ms') or baseline.get('avg_latency_ms', 0)
        if s_lat < b_lat:
            verdicts['latency'] = f"S-GAS wins ({s_lat:.0f}ms vs {b_lat:.0f}ms)"
        elif b_lat < s_lat:
            verdicts['latency'] = f"Baseline wins ({b_lat:.0f}ms vs {s_lat:.0f}ms)"
        else:
            verdicts['latency'] = f"Tie ({s_lat:.0f}ms)"

        # Overall
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
        """Log comparison results to console."""
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

    def _upload_document(self, session_id: str, doc_path: Path, document_processor,
                         chunk_size: Optional[int] = None, chunk_overlap: Optional[int] = None):
        """Upload a document for benchmarking.

        When the scenario declares ``chunk_size`` / ``chunk_overlap``, the
        chunking service is temporarily patched so the production config is
        not affected and real-time chat keeps its own settings.
        """
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
