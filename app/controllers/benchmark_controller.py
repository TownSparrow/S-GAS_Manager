import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

from fastapi import HTTPException
from fastapi.responses import FileResponse, JSONResponse

from app.consts.defaults import STATIC_DIR
from app.utils.serialization import serialize_json_safe

logger = logging.getLogger(__name__)


class BenchmarkController:
    def __init__(self, benchmark_runner, sessions: Dict[str, Any], document_processor, graph_service=None):
        self._benchmark_runner = benchmark_runner
        self._sessions = sessions
        self._document_processor = document_processor
        self._graph_service = graph_service

    async def benchmark_ui(self):
        static_benchmark = STATIC_DIR / "benchmark.html"
        if static_benchmark.exists():
            return FileResponse(str(static_benchmark))
        return JSONResponse(
            {"message": "Benchmark page not found", "hint": "Create static/benchmark.html"},
            status_code=404,
        )

    async def list_scenarios(self):
        try:
            from app.services.testing.scenario_loader import ScenarioLoader
            scenario_loader = ScenarioLoader(scenarios_dir="tests/scenarios")
            scenarios = scenario_loader.list_available_scenarios()
            return {"status": "success", "scenarios": scenarios, "count": len(scenarios)}
        except Exception as e:
            logger.error(f"Failed to list scenarios: {e}")
            return {"status": "error", "message": str(e), "scenarios": []}

    async def get_progress(self, scenario_name: str):
        try:
            return self._benchmark_runner.get_progress(scenario_name)
        except Exception as e:
            logger.error(f"Failed to get benchmark progress: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    async def run_benchmark(self, scenario_name: str, session_id: Optional[str] = None,
                            fresh_server: bool = False):
        try:
            result = await self._benchmark_runner.run_scenario(
                scenario_name=scenario_name,
                sessions=self._sessions,
                document_processor=self._document_processor,
                session_id=session_id,
                fresh_server=fresh_server,
            )
            return {
                "status": result.get("status", "success"),
                "scenario": result['scenario'],
                "modes": {
                    mode: {
                        "label": mode_result.get("label", mode),
                        "session_id": mode_result["session_id"],
                        "summary": mode_result["summary"],
                        "files": {"csv": mode_result["csv_file"], "json": mode_result["json_file"]},
                    }
                    for mode, mode_result in result.get("modes", {}).items()
                },
                "sgas": {
                    "session_id": result['sgas']['session_id'],
                    "summary": result['sgas']['summary'],
                    "files": {"csv": result['sgas']['csv_file'], "json": result['sgas']['json_file']},
                },
                "baseline": {
                    "session_id": result['baseline']['session_id'],
                    "summary": result['baseline']['summary'],
                    "files": {"csv": result['baseline']['csv_file'], "json": result['baseline']['json_file']},
                },
                "comparison": result['comparison'],
                "report_file": result.get('docx_report_file', ''),
                "files": {
                    "comparison": result.get('comparison_file', ''),
                    "docx_report": result.get('docx_report_file', ''),
                },
            }
        except Exception as e:
            logger.error(f"Benchmark failed: {e}")
            import traceback
            logger.error(f"Traceback:\n{traceback.format_exc()}")
            raise HTTPException(status_code=500, detail=str(e))

    async def run_single_mode(self, scenario_name: str, mode: str = "sgas",
                              fresh_server: bool = False):
        """Running benchmark in a single mode only."""
        valid_modes = ("baseline", "hybrid_rag", "sgas_no_filtering", "sgas")
        if mode not in valid_modes:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid mode: {mode}. Use one of: {', '.join(valid_modes)}.",
            )
        try:
            result = await self._benchmark_runner.run_single_mode(
                scenario_name=scenario_name,
                sessions=self._sessions,
                mode=mode,
                document_processor=self._document_processor,
                fresh_server=fresh_server,
            )
            return {
                "status": result.get("status", "success"),
                "scenario": result['scenario'],
                "mode": result['mode'],
                "label": result.get('label', result['mode']),
                "session_id": result['session_id'],
                "summary": result['summary'],
                "files": {"csv": result['csv_file'], "json": result['json_file']},
            }
        except Exception as e:
            logger.error(f"Single-mode benchmark failed: {e}")
            import traceback
            logger.error(f"Traceback:\n{traceback.format_exc()}")
            raise HTTPException(status_code=500, detail=str(e))

    async def generate_report(self, scenario_name: str):
        try:
            from app.services.testing.report_generator import ReportGenerator

            results_dir = Path("logs/benchmarks")
            csv_files = list(results_dir.glob(f"{scenario_name}_*.csv"))
            if not csv_files:
                raise HTTPException(status_code=404, detail=f"No results found for scenario {scenario_name}")

            latest_csv = max(csv_files, key=lambda f: f.stat().st_mtime)
            generator = ReportGenerator(results_dir=str(results_dir))
            plots_dir = generator.generate_plots(str(latest_csv))

            json_file = Path(str(latest_csv).replace('.csv', '.json'))
            if json_file.exists():
                html_file = generator.generate_html_report(str(json_file), str(plots_dir))
                return {
                    "status": "success", "html_report": html_file,
                    "plots_dir": str(plots_dir),
                    "plots": [f.name for f in plots_dir.glob("*.png")],
                }
            return {
                "status": "success", "plots_dir": str(plots_dir),
                "plots": [f.name for f in plots_dir.glob("*.png")],
            }
        except Exception as e:
            logger.error(f"Report generation failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    async def graph_ui(self):
        from app.consts.defaults import STATIC_DIR
        graph_page = STATIC_DIR / "graph.html"
        if graph_page.exists():
            return FileResponse(str(graph_page))
        return JSONResponse(
            {"message": "Graph visualization page not found", "hint": "Create static/graph.html"},
            status_code=404,
        )

    async def get_graph_data(self):
        """Returning the current knowledge graph as JSON for the visualization client."""
        if self._graph_service is None:
            return JSONResponse(
                {"status": "error", "message": "Graph service not available"},
                status_code=503,
            )
        try:
            data = self._graph_service.export_graph_info()
            nodes = data.get('nodes', [])

            # If the live graph is empty, try the last persisted snapshot.
            if not nodes:
                snapshot_path = Path("logs/benchmarks/latest_sgas_graph.json")
                if snapshot_path.exists():
                    try:
                        with open(snapshot_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        nodes = data.get('nodes', [])
                        logger.info(
                            f"Live graph empty — serving snapshot: {len(nodes)} nodes "
                            f"from {snapshot_path}"
                        )
                    except Exception as snap_err:
                        logger.warning(f"Could not read graph snapshot: {snap_err}")

            stats = data.get('statistics', {})
            return serialize_json_safe({
                "status": "success",
                "statistics": stats,
                "nodes": nodes,
                "edges": data.get('edges', []),
                "node_count": len(nodes),
                "edge_count": len(data.get('edges', [])),
            })
        except Exception as e:
            logger.error(f"Graph data export failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    async def get_benchmark_results(self, scenario_name: str):
        try:
            results_dir = Path("logs/benchmarks")

            # Try comparison file first (newest format)
            comparison_files = list(results_dir.glob(f"{scenario_name}_comparison_*.json"))
            if comparison_files:
                latest = max(comparison_files, key=lambda f: f.stat().st_mtime)
                with open(latest, 'r', encoding='utf-8') as f:
                    comparison = json.load(f)
                return {"status": "success", "scenario": scenario_name, "type": "comparison", "results": comparison}

            # Fallback to any JSON result
            json_files = list(results_dir.glob(f"{scenario_name}_*.json"))
            if not json_files:
                return {"status": "not_found", "message": f"No results found for {scenario_name}"}

            latest_json = max(json_files, key=lambda f: f.stat().st_mtime)
            with open(latest_json, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return {"status": "success", "scenario": scenario_name, "type": "single", "results": data}
        except Exception as e:
            logger.error(f"Failed to get results: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    async def download_result_file(self, filename: str):
        """Safely download a generated benchmark artifact from logs/benchmarks."""
        try:
            results_dir = Path("logs/benchmarks").resolve()
            file_path = (results_dir / filename).resolve()
            if results_dir not in file_path.parents and file_path != results_dir:
                raise HTTPException(status_code=400, detail="Invalid benchmark file path")
            if not file_path.exists() or not file_path.is_file():
                raise HTTPException(status_code=404, detail=f"Benchmark file not found: {filename}")
            return FileResponse(str(file_path), filename=file_path.name)
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to download benchmark file: {e}")
            raise HTTPException(status_code=500, detail=str(e))
