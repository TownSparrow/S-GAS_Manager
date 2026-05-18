"""S-GAS Manager — Entry Point. Initializes all services via constructor DI, registers FastAPI routes."""

import logging
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse

from config import Settings
from app.consts.defaults import UPLOADS_DIR, STATIC_DIR, API_VERSION
from app.models.api import SessionCreateRequest, SessionCreateResponse, SearchRequest, SearchResponse, ChatRequest, ChatResponse
from app.loaders.pdf_loader import PDFLoader
from app.loaders.text_loader import TextLoader
from app.loaders.docx_loader import DOCXLoader
from app.services.embedding_service import EmbeddingService
from app.services.vector_store_service import ChromaVectorStoreService
from app.services.chunking_service import ChunkingService
from app.services.document_processor_service import DocumentProcessorService
from app.services.graph_service import GraphService
from app.services.scoring_service import ScoringService
from app.services.swap_service import SwapService
from app.services.vllm_service import VLLMService
from app.services.chat_service import ChatService
from app.services.monitoring.kv_monitor import KVCacheMonitor
from app.services.testing.benchmark_runner import BenchmarkRunner
from app.controllers.health_controller import HealthController
from app.controllers.session_controller import SessionController
from app.controllers.document_controller import DocumentController
from app.controllers.search_controller import SearchController
from app.controllers.chat_controller import ChatController
from app.controllers.benchmark_controller import BenchmarkController

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_app(settings: Settings) -> FastAPI:
    sessions = {}

    # ── Services ──────────────────────────────────────────────────────
    embedding_service = EmbeddingService(
        model_name=settings['embeddings']['model'],
        api_base=settings['vllm']['api_base'],
        embedding_api_url=settings.get('embeddings', {}).get('api_url', ''),
    )
    vector_store_service = ChromaVectorStoreService(persist_directory=settings['database']['chroma_persist_dir'], collection_name=settings['database'].get('collection_name', 'documents'))
    chunking_service = ChunkingService(max_chunk_size=settings['chunking']['max_chunk_size'], overlap_size=settings['chunking']['overlap_size'], spacy_model=settings.get('graph', {}).get('spacy_model', 'ru_core_news_md'))
    loaders = {'.pdf': PDFLoader(), '.txt': TextLoader(), '.docx': DOCXLoader(), '.doc': DOCXLoader()}
    document_processor_service = DocumentProcessorService(vector_store=vector_store_service, chunker=chunking_service, embedding_service=embedding_service, loaders=loaders)

    graph_config = settings.get('graph', {})
    graph_service = GraphService(
        priority_model_for_en=graph_config.get('priority_model_for_en', 'en_core_web_sm'),
        priority_model_for_ru=graph_config.get('priority_model_for_ru', 'natasha'),
        priority_kw_extractor_for_ru=graph_config.get('priority_kw_extractor_for_ru', 'yake'),
        priority_kw_extractor_for_en=graph_config.get('priority_kw_extractor_for_en', 'yake'),
        use_gpu=False,
    )

    graph_service.set_embedding_service(embedding_service)
    graph_service.set_edge_top_p(graph_config.get('edge_top_p', 0.7))

    reranker_config = settings.get('reranker', {})
    retrieval_config = settings.get('retrieval', {})
    scoring_service = ScoringService(
        alpha=settings['graph']['alpha'],
        beta=settings['graph']['beta'],
        cross_encoder_model=reranker_config.get('cross_encoder_model', ''),
        cross_encoder_top_n=reranker_config.get('cross_encoder_top_n', 15),
        cross_encoder_weight=reranker_config.get('cross_encoder_weight', 0.0),
        semantic_anchor_threshold=retrieval_config.get('semantic_anchor_threshold', 0.75),
        min_return_chunks=retrieval_config.get('min_return_chunks', 3),
        enable_dynamic_weights=retrieval_config.get('enable_dynamic_weights', False),
        low_semantic_warning_threshold=retrieval_config.get('low_semantic_warning_threshold', 0.3),
    )
    swap_config = settings.get('swap', {})
    swap_service = SwapService(threshold=swap_config['threshold'], prefetch_count=swap_config['prefetch_count'], memory_check_interval_ms=swap_config['memory_check_interval'], max_gpu_memory_tokens=settings['vllm']['max_model_len'], force_offload_on_iteration=swap_config.get('force_offload_on_iteration', -1), retain_top_centrality_pct=swap_config.get('retain_top_centrality_pct', 0.15), eviction_ram_threshold_gb=swap_config.get('eviction_ram_threshold_gb', 12.0), proactive_offload=swap_config.get('proactive_offload', False), gpu_pressure_free_ratio=swap_config.get('gpu_pressure_free_ratio', 0.15), enabled=swap_config.get('enabled', False))
    vllm_service = VLLMService(api_base=settings['vllm']['api_base'], model_name=settings['vllm'].get('served_model_name', settings['vllm']['model_name']), default_temperature=settings['vllm']['temperature'], default_top_p=settings['vllm']['top_p'], default_max_tokens=settings['vllm']['max_tokens'], max_model_len=settings['vllm']['max_model_len'], request_timeout=settings['vllm'].get('request_timeout', 300))

    kv_monitor = KVCacheMonitor()

    chat_service = ChatService(embedding_service=embedding_service, vector_store=vector_store_service, chunker=chunking_service, graph_builder=graph_service, scorer=scoring_service, swap_manager=swap_service, vllm_client=vllm_service, prompt_config=settings.get('prompt', {}), model_name=settings['vllm']['model_name'], kv_monitor=kv_monitor, min_similarity_score=retrieval_config.get('min_similarity_score', 0.0), bm25_weight=retrieval_config.get('bm25_weight', 0.0), rrf_k=retrieval_config.get('rrf_k', 60), enable_sgas_filtering=retrieval_config.get('enable_sgas_filtering', True))
    chat_service._exclude_used_chunks = retrieval_config.get('exclude_used_chunks', False)

    benchmark_runner = BenchmarkRunner(chat_service=chat_service, document_processor=document_processor_service)

    # ── Controllers ───────────────────────────────────────────────────
    health_ctrl = HealthController(vllm_client=vllm_service, vector_store=vector_store_service, chunker=chunking_service, model_name=settings['vllm']['model_name'], sessions=sessions)
    session_ctrl = SessionController(sessions=sessions, document_processor=document_processor_service, chunker=chunking_service, vector_store=vector_store_service)
    document_ctrl = DocumentController(document_processor=document_processor_service, sessions=sessions)
    search_ctrl = SearchController(embedding_service=embedding_service, vector_store=vector_store_service, sessions=sessions)
    chat_ctrl = ChatController(chat_service=chat_service, sessions=sessions)
    benchmark_ctrl = BenchmarkController(benchmark_runner=benchmark_runner, sessions=sessions, document_processor=document_processor_service, graph_service=graph_service)

    # ── Lifespan ──────────────────────────────────────────────────────
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
        STATIC_DIR.mkdir(exist_ok=True)
        logger.info("S-GAS Manager started successfully")
        yield
        swap_service.cleanup()
        logger.info("S-GAS Manager shutdown complete")

    # ── FastAPI app ───────────────────────────────────────────────────
    app = FastAPI(title="S-GAS Manager API", description="Semantic-Graph Adaptive Swapping (S-GAS) for Small Language Models", version=API_VERSION, lifespan=lifespan)
    app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "http://localhost:8080", "http://127.0.0.1:8080", f"http://{settings['api']['host']}:{settings['api']['port']}"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    # ── Routes ────────────────────────────────────────────────────────

    @app.get("/")
    async def serve_web_client():
        idx = STATIC_DIR / "index.html"
        if idx.exists():
            return FileResponse(str(idx))
        return JSONResponse({
            "message": "S-GAS Manager API",
            "version": API_VERSION,
            "endpoints": [
                "/api/session/new", "/api/session/{session_id}/upload-document",
                "/api/session/{session_id}/search", "/api/session/{session_id}/chat",
                "/api/session/{session_id}/clear",
                "/api/session/{session_id}/info", "/api/session/{session_id}/documents",
                "/api/sgas-statistics", "/health",
                "/benchmark", "/api/benchmark/scenarios",
                "/api/benchmark/run/{scenario_name}",
                "/api/benchmark/generate-report/{scenario_name}",
                "/api/benchmark/results/{scenario_name}",
                "/graph", "/api/graph/data",
            ],
        })

    @app.get("/health")
    async def health_check():
        return await health_ctrl.health_check()

    @app.post("/api/session/new", response_model=SessionCreateResponse)
    async def create_new_session(request: SessionCreateRequest):
        return session_ctrl.create_session(user_id=request.user_id)

    @app.post("/api/session/{session_id}/upload-document")
    async def upload_document(session_id: str, file: UploadFile = File(...), document_type: str = "general"):
        return await document_ctrl.upload_document(session_id, file, document_type)

    @app.post("/api/session/{session_id}/search", response_model=SearchResponse)
    async def search_documents(session_id: str, request: SearchRequest):
        return await search_ctrl.search(session_id, request.query, request.top_k)

    @app.post("/api/session/{session_id}/chat", response_model=ChatResponse)
    async def chat_endpoint(session_id: str, request: ChatRequest):
        return await chat_ctrl.chat(session_id=session_id, message=request.message, use_rag=request.use_rag, n_chunks=request.n_chunks, temperature=request.temperature, top_p=request.top_p, max_tokens=request.max_tokens)

    @app.delete("/api/session/{session_id}/clear")
    async def clear_session(session_id: str):
        return session_ctrl.clear_session(session_id)

    @app.get("/api/session/{session_id}/info")
    async def get_session_info(session_id: str):
        return session_ctrl.get_session_info(session_id)

    @app.get("/api/session/{session_id}/documents")
    async def list_session_documents(session_id: str):
        return session_ctrl.list_session_documents(session_id)

    @app.get("/api/sgas-statistics")
    async def get_sgas_statistics():
        try:
            return JSONResponse(content={"graph": graph_service.get_graph_statistics(), "swap": swap_service.get_statistics(), "scorer": {"alpha": scoring_service.alpha, "beta": scoring_service.beta}})
        except Exception as e:
            return JSONResponse(content={"error": str(e)}, status_code=500)

    # ── Benchmark endpoints ───────────────────────────────────────────

    @app.get("/benchmark")
    async def benchmark_ui():
        return await benchmark_ctrl.benchmark_ui()

    @app.get("/api/benchmark/scenarios")
    async def list_scenarios():
        return await benchmark_ctrl.list_scenarios()

    @app.get("/api/benchmark/progress/{scenario_name}")
    async def get_benchmark_progress(scenario_name: str):
        return await benchmark_ctrl.get_progress(scenario_name)

    @app.post("/api/benchmark/run/{scenario_name}")
    async def run_benchmark(scenario_name: str, session_id: Optional[str] = None,
                            fresh_server: bool = False):
        return await benchmark_ctrl.run_benchmark(scenario_name, session_id,
                                                  fresh_server=fresh_server)

    @app.post("/api/benchmark/run/{scenario_name}/{mode}")
    async def run_single_benchmark(scenario_name: str, mode: str,
                                   fresh_server: bool = False):
        return await benchmark_ctrl.run_single_mode(scenario_name, mode,
                                                    fresh_server=fresh_server)

    @app.post("/api/benchmark/generate-report/{scenario_name}")
    async def generate_report(scenario_name: str):
        return await benchmark_ctrl.generate_report(scenario_name)

    @app.get("/api/benchmark/results/{scenario_name}")
    async def get_benchmark_results(scenario_name: str):
        return await benchmark_ctrl.get_benchmark_results(scenario_name)

    @app.get("/api/benchmark/download/{filename}")
    async def download_benchmark_file(filename: str):
        return await benchmark_ctrl.download_result_file(filename)

    @app.get("/graph")
    async def graph_visualization_ui():
        return await benchmark_ctrl.graph_ui()

    @app.get("/api/graph/data")
    async def get_graph_data():
        return await benchmark_ctrl.get_graph_data()

    # ── Error handlers ────────────────────────────────────────────────

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request, exc):
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

    @app.exception_handler(Exception)
    async def general_exception_handler(request, exc):
        logger.error(f"Unhandled exception: {exc}")
        return JSONResponse(status_code=500, content={"detail": "Internal server error"})

    return app


settings = Settings()
app = create_app(settings)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("run:app", host=settings['api']['host'], port=settings['api']['port'], reload=True)
