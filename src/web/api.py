from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import httpx
import logging
import shutil
from pathlib import Path
from datetime import datetime, timezone
from contextlib import asynccontextmanager
import time

# Configs and modules
from src.config import settings
from src.modules.retrieval.vector_store import ChromaVectorStore
from src.modules.retrieval.chunking import SemanticChunker
from src.modules.retrieval.document_processor import DocumentProcessor
from src.modules.retrieval.embedder import get_embeddings
from src.modules.graph.graph_builder import KnowledgeGraphBuilder
from src.core.scoring import HybridScorer
from src.modules.swap.swap_manager import SwapManager

# -----------------------------------------------------------------------------
# Init of app and basic services
# -----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

UPLOADS_DIR = Path("data/uploads")
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(exist_ok=True)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Services init with settings['...']
    vector_store = ChromaVectorStore(
        persist_directory=settings['database']['chroma_persist_dir'],
        collection_name=settings['database'].get('collection_name', 'documents'),
    )
    chunker = SemanticChunker(
        max_chunk_size=settings['chunking']['max_chunk_size'],
        overlap_size=settings['chunking']['overlap_size'],
    )
    try:
        document_processor = DocumentProcessor(vector_store=vector_store, chunker=chunker)
        
        # 1. Graph Builder
        graph_builder = KnowledgeGraphBuilder(
            spacy_model="ru_core_news_md", # or "en_core_web_sm"
            use_gpu=False # True if GPU is free
        )
        logger.info("✅ KnowledgeGraphBuilder is initialized")

        # 2. Hybrid Scorer
        hybrid_scorer = HybridScorer(
            alpha=settings['graph']['alpha'],
            beta=settings['graph']['beta']
        )
        logger.info("✅ HybridScorer is initialized")

        # 3. Swap Manager
        swap_manager = SwapManager(
            threshold=settings['swap']['threshold'],
            prefetch_count=settings['swap']['prefetch_count'],
            memory_check_interval_ms=settings['swap']['memory_check_interval'],
            max_gpu_memory_tokens=settings['vllm']['max_model_len']
        )
        logger.info("✅ SwapManager is initialized")

        # Standart components for inference
        app.state.vector_store = vector_store
        app.state.chunker = chunker
        app.state.document_processor = document_processor

        # Algorithm components
        app.state.graph_builder = graph_builder
        app.state.hybrid_scorer = hybrid_scorer
        app.state.swap_manager = swap_manager

        logger.info("✅ All S-GAS components initialized with config parameters")
    except Exception as e:
        logger.error(f"❌ Failed to initialize components: {e}")
    yield
    
    # Cleanup
    app.state.vector_store = None
    app.state.chunker = None
    app.state.document_processor = None
    app.state.graph_builder = None
    app.state.hybrid_scorer = None
    app.state.swap_manager = None

app = FastAPI(
    title="S-GAS Manager API",
    version="0.1.3",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000", "http://127.0.0.1:3000",
        "http://localhost:8080", "http://127.0.0.1:8080",
        f"http://{settings['api']['host']}:{settings['api']['port']}",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# -----------------------------------------------------------------------------
# Models of requests and responses
# -----------------------------------------------------------------------------
class ChatRequest(BaseModel):
    message: str
    use_rag: bool = True
    n_chunks: int = 5
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None

class ChatResponse(BaseModel):
    response: str
    metadata: Dict[str, Any] = {}

# -----------------------------------------------------------------------------
# Additional functions
# -----------------------------------------------------------------------------
def build_prompt(context_chunks: List[Dict[str, Any]], user_message: str) -> str:
    if context_chunks:
        context_text = "\n\n".join([c.get("text") or c.get("document") or "" for c in context_chunks if c])
        return (
            f"Context from the knowledge base:\n{context_text}\n\n"
            f"User's request: {user_message}\n\n"
            f"Instructions: Analyze the context and give a brief, precise, and specific answer based on the fragments provided. If the context lacks information, state so and respond briefly using general knowledge.\n\n"
            f"Response:"
        )
    else:
        return (
            f"User's request: {user_message}\n\n"
            f"Give a short and precise answer.\n\n"
            f"Response:"
        )

async def call_vllm_chat(prompt: str, overrides: Dict[str, Any]) -> Dict[str, Any]:
    api_base = settings['vllm']['api_base']
    model_name = settings['vllm']['model_name']
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": overrides.get("temperature", settings['vllm']['temperature']),
        "max_tokens": overrides.get("max_tokens", settings['vllm']['max_tokens']),
        "top_p": overrides.get("top_p", settings['vllm']['top_p']),
        "stop": ["\n\nRequest:", "\n\nUser:", "\n\nAssistant:"],
        "stream": False,
    }
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(f"{api_base}/chat/completions", json=payload)
        if resp.status_code != 200:
            raise HTTPException(status_code=502, detail=f"vLLM error {resp.status_code}: {resp.text}")
        data = resp.json()
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices or "message" not in choices[0]:
            raise HTTPException(status_code=502, detail=f"vLLM malformed response: {data}")
        if "content" not in choices[0]["message"]:
            raise HTTPException(status_code=502, detail=f"vLLM response without content: {data}")
        return data

# -----------------------------------------------------------------------------
# Endpoints
# -----------------------------------------------------------------------------
@app.get("/")
async def serve_web_client():
    static_index = STATIC_DIR / "index.html"
    if static_index.exists():
        return FileResponse(str(static_index))
    return JSONResponse(
        {
            "message": "S-GAS Manager API",
            "version": "1.0",
            "endpoints": ["/api/chat", "/health", "/api/upload-document", "/api/documents"],
            "web_client": "No files in src/web/static/",
        },
        status_code=200,
    )

@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        logger.info(f"Request received: {request.message}")

        # 1) Embedding of request
        try:
            query_embedding = await get_embeddings([request.message])
            query_embedding_vec = query_embedding[0] if len(query_embedding.shape) > 1 else query_embedding
            logger.info(f"✅ Embedding received. Shape: {query_embedding.shape}")
        except (AttributeError, IndexError, TypeError) as e:
            logger.error(f"❌ Failed to process embedding shape: {e}")
            raise HTTPException(status_code=500, detail="Embedding processing failed")

        # 2) Search of context (optionally)
        context_chunks: List[Dict[str, Any]] = []
        reranked_chunks: List[Dict[str, Any]] = []

        if request.use_rag:
            try:
                vector_store = app.state.vector_store
                if vector_store is not None:
                    # Recieving more chunks for the next reranking
                    initial_top_k = max(20, request.n_chunks * 3)
                    
                    context_chunks = await vector_store.search(
                        query_embedding,
                        top_k = initial_top_k, #top_k=max(1, int(request.n_chunks)),
                    )
                    logger.info(f"Retrieved {len(context_chunks)} starting context chunks")
                else:
                    logger.warning("⚠️ Vector store is None, skipping RAG")
            except Exception as e:
                logger.warning(f"❌ RAG search failed: {e}")
                context_chunks = []

        reranked_chunks = context_chunks.copy()

        # 3) S-GAS Algorithm: Reranking
        if request.use_rag and len(context_chunks) > 0:
            # Algorithm Stage 1. Building graph and calculating distances
            try:
                logger.info("🔄 S-GAS Algorithm Stage 1: Building knowledge-graph...")

                # Recieving embeddings of all chunks
                chunk_texts = [c.get('text', '') for c in context_chunks]
                chunk_embeddings = await get_embeddings(chunk_texts)

                # Building knowledge-graph
                graph_builder = app.state.graph_builder
                
                if graph_builder is None:
                    raise ValueError("Graph builder is not initialized")
                
                graph = graph_builder.build_graph(context_chunks, chunk_embeddings)

                # Calculating graph distances
                chunk_ids = [c.get('id', f'chunk_{i}') for i, c in enumerate(context_chunks)]
                graph_distances = graph_builder.compute_graph_distances(
                    request.message, 
                    chunk_ids
                )

                logger.info(f"✅ Knowledge-graph built with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges")
                logger.info(f"✅ Graph distances calculated for {len(graph_distances)} chunks")
            except Exception as e:
                logger.warning(f"❌ Knowledge-graph building failed: {e}")
                # Continue with original context chunks if graph building fails
                graph_distances = {}
                chunk_embeddings = await get_embeddings(chunk_texts) if 'chunk_texts' in locals() else None

            # Algorithm Stage 2. Reranking
            try:
                logger.info("🔄 S-GAS Algorithm Stage 2: Reranking chunks via S-GAS...")

                # Calculating scores of chunks
                hybrid_scorer = app.state.hybrid_scorer
                if hybrid_scorer is None:
                    raise ValueError("Hybrid scorer is not initialized")

                if 'chunk_embeddings' not in locals() or chunk_embeddings is None:
                    chunk_texts = [c.get('text', '') for c in context_chunks]
                    chunk_embeddings = await get_embeddings(chunk_texts)

                if not graph_distances:
                    logger.warning("⚠️ Graph distances are empty, using default distances")
                    graph_distances = {c.get('id', f'chunk_{i}'): 1000.0 for i, c in enumerate(context_chunks)}

                # Reranking the chunks
                reranked_chunks = hybrid_scorer.rerank_chunks(
                    query_embedding_vec,
                    context_chunks,
                    chunk_embeddings,
                    graph_distances,
                    top_k=request.n_chunks
                )

                logger.info(f"✅ Chunks were reranked via S-GAS Algorithm. Top-{request.n_chunks} were selected.")
            except Exception as e:
                logger.warning(f"❌ Reranking process failed: {e}")
                logger.info("⚠️ Using original context chunks as fallback")

            # Algorithm Stage 3. Swapping data
            try:
                logger.info("🔄 S-GAS Algorithm Stage 3. Managing memory swap...")

                swap_manager = app.state.swap_manager
                if swap_manager is None:
                    raise ValueError("Swap manager is not initialized")
                        
                # Initializing chunks in swap manager (for the first time)
                if not hasattr(app.state, '_swap_initialized'):
                    swap_manager.initialize_chunks(reranked_chunks)
                    app.state._swap_initialized = True
                    logger.debug(f"🔧 Swap manager initialized with {len(reranked_chunks)} chunks")
                        
                # Updating the prefetch buffer
                swap_manager.update_prefetch_buffer(reranked_chunks)
                logger.debug(f"🔧 Prefetch buffer updated with {len(reranked_chunks)} chunks")
                        
                # Making a decision about swapping
                current_context_tokens = sum(
                    c.get('metadata', {}).get('chunk_size', 500) 
                    for c in reranked_chunks
                )
                        
                swap_decision = swap_manager.decide_swap_action(
                    reranked_chunks,
                    current_context_tokens
                )
                        
                # Perform swapping (if necessary)
                swap_manager.execute_swap_decision(swap_decision)
                        
                logger.info(f"✅ Swapping completed: {swap_decision.get('action', 'none')}")
            except Exception as e:
                logger.warning(f"❌ Swapping failed: {e}")
                logger.info("⚠️ Continuing without swap optimization")

        # 4) Generate prompt with context
        final_context = reranked_chunks if reranked_chunks else context_chunks
        enhanced_prompt = build_prompt(final_context, request.message)
        logger.debug(f"📝 Prompt built with {len(final_context)} context chunks")

        # 5) Calling vLLM
        try:
            overrides = {
                "temperature": request.temperature,
                "top_p": request.top_p,
                "max_tokens": request.max_tokens,
            }
            
            # Measuring inference time for swap_manager
            inference_start = time.time()
            data = await call_vllm_chat(enhanced_prompt, overrides)
            inference_time = time.time() - inference_start
            
            # Recording the T_comp metric
            swap_manager = app.state.swap_manager
            if swap_manager is not None:
                swap_manager.record_compute_time(inference_time)
            
            logger.info(f"⏱️⏱️⏱️ Inference completed in {inference_time:.3f}s")
            
        except Exception as e:
            logger.error(f"❌ vLLM inference failed: {e}")
            raise HTTPException(status_code=500, detail="Inference generation failed")

        # 6) Response
        try:
            msg = data["choices"][0]["message"]
            answer = msg.get("content", "")
            usage = data.get("usage", {})
        except (KeyError, IndexError, TypeError) as e:
            logger.error(f"❌ Failed to parse vLLM response: {e}")
            raise HTTPException(status_code=500, detail="Failed to parse model response")
        
        # 7) S-GAS Statistics
        sgas_stats = {}
        try:
            graph_builder = app.state.graph_builder
            swap_manager = app.state.swap_manager
            
            if graph_builder is not None:
                sgas_stats['graph_stats'] = graph_builder.get_graph_statistics()
            else:
                sgas_stats['graph_stats'] = {"error": "Graph builder not initialized"}
            
            if swap_manager is not None:
                sgas_stats['swap_stats'] = swap_manager.get_statistics()
            else:
                sgas_stats['swap_stats'] = {"error": "Swap manager not initialized"}
                
            logger.debug("✅ S-GAS statistics collected")
            
        except Exception as e:
            logger.warning(f"❌ Failed to collect S-GAS statistics: {e}")
            sgas_stats = {
                "error": str(e),
                "graph_stats": None,
                "swap_stats": None
            }
        
        # 8) Return response
        return ChatResponse(
            response=answer,
            metadata={
                "usage": usage,
                "model_used": settings['vllm']['model_name'],
                "use_rag": request.use_rag,
                "context_chunks_used": len(final_context),
                "inference_time_sec": inference_time,
                "sgas_statistics": sgas_stats,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"❌ Unexpected error in /api/chat: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/api/upload-document")
async def upload_document(file: UploadFile = File(...)):
    if not file.filename or '..' in file.filename or '/' in file.filename:
        raise HTTPException(status_code=400, detail="Invalid filename")
    file_path = UPLOADS_DIR / file.filename
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        logger.info(f"✅ File uploaded: {file.filename} ({file_path.stat().st_size} bytes)")
        result = await app.state.document_processor.process_document(file_path)
        return result
    except Exception as e:
        logger.error(f"❌ Error uploading document: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.get("/api/documents")
async def list_documents():
    documents = [
        {"filename": f.name, "size": f.stat().st_size, "modified": f.stat().st_mtime}
        for f in UPLOADS_DIR.iterdir()
        if f.is_file()
    ]
    return {"documents": documents}

@app.delete("/api/documents/{filename}")
async def delete_document(filename: str):
    file_path = UPLOADS_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Document not found")
    try:
        file_path.unlink()
        logger.info(f"✅ Deleted document: {filename}")
        return {"message": f"Document {filename} deleted successfully"}
    except Exception as e:
        logger.error(f"❌ Error deleting document: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete document")

@app.get("/health")
async def health_check():
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{settings['vllm']['api_base']}/models")
        vllm_status = "healthy" if response.status_code == 200 else "unhealthy"
    except Exception:
        vllm_status = "unavailable"
    vector_store_status = "ready" if app.state.vector_store is not None else "not_initialized"
    return {
        "status": "healthy",
        "vllm_status": vllm_status,
        "vector_store_status": vector_store_status,
        "model": settings['vllm']['model_name'],
        "time": datetime.now(timezone.utc).isoformat(),
    }

@app.get("/api/sgas-statistics")
async def get_sgas_statistics():
    try:
        graph_builder = app.state.graph_builder
        hybrid_scorer = app.state.hybrid_scorer
        swap_manager = app.state.swap_manager
        
        statistics = {
            "graph": graph_builder.get_graph_statistics() if graph_builder else {},
            "swap": swap_manager.get_statistics() if swap_manager else {},
            "scorer": {
                "alpha": hybrid_scorer.alpha if hybrid_scorer else 0.6,
                "beta": hybrid_scorer.beta if hybrid_scorer else 0.4
            }
        }
        
        return JSONResponse(content=statistics, status_code=200)
        
    except Exception as e:
        logger.error(f"❌ Error of recieving the statistics of S-GAS Algorithm: {e}")
        return JSONResponse(
            content={"error": str(e)}, 
            status_code=500
        )