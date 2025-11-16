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

# ============================================================================
# Init of app and basic services
# ============================================================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

UPLOADS_DIR = Path("data/uploads")
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(exist_ok=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize services at startup"""
    logger.info("🚀 Starting S-GAS Manager API initialization...")
    
    try:
        # Initialize vector store
        vector_store = ChromaVectorStore(
            persist_directory=settings['database']['chroma_persist_dir'],
            collection_name=settings['database'].get('collection_name', 'documents'),
        )
        logger.info("✅ ChromaVectorStore initialized")
        
        # Initialize chunker with adaptive tracking
        chunker = SemanticChunker(
            max_chunk_size=settings['chunking']['max_chunk_size'],
            overlap_size=settings['chunking']['overlap_size'],
        )
        logger.info("✅ SemanticChunker initialized")
        
        # Initialize document processor
        document_processor = DocumentProcessor(vector_store=vector_store, chunker=chunker)
        logger.info("✅ DocumentProcessor initialized")
        
        # Initialize Graph Builder
        graph_builder = KnowledgeGraphBuilder(
            spacy_model="ru_core_news_md", # or "en_core_news_md"
            use_gpu=False
        )
        logger.info("✅ KnowledgeGraphBuilder initialized")
        
        # Initialize Hybrid Scorer
        hybrid_scorer = HybridScorer(
            alpha=settings['graph']['alpha'],
            beta=settings['graph']['beta']
        )
        logger.info("✅ HybridScorer initialized")
        
        # Initialize Swap Manager
        swap_manager = SwapManager(
            threshold=settings['swap']['threshold'],
            prefetch_count=settings['swap']['prefetch_count'],
            memory_check_interval_ms=settings['swap']['memory_check_interval'],
            max_gpu_memory_tokens=settings['vllm']['max_model_len']
        )
        logger.info("✅ SwapManager initialized")
        
        # Store all components in app.state
        app.state.vector_store = vector_store
        app.state.chunker = chunker
        app.state.document_processor = document_processor
        app.state.graph_builder = graph_builder
        app.state.hybrid_scorer = hybrid_scorer
        app.state.swap_manager = swap_manager
        
        # Initialize iteration tracking
        app.state.current_iteration = 0
        app.state.session_chunks_explored = 0
        app.state._swap_initialized = False
        
        logger.info("✅ All S-GAS components initialized successfully!")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize components: {e}")
        raise
    
    yield
    
    # Cleanup on shutdown
    logger.info("🛑 Cleaning up resources...")
    app.state.vector_store = None
    app.state.chunker = None
    app.state.document_processor = None
    app.state.graph_builder = None
    app.state.hybrid_scorer = None
    app.state.swap_manager = None


# Create FastAPI app with lifespan
app = FastAPI(
    title="S-GAS Manager API",
    description="Semantic-Graph Adaptive Swapping (S-GAS) for Small Language Models",
    version="0.1.4",
    lifespan=lifespan,
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8080",
        "http://127.0.0.1:8080",
        f"http://{settings['api']['host']}:{settings['api']['port']}",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# ============================================================================
# Data Models
# ============================================================================


class ChatRequest(BaseModel):
    """Chat request model"""
    message: str
    use_rag: bool = True
    n_chunks: int = 5
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None


class ChatResponse(BaseModel):
    """Chat response model"""
    response: str
    metadata: Dict[str, Any] = {}

# ============================================================================
# Helper Functions
# ============================================================================


def build_prompt(context_chunks: List[Dict[str, Any]], user_message: str) -> str:
    """Build prompt with context"""
    if context_chunks:
        context_text = "\n\n".join([
            c.get("text") or c.get("document") or ""
            for c in context_chunks if c
        ])
        return (
            f"Context from the knowledge base:\n{context_text}\n\n"
            f"User's request: {user_message}\n\n"
            f"Instructions: Analyze the context and give a brief, precise, and specific answer based on the fragments provided. "
            f"If the context lacks information, state so and respond briefly using general knowledge. Use the same language as the user uses.\n\n"
            f"Response:"
        )
    else:
        return (
            f"User's request: {user_message}\n\n"
            f"Give a short and precise answer. Use the same language as the user uses.\n\n"
            f"Response:"
        )


async def call_vllm_chat(prompt: str, overrides: Dict[str, Any]) -> Dict[str, Any]:
    """Call vLLM API"""
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
            raise HTTPException(
                status_code=502,
                detail=f"vLLM error {resp.status_code}: {resp.text}"
            )
        
        data = resp.json()
        choices = data.get("choices")
        
        if not isinstance(choices, list) or not choices or "message" not in choices[0]:
            raise HTTPException(
                status_code=502,
                detail=f"vLLM malformed response: {data}"
            )
        
        if "content" not in choices[0]["message"]:
            raise HTTPException(
                status_code=502,
                detail=f"vLLM response without content: {data}"
            )
        
        return data

# ============================================================================
# API Endpoints
# ============================================================================


@app.get("/")
async def serve_web_client():
    """Serve web client or show API info"""
    static_index = STATIC_DIR / "index.html"
    if static_index.exists():
        return FileResponse(str(static_index))
    
    return JSONResponse(
        {
            "message": "S-GAS Manager API",
            "version": "0.1.4",
            "endpoints": [
                "/api/chat",
                "/health",
                "/api/upload-document",
                "/api/documents",
                "/api/sgas-statistics"
            ],
            "web_client": "No files in src/web/static/",
        },
        status_code=200,
    )


@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Main chat endpoint with S-GAS adaptive chunking
    
    Implements 5-step algorithm:
    1. Determine excluded chunks from previous iterations
    2. Search for new chunks excluding previously used ones
    3. Evaluate and rank chunks using S-GAS algorithm
    4. Manage memory swapping
    5. Mark used chunks for next iteration
    """
    try:
        # ================================================================
        # STEP 1: Increment iteration counter and get excluded chunks
        # ================================================================
        app.state.current_iteration += 1
        iteration = app.state.current_iteration
        
        logger.info(f"🔄 Iteration {iteration}: Query: {request.message[:50]}...")
        
        excluded_chunks = []
        if iteration > 1:
            excluded_chunks = list(app.state.chunker.get_excluded_chunk_ids())
            logger.info(f"Excluding {len(excluded_chunks)} previously used chunks")
        
        # ================================================================
        # STEP 2: Search for chunks with exclusion
        # ================================================================
        
        query_embedding = await get_embeddings([request.message])
        query_embedding_vec = query_embedding[0] if len(query_embedding.shape) > 1 else query_embedding
        
        context_chunks = []
        
        if request.use_rag:
            try:
                vector_store = app.state.vector_store
                
                initial_top_k = max(20, request.n_chunks * 3)
                
                # Search for more candidates if we need to exclude some
                if excluded_chunks:
                    search_k = initial_top_k * 2
                else:
                    search_k = initial_top_k
                
                candidates = await vector_store.search(
                    query_embedding,
                    top_k=search_k,
                )
                
                # Filter out previously used chunks
                if excluded_chunks:
                    excluded_set = set(excluded_chunks)
                    context_chunks = [c for c in candidates if c['id'] not in excluded_set]
                    context_chunks = context_chunks[:initial_top_k]
                    logger.info(f"✅ Found {len(context_chunks)} NEW chunks (filtered from {len(candidates)} candidates)")
                else:
                    context_chunks = candidates[:initial_top_k]
                    logger.info(f"✅ Found {len(context_chunks)} chunks (iteration 1 - full search)")
                
            except Exception as e:
                logger.warning(f"❌ RAG search failed: {e}")
        
        # ================================================================
        # STEP 3: S-GAS Algorithm - Graph building and ranking
        # ================================================================
        
        reranked_chunks = context_chunks.copy()
        
        if request.use_rag and len(context_chunks) > 0:
            
            # --- Stage 1: Build knowledge graph ---
            try:
                logger.info("🔄 Stage 1: Building knowledge graph...")
                
                chunk_texts = [c.get('text', '') for c in context_chunks]
                chunk_embeddings = await get_embeddings(chunk_texts)
                
                graph_builder = app.state.graph_builder
                if graph_builder is None:
                    raise ValueError("Graph builder not initialized")
                
                graph = graph_builder.build_graph(context_chunks, chunk_embeddings)
                
                chunk_ids = [c.get('id', f'chunk_{i}') for i, c in enumerate(context_chunks)]
                graph_distances = graph_builder.compute_graph_distances(
                    request.message,
                    chunk_ids
                )
                
                logger.info(f"✅ Graph built: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
                
            except Exception as e:
                logger.warning(f"❌ Graph building failed: {e}")
                graph_distances = {}
            
            # --- Stage 2: Hybrid reranking ---
            try:
                logger.info("🔄 Stage 2: Hybrid reranking...")
                
                hybrid_scorer = app.state.hybrid_scorer
                if hybrid_scorer is None:
                    raise ValueError("Hybrid scorer not initialized")
                
                if 'chunk_embeddings' not in locals() or chunk_embeddings is None:
                    chunk_texts = [c.get('text', '') for c in context_chunks]
                    chunk_embeddings = await get_embeddings(chunk_texts)
                
                if not graph_distances:
                    logger.warning("⚠️ Graph distances are empty, using default distances")
                    graph_distances = {
                        c.get('id', f'chunk_{i}'): 1000.0
                        for i, c in enumerate(context_chunks)
                    }
                
                reranked_chunks = hybrid_scorer.rerank_chunks(
                    query_embedding_vec,
                    context_chunks,
                    chunk_embeddings,
                    graph_distances,
                    top_k=request.n_chunks
                )
                
                logger.info(f"✅ Reranked to top {request.n_chunks}")
                
            except Exception as e:
                logger.warning(f"❌ Reranking failed: {e}")
        
        # ================================================================
        # STEP 4: Memory management and swapping
        # ================================================================
        
        final_context = reranked_chunks if reranked_chunks else context_chunks
        selected_ids = [c.get('id', '') for c in final_context]
        
        try:
            logger.info("🔄 Stage 3: Memory management...")
            
            swap_manager = app.state.swap_manager
            
            if not app.state._swap_initialized:
                swap_manager.initialize_chunks(final_context)
                app.state._swap_initialized = True
            
            swap_manager.update_prefetch_buffer(final_context)
            
            current_context_tokens = sum(
                c.get('metadata', {}).get('chunk_size', 500)
                for c in final_context
            )
            
            swap_decision = swap_manager.decide_swap_action(
                final_context,
                current_context_tokens
            )
            
            swap_manager.execute_swap_decision(swap_decision)
            
            logger.info(f"✅ Swap action: {swap_decision.get('action', 'none')}")
            
        except Exception as e:
            logger.warning(f"❌ Swap management failed: {e}")
        
        # ================================================================
        # STEP 5: Mark used chunks for next iteration
        # ================================================================
        
        app.state.chunker.mark_chunks_used(selected_ids, iteration)
        
        stats = app.state.chunker.get_statistics()
        logger.info(f"📊 Coverage: {stats['coverage_percent']:.1f}% ({stats['used_chunks']}/{stats['total_chunks_in_pool']})")
        
        # ================================================================
        # Inference
        # ================================================================
        
        enhanced_prompt = build_prompt(final_context, request.message)
        
        overrides = {
            "temperature": request.temperature,
            "top_p": request.top_p,
            "max_tokens": request.max_tokens,
        }
        
        inference_start = time.time()
        data = await call_vllm_chat(enhanced_prompt, overrides)
        inference_time = time.time() - inference_start
        
        swap_manager = app.state.swap_manager
        if swap_manager is not None:
            swap_manager.record_compute_time(inference_time)
        
        logger.info(f"✅ Inference complete: {inference_time:.3f}s")
        
        # Extract response
        msg = data["choices"][0]["message"]
        answer = msg.get("content", "")
        usage = data.get("usage", {})
        
        # Collect statistics
        sgas_stats = {}
        try:
            sgas_stats = {
                'graph_stats': app.state.graph_builder.get_graph_statistics(),
                'swap_stats': app.state.swap_manager.get_statistics(),
                'chunking_stats': stats
            }
        except Exception as e:
            logger.warning(f"Failed to collect statistics: {e}")
        
        return ChatResponse(
            response=answer,
            metadata={
                "usage": usage,
                "model_used": settings['vllm']['model_name'],
                "use_rag": request.use_rag,
                "context_chunks_used": len(final_context),
                "iteration": iteration,
                "new_chunks_in_this_iteration": len([
                    c for c in final_context if c['id'] not in excluded_chunks
                ]) if excluded_chunks else len(final_context),
                "total_chunks_explored": stats['used_chunks'],
                "inference_time_sec": inference_time,
                "sgas_statistics": sgas_stats,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"❌ Error in /api/chat: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/api/upload-document")
async def upload_document(file: UploadFile = File(...)):
    """Upload and process a document"""
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
    """List all uploaded documents"""
    documents = [
        {
            "filename": f.name,
            "size": f.stat().st_size,
            "modified": f.stat().st_mtime
        }
        for f in UPLOADS_DIR.iterdir()
        if f.is_file()
    ]
    return {"documents": documents}


@app.delete("/api/documents/{filename}")
async def delete_document(filename: str):
    """Delete a document"""
    file_path = UPLOADS_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Document not found")
    
    try:
        file_path.unlink()
        logger.info(f"✅ Deleted document: {filename}")
        
        # Reset chunker tracking for new document session
        app.state.chunker.reset_usage_tracking()
        app.state.current_iteration = 0
        
        return {"message": f"✅ Document {filename} deleted successfully"}
        
    except Exception as e:
        logger.error(f"❌ Error deleting document: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete document")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{settings['vllm']['api_base']}/models")
            vllm_status = "healthy" if response.status_code == 200 else "unhealthy"
    except Exception:
        vllm_status = "unavailable"
    
    vector_store_status = "ready" if app.state.vector_store is not None else "not_initialized"
    chunker_status = "ready" if app.state.chunker is not None else "not_initialized"
    
    return {
        "status": "healthy",
        "vllm_status": vllm_status,
        "vector_store_status": vector_store_status,
        "chunker_status": chunker_status,
        "model": settings['vllm']['model_name'],
        "current_iteration": app.state.current_iteration,
        "time": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/api/sgas-statistics")
async def get_sgas_statistics():
    """Get S-GAS algorithm statistics"""
    try:
        graph_builder = app.state.graph_builder
        swap_manager = app.state.swap_manager
        chunker = app.state.chunker
        hybrid_scorer = app.state.hybrid_scorer
        
        statistics = {
            "graph": graph_builder.get_graph_statistics() if graph_builder else {},
            "swap": swap_manager.get_statistics() if swap_manager else {},
            "chunking": chunker.get_statistics() if chunker else {},
            "scorer": {
                "alpha": hybrid_scorer.alpha if hybrid_scorer else 0.6,
                "beta": hybrid_scorer.beta if hybrid_scorer else 0.4
            }
        }
        
        return JSONResponse(content=statistics, status_code=200)
        
    except Exception as e:
        logger.error(f"❌ Error retrieving S-GAS statistics: {e}")
        return JSONResponse(
            content={"error": str(e)},
            status_code=500
        )