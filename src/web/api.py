import uuid
import time
from pathlib import Path
from datetime import datetime, timezone
from contextlib import asynccontextmanager
from typing import List, Optional, Dict, Any, Set
#from curses import meta, tparm
#import imp

#from nt import unlink
#from re import search
#from socketserver import _RequestType
#import stat

import httpx
import logging
import shutil
import aiofiles

#from networkx import topological_generations
#from networkx.algorithms.community.quality import require_partition

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

# Configs and modules
from src.config import settings
from src.modules.retrieval import vector_store
from src.modules.retrieval.vector_store import ChromaVectorStore
from src.modules.retrieval.chunking import SemanticChunker
from src.modules.retrieval.document_processor import DocumentProcessor
from src.modules.retrieval.embedder import embedding_manager, get_embeddings
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
        app.state.sessions: Dict[str, Dict[str, Any]] = {}
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
    version="0.1.5",
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

class SessionCreateRequest(BaseModel):
    user_id: Optional[str] = None


class SessionCreateResponse(BaseModel):
    session_id: str
    created_at: str


class SearchRequest(BaseModel):
    query: str
    top_k: int = 10


class SearchResponse(BaseModel):
    query: str
    session_id: str
    results: List[Dict[str, Any]]


class ChatRequest(BaseModel):
    """Chat request with S-GAS parameters"""
    message: str
    use_rag: bool = True
    n_chunks: int = 5
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None


class ChatResponse(BaseModel):
    """Chat response with full S-GAS statistics"""
    response: str
    metadata: Optional[Dict[str, Any]] = None


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
            f"Instructions: Analyze the context and give a brief, precise, full and specific answer based on the fragments provided. "
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
                detail=f"❌ vLLM error {resp.status_code}: {resp.text}"
            )
        
        data = resp.json()
        choices = data.get("choices")
        
        if not isinstance(choices, list) or not choices or "message" not in choices[0]:
            raise HTTPException(
                status_code=502,
                detail=f"❌ vLLM malformed response: {data}"
            )
        
        if "content" not in choices[0]["message"]:
            raise HTTPException(
                status_code=502,
                detail=f"❌ vLLM response without content: {data}"
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
            "version": "0.1.5",
            "endpoints": [
                "/api/session/new",
                "/api/session/{session_id}/upload-document",
                "/api/session/{session_id}/search",
                "/api/session/{session_id}/chat",
                "/api/session/{session_id}/clear",
                "/health",
            ],
            "web_client": "No files in src/web/static/",
        },
        status_code=200,
    )


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
        "active_sessions": len(app.state.sessions),
        "time": datetime.now(timezone.utc).isoformat(),
    }


@app.post("/api/session/new")
async def create_new_seession(request: SessionCreateRequest):
    """Create a new session"""

    session_id = f"session_{uuid.uuid4().hex[:12]}"

    app.state.sessions[session_id] = {
        'iteration': 0,
        'excluded_chunks': [],
        'created_at': datetime.now(timezone.utc).isoformat(),
        'user_id': request.user_id if hasattr(request, 'user_id') else None,
    }
    
    logger.info(f"✅ Created new session: {session_id}")

    return SessionCreateResponse(
        session_id=session_id,
        created_at=datetime.now(timezone.utc).isoformat()
    )


@app.post("/api/session/{session_id}/upload-document")
async def upload_document(
    session_id: str,
    file: UploadFile,
    document_type: str = "general"
):
    """Upload a document to a session"""
    if session_id not in app.state.sessions:
        raise HTTPException(status_code=404, detail="❌ Session not found")

    # Save temporary file
    temp_path = Path(f"/tmp/{file.filename}")
    async with aiofiles.open(temp_path, 'wb') as f:
        await f.write(await file.read())

    # Process document with session_id
    try:
        result = await app.state.document_processor.process_document(
            temp_path,
            session_id=session_id,
            metadata={"document_type": document_type}
        )
        logger.info(f"✅ Document uploaded to session {session_id}: {file.filename}")
    finally:
        # Delete temporary file
        temp_path.unlink()

    return result


@app.post("/api/session/{session_id}/search")
async def search_documents(session_id: str, request: SearchRequest):
    """Search documents in session context"""
    if session_id not in app.state.sessions:
        raise HTTPException(status_code=404, detail="❌ Session not found")

    # Embed query
    query_embeddings = await get_embeddings([request.query])
    query_embedding = query_embeddings[0]

    # Search with session isolation
    results = await app.state.vector_store.search(
        query_embedding,
        session_id=session_id,
        top_k=request.top_k
    )

    return SearchResponse(
        query=request.query,
        session_id=session_id,
        results=results
    )


@app.delete("/api/session/{session_id}/clear")
async def clear_session(session_id: str):
    """Clear a session (delete all chunks and reset state)"""
    if session_id not in app.state.sessions:
        raise HTTPException(status_code=404, detail="❌ Session not found")

    # Delete chunks from vector store
    result = await app.state.vector_store.delete_session_chunks(session_id)

    # Reset chunker tracking
    app.state.chunker.reset_session_tracking(session_id)

    # Remove session from tracking
    app.state.sessions.pop(session_id)

    logger.info(f"✅ Session {session_id} cleared: {result['deleted']} chunks deleted")

    return {
        "status": "success",
        "cleared_chunks": result['deleted'],
        "session_id": session_id
    }

@app.post("/api/session/{session_id}/chat")
async def chat_endpoint(session_id: str, request: ChatRequest):
    """
    Chat with S-GAS adaptive chunking
    
    1. Get excluded chunks from previous iterations (if not the first iteration)
    2. Search for NEW chunks excluding the old ones
    3. Graph analysis and Hybrid reranking
    4. Memory swapping management
    5. Mark used chunks for the next iteration
    """
    
    # ════════════════════════════════════════════════════════════════════════════
    # STEP 1: Initialize iteration and get excluded chunks
    # ════════════════════════════════════════════════════════════════════════════
    
    if session_id not in app.state.sessions:
        raise HTTPException(status_code=404, detail="❌ Session not found")
    
    session_data = app.state.sessions[session_id]
    
    # Increment iteration for this session
    session_data['iteration'] += 1
    iteration = session_data['iteration']
    
    logger.info(f"🔄 Session {session_id} | Iteration {iteration}: Query: {request.message[:50]}...")
    
    # Get chunks excluded in previous iterations using session state
    excluded_chunks = session_data.get('excluded_chunks', [])
    logger.info(f"📌 Excluding {len(excluded_chunks)} chunks from previous iterations")
    
    # ════════════════════════════════════════════════════════════════════════════
    # STEP 2: Search for chunks with exclusion
    # ════════════════════════════════════════════════════════════════════════════
    
    try:
        # Embed query
        query_embeddings = await get_embeddings([request.message])
        query_embedding_vec = query_embeddings[0] if len(query_embeddings.shape) > 1 else query_embeddings
        
        context_chunks = []
        
        if request.use_rag:
            try:
                vector_store = app.state.vector_store
                initial_top_k = max(20, request.n_chunks * 3)
                
                # Search: get more candidates to account for filtering
                if excluded_chunks:
                    search_k = initial_top_k * 2  # Get 2x to have enough after filtering
                else:
                    search_k = initial_top_k
                
                # Pass session_id to search
                candidates = await vector_store.search(
                    query_embedding_vec,
                    session_id=session_id,  # Session isolation
                    top_k=search_k
                )
                
                logger.info(f"📊 Initial search returned {len(candidates)} candidates")
                
                # Filter out previously used chunks (from session history)
                if excluded_chunks:
                    excluded_set = set(excluded_chunks)
                    context_chunks = [c for c in candidates if c.get('id') not in excluded_set]
                    context_chunks = context_chunks[:initial_top_k]
                    logger.info(f"✅ Filtered to {len(context_chunks)} NEW chunks from {len(candidates)} candidates")
                else:
                    context_chunks = candidates[:initial_top_k]
                    logger.info(f"✅ Iteration 1: Got {len(context_chunks)} chunks (no filtering needed)")
                    
            except Exception as e:
                logger.warning(f"❌ RAG search failed: {e}")
                context_chunks = []
        
        # ════════════════════════════════════════════════════════════════════════════
        # STEP 3: S-GAS Algorithm - Graph building and ranking
        # ════════════════════════════════════════════════════════════════════════════
        
        reranked_chunks = context_chunks.copy()
        
        if request.use_rag and len(context_chunks) > 0:
            
            # --- Stage 1: Build knowledge graph from chunks ---
            try:
                logger.info("🔄 Stage 1: Building knowledge graph...")
                
                # Get chunk texts and embeddings for graph building
                chunk_texts = [c.get('text', '') for c in context_chunks]
                chunk_embeddings = await get_embeddings(chunk_texts)
                
                graph_builder = app.state.graph_builder
                if graph_builder is None:
                    raise ValueError("❌ Graph builder not initialized")
                
                # Build graph with semantic relationships
                graph = graph_builder.build_graph(context_chunks, chunk_embeddings)
                
                # Compute distances from query to each chunk via graph
                chunk_ids = [c.get('id', f'chunk_{i}') for i, c in enumerate(context_chunks)]
                graph_distances = graph_builder.compute_graph_distances(
                    request.message,  # Query for semantic relevance
                    chunk_ids
                )
                
                logger.info(f"✅ Graph built: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
                logger.info(f"✅ Graph distances computed for {len(graph_distances)} chunks")
                
            except Exception as e:
                logger.warning(f"❌ Graph building failed: {e}")
                graph_distances = {}
            
            # --- Stage 2: Hybrid reranking (semantic + graph-based) ---
            try:
                logger.info("🔄 Stage 2: Hybrid reranking...")
                
                hybrid_scorer = app.state.hybrid_scorer
                if hybrid_scorer is None:
                    raise ValueError("❌ Hybrid scorer not initialized")
                
                # Ensure we have embeddings
                if 'chunk_embeddings' not in locals() or chunk_embeddings is None:
                    chunk_texts = [c.get('text', '') for c in context_chunks]
                    chunk_embeddings = await get_embeddings(chunk_texts)
                
                # Handle empty graph_distances (use default)
                if not graph_distances:
                    logger.warning("⚠️ Graph distances empty, using default distances")
                    graph_distances = {
                        c.get('id', f'chunk_{i}'): 1000.0
                        for i, c in enumerate(context_chunks)
                    }
                
                # Rerank by combining:
                # - Semantic similarity (query embedding vs chunk embedding)
                # - Graph proximity (distance in knowledge graph)
                reranked_chunks = hybrid_scorer.rerank_chunks(
                    query_embedding_vec,
                    context_chunks,
                    chunk_embeddings,
                    graph_distances,
                    top_k=request.n_chunks
                )
                
                logger.info(f"✅ Reranked to top {request.n_chunks} chunks")
                if reranked_chunks:
                    logger.info(f"   Top chunk: {reranked_chunks[0].get('id', 'unknown')} (sim: {reranked_chunks[0].get('similarity', 'N/A')})")
                
            except Exception as e:
                logger.warning(f"❌ Reranking failed: {e}")
                # Fallback: use chunks as-is if reranking fails
                reranked_chunks = context_chunks[:request.n_chunks]
        
        # ════════════════════════════════════════════════════════════════════════════
        # STEP 4: Memory management and swapping
        # ════════════════════════════════════════════════════════════════════════════
        
        final_context = reranked_chunks if reranked_chunks else context_chunks
        selected_ids = [c.get('id', '') for c in final_context]
        
        try:
            logger.info("🔄 Stage 3: Memory management...")
            
            swap_manager = app.state.swap_manager
            
            # Initialize swap manager on first iteration
            if not app.state._swap_initialized:
                swap_manager.initialize_chunks(final_context)
                app.state._swap_initialized = True
            
            # Update prefetch buffer with current chunks
            swap_manager.update_prefetch_buffer(final_context)
            
            # Calculate current context size
            current_context_tokens = sum(
                c.get('metadata', {}).get('chunk_size', 500)
                for c in final_context
            )
            
            # Decide if swapping is needed
            swap_decision = swap_manager.decide_swap_action(
                final_context,
                current_context_tokens
            )
            
            # Execute swap (GPU ↔ CPU memory transfer if needed)
            swap_manager.execute_swap_decision(swap_decision)
            
            logger.info(f"✅ Memory action: {swap_decision.get('action', 'none')}")
            logger.info(f"   Total tokens in context: {current_context_tokens}")
            
        except Exception as e:
            logger.warning(f"❌ Swap management failed: {e}")
        
        # ════════════════════════════════════════════════════════════════════════════
        # STEP 5: Mark used chunks for next iteration
        # ════════════════════════════════════════════════════════════════════════════
        
        # Mark these chunks as used so they're excluded next iteration
        app.state.chunker.mark_chunks_used(session_id, selected_ids, iteration)
        
        # ✅ UPDATE session_data with used chunks for next iteration!
        session_data['excluded_chunks'].extend(selected_ids)
        
        # Get statistics for response
        chunking_stats = app.state.chunker.get_statistics(session_id)
        new_chunks_count = len([c for c in final_context if c.get('id') not in excluded_chunks]) if excluded_chunks else len(final_context)
        
        logger.info(f"📊 Chunks marked as used: {len(selected_ids)}")
        logger.info(f"📊 Coverage: {chunking_stats['coverage_percent']:.1f}% ({chunking_stats['used_chunks']}/{chunking_stats['total_chunks_in_pool']})")
        
        # ════════════════════════════════════════════════════════════════════════════
        # Inference: Call LLM with prepared context
        # ════════════════════════════════════════════════════════════════════════════
        
        enhanced_prompt = build_prompt(final_context, request.message)
        
        overrides = {
            "temperature": request.temperature,
            "top_p": request.top_p,
            "max_tokens": request.max_tokens,
        }
        
        inference_start = time.time()
        
        try:
            data = await call_vllm_chat(enhanced_prompt, overrides)
            inference_time = time.time() - inference_start
            
            # Record inference time for swap manager statistics
            if app.state.swap_manager is not None:
                app.state.swap_manager.record_compute_time(inference_time)
            
            logger.info(f"✅ Inference complete: {inference_time:.3f}s")
            
        except Exception as e:
            logger.error(f"❌ Inference failed: {e}")
            raise HTTPException(status_code=502, detail="LLM inference failed")
        
        # Extract response
        msg = data.get("choices", [{}])[0].get("message", {})
        answer = msg.get("content", "")
        usage = data.get("usage", {})
        
        # ════════════════════════════════════════════════════════════════════════════
        # Collect detailed statistics for response
        # ════════════════════════════════════════════════════════════════════════════
        
        sgas_stats = {}
        try:
            sgas_stats = {
                'graph_stats': app.state.graph_builder.get_graph_statistics() if app.state.graph_builder else {},
                'swap_stats': app.state.swap_manager.get_statistics() if app.state.swap_manager else {},
                'chunking_stats': chunking_stats
            }
        except Exception as e:
            logger.warning(f"⚠️ Failed to collect statistics: {e}")
        
        # ════════════════════════════════════════════════════════════════════════════
        # Return detailed response
        # ════════════════════════════════════════════════════════════════════════════
        
        return ChatResponse(
            response=answer,
            metadata={
                "usage": usage,
                "model_used": settings['vllm']['model_name'],
                "use_rag": request.use_rag,
                "session_id": session_id,
                "iteration": iteration,
                
                # Chunk statistics
                "context_chunks_used": len(final_context),
                "new_chunks_in_this_iteration": new_chunks_count,
                "chunks_from_previous_iterations": len(excluded_chunks),
                "total_chunks_explored": chunking_stats['used_chunks'],
                "total_chunks_available": chunking_stats['total_chunks_in_pool'],
                "coverage_percent": chunking_stats['coverage_percent'],
                
                # Performance metrics
                "inference_time_sec": inference_time,
                "tokens_generated": usage.get('completion_tokens', 0),
                "tokens_in_context": sum(c.get('metadata', {}).get('chunk_size', 500) for c in final_context),
                
                # S-GAS algorithm statistics
                "sgas_statistics": sgas_stats,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"❌ Error in /api/session/{session_id}/chat: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.get("/api/session/{session_id}/documents")
async def list_session_documents(session_id: str):
    """List documents in session"""
    if session_id not in app.state.sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Get chunks for this session
    all_chunks = app.state.chunker.get_all_chunks(session_id)
    
    # Group by document
    documents = {}
    for chunk in all_chunks:
        doc_name = chunk.get('metadata', {}).get('document_name', 'unknown')
        if doc_name not in documents:
            documents[doc_name] = {
                'name': doc_name,
                'chunks': 0,
                'size': 0
            }
        documents[doc_name]['chunks'] += 1
        documents[doc_name]['size'] += chunk.get('metadata', {}).get('chunk_size', 0)
    
    return {
        "session_id": session_id,
        "total_chunks": len(all_chunks),
        "documents": list(documents.values())
    }


@app.get("/api/sgas-statistics")
async def get_sgas_statistics():
    """Get S-GAS algorithm global statistics"""
    try:
        graph_builder = app.state.graph_builder
        swap_manager = app.state.swap_manager
        hybrid_scorer = app.state.hybrid_scorer

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
        logger.error(f"❌ Error retrieving S-GAS statistics: {e}")
        return JSONResponse(
            content={"error": str(e)},
            status_code=500
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api:app",
        host=settings['api']['host'],
        port=settings['api']['port'],
        reload=True
    )
